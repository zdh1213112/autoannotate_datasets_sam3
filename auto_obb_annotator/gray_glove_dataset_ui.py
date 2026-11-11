"""Dataset-mode integration for the simplified gray-glove editor."""
from __future__ import annotations

import copy
from pathlib import Path
import time

from PyQt5.QtCore import QThread, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QProgressDialog

from .generic_core import sync_candidates
from .gray_glove_dataset import YoloGloveDataset, CLASSES


class DatasetJob(QThread):
    progress_signal = pyqtSignal(int, int, str)

    def __init__(self, task, parent):
        super().__init__(parent)
        self.task = task
        self.result = None
        self.error = None
        self.cancelled = False
        self.last_progress = 0

    def report(self, done, total, stage):
        now = time.monotonic()
        if done in {0, total} or now - self.last_progress > 0.1:
            self.progress_signal.emit(done, total, stage)
            self.last_progress = now

    def run(self):
        try:
            self.result = self.task(self.report, lambda: not self.isInterruptionRequested())
        except InterruptedError:
            self.cancelled = True
        except Exception as exc:
            self.error = str(exc)


class YoloDatasetMixin:
    """Native review projects keep their original code paths and file formats."""

    def run_dataset_job(self, title, task):
        dialog = QProgressDialog(title, '取消', 0, 0, self)
        dialog.setWindowTitle(title)
        dialog.setWindowModality(Qt.ApplicationModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        worker = DatasetJob(task, self)
        self._dataset_job = worker

        def progress(done, total, stage):
            dialog.setRange(0, max(1, total))
            dialog.setValue(done)
            dialog.setLabelText(f'{stage}：{done}/{total}')

        worker.progress_signal.connect(progress)
        worker.finished.connect(dialog.accept)
        dialog.canceled.connect(worker.requestInterruption)
        QTimer.singleShot(0, worker.start)
        dialog.exec_()
        if worker.isRunning():
            worker.requestInterruption()
            worker.wait()
        self._dataset_job = None
        result, error, cancelled = worker.result, worker.error, worker.cancelled
        worker.deleteLater()
        dialog.deleteLater()
        self.activateWindow()
        self.canvas.setFocus()
        self.button_shortcuts.update(None, QApplication.focusWidget())
        if error:
            QMessageBox.warning(self, title + '失败', error)
        elif cancelled:
            self.status.setText(title + '已取消。')
        return None if error or cancelled else result

    def open_review_project(self):
        if self.dataset_busy():
            self.status.setText('请先停止当前任务，再打开项目或数据集。')
            return
        folder = QFileDialog.getExistingDirectory(
            self, '打开复核项目或 YOLO 数据集（也可选择 images / labels）',
            str(self.active_output or self.output_dir()))
        if folder:
            self.open_path(folder)

    def dataset_busy(self):
        return (self._dataset_job is not None or self.export_thread is not None or
                any(worker is not None and worker.isRunning() for worker in (self.thread, self.video_thread)))

    def open_path(self, folder, preferred=None):
        if self.dataset_busy():
            self.status.setText('请先停止当前任务，再打开项目或数据集。')
            return False
        folder = Path(folder).expanduser().resolve()
        if (folder / 'project.json').is_file():
            error = self.review_project_error(folder, validate_records=True)
            if error:
                QMessageBox.warning(self, '无法打开复核项目', error)
                return False
            self._yolo_dataset = None
            self.output_edit.setText(str(folder))
            self.load_project()
            self.set_config_controls(True)
            self.dataset_location.hide()
            self.resume_review_image(preferred)
        else:
            dataset = self.run_dataset_job('打开 YOLO 数据集',
                lambda progress, running: YoloGloveDataset.load(folder, progress, running))
            if dataset is None:
                return False
            self.activate_dataset(dataset, preferred)
        self.remember_review_project()
        return True

    def activate_dataset(self, dataset, preferred=None):
        self._yolo_dataset = dataset
        self._loaded_video_folder = None
        self.video_list.clear()
        self.video_edit.clear()
        self.video_progress.setText('当前编辑 YOLO 数据集；新建检测项目后可添加录像。')
        self.clear_undo_history()
        self._loaded_review_ui_config = None
        self._loaded_review_project_config = None
        self.active_source = dataset.root
        self.active_output = dataset.root
        self.active_config = {'classes': copy.deepcopy(CLASSES)}
        self.input_edit.setText(str(dataset.root / 'images'))
        self.output_edit.setText(str(dataset.root))
        self.roi = [0., 0., 1., 1.]
        self.set_classes(CLASSES)
        self.records = copy.deepcopy(dataset.records)
        self.current_name = None
        self.clear_preview()
        self.table.blockSignals(True)
        self.table.setUpdatesEnabled(False)
        try:
            self.table.setRowCount(0)
            self.row_for_name.clear()
            self._visible_rows = 0
            for record in self.records.values():
                self.update_table(record)
        finally:
            self.table.setUpdatesEnabled(True)
            self.table.blockSignals(False)
        self.search_edit.clear()
        self.review_filter.setChecked(False)
        self.status_filter.setCurrentIndex(self.status_filter.findData('all'))
        self.follow_check.setChecked(False)
        self.format_combo.setCurrentIndex(self.format_combo.findData(dataset.default_format))
        self.format_combo.setToolTip('仅用于导出副本；日常修改按各标签的原格式写回。')
        self.set_config_controls(True)
        self.apply_filter()
        self.update_counts()
        self.resume_review_image(preferred)
        errors = sum(bool(record.get('yolo_error')) for record in self.records.values())
        message = (f'已打开 YOLO 数据集：{len(self.records)} 张。修改自动写回 labels；'
                   '首次修改会备份原标签，Ctrl+Z 可撤销。')
        if errors:
            message += f' {errors} 张标签或图片有错误，已显示原因并禁止覆盖。'
        if dataset.classes_assumed:
            message += ' 未找到类别文件，按 0=left、1=right 读取。'
        self.status.setText(message)
        self.dataset_location.setText(f'YOLO 数据集 · {dataset.root} · 修改自动写回标签')
        self.dataset_location.setToolTip(str(dataset.root))
        self.dataset_location.show()

    def load_project(self):
        if self._yolo_dataset is None:
            return super().load_project()

    def set_config_controls(self, enabled):
        super().set_config_controls(enabled)
        dataset_mode = self._yolo_dataset is not None
        if hasattr(self, 'settings_group'):
            self.settings_group.setEnabled(not dataset_mode)
        if hasattr(self, 'video_group'):
            self.video_group.setEnabled(not dataset_mode)
        if dataset_mode:
            self.start_button.setEnabled(False)
            self.recheck_review_button.setEnabled(False)
            self.start_button.setToolTip('当前直接编辑已有 YOLO 标签；打开原复核项目可继续 SAM3 检测。')
        elif enabled:
            self.start_button.setEnabled(True)
            self.start_button.setToolTip('开始或继续原项目的自动检测')

    def start_detection(self):
        if self._yolo_dataset is None:
            return super().start_detection()

    def recheck_review_images(self):
        if self._yolo_dataset is None:
            return super().recheck_review_images()

    def import_video(self, detect_after=False):
        if self._yolo_dataset is None:
            return super().import_video(detect_after)

    def update_table(self, record):
        super().update_table(record)
        row = self.row_for_name.get(record['name'])
        if row is not None:
            self.table.item(row, 0).setToolTip(record['name'])
            self.table.item(row, 1).setToolTip(record.get('reason', ''))

    def record_image_available(self, name):
        if self._yolo_dataset is not None:
            return name in self._yolo_dataset.labels and (self.active_source / name).is_file()
        return super().record_image_available(name)

    def resume_review_image(self, preferred=None):
        if self._yolo_dataset is None:
            return super().resume_review_image(preferred)
        names = ([preferred] if preferred in self.records else []) + list(self.records)
        if names:
            row = self.row_for_name.get(names[0])
            if row is not None:
                self.table.selectRow(row)
                self.table.scrollToItem(self.table.item(row, 0))
            self.show_record(names[0])

    def current_record(self):
        record = super().current_record()
        if self._yolo_dataset is not None and record and record.get('yolo_error'):
            QMessageBox.warning(self, '原标签无法读取',
                record['yolo_error'] + '\n为避免丢失标注，本张暂不可修改；修正原文件后重新打开。')
            return None
        return record

    def save_review(self, record):
        if self._yolo_dataset is None:
            return super().save_review(record)
        dataset = self._yolo_dataset
        record = sync_candidates(copy.deepcopy(record))
        name = record['name']
        previous = copy.deepcopy(self.records[name])
        objects = record.get('objects', [])
        classes = [obj['class_id'] for obj in objects]
        if not objects:
            record['side_pending'] = False
        elif len(objects) == 1 and not previous.get('objects'):
            record['side_pending'] = record.get('reason') not in {'人工指定左右类别', '人工修正类别'}
        if record.get('reason') in {'人工指定左右类别', '人工修正类别'}:
            record['side_pending'] = False
        if len(objects) > 2 or len(set(classes)) != len(classes):
            record.update(status='review', reason='请核对左右类别：left/right 各最多一个')
        elif record.get('side_pending'):
            record.update(status='review', reason='请按 1 / 2 确认单手类别')
        before_bytes = dataset.expected[name]
        try:
            # Confirming an untouched import preserves its exact text precision.
            data = before_bytes if objects == previous.get('objects', []) and before_bytes is not None else dataset.encode(record)
            warning = dataset.write(record, data)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, '保存失败', f'本次修改未保存：{exc}')
            return False
        if previous != record or before_bytes != data:
            self._undo_stack.append({'name': name, 'before': previous, 'after': copy.deepcopy(record),
                                     'label_before': before_bytes, 'label_after': data,
                                     'output': self.active_output, 'source': self.active_source})
        self.records[name] = record
        self._update_undo_button()
        self.update_table(record)
        self.show_record(name)
        self.update_counts()
        self.status.setText(warning or f'已写回标签：{dataset.labels[name].relative_to(dataset.root)} · Ctrl+Z 撤销')
        return True

    def undo_last_review(self):
        if self._yolo_dataset is None:
            return super().undo_last_review()
        if not self._undo_stack:
            self.status.setText('当前没有可撤销的人工标注操作。')
            return
        entry = self._undo_stack[-1]
        name = entry['name']
        if self.active_output != entry['output'] or self.records.get(name) != entry['after']:
            self.clear_undo_history()
            self.status.setText('项目或标注已改变，旧的撤销历史已失效。')
            return
        record = copy.deepcopy(entry['before'])
        try:
            warning = self._yolo_dataset.write(record, entry['label_before'])
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, '撤销失败', str(exc))
            return
        self._undo_stack.pop()
        self._update_undo_button()
        self.records[name] = record
        self.update_table(record)
        self.apply_filter()
        row = self.row_for_name[name]
        if self.table.isRowHidden(row):
            self.search_edit.clear()
            self.review_filter.setChecked(False)
            self.status_filter.setCurrentIndex(self.status_filter.findData('all'))
        self.table.selectRow(row)
        self.table.scrollToItem(self.table.item(row, 0))
        self.show_record(name)
        self.update_counts()
        self.status.setText(warning or '已撤销，原 YOLO 标签文件已同步恢复。')

    def export_dataset_copy(self, split):
        if self.dataset_busy():
            return
        if split:
            for record in self.records.values():
                if record['status'] not in {'auto', 'confirmed', 'empty_confirmed'}:
                    continue
                classes = [obj['class_id'] for obj in record.get('objects', [])]
                if record.get('side_pending') or len(classes) > 2 or len(classes) != len(set(classes)):
                    QMessageBox.warning(self, '导出前需复核', record['name'] + ' 的左右类别仍有冲突。')
                    return
        output = QFileDialog.getExistingDirectory(self, '选择导出副本的存放目录', str(self.active_output.parent))
        if not output:
            return
        records = copy.deepcopy(self.records)
        dataset = self._yolo_dataset
        box_format = self.format_combo.currentData()
        result = self.run_dataset_job('导出数据集副本', lambda progress, running:
            dataset.export(output, records, box_format, split, progress, running))
        if result is not None:
            target, images, objects = result
            self.status.setText(f'副本已导出：{target} · 有框图片 {images} 张，共 {objects} 个框')
            QMessageBox.information(self, '导出完成', str(target))

    def export(self):
        if self._yolo_dataset is None:
            return super().export()
        return self.export_dataset_copy(split=True)

    def export_all_dataset(self):
        if self._yolo_dataset is None:
            return super().export_all_dataset()
        return self.export_dataset_copy(split=False)
