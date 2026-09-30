"""Dedicated single-barcode annotation and review window for D435 images."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PyQt5.QtCore import QCoreApplication, QLibraryInfo, QSettings, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QShortcut, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .app import RotatableROILabel, SAM3TextSegmenter, torch
from .barcode_core import (
    append_record, detect_barcode, export_dataset, image_files, load_records,
    normalize_roi, polygon_in_roi, roi_pixels,
)
from .project_paths import SAM3_MODEL


DEFAULT_INPUT = Path("/home/p3/yolo_train/raw_data/D435_barcode_0923/color")
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "barcode_annotations"
STATUS_TEXT = {"auto": "自动候选", "review": "待人工复核",
               "confirmed": "人工确认", "empty_confirmed": "人工确认无目标"}


def _pixmap(image_bgr) -> QPixmap:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, width, height, width * 3,
                                    QImage.Format_RGB888).copy())


class SingleBoxDialog(QDialog):
    """Edit one rotated rectangle; also used to select the fixed search ROI."""

    def __init__(self, image_path: Path, title: str, initial=None, roi_mode=False,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.roi_mode = roi_mode
        self.result_polygon = None
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        self.height, self.width = image.shape[:2]
        screen = QApplication.primaryScreen().availableGeometry()
        max_width = min(1100, max(450, screen.width() - 100))
        max_height = min(620, max(270, screen.height() - 210))
        self.scale = min(1.0, max_width / self.width, max_height / self.height)
        display = cv2.resize(
            image, (round(self.width * self.scale), round(self.height * self.scale)),
            interpolation=cv2.INTER_AREA,
        )

        layout = QVBoxLayout(self)
        self.canvas = RotatableROILabel()
        self.canvas.setPixmap(_pixmap(display))
        self.canvas.setFixedSize(display.shape[1], display.shape[0])
        layout.addWidget(self.canvas, alignment=Qt.AlignCenter)
        if initial is not None:
            points = np.asarray(initial, dtype=np.float32) * self.scale
            (cx, cy), (width, height), angle = cv2.minAreaRect(points)
            self.canvas._cx, self.canvas._cy = cx, cy
            self.canvas._w, self.canvas._h = width, height
            self.canvas._angle = angle
            self.canvas._has_pending = True
            self.canvas.update()

        hint = (
            "拖动鼠标框选区域；已有框可拖动、拉控制点。只保存一个框。"
            if roi_mode else
            "拖动鼠标重画框；拉白点调大小，拖橙色点旋转。Tab 可重新编辑已确认框；只保存一个框。"
        )
        layout.addWidget(QLabel(hint))
        buttons = QHBoxLayout()
        clear = QPushButton("清除框")
        clear.clicked.connect(self.canvas.clear_rois)
        save = QPushButton("保存这个框")
        save.clicked.connect(self.accept)
        cancel = QPushButton("取消")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(clear)
        buttons.addStretch()
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout.addLayout(buttons)

    def accept(self):
        self.canvas.confirm_pending()
        if len(self.canvas.rois) != 1:
            QMessageBox.warning(self, "需要一个框", "请只保留一个框，然后保存。")
            return
        cx, cy, width, height, angle = self.canvas.rois[0]
        points = cv2.boxPoints(((cx, cy), (width, height), angle)) / self.scale
        points[:, 0] = np.clip(points[:, 0], 0, self.width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, self.height - 1)
        if self.roi_mode:
            lower = points.min(axis=0)
            upper = points.max(axis=0)
            self.result_polygon = [[float(lower[0]), float(lower[1])],
                                   [float(upper[0]), float(lower[1])],
                                   [float(upper[0]), float(upper[1])],
                                   [float(lower[0]), float(upper[1])]]
        else:
            self.result_polygon = points.round(2).tolist()
        super().accept()


class BarcodeThread(QThread):
    result_signal = pyqtSignal(object)
    progress_signal = pyqtSignal(int, int)
    message_signal = pyqtSignal(str)

    def __init__(self, files: list[Path], roi, done_names: set[str]):
        super().__init__()
        self.files = files
        self.roi = roi
        self.done_names = done_names
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        try:
            todo = [path for path in self.files if path.name not in self.done_names]
            total = len(self.files)
            completed = total - len(todo)
            self.progress_signal.emit(completed, total)
            if not todo:
                self.message_signal.emit("所有图片已有检测结果；可继续人工复核或导出。")
                return
            device = "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"
            self.message_signal.emit(f"正在加载 SAM3（{device}），本次检测 {len(todo)} 张...")
            segmenter = SAM3TextSegmenter(
                model_path=str(SAM3_MODEL), device=device, conf=0.15)
            for path in todo:
                if not self.running:
                    break
                try:
                    image = cv2.imread(str(path))
                    if image is None:
                        raise ValueError("图片无法解码")
                    result = detect_barcode(image, self.roi, segmenter)
                    result.update(name=path.name, width=image.shape[1],
                                  height=image.shape[0])
                except Exception as exc:
                    result = {"name": path.name, "polygon": None,
                              "score": 0.0, "status": "review",
                              "reason": f"检测失败: {exc}", "candidates": 0}
                self.result_signal.emit(result)
                completed += 1
                self.progress_signal.emit(completed, total)
            self.message_signal.emit("检测已停止。" if not self.running else "检测完成，请复核待处理图片。")
        except Exception as exc:
            self.message_signal.emit(f"检测无法启动: {exc}")


class BarcodeAnnotatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("D435 单条形码标注与人工复核")
        self.resize(1400, 900)
        old_settings = QSettings("AutoOBBAnnotator", "AutoOBBAnnotator")
        self.roi = None
        if (old_settings.contains("workspace_roi") and
                old_settings.value("workspace_roi_enabled", False, type=bool)):
            self.roi = normalize_roi(
                str(old_settings.value("workspace_roi")).split(","))
        self.records: dict[str, dict] = {}
        self.row_for_name: dict[str, int] = {}
        self.current_name = None
        self.active_source = None
        self.active_output = None
        self.thread = None
        self.preview_pixmap = None

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        settings = QGroupBox("数据与标注区域")
        grid = QGridLayout(settings)
        self.input_edit = QLineEdit(str(DEFAULT_INPUT))
        self.output_edit = QLineEdit(str(DEFAULT_OUTPUT))
        self.path_buttons = []
        for row, (label, edit, chooser) in enumerate((
            ("图片目录", self.input_edit, self.choose_input),
            ("结果目录", self.output_edit, self.choose_output),
        )):
            button = QPushButton("浏览")
            button.clicked.connect(chooser)
            self.path_buttons.append(button)
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(button, row, 2)
        self.roi_label = QLabel("尚未设置标注区域；点击右侧按钮在样图上框选")
        self.roi_button = QPushButton("框选/修改标注区域")
        self.roi_button.clicked.connect(self.choose_roi)
        grid.addWidget(self.roi_label, 2, 0, 1, 2)
        grid.addWidget(self.roi_button, 2, 2)
        outer.addWidget(settings)

        controls = QHBoxLayout()
        self.start_button = QPushButton("开始/继续自动检测")
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_detection)
        self.follow_check = QCheckBox("实时跟随每张结果")
        self.follow_check.setChecked(True)
        self.format_combo = QComboBox()
        self.format_combo.addItem("YOLO-OBB 旋转框", "obb")
        self.format_combo.addItem("YOLO-HBB 普通框", "hbb")
        self.export_button = QPushButton("导出训练数据集")
        self.export_button.clicked.connect(self.export)
        for widget in (self.start_button, self.stop_button, self.follow_check,
                       self.format_combo, self.export_button):
            controls.addWidget(widget)
        outer.addLayout(controls)

        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["图片", "状态", "置信度", "原因"])
        self.table.setColumnWidth(0, 115)
        self.table.setColumnWidth(1, 120)
        self.table.setColumnWidth(2, 75)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemSelectionChanged.connect(self.select_table_row)
        left_layout.addWidget(self.table)
        filter_row = QHBoxLayout()
        self.review_filter = QCheckBox("只看待复核")
        self.review_filter.toggled.connect(self.apply_filter)
        next_button = QPushButton("下一张待复核 [D]")
        next_button.clicked.connect(self.next_review)
        filter_row.addWidget(self.review_filter)
        filter_row.addWidget(next_button)
        left_layout.addLayout(filter_row)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.preview_title = QLabel("检测开始后，这里会逐张显示标注结果")
        self.preview_title.setWordWrap(True)
        self.preview = QLabel("等待图片")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(650, 400)
        self.preview.setStyleSheet("background:#202020;color:white;")
        right_layout.addWidget(self.preview_title)
        right_layout.addWidget(self.preview, 1)
        edit_row = QHBoxLayout()
        edit_button = QPushButton("修改/补画旋转框")
        edit_button.clicked.connect(self.edit_current)
        confirm_button = QPushButton("确认当前框 [Space]")
        confirm_button.clicked.connect(self.confirm_current)
        next_button = QPushButton("下一张")
        next_button.clicked.connect(self.show_next_image)
        empty_button = QPushButton("确认无条形码 [Q]")
        empty_button.clicked.connect(self.mark_empty)
        for widget in (edit_button, confirm_button, next_button, empty_button):
            edit_row.addWidget(widget)
        right_layout.addLayout(edit_row)
        splitter.addWidget(right)
        splitter.setSizes([440, 960])
        outer.addWidget(splitter, 1)

        self.progress = QProgressBar()
        outer.addWidget(self.progress)
        self.status = QLabel(
            "请先核对主程序保存的标注区域，再开始检测。" if self.roi else
            "先框选标注区域，再开始检测。低置信度和多候选图片会进入待复核列表。")
        outer.addWidget(self.status)
        self.show_roi()
        self.load_project()

        self.shortcuts = []
        for key, action in ((Qt.Key_Space, self.confirm_current),
                            (Qt.Key_D, self.next_review),
                            (Qt.Key_Q, self.mark_empty)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(action)
            self.shortcuts.append(shortcut)
        QApplication.instance().focusChanged.connect(self.update_shortcuts)

    def update_shortcuts(self, _old, current):
        # Keep ordinary typing in the input/output path fields available.
        enabled = not (isinstance(current, QLineEdit) and self.isAncestorOf(current))
        for shortcut in self.shortcuts:
            shortcut.setEnabled(enabled)

    def source_dir(self) -> Path:
        return Path(self.input_edit.text().strip()).expanduser().resolve()

    def output_dir(self) -> Path:
        return Path(self.output_edit.text().strip()).expanduser().resolve()

    def choose_input(self):
        path = QFileDialog.getExistingDirectory(self, "选择图片目录", str(self.source_dir()))
        if path:
            self.input_edit.setText(path)

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择结果目录", str(self.output_dir()))
        if path:
            self.output_edit.setText(path)
            self.load_project()

    def choose_roi(self):
        files = image_files(self.source_dir()) if self.source_dir().is_dir() else []
        if not files:
            QMessageBox.warning(self, "没有图片", "请先选择包含图片的目录。")
            return
        initial = None
        image = cv2.imread(str(files[0]))
        if image is None:
            QMessageBox.warning(self, "无法读取", str(files[0]))
            return
        if self.roi is not None:
            x1, y1, x2, y2 = roi_pixels(self.roi, image.shape[1], image.shape[0])
            initial = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        dialog = SingleBoxDialog(files[0], "框选所有图片共用的标注区域", initial,
                                 roi_mode=True, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            points = np.asarray(dialog.result_polygon)
            self.roi = normalize_roi([
                points[:, 0].min() / image.shape[1],
                points[:, 1].min() / image.shape[0],
                points[:, 0].max() / image.shape[1],
                points[:, 1].max() / image.shape[0],
            ])
            self.show_roi()

    def show_roi(self):
        if self.roi is None:
            self.roi_label.setText("尚未设置标注区域")
        else:
            self.roi_label.setText("标注区域比例：" + ", ".join(f"{value:.3f}" for value in self.roi))

    def load_project(self):
        config_path = self.output_dir() / "project.json"
        if not config_path.exists():
            self.active_source = None
            self.active_output = None
            self.records.clear()
            self.table.setRowCount(0)
            self.row_for_name.clear()
            self.current_name = None
            self.clear_preview()
            return
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.input_edit.setText(config["source_dir"])
            self.roi = normalize_roi(config["roi"])
            if self.roi is None:
                raise ValueError("项目中的标注区域无效")
            self.show_roi()
            self.active_source = Path(config["source_dir"])
            self.active_output = self.output_dir()
            self.records = load_records(self.output_dir() / "review.jsonl")
            self.table.setRowCount(0)
            self.row_for_name.clear()
            self.current_name = None
            self.clear_preview()
            for record in sorted(self.records.values(), key=lambda item: item["name"]):
                self.update_table(record)
            self.update_counts()
        except (OSError, ValueError, KeyError) as exc:
            self.status.setText(f"读取旧项目失败: {exc}")

    def ensure_project(self):
        source, output = self.source_dir(), self.output_dir()
        if self.roi is None:
            raise ValueError("请先在样图上框选标注区域")
        if not source.is_dir():
            raise ValueError(f"图片目录不存在: {source}")
        if output == source or output.is_relative_to(source):
            raise ValueError("结果目录不能位于图片目录内部")
        config_path = output / "project.json"
        config = {"source_dir": str(source), "roi": self.roi}
        if config_path.exists():
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous != config:
                raise ValueError("该结果目录已有其他图片目录或标注区域；请换一个新结果目录")
            self.records = load_records(output / "review.jsonl")
        else:
            if (output / "review.jsonl").exists():
                raise ValueError("结果目录有 review.jsonl 但缺少 project.json，请换目录")
            output.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
            self.records = {}
        if self.active_output != output:
            self.current_name = None
            self.table.setRowCount(0)
            self.row_for_name.clear()
            self.clear_preview()
        self.active_source = source
        self.active_output = output

    def start_detection(self):
        try:
            self.ensure_project()
            files = image_files(self.source_dir())
            if not files:
                raise ValueError("图片目录为空")
            if not SAM3_MODEL.is_file():
                raise FileNotFoundError(f"找不到 SAM3 模型: {SAM3_MODEL}")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法开始", str(exc))
            return
        self.table.setRowCount(0)
        self.row_for_name.clear()
        for record in sorted(self.records.values(), key=lambda item: item["name"]):
            self.update_table(record)
        self.thread = BarcodeThread(files, self.roi, set(self.records))
        self.thread.result_signal.connect(self.receive_result)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.message_signal.connect(self.status.setText)
        self.thread.finished.connect(self.detection_finished)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.roi_button.setEnabled(False)
        self.input_edit.setEnabled(False)
        self.output_edit.setEnabled(False)
        for button in self.path_buttons:
            button.setEnabled(False)
        self.thread.start()

    def stop_detection(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.status.setText("正在停止；当前图片完成后退出。")

    def detection_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.export_button.setEnabled(True)
        self.roi_button.setEnabled(True)
        self.input_edit.setEnabled(True)
        self.output_edit.setEnabled(True)
        for button in self.path_buttons:
            button.setEnabled(True)
        self.update_counts()
        self.thread = None

    def receive_result(self, record):
        append_record(self.active_output / "review.jsonl", record)
        self.records[record["name"]] = record
        self.update_table(record)
        self.update_counts()
        if self.follow_check.isChecked():
            self.table.blockSignals(True)
            self.table.selectRow(self.row_for_name[record["name"]])
            self.table.blockSignals(False)
            self.show_record(record["name"])

    def update_progress(self, done, total):
        self.progress.setMaximum(total)
        self.progress.setValue(done)

    def update_table(self, record):
        name = record["name"]
        if name not in self.row_for_name:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.row_for_name[name] = row
        row = self.row_for_name[name]
        values = [name, STATUS_TEXT.get(record["status"], record["status"]),
                  f"{record.get('score', 0):.2f}", record.get("reason", "")]
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column == 1 and record["status"] == "review":
                item.setForeground(QColor("#bf5b00"))
            self.table.setItem(row, column, item)
        self.table.setRowHidden(row, self.review_filter.isChecked() and
                                record["status"] != "review")

    def apply_filter(self):
        if self.review_filter.isChecked():
            self.follow_check.setChecked(False)
        for name, row in self.row_for_name.items():
            self.table.setRowHidden(row, self.review_filter.isChecked() and
                                    self.records[name]["status"] != "review")

    def select_table_row(self):
        row = self.table.currentRow()
        if row < 0 or self.table.item(row, 0) is None:
            return
        name = self.table.item(row, 0).text()
        if name != self.current_name:
            self.follow_check.setChecked(False)
            self.show_record(name)

    def show_record(self, name):
        record = self.records.get(name)
        if record is None:
            return
        self.current_name = name
        image = cv2.imread(str(self.active_source / name))
        if image is None:
            self.preview.setText("无法读取图片")
            return
        x1, y1, x2, y2 = roi_pixels(self.roi, image.shape[1], image.shape[0])
        cv2.rectangle(image, (x1, y1), (x2, y2), (30, 30, 230), 2)
        polygon = record.get("polygon")
        if polygon is not None:
            points = np.rint(polygon).astype(np.int32)
            color = (0, 210, 0) if record["status"] in {"auto", "confirmed"} else (0, 180, 255)
            cv2.polylines(image, [points], True, color, 3)
        self.preview_pixmap = _pixmap(image)
        self.refresh_preview()
        self.preview_title.setText(
            f"{name}  |  {STATUS_TEXT.get(record['status'], record['status'])}  |  "
            f"置信度 {record.get('score', 0):.2f}  |  {record.get('reason', '')}"
        )

    def clear_preview(self):
        self.preview_pixmap = None
        self.preview.setPixmap(QPixmap())
        self.preview.setText("等待图片")
        self.preview_title.setText("检测开始后，这里会逐张显示标注结果")

    def refresh_preview(self):
        if self.preview_pixmap:
            self.preview.setPixmap(self.preview_pixmap.scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_preview()

    def current_record(self):
        if self.current_name is None or self.current_name not in self.records:
            QMessageBox.information(self, "先选图片", "请在左侧选一张图片。")
            return None
        return self.records[self.current_name]

    def save_review(self, record):
        append_record(self.active_output / "review.jsonl", record)
        self.records[record["name"]] = record
        self.update_table(record)
        self.show_record(record["name"])
        self.update_counts()

    def edit_current(self):
        record = self.current_record()
        if record is None:
            return
        self.follow_check.setChecked(False)
        path = self.active_source / record["name"]
        dialog = SingleBoxDialog(path, f"人工修正：{record['name']}",
                                 record.get("polygon"), parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        image = cv2.imread(str(path))
        if not polygon_in_roi(dialog.result_polygon, self.roi, image.shape[1], image.shape[0]):
            QMessageBox.warning(self, "框超出区域", "条形码框中心须在标注区域内，且至少一半面积在区域内。")
            return
        self.save_review({**record, "polygon": dialog.result_polygon,
                          "status": "confirmed", "reason": "人工修正的唯一条形码"})

    def confirm_current(self):
        record = self.current_record()
        if record is None:
            return
        if record.get("polygon") is None:
            QMessageBox.information(self, "尚无框", "请先补画条形码框，或确认无条形码。")
            return
        self.save_review({**record, "status": "confirmed", "reason": "人工确认的唯一条形码"})

    def mark_empty(self):
        record = self.current_record()
        if record is not None:
            self.save_review({**record, "polygon": None, "status": "empty_confirmed",
                              "reason": "人工确认区域内无可见条形码"})

    def show_next_image(self):
        if not self.row_for_name:
            self.status.setText("还没有可查看的图片。")
            return
        start = self.row_for_name.get(self.current_name, -1)
        for row in range(start + 1, self.table.rowCount()):
            if self.table.isRowHidden(row):
                continue
            name = self.table.item(row, 0).text()
            self.follow_check.setChecked(False)
            self.table.selectRow(row)
            self.show_record(name)
            return
        self.status.setText("已经是最后一张图片。")

    def next_review(self):
        if not self.row_for_name:
            return
        count = self.table.rowCount()
        start = self.row_for_name.get(self.current_name, -1)
        for offset in range(1, count + 1):
            row = (start + offset) % count
            name = self.table.item(row, 0).text()
            if self.records[name]["status"] == "review":
                self.table.selectRow(row)
                self.show_record(name)
                return
        self.status.setText("没有待复核图片。")

    def update_counts(self):
        counts = {status: 0 for status in STATUS_TEXT}
        for record in self.records.values():
            counts[record["status"]] = counts.get(record["status"], 0) + 1
        self.status.setText(
            f"已处理 {len(self.records)} 张；自动候选 {counts['auto']}，"
            f"待复核 {counts['review']}，人工确认 {counts['confirmed']}，"
            f"确认无目标 {counts['empty_confirmed']}。"
        )

    def export(self):
        try:
            self.ensure_project()
            target, positive, negative = export_dataset(
                self.source_dir(), self.output_dir(), self.records,
                self.format_combo.currentData())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        unresolved = sum(record["status"] == "review" for record in self.records.values())
        self.status.setText(
            f"已导出 {positive} 张条形码图、{negative} 张确认无目标图；"
            f"{unresolved} 张待复核未导出。目录：{target}"
        )
        QMessageBox.information(self, "导出完成", f"数据集：{target}\n待复核未导出：{unresolved} 张")

    def closeEvent(self, event):
        if self.thread:
            if self.thread.isRunning():
                self.thread.stop()
                self.thread.wait()
            QApplication.processEvents()
        super().closeEvent(event)


def main() -> int:
    # cv2 changes this environment variable to its own incompatible Qt build.
    pyqt_plugin_root = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyqt_plugin_root
    QCoreApplication.setLibraryPaths([pyqt_plugin_root])
    app = QApplication(sys.argv)
    window = BarcodeAnnotatorApp()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
