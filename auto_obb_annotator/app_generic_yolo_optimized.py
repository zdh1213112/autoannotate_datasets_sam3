"""Usability-focused copy of the YOLO detection annotation window."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import cv2
from PyQt5.QtCore import QCoreApplication, QLibraryInfo, Qt, QTimer
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QScrollArea, QShortcut, QSpinBox,
    QVBoxLayout,
)
from ultralytics import YOLO

from .app import torch
from .app_generic import GenericAnnotatorApp, GenericThread
from .generic_core import detect_generic
from .project_paths import ROOT_DIR, resolve_path


DEFAULT_YOLO_MODEL = ROOT_DIR / "models" / "best_top_barcode_0929.pt"
DEFAULT_YOLO_OUTPUT = ROOT_DIR / "data" / "generic_yolo_optimized_annotations"


class LargePreviewDialog(QDialog):
    """Inspect the rendered detection image at its native pixel resolution."""

    def __init__(self, pixmap, name: str, parent=None):
        super().__init__(parent)
        self.source_pixmap = pixmap
        self.setWindowTitle(f"放大查看检测图：{name}")
        self.resize(1100, 760)
        layout = QVBoxLayout(self)
        toolbar = QHBoxLayout()
        toolbar.addWidget(QLabel("缩放"))
        self.zoom_spin = QSpinBox()
        self.zoom_spin.setRange(25, 400)
        self.zoom_spin.setSuffix("%")
        self.zoom_spin.setValue(100)
        self.zoom_spin.valueChanged.connect(self.update_zoom)
        toolbar.addWidget(self.zoom_spin)
        fit_button = QPushButton("适合窗口")
        fit_button.clicked.connect(self.fit_to_window)
        toolbar.addWidget(fit_button)
        native_button = QPushButton("100% 原尺寸")
        native_button.clicked.connect(lambda: self.zoom_spin.setValue(100))
        toolbar.addWidget(native_button)
        toolbar.addStretch()
        layout.addLayout(toolbar)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.image_label)
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.scroll)
        self.update_zoom()
        QTimer.singleShot(0, self.fit_to_window)

    def update_zoom(self):
        scale = self.zoom_spin.value() / 100
        target = self.source_pixmap.size() * scale
        self.image_label.setPixmap(self.source_pixmap.scaled(
            target, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.image_label.resize(self.image_label.pixmap().size())

    def fit_to_window(self):
        viewport = self.scroll.viewport().size()
        if self.source_pixmap.width() and self.source_pixmap.height():
            scale = min(viewport.width() / self.source_pixmap.width(),
                        viewport.height() / self.source_pixmap.height())
            self.zoom_spin.setValue(max(25, min(400, int(scale * 100))))


def model_classes(model_path: Path) -> list[dict]:
    """Read the fixed class IDs in a YOLO detection checkpoint."""
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 YOLO 检测权重: {model_path}")
    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        raise ValueError(f"无法加载 YOLO 检测权重 {model_path}: {exc}") from exc
    if model.task != "detect":
        raise ValueError(f"仅支持 YOLO 普通目标检测权重；当前权重任务为 {model.task}")
    names = model.names
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    elif isinstance(names, dict):
        names = {int(index): name for index, name in names.items()}
    else:
        raise ValueError("YOLO 权重没有可读取的类别名称")
    if not names or sorted(names) != list(range(len(names))):
        raise ValueError("YOLO 权重的类别编号不是从 0 连续排列")
    return [{"name": str(names[index]), "prompt": str(names[index])}
            for index in range(len(names))]


class GenericYOLODetector:
    """Expose YOLO boxes through the existing generic selection pipeline."""

    def __init__(self, model_path: str, device: str, conf: float):
        self.model_path = str(Path(model_path).resolve())
        self.model = YOLO(self.model_path)
        if self.model.task != "detect":
            raise ValueError(f"仅支持 YOLO 普通目标检测权重；当前权重任务为 {self.model.task}")
        self.device = device
        self.conf = float(conf)

    def detect_many(self, image_bgr, _prompts: list[str]) -> list[dict]:
        results = self.model.predict(source=image_bgr, device=self.device,
                                     conf=self.conf, verbose=False)
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        coordinates = boxes.xyxy.detach().cpu().numpy().tolist()
        scores = boxes.conf.detach().cpu().numpy().tolist()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int).tolist()
        return [{"class_id": class_id, "box": box, "score": score}
                for class_id, box, score in zip(class_ids, coordinates, scores)]


class GenericYOLOThread(GenericThread):
    def run(self):
        try:
            todo = [path for path in self.files if path.name not in self.done_names]
            total = len(self.files)
            completed = total - len(todo)
            self.progress_signal.emit(completed, total)
            if not todo:
                self.message_signal.emit("所有抽中图片已有检测结果；可继续复核或导出。")
                return
            device = "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"
            model_path = str(Path(self.config["model_path"]).resolve())
            if (self.segmenter is None or self.segmenter.device != device or
                    self.segmenter.model_path != model_path):
                self.message_signal.emit(f"正在加载 YOLO 检测权重（{device}），本次检测 {len(todo)} 张...")
                self.segmenter = GenericYOLODetector(
                    model_path, device, self.config["confidence"])
            else:
                self.segmenter.conf = float(self.config["confidence"])
                self.message_signal.emit(f"复用已加载的 YOLO 权重（{device}），本次检测 {len(todo)} 张...")
            for path in todo:
                if not self.running:
                    break
                try:
                    image = cv2.imread(str(path))
                    if image is None:
                        raise ValueError("图片无法解码")
                    result = detect_generic(
                        image, self.config["roi"], self.segmenter,
                        self.config["classes"], self.config["count_mode"],
                        self.config["target_count"],
                        self.config.get("review_reasons"),
                        self.config.get("selection_rule"),
                    )
                    result.update(name=path.name, width=image.shape[1],
                                  height=image.shape[0])
                except Exception as exc:
                    result = {"name": path.name, "objects": [], "candidates": [],
                              "status": "review", "reason": f"检测失败: {exc}",
                              "candidate_count": 0}
                self.result_signal.emit(result)
                completed += 1
                self.progress_signal.emit(completed, total)
            self.message_signal.emit(
                "检测已停止。" if not self.running else "检测完成，请复核待处理图片。")
        except Exception as exc:
            self.message_signal.emit(f"YOLO 检测无法启动: {exc}")


class OptimizedYOLOAnnotatorApp(GenericAnnotatorApp):
    def __init__(self):
        self._building_yolo_window = True
        self._class_model_path = None
        self._class_model_mtime = None
        self._run_started_at = None
        self._run_start_done = None
        super().__init__()
        self._building_yolo_window = False
        self.setWindowTitle("YOLO 检测标注与复核 · 优化版")
        self.model_path_edit = QLineEdit(str(DEFAULT_YOLO_MODEL))
        self.model_path_edit.editingFinished.connect(self.sync_model_classes)
        self.model_button = QPushButton("选择 YOLO 权重 [Alt+M]")
        self.model_button.clicked.connect(self.choose_model)
        self.model_shortcut = QShortcut(QKeySequence("Alt+M"), self)
        self.model_shortcut.setContext(Qt.WindowShortcut)
        self.model_shortcut.activated.connect(self.model_button.click)
        settings = self.centralWidget().layout().itemAt(0).widget()
        grid = settings.layout()
        grid.addWidget(QLabel("检测模型权重"), 8, 0)
        grid.addWidget(self.model_path_edit, 8, 1, 1, 3)
        grid.addWidget(self.model_button, 8, 4)
        self.class_table.setHorizontalHeaderLabels(["模型类别名", "模型类别"])
        self.class_table.setColumnHidden(1, True)
        self.class_table.setToolTip("YOLO 类别由权重文件决定；检测提示词不参与推理。")
        self.confidence_spin.setToolTip("YOLO 最低检测置信度；提高可减少误检，也可能漏检。")
        self.output_edit.setText(str(DEFAULT_YOLO_OUTPUT))
        self.load_project()
        self.format_combo.setCurrentIndex(self.format_combo.findData("hbb"))

        left_layout = self.table.parentWidget().layout()
        search_row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("输入图片名，快速定位")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self.apply_filter)
        self.status_filter = QComboBox()
        for label, status in (("全部状态", "all"), ("待人工复核", "review"),
                              ("待重新检测", "unlabeled"), ("自动候选", "auto"),
                              ("人工确认", "confirmed"),
                              ("确认无目标", "empty_confirmed"), ("未抽中", "skipped")):
            self.status_filter.addItem(label, status)
        self.status_filter.currentIndexChanged.connect(self.apply_filter)
        search_row.addWidget(self.search_edit, 2)
        search_row.addWidget(self.status_filter, 1)
        left_layout.insertLayout(0, search_row)
        self.visible_count = QLabel()
        left_layout.insertWidget(2, self.visible_count)

        right_layout = self.preview.parentWidget().layout()
        self.large_preview_button = QPushButton("放大查看检测图 [Alt+Z]")
        self.large_preview_button.clicked.connect(self.open_large_preview)
        right_layout.insertWidget(3, self.large_preview_button)
        self.large_preview_shortcut = QShortcut(QKeySequence("Alt+Z"), self)
        self.large_preview_shortcut.setContext(Qt.WindowShortcut)
        self.large_preview_shortcut.activated.connect(self.large_preview_button.click)

        outer = self.centralWidget().layout()
        self.progress_detail = QLabel("检测尚未开始")
        self.summary = QLabel()
        outer.insertWidget(outer.count() - 1, self.progress_detail)
        outer.insertWidget(outer.count() - 1, self.summary)
        self.apply_filter()
        self.update_counts()
        content = self.takeCentralWidget()
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setWidget(content)
        self.setCentralWidget(scroller)
        self.resize(1500, 960)

    def set_config_controls(self, enabled: bool):
        super().set_config_controls(enabled)
        self.class_table.setEnabled(False)
        for button in self.class_buttons:
            button.setEnabled(False)
        if hasattr(self, "model_path_edit"):
            self.model_path_edit.setEnabled(enabled)
            self.model_button.setEnabled(enabled)

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 检测权重", self.model_path_edit.text(),
            "PyTorch 权重 (*.pt)")
        if path:
            self.model_path_edit.setText(path)
            self.sync_model_classes()

    def sync_model_classes(self, show_warning: bool = True):
        try:
            path = resolve_path(self.model_path_edit.text().strip())
            mtime = path.stat().st_mtime_ns
            if (self._class_model_path != path or
                    self._class_model_mtime != mtime):
                classes = model_classes(path)
                self.set_classes(classes)
                self._class_model_path = path
                self._class_model_mtime = mtime
            self.model_path_edit.setText(str(path))
        except (OSError, ValueError, RuntimeError) as exc:
            if show_warning:
                QMessageBox.warning(self, "无法读取模型", str(exc))
            else:
                raise

    def config(self) -> dict:
        config = super().config()
        config.update(model_type="yolo_detect",
                      model_path=str(resolve_path(self.model_path_edit.text().strip())))
        return config

    def load_project(self):
        if self._building_yolo_window:
            return
        self._visible_rows = 0
        try:
            project_path = self.output_dir() / "project.json"
            if project_path.exists():
                saved = json.loads(project_path.read_text(encoding="utf-8"))
                if saved.get("model_type") != "yolo_detect":
                    raise ValueError("结果目录属于其他标注项目，请为 YOLO 选择单独的结果目录")
                self.model_path_edit.setText(saved["model_path"])
            super().load_project()
            self.sync_model_classes(show_warning=False)
            self.set_config_controls(True)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            self.status.setText(f"读取 YOLO 项目失败: {exc}")

    def ensure_project(self, allow_reconfigure: bool = False) -> dict:
        path = self.output_dir() / "project.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("model_type") != "yolo_detect":
                raise ValueError("结果目录属于其他标注项目，请为 YOLO 选择单独的结果目录")
        self.sync_model_classes(show_warning=False)
        return super().ensure_project(allow_reconfigure=allow_reconfigure)

    def validate_model_ready(self):
        self.sync_model_classes(show_warning=False)

    def make_detection_thread(self, files, config, done_names):
        return GenericYOLOThread(files, config, done_names, self.segmenter)

    def apply_filter(self, *_args):
        if not hasattr(self, "search_edit"):
            return super().apply_filter()
        name_part = self.search_edit.text().strip().casefold()
        wanted = self.status_filter.currentData()
        review_only = self.review_filter.isChecked()
        if name_part or wanted != "all" or review_only:
            self.follow_check.setChecked(False)
        visible = 0
        for name, row in self.row_for_name.items():
            status = self.records[name]["status"]
            show = (name_part in name.casefold() and
                    (wanted == "all" or status == wanted) and
                    (not review_only or status in {"review", "unlabeled"}))
            self.table.setRowHidden(row, not show)
            visible += show
        self.visible_count.setText(f"显示 {visible}/{len(self.row_for_name)} 张")
        self._visible_rows = visible

    def update_table(self, record):
        tracked = hasattr(self, "search_edit")
        previous_row = self.row_for_name.get(record["name"])
        was_visible = (tracked and previous_row is not None and
                       not self.table.isRowHidden(previous_row))
        super().update_table(record)
        if tracked:
            name = record["name"]
            status = record["status"]
            visible = (self.search_edit.text().strip().casefold() in name.casefold()
                       and self.status_filter.currentData() in {"all", status}
                       and (not self.review_filter.isChecked() or
                            status in {"review", "unlabeled"}))
            self.table.setRowHidden(self.row_for_name[name], not visible)
            self._visible_rows += int(visible) - int(was_visible)
            self.visible_count.setText(
                f"显示 {self._visible_rows}/{len(self.row_for_name)} 张")

    def update_counts(self):
        if not hasattr(self, "summary"):
            return super().update_counts()
        counts = {status: 0 for status in
                  ("auto", "review", "confirmed", "empty_confirmed", "unlabeled", "skipped")}
        for record in self.records.values():
            status = record.get("status")
            counts[status] = counts.get(status, 0) + 1
        self.summary.setText(
            f"图片 {len(self.records)} 张  ·  待复核 {counts['review']}  ·  待重检 {counts['unlabeled']}"
            f"  ·  已确认 {counts['confirmed']}  ·  无目标 {counts['empty_confirmed']}"
            f"  ·  自动候选 {counts['auto']}  ·  未抽中 {counts['skipped']}")

    def start_detection(self):
        self._run_started_at = time.monotonic()
        self._run_start_done = None
        self._visible_rows = 0
        super().start_detection()
        self.apply_filter()
        if self.thread is not None and self.thread.isRunning():
            self.progress_detail.setText("正在准备检测...")

    def update_progress(self, done, total):
        super().update_progress(done, total)
        if self._run_started_at is None:
            self._run_started_at = time.monotonic()
        if self._run_start_done is None:
            self._run_start_done = done
        if done < self._run_start_done:
            self._run_start_done = done
        processed = done - self._run_start_done
        detail = f"检测进度 {done}/{total} 张"
        if processed > 0 and done < total:
            elapsed = time.monotonic() - self._run_started_at
            seconds = round(elapsed * (total - done) / processed)
            detail += f"  ·  预计剩余 {seconds // 60} 分 {seconds % 60} 秒"
        self.progress_detail.setText(detail)

    def detection_finished(self):
        stopped = self.thread is not None and not self.thread.running
        super().detection_finished()
        self.progress_detail.setText(
            f"检测{'已停止' if stopped else '已完成'}："
            f"{self.progress.value()}/{self.progress.maximum()} 张")
        self._run_started_at = None

    def open_large_preview(self):
        if self.current_name is None or self.preview_pixmap is None:
            self.status.setText("请先在左侧选择一张已检测的图片。")
            return
        dialog = LargePreviewDialog(self.preview_pixmap, self.current_name, self)
        dialog.exec_()

    def clear_all_labels(self):
        super().clear_all_labels()
        self.apply_filter()


def main() -> int:
    pyqt_plugin_root = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyqt_plugin_root
    QCoreApplication.setLibraryPaths([pyqt_plugin_root])
    app = QApplication(sys.argv)
    window = OptimizedYOLOAnnotatorApp()
    window.show()
    return app.exec_()
