"""Reusable SAM3 image annotator with custom classes and object counts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PyQt5.QtCore import QCoreApplication, QLibraryInfo, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox,
    QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QDialog,
)

from .app import SAM3SemanticPredictor, SAM3TextSegmenter, UltralyticsModel, torch
from .app_barcode import SingleBoxDialog
from .barcode_core import (
    append_record, image_files, load_records, normalize_roi, polygon_in_roi,
    roi_pixels,
)
from .generic_core import detect_generic, export_generic, validate_classes
from .project_paths import ROOT_DIR, SAM3_MODEL


DEFAULT_OUTPUT = ROOT_DIR / "data" / "generic_annotations"
STATUS_TEXT = {"auto": "自动候选", "review": "待人工复核",
               "confirmed": "人工确认", "empty_confirmed": "人工确认无目标"}
COLORS = [(0, 210, 0), (0, 190, 255), (255, 180, 0), (220, 0, 200),
          (100, 220, 220), (170, 80, 255)]


def _pixmap(image_bgr) -> QPixmap:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, width, height, width * 3,
                                    QImage.Format_RGB888).copy())


class GenericSAM3Segmenter(SAM3TextSegmenter):
    """Run all text classes in one SAM3 image encoding."""

    def detect_many(self, image_bgr, prompts: list[str]) -> list[dict]:
        results = UltralyticsModel.predict(
            self.model, source=image_bgr, predictor=SAM3SemanticPredictor,
            prompts={"text": prompts}, device=self.device, conf=self.conf,
            verbose=False,
        )
        if not results or results[0].boxes is None:
            return []
        result = results[0]
        boxes = result.boxes.xyxy.detach().cpu().numpy().tolist()
        scores = (result.boxes.conf.detach().cpu().numpy().tolist()
                  if result.boxes.conf is not None else [1.0] * len(boxes))
        class_ids = (result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
                     if result.boxes.cls is not None else [0] * len(boxes))
        masks = (result.masks.data.detach().cpu().numpy().astype(np.uint8)
                 if result.masks is not None else [])
        return [{"class_id": class_ids[index], "box": box,
                 "score": scores[index],
                 "mask": masks[index] if index < len(masks) else None}
                for index, box in enumerate(boxes)]


class GenericThread(QThread):
    result_signal = pyqtSignal(object)
    progress_signal = pyqtSignal(int, int)
    message_signal = pyqtSignal(str)

    def __init__(self, files: list[Path], config: dict, done_names: set[str]):
        super().__init__()
        self.files = files
        self.config = config
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
            segmenter = GenericSAM3Segmenter(
                model_path=str(SAM3_MODEL), device=device,
                conf=self.config["confidence"])
            for path in todo:
                if not self.running:
                    break
                try:
                    image = cv2.imread(str(path))
                    if image is None:
                        raise ValueError("图片无法解码")
                    result = detect_generic(
                        image, self.config["roi"], segmenter,
                        self.config["classes"], self.config["count_mode"],
                        self.config["target_count"],
                    )
                    result.update(name=path.name, width=image.shape[1],
                                  height=image.shape[0])
                except Exception as exc:
                    result = {"name": path.name, "objects": [],
                              "status": "review", "reason": f"检测失败: {exc}",
                              "candidate_count": 0}
                self.result_signal.emit(result)
                completed += 1
                self.progress_signal.emit(completed, total)
            self.message_signal.emit("检测已停止。" if not self.running else "检测完成，请复核待处理图片。")
        except Exception as exc:
            self.message_signal.emit(f"检测无法启动: {exc}")


class GenericAnnotatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("通用 SAM3 标注与人工复核")
        self.resize(1500, 960)
        self.roi = [0.0, 0.0, 1.0, 1.0]
        self.records: dict[str, dict] = {}
        self.row_for_name: dict[str, int] = {}
        self.current_name = None
        self.active_source = None
        self.active_output = None
        self.active_config = None
        self.thread = None
        self.preview_pixmap = None
        self._loading_objects = False
        self.config_locked = False

        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        settings = QGroupBox("项目设置")
        grid = QGridLayout(settings)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("选择包含图片的目录")
        self.output_edit = QLineEdit(str(DEFAULT_OUTPUT))
        self.output_edit.editingFinished.connect(self.load_project)
        self.path_buttons = []
        for row, (label, edit, chooser) in enumerate((
            ("图片目录", self.input_edit, self.choose_input),
            ("结果目录", self.output_edit, self.choose_output),
        )):
            button = QPushButton("浏览")
            button.clicked.connect(chooser)
            self.path_buttons.append(button)
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1, 1, 3)
            grid.addWidget(button, row, 4)
        self.roi_label = QLabel()
        self.roi_button = QPushButton("在样图上框选区域")
        self.roi_button.clicked.connect(self.choose_roi)
        self.full_button = QPushButton("使用整张图片")
        self.full_button.clicked.connect(self.use_full_image)
        grid.addWidget(self.roi_label, 2, 0, 1, 3)
        grid.addWidget(self.roi_button, 2, 3)
        grid.addWidget(self.full_button, 2, 4)

        self.class_table = QTableWidget(1, 2)
        self.class_table.setHorizontalHeaderLabels(["导出类别名", "SAM3 检测提示词（建议英文）"])
        self.class_table.horizontalHeader().setStretchLastSection(True)
        self.class_table.setMaximumHeight(135)
        self.class_table.setItem(0, 0, QTableWidgetItem("object"))
        self.class_table.setItem(0, 1, QTableWidgetItem("object"))
        add_class = QPushButton("添加类别")
        add_class.clicked.connect(self.add_class)
        remove_class = QPushButton("删除选中类别")
        remove_class.clicked.connect(self.remove_class)
        self.class_buttons = [add_class, remove_class]
        grid.addWidget(QLabel("类别与提示词"), 3, 0)
        grid.addWidget(self.class_table, 3, 1, 1, 2)
        grid.addWidget(add_class, 3, 3)
        grid.addWidget(remove_class, 3, 4)

        self.count_combo = QComboBox()
        self.count_combo.addItem("保留全部检测目标", "all")
        self.count_combo.addItem("每张期望固定数量", "fixed")
        self.count_combo.setToolTip("固定数量按所有类别合计；不足或有额外高分候选会进入待复核。")
        self.count_combo.currentIndexChanged.connect(self.update_count_ui)
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 100)
        self.count_spin.setValue(1)
        self.count_spin.setEnabled(False)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.05, 0.95)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setValue(0.25)
        self.confidence_spin.setToolTip("SAM3 最低候选置信度；提高可减少误检，也可能漏检。")
        grid.addWidget(QLabel("目标数量"), 4, 0)
        grid.addWidget(self.count_combo, 4, 1)
        grid.addWidget(self.count_spin, 4, 2)
        grid.addWidget(QLabel("最低检测置信度"), 4, 3)
        grid.addWidget(self.confidence_spin, 4, 4)
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
        self.table.setHorizontalHeaderLabels(["图片", "状态", "已选/候选", "原因"])
        self.table.setColumnWidth(0, 115)
        self.table.setColumnWidth(1, 120)
        self.table.setColumnWidth(2, 85)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.itemSelectionChanged.connect(self.select_table_row)
        left_layout.addWidget(self.table)
        filter_row = QHBoxLayout()
        self.review_filter = QCheckBox("只看待复核")
        self.review_filter.toggled.connect(self.apply_filter)
        next_button = QPushButton("下一张待复核")
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
        self.preview.setMinimumSize(650, 340)
        self.preview.setStyleSheet("background:#202020;color:white;")
        right_layout.addWidget(self.preview_title)
        right_layout.addWidget(self.preview, 1)

        self.object_table = QTableWidget(0, 3)
        self.object_table.setHorizontalHeaderLabels(["编号", "类别", "置信度"])
        self.object_table.setMaximumHeight(145)
        self.object_table.horizontalHeader().setStretchLastSection(True)
        self.object_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.object_table.itemSelectionChanged.connect(self.select_object)
        right_layout.addWidget(self.object_table)
        edit_row = QHBoxLayout()
        self.object_class_combo = QComboBox()
        self.object_class_combo.currentIndexChanged.connect(self.change_object_class)
        add_object = QPushButton("添加框")
        add_object.clicked.connect(self.add_object)
        edit_object = QPushButton("修改选中框")
        edit_object.clicked.connect(self.edit_object)
        delete_object = QPushButton("删除选中框")
        delete_object.clicked.connect(self.delete_object)
        for widget in (self.object_class_combo, add_object, edit_object, delete_object):
            edit_row.addWidget(widget)
        right_layout.addLayout(edit_row)
        review_row = QHBoxLayout()
        confirm_button = QPushButton("确认本张所有框")
        confirm_button.clicked.connect(self.confirm_current)
        empty_button = QPushButton("确认本张无目标")
        empty_button.clicked.connect(self.mark_empty)
        review_row.addWidget(confirm_button)
        review_row.addWidget(empty_button)
        right_layout.addLayout(review_row)
        splitter.addWidget(right)
        splitter.setSizes([430, 1070])
        outer.addWidget(splitter, 1)

        self.progress = QProgressBar()
        outer.addWidget(self.progress)
        self.status = QLabel("选择图片目录并设置类别后开始检测；区域默认是整张图片。")
        outer.addWidget(self.status)
        self.show_roi()
        self.load_project()

    def source_dir(self) -> Path:
        value = self.input_edit.text().strip()
        if not value:
            raise ValueError("请先选择图片目录")
        return Path(value).expanduser().resolve()

    def output_dir(self) -> Path:
        value = self.output_edit.text().strip()
        if not value:
            raise ValueError("请先选择结果目录")
        return Path(value).expanduser().resolve()

    def classes(self) -> list[dict]:
        rows = []
        for row in range(self.class_table.rowCount()):
            name_item = self.class_table.item(row, 0)
            prompt_item = self.class_table.item(row, 1)
            rows.append({"name": name_item.text() if name_item else "",
                         "prompt": prompt_item.text() if prompt_item else ""})
        return validate_classes(rows)

    def set_classes(self, classes):
        self.class_table.setRowCount(len(classes))
        for row, spec in enumerate(classes):
            self.class_table.setItem(row, 0, QTableWidgetItem(spec["name"]))
            self.class_table.setItem(row, 1, QTableWidgetItem(spec["prompt"]))
        self.refresh_object_classes()

    def add_class(self):
        row = self.class_table.rowCount()
        self.class_table.insertRow(row)
        self.class_table.setItem(row, 0, QTableWidgetItem(""))
        self.class_table.setItem(row, 1, QTableWidgetItem(""))
        self.class_table.setCurrentCell(row, 0)
        self.refresh_object_classes()

    def remove_class(self):
        row = self.class_table.currentRow()
        if row >= 0 and self.class_table.rowCount() > 1:
            self.class_table.removeRow(row)
            self.refresh_object_classes()

    def refresh_object_classes(self):
        current = self.object_class_combo.currentIndex()
        self._loading_objects = True
        self.object_class_combo.clear()
        for index in range(self.class_table.rowCount()):
            item = self.class_table.item(index, 0)
            self.object_class_combo.addItem(item.text().strip() if item else "", index)
        if current >= 0 and current < self.object_class_combo.count():
            self.object_class_combo.setCurrentIndex(current)
        self._loading_objects = False

    def update_count_ui(self):
        self.count_spin.setEnabled(
            not self.config_locked and self.count_combo.currentData() == "fixed")

    def set_config_controls(self, enabled: bool):
        self.config_locked = not enabled
        for widget in (self.input_edit, self.roi_button, self.full_button,
                       self.class_table, self.count_combo, self.confidence_spin,
                       self.path_buttons[0], *self.class_buttons):
            widget.setEnabled(enabled)
        self.update_count_ui()

    def choose_input(self):
        start = self.input_edit.text().strip() or str(ROOT_DIR / "data")
        path = QFileDialog.getExistingDirectory(self, "选择图片目录", start)
        if path:
            self.input_edit.setText(path)
            self.roi = [0.0, 0.0, 1.0, 1.0]
            self.show_roi()

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择结果目录", str(self.output_dir()))
        if path:
            self.output_edit.setText(path)
            self.load_project()

    def choose_roi(self):
        try:
            source = self.source_dir()
            files = image_files(source)
            if not files:
                raise ValueError("图片目录为空")
            image = cv2.imread(str(files[0]))
            if image is None:
                raise ValueError(f"无法读取图片: {files[0]}")
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法框选", str(exc))
            return
        x1, y1, x2, y2 = roi_pixels(self.roi, image.shape[1], image.shape[0])
        initial = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        dialog = SingleBoxDialog(files[0], "框选本项目的检测区域", initial,
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

    def use_full_image(self):
        self.roi = [0.0, 0.0, 1.0, 1.0]
        self.show_roi()

    def show_roi(self):
        if self.roi == [0.0, 0.0, 1.0, 1.0]:
            self.roi_label.setText("检测区域：整张图片")
        else:
            self.roi_label.setText("检测区域比例：" + ", ".join(f"{value:.3f}" for value in self.roi))

    def config(self) -> dict:
        source = self.source_dir()
        return {"source_dir": str(source), "roi": self.roi,
                "classes": self.classes(), "count_mode": self.count_combo.currentData(),
                "target_count": self.count_spin.value(),
                "confidence": self.confidence_spin.value()}

    def load_project(self):
        path = self.output_dir() / "project.json"
        if not path.exists():
            self.active_source = None
            self.active_output = None
            self.active_config = None
            self.records.clear()
            self.table.setRowCount(0)
            self.row_for_name.clear()
            self.current_name = None
            self.clear_preview()
            self.set_config_controls(True)
            return
        try:
            config = json.loads(path.read_text(encoding="utf-8"))
            self.input_edit.setText(config["source_dir"])
            self.roi = normalize_roi(config["roi"])
            if self.roi is None:
                raise ValueError("项目中的检测区域无效")
            self.show_roi()
            self.set_classes(validate_classes(config["classes"]))
            self.count_combo.setCurrentIndex(
                self.count_combo.findData(config["count_mode"]))
            self.count_spin.setValue(int(config["target_count"]))
            self.confidence_spin.setValue(float(config["confidence"]))
            self.active_source = Path(config["source_dir"])
            self.active_output = self.output_dir()
            self.active_config = config
            self.records = load_records(self.active_output / "review.jsonl")
            self.table.setRowCount(0)
            self.row_for_name.clear()
            self.current_name = None
            self.clear_preview()
            for record in sorted(self.records.values(), key=lambda item: item["name"]):
                self.update_table(record)
            self.update_counts()
            self.set_config_controls(False)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.status.setText(f"读取旧项目失败: {exc}")

    def ensure_project(self) -> dict:
        config = self.config()
        source, output = self.source_dir(), self.output_dir()
        if not source.is_dir():
            raise ValueError(f"图片目录不存在: {source}")
        if output == source or output.is_relative_to(source):
            raise ValueError("结果目录不能位于图片目录内部")
        path = output / "project.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous != config:
                raise ValueError("该结果目录已有其他类别、数量、区域或图片源；请换一个新结果目录")
            self.records = load_records(output / "review.jsonl")
        else:
            if (output / "review.jsonl").exists():
                raise ValueError("结果目录有 review.jsonl 但缺少 project.json，请换目录")
            output.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2),
                            encoding="utf-8")
            self.records = {}
        if self.active_output != output:
            self.current_name = None
            self.table.setRowCount(0)
            self.row_for_name.clear()
            self.clear_preview()
        self.active_source = source
        self.active_output = output
        self.active_config = config
        self.refresh_object_classes()
        self.set_config_controls(False)
        return config

    def start_detection(self):
        try:
            config = self.ensure_project()
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
        self.thread = GenericThread(files, config, set(self.records))
        self.thread.result_signal.connect(self.receive_result)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.message_signal.connect(self.status.setText)
        self.thread.finished.connect(self.detection_finished)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.output_edit.setEnabled(False)
        self.path_buttons[1].setEnabled(False)
        self.thread.start()

    def stop_detection(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.status.setText("正在停止；当前图片完成后退出。")

    def detection_finished(self):
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.export_button.setEnabled(True)
        self.output_edit.setEnabled(True)
        self.path_buttons[1].setEnabled(True)
        self.set_config_controls(False)
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
        objects = record.get("objects", [])
        values = [name, STATUS_TEXT.get(record["status"], record["status"]),
                  f"{len(objects)}/{record.get('candidate_count', 0)}",
                  record.get("reason", "")]
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
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (30, 30, 230), 2)
        self._loading_objects = True
        self.object_table.setRowCount(0)
        captions = []
        for index, obj in enumerate(record.get("objects", [])):
            class_id = int(obj["class_id"])
            color = COLORS[class_id % len(COLORS)]
            points = np.rint(obj["polygon"]).astype(np.int32)
            cv2.polylines(image, [points], True, color, 3)
            label = self.active_config["classes"][class_id]["name"]
            anchor = points.min(axis=0)
            captions.append((max(2, int(anchor[0])), max(24, int(anchor[1]) - 5),
                             f"{index + 1}: {label}", color))
            self.object_table.insertRow(index)
            for column, text in enumerate((str(index + 1), label,
                                           f"{obj.get('score', 0):.2f}")):
                self.object_table.setItem(index, column, QTableWidgetItem(text))
        self._loading_objects = False
        self.preview_pixmap = _pixmap(image)
        painter = QPainter(self.preview_pixmap)
        painter.setFont(QFont("Sans Serif", 12))
        for text_x, text_y, caption, color in captions:
            width = painter.fontMetrics().horizontalAdvance(caption) + 8
            painter.fillRect(text_x - 2, text_y - 19, width, 23, QColor(0, 0, 0, 180))
            painter.setPen(QColor(color[2], color[1], color[0]))
            painter.drawText(text_x + 2, text_y, caption)
        painter.end()
        self.refresh_preview()
        self.preview_title.setText(
            f"{name}  |  {STATUS_TEXT.get(record['status'], record['status'])}  |  "
            f"已选 {len(record.get('objects', []))} / 候选 {record.get('candidate_count', 0)}  |  "
            f"{record.get('reason', '')}"
        )

    def clear_preview(self):
        self.preview_pixmap = None
        self.preview.setPixmap(QPixmap())
        self.preview.setText("等待图片")
        self.preview_title.setText("检测开始后，这里会逐张显示标注结果")
        self.object_table.setRowCount(0)

    def refresh_preview(self):
        if self.preview_pixmap:
            self.preview.setPixmap(self.preview_pixmap.scaled(
                self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_preview()

    def select_object(self):
        if self._loading_objects or self.current_name is None:
            return
        self.follow_check.setChecked(False)
        row = self.object_table.currentRow()
        objects = self.records[self.current_name].get("objects", [])
        if 0 <= row < len(objects):
            self._loading_objects = True
            self.object_class_combo.setCurrentIndex(int(objects[row]["class_id"]))
            self._loading_objects = False

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

    def add_object(self):
        record = self.current_record()
        if record is None:
            return
        self.follow_check.setChecked(False)
        path = self.active_source / record["name"]
        dialog = SingleBoxDialog(path, f"添加目标：{record['name']}", parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        image = cv2.imread(str(path))
        if not polygon_in_roi(dialog.result_polygon, self.roi, image.shape[1], image.shape[0]):
            QMessageBox.warning(self, "框超出区域", "目标框中心须在区域内，且至少一半面积位于区域内。")
            return
        objects = list(record.get("objects", []))
        objects.append({"class_id": self.object_class_combo.currentIndex(),
                        "polygon": dialog.result_polygon, "score": 1.0})
        self.save_review({**record, "objects": objects, "status": "confirmed",
                          "reason": "人工添加或修正目标"})
        self.object_table.selectRow(len(objects) - 1)

    def edit_object(self):
        record = self.current_record()
        if record is None:
            return
        index = self.object_table.currentRow()
        objects = list(record.get("objects", []))
        if not 0 <= index < len(objects):
            QMessageBox.information(self, "先选目标", "请先在图片下方选择一个目标。")
            return
        self.follow_check.setChecked(False)
        path = self.active_source / record["name"]
        dialog = SingleBoxDialog(path, f"修正目标 {index + 1}：{record['name']}",
                                 objects[index]["polygon"], parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        image = cv2.imread(str(path))
        if not polygon_in_roi(dialog.result_polygon, self.roi, image.shape[1], image.shape[0]):
            QMessageBox.warning(self, "框超出区域", "目标框中心须在区域内，且至少一半面积位于区域内。")
            return
        objects[index] = {**objects[index], "polygon": dialog.result_polygon,
                          "class_id": self.object_class_combo.currentIndex()}
        self.save_review({**record, "objects": objects, "status": "confirmed",
                          "reason": "人工修正目标"})
        self.object_table.selectRow(index)

    def delete_object(self):
        record = self.current_record()
        if record is None:
            return
        self.follow_check.setChecked(False)
        index = self.object_table.currentRow()
        objects = list(record.get("objects", []))
        if not 0 <= index < len(objects):
            QMessageBox.information(self, "先选目标", "请先在图片下方选择一个目标。")
            return
        objects.pop(index)
        self.save_review({**record, "objects": objects,
                          "status": "confirmed" if objects else "review",
                          "reason": "人工删除误检目标" if objects else "已删除全部框，请补框或确认无目标"})

    def change_object_class(self):
        if self._loading_objects or self.current_name is None:
            return
        index = self.object_table.currentRow()
        record = self.records[self.current_name]
        objects = list(record.get("objects", []))
        if not 0 <= index < len(objects):
            return
        class_id = self.object_class_combo.currentIndex()
        if class_id < 0 or objects[index]["class_id"] == class_id:
            return
        objects[index] = {**objects[index], "class_id": class_id}
        self.save_review({**record, "objects": objects, "status": "confirmed",
                          "reason": "人工修正类别"})
        self.object_table.selectRow(index)

    def confirm_current(self):
        record = self.current_record()
        if record is None:
            return
        self.follow_check.setChecked(False)
        if not record.get("objects"):
            QMessageBox.information(self, "尚无目标", "请先补画目标框，或确认本张无目标。")
            return
        self.save_review({**record, "status": "confirmed", "reason": "人工确认所有目标"})

    def mark_empty(self):
        record = self.current_record()
        if record is not None:
            self.follow_check.setChecked(False)
            self.save_review({**record, "objects": [], "status": "empty_confirmed",
                              "reason": "人工确认区域内无目标"})

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
            config = self.ensure_project()
            target, images, objects = export_generic(
                self.active_source, self.active_output, self.records,
                config["classes"], self.format_combo.currentData())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        unresolved = sum(record["status"] == "review" for record in self.records.values())
        self.status.setText(
            f"已导出 {images} 张有目标图片、{objects} 个目标；"
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
    pyqt_plugin_root = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyqt_plugin_root
    QCoreApplication.setLibraryPaths([pyqt_plugin_root])
    app = QApplication(sys.argv)
    window = GenericAnnotatorApp()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
