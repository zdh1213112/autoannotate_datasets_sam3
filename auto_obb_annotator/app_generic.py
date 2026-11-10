"""Reusable SAM3 image annotator with custom classes and object counts."""

from __future__ import annotations

import copy
import json
import os
import secrets
import shutil
from datetime import datetime
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from PyQt5.QtCore import QCoreApplication, QLibraryInfo, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout, QGroupBox,
    QDoubleSpinBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QShortcut, QSpinBox, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QDialog,
)

from .app import RotatableROILabel, SAM3SemanticPredictor, SAM3TextSegmenter, UltralyticsModel, torch
from .app_barcode import SingleBoxDialog
from .barcode_core import append_record, image_files, load_records
from .button_shortcuts import ButtonShortcuts
from .generic_core import (
    REVIEW_RULES, detect_generic, export_generic, has_duplicate_selected_boxes,
    normalize_review_reasons,
    normalize_review_score, normalize_sample_percent, normalize_sample_seed,
    normalize_selection_rule, recheck_review_scores, sample_image_files,
    sync_candidates, validate_classes,
)
from .generic_region import (
    FULL_REGION, circle_pixels, normalize_region, polygon_in_region, region_bounds,
)
from .generic_region_dialog import GenericRegionDialog
from .project_paths import ROOT_DIR, SAM3_MODEL


DEFAULT_OUTPUT = ROOT_DIR / "data" / "generic_annotations"
STATUS_TEXT = {"auto": "自动候选", "review": "待人工复核",
               "confirmed": "人工确认", "empty_confirmed": "人工确认无目标",
               "unlabeled": "待重新检测", "skipped": "未抽中"}
SELECTED_COLOR = (0, 200, 0)  # BGR
UNSELECTED_COLOR = (11, 134, 184)  # dark goldenrod, BGR


class ClickablePreview(QLabel):
    clicked = pyqtSignal(object)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.pos())
            event.accept()
        else:
            super().mousePressEvent(event)


def _pixmap(image_bgr) -> QPixmap:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return QPixmap.fromImage(QImage(rgb.data, width, height, width * 3,
                                    QImage.Format_RGB888).copy())


def replace_review_records(path: Path, records: dict[str, dict]) -> None:
    """Atomically replace effective records when clearing a whole image directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent,
                prefix=".review_", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            for record in sorted(records.values(), key=lambda item: item["name"]):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def replace_project_config(path: Path, config: dict) -> None:
    """Commit a changed project configuration without a partially written JSON file."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent,
                prefix=".project_", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(config, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def backup_project(output: Path) -> Path:
    """Preserve the previous configuration and labels before reprocessing."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = output / f"backup_before_config_{stamp}"
    suffix = 2
    while backup.exists():
        backup = output / f"backup_before_config_{stamp}_{suffix}"
        suffix += 1
    backup.mkdir()
    shutil.copy2(output / "project.json", backup / "project.json")
    review_path = output / "review.jsonl"
    if review_path.exists():
        shutil.copy2(review_path, backup / "review.jsonl")
    return backup


class UndoableBoxCanvas(RotatableROILabel):
    """Keep the current dialog's frame edits separate from saved review undo."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._edit_history = []
        self._mouse_before = None

    def _snapshot(self):
        return (list(self.rois), self._has_pending, self._cx, self._cy,
                self._w, self._h, self._angle)

    def _remember(self, before):
        if before != self._snapshot():
            self._edit_history.append(before)
            if len(self._edit_history) > 100:
                self._edit_history.pop(0)

    def undo_edit(self):
        if not self._edit_history:
            return
        (self.rois, self._has_pending, self._cx, self._cy,
         self._w, self._h, self._angle) = self._edit_history.pop()
        self._state = self._ST_IDLE
        self._drag_handle = -1
        self._mouse_before = None
        self.rotation_changed.emit(self._angle)
        self.update()

    def mousePressEvent(self, event):
        self._mouse_before = self._snapshot() if event.button() == Qt.LeftButton else None
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton and self._mouse_before is not None:
            self._remember(self._mouse_before)
            self._mouse_before = None

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Z and event.modifiers() & Qt.ControlModifier:
            self.undo_edit()
            event.accept()
            return
        before = self._snapshot()
        super().keyPressEvent(event)
        self._remember(before)

    def clear_rois(self):
        before = self._snapshot()
        super().clear_rois()
        self._remember(before)


class GenericBoxDialog(SingleBoxDialog):
    """Add the review shortcuts to the existing rotated-box editor."""

    canvas_type = UndoableBoxCanvas

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        labels = {"清除框": "清除框 [Ctrl+Backspace]",
                  "保存这个框": "保存这个框 [Ctrl+S]", "取消": "取消 [Esc]"}
        for button in self.findChildren(QPushButton):
            button.setText(labels[button.text()])
        self.button_shortcuts = ButtonShortcuts(self)
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.setContext(Qt.WindowShortcut)
        self.undo_shortcut.setAutoRepeat(False)
        self.undo_shortcut.activated.connect(self.canvas.undo_edit)
        self.canvas.setFocus()


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

    def __init__(self, files: list[Path], config: dict, done_names: set[str],
                 segmenter=None):
        super().__init__()
        self.files = files
        self.config = config
        self.done_names = done_names
        self.segmenter = segmenter
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
            if self.segmenter is None or self.segmenter.device != device:
                self.message_signal.emit(f"正在加载 SAM3（{device}），本次检测 {len(todo)} 张...")
                self.segmenter = GenericSAM3Segmenter(
                    model_path=str(SAM3_MODEL), device=device,
                    conf=self.config["confidence"])
            else:
                self.segmenter.conf = float(self.config["confidence"])
                self.message_signal.emit(f"复用已加载的 SAM3（{device}），本次检测 {len(todo)} 张...")
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
                        self.config.get("review_score"),
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
            self.message_signal.emit("检测已停止。" if not self.running else "检测完成，请复核待处理图片。")
        except Exception as exc:
            self.message_signal.emit(f"检测无法启动: {exc}")


class GenericAnnotatorApp(QMainWindow):
    previous_image_shortcut = "Shift+N"

    def __init__(self):
        super().__init__()
        self.setWindowTitle("通用 SAM3 标注与人工复核")
        self.resize(1500, 960)
        self.roi = FULL_REGION.copy()
        self.records: dict[str, dict] = {}
        self.row_for_name: dict[str, int] = {}
        self.current_name = None
        self.active_source = None
        self.active_output = None
        self.active_config = None
        self.thread = None
        self.segmenter = None
        self.sample_seed = secrets.randbits(64)
        self._deleted_during_run = set()
        self._undo_stack = []
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
        for row, (label, edit, chooser, shortcut) in enumerate((
            ("图片目录", self.input_edit, self.choose_input, "Alt+I"),
            ("结果目录", self.output_edit, self.choose_output, "Alt+O"),
        )):
            button = QPushButton(f"浏览 [{shortcut}]")
            button.clicked.connect(chooser)
            self.path_buttons.append(button)
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1, 1, 3)
            grid.addWidget(button, row, 4)
        self.roi_label = QLabel()
        self.roi_button = QPushButton("在样图上选择区域 [Alt+R]")
        self.roi_button.clicked.connect(self.choose_roi)
        self.full_button = QPushButton("使用整张图片 [Alt+F]")
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
        add_class = QPushButton("添加类别 [Ctrl+Shift+A]")
        add_class.clicked.connect(self.add_class)
        remove_class = QPushButton("删除选中类别 [Ctrl+Shift+Delete]")
        remove_class.clicked.connect(self.remove_class)
        self.class_buttons = [add_class, remove_class]
        grid.addWidget(QLabel("类别与提示词"), 3, 0)
        grid.addWidget(self.class_table, 3, 1, 1, 2)
        grid.addWidget(add_class, 3, 3)
        grid.addWidget(remove_class, 3, 4)

        self.count_combo = QComboBox()
        self.count_combo.addItem("保留全部检测目标", "all")
        self.count_combo.addItem("每张期望固定数量", "fixed")
        self.count_combo.setToolTip("固定数量按所有类别合计；标签框按下方选择方式确定。数量不足是否待复核由勾选项决定。")
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
        self.selection_combo = QComboBox()
        self.selection_combo.addItem("置信度最高的期望固定数量", "confidence")
        self.selection_combo.addItem("框面积最大的期望固定数量", "area")
        self.selection_combo.setToolTip("仅在固定数量模式下生效；面积按预测框的实际像素面积计算。")
        self.selection_combo.setEnabled(False)
        grid.addWidget(QLabel("已选标签框判断方式"), 5, 0)
        grid.addWidget(self.selection_combo, 5, 1, 1, 4)
        grid.addWidget(QLabel("待人工复核条件"), 6, 0)
        reasons_row = QHBoxLayout()
        self.review_reason_checks = {}
        for key, label in REVIEW_RULES:
            check = QCheckBox(label)
            check.setChecked(True)
            check.setToolTip("执行前取消勾选后，该条件不再使图片进入待人工复核。")
            self.review_reason_checks[key] = check
            reasons_row.addWidget(check)
        grid.addLayout(reasons_row, 6, 1, 1, 4)
        self.review_score_spin = QDoubleSpinBox()
        self.review_score_spin.setRange(0.05, 0.95)
        self.review_score_spin.setSingleStep(0.05)
        self.review_score_spin.setDecimals(2)
        self.review_score_spin.setValue(0.50)
        self.review_score_spin.setToolTip(
            "已选标签框低于此置信度时进入人工复核；与最低检测置信度分别设置。")
        reasons_row.addWidget(QLabel("低分复核阈值"))
        reasons_row.addWidget(self.review_score_spin)
        self.sample_percent_spin = QSpinBox()
        self.sample_percent_spin.setRange(1, 100)
        self.sample_percent_spin.setValue(100)
        self.sample_percent_spin.setSuffix("%")
        self.sample_percent_spin.setToolTip(
            "从目录随机抽取指定比例的图片；至少抽取 1 张。本次随机抽样会保存，续跑仍使用同一批图片。")
        self.sample_summary = QLabel("选择图片目录后显示预计抽取数量")
        self.sample_percent_spin.valueChanged.connect(self.update_sample_summary)
        self.input_edit.editingFinished.connect(self.update_sample_summary)
        grid.addWidget(QLabel("自动标注图片比例"), 7, 0)
        grid.addWidget(self.sample_percent_spin, 7, 1)
        grid.addWidget(self.sample_summary, 7, 2, 1, 3)
        outer.addWidget(settings)

        controls = QHBoxLayout()
        self.start_button = QPushButton("开始/继续自动检测 [F5]")
        self.start_button.clicked.connect(self.start_detection)
        self.stop_button = QPushButton("停止 [Esc]")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_detection)
        self.follow_check = QCheckBox("实时跟随每张结果 [F]")
        self.follow_check.setChecked(True)
        self.format_combo = QComboBox()
        self.format_combo.addItem("YOLO-OBB 旋转框", "obb")
        self.format_combo.addItem("YOLO-HBB 普通框", "hbb")
        self.export_button = QPushButton("导出训练数据集 [Ctrl+E]")
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
        self.review_filter = QCheckBox("只看待复核/待重检 [R]")
        self.review_filter.toggled.connect(self.apply_filter)
        previous_button = QPushButton("上一张待复核 [Shift+D]")
        previous_button.clicked.connect(self.previous_review)
        next_button = QPushButton("下一张待复核 [D]")
        next_button.clicked.connect(self.next_review)
        filter_row.addWidget(self.review_filter)
        filter_row.addWidget(previous_button)
        filter_row.addWidget(next_button)
        left_layout.addLayout(filter_row)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.preview_title = QLabel("检测开始后，这里会逐张显示标注结果")
        self.preview_title.setWordWrap(True)
        self.preview = ClickablePreview("等待图片")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(650, 340)
        self.preview.setStyleSheet("background:#202020;color:white;")
        self.preview.clicked.connect(self.preview_clicked)
        right_layout.addWidget(self.preview_title)
        self.preview_legend = QLabel(
            "绿色：输出标签框；暗黄色：未选检测框。点击框可切换，框旁显示类别和置信度。")
        right_layout.addWidget(self.preview_legend)
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
        add_object = QPushButton("添加框 [A]")
        add_object.clicked.connect(self.add_object)
        edit_object = QPushButton("修改选中框 [E]")
        edit_object.clicked.connect(self.edit_object)
        delete_object = QPushButton("删除选中框 [Delete]")
        delete_object.clicked.connect(self.delete_object)
        for widget in (self.object_class_combo, add_object, edit_object, delete_object):
            edit_row.addWidget(widget)
        right_layout.addLayout(edit_row)
        review_row = QHBoxLayout()
        confirm_button = QPushButton("确认并下一张待复核 [Space]")
        confirm_button.clicked.connect(self.confirm_current)
        previous_image_button = QPushButton(f"上一张 [{self.previous_image_shortcut}]")
        previous_image_button.clicked.connect(self.show_previous_image)
        next_image_button = QPushButton("下一张 [N]")
        next_image_button.clicked.connect(self.show_next_image)
        empty_button = QPushButton("确认无目标并下一张待复核 [Q]")
        empty_button.clicked.connect(self.mark_empty)
        clear_labels_button = QPushButton("删除本张标签并待重检 [Ctrl+Delete]")
        clear_labels_button.clicked.connect(self.clear_current_labels)
        review_row.addWidget(confirm_button)
        review_row.addWidget(previous_image_button)
        review_row.addWidget(next_image_button)
        review_row.addWidget(empty_button)
        review_row.addWidget(clear_labels_button)
        self.undo_button = QPushButton("撤销上一步 [Ctrl+Z]")
        self.undo_button.setEnabled(False)
        self.undo_button.setToolTip("撤销当前会话最近一次人工标注操作，恢复对应图片。")
        self.undo_button.clicked.connect(self.undo_last_review)
        review_row.addWidget(self.undo_button)
        right_layout.addLayout(review_row)
        self.clear_all_button = QPushButton("删除目录全部标签并待重检 [Ctrl+Shift+R]")
        self.clear_all_button.clicked.connect(self.clear_all_labels)
        self.clear_all_button.setStyleSheet("color:#b71c1c;")
        right_layout.addWidget(self.clear_all_button, alignment=Qt.AlignRight)
        splitter.addWidget(right)
        splitter.setSizes([430, 1070])
        outer.addWidget(splitter, 1)

        self.progress = QProgressBar()
        outer.addWidget(self.progress)
        self.status = QLabel("选择图片目录并设置类别后开始检测；区域默认是整张图片。")
        outer.addWidget(self.status)
        self.show_roi()
        self.load_project()
        self.button_shortcuts = ButtonShortcuts(self, (self.class_table,))
        self.table.setFocus()

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
        fixed = not self.config_locked and self.count_combo.currentData() == "fixed"
        self.count_spin.setEnabled(fixed)
        self.selection_combo.setEnabled(fixed)

    def set_config_controls(self, enabled: bool):
        self.config_locked = not enabled
        for widget in (self.input_edit, self.roi_button, self.full_button,
                       self.class_table, self.count_combo, self.confidence_spin,
                       self.review_score_spin,
                       self.sample_percent_spin,
                       self.path_buttons[0], *self.class_buttons,
                       *self.review_reason_checks.values()):
            widget.setEnabled(enabled)
        self.update_count_ui()

    def choose_input(self):
        start = self.input_edit.text().strip() or str(ROOT_DIR / "data")
        path = QFileDialog.getExistingDirectory(self, "选择图片目录", start)
        if path:
            self.input_edit.setText(path)
            self.roi = FULL_REGION.copy()
            self.show_roi()
            self.update_sample_summary()

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
        dialog = GenericRegionDialog(files[0], self.roi, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            self.roi = dialog.result_region
            self.show_roi()

    def use_full_image(self):
        self.roi = FULL_REGION.copy()
        self.show_roi()

    def show_roi(self):
        if self.roi == FULL_REGION:
            self.roi_label.setText("检测区域：整张图片")
        elif isinstance(self.roi, dict):
            cx, cy = self.roi["center"]
            self.roi_label.setText(
                f"检测区域：圆形（中心 {cx:.3f}, {cy:.3f}；半径 {self.roi['radius']:.3f} × 短边）")
        else:
            self.roi_label.setText("检测区域：矩形（" + ", ".join(
                f"{value:.3f}" for value in self.roi) + "）")

    def update_sample_summary(self):
        try:
            files = image_files(self.source_dir())
            count = len(sample_image_files(files, self.sample_percent_spin.value(),
                                           self.sample_seed))
        except (OSError, ValueError):
            self.sample_summary.setText("选择图片目录后显示预计抽取数量")
            return
        self.sample_summary.setText(
            f"预计随机抽取 {count}/{len(files)} 张")

    def config(self) -> dict:
        source = self.source_dir()
        return {"source_dir": str(source), "roi": self.roi,
                "classes": self.classes(), "count_mode": self.count_combo.currentData(),
                "target_count": self.count_spin.value(),
                "selection_rule": normalize_selection_rule(
                    self.selection_combo.currentData()),
                "sample_percent": normalize_sample_percent(
                    self.sample_percent_spin.value()),
                "sample_seed": normalize_sample_seed(self.sample_seed),
                "confidence": self.confidence_spin.value(),
                "review_score": normalize_review_score(
                    self.review_score_spin.value()),
                "review_reasons": [key for key, check in self.review_reason_checks.items()
                                   if check.isChecked()]}

    def load_project(self):
        self.clear_undo_history()
        path = self.output_dir() / "project.json"
        if not path.exists():
            self.sample_seed = secrets.randbits(64)
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
            self.roi = normalize_region(config["roi"])
            if self.roi is None:
                raise ValueError("项目中的检测区域无效")
            reasons = normalize_review_reasons(config.get("review_reasons"))
            config["review_reasons"] = reasons
            config["review_score"] = normalize_review_score(
                config.get("review_score"))
            config["selection_rule"] = normalize_selection_rule(
                config.get("selection_rule"))
            config["sample_percent"] = normalize_sample_percent(
                config.get("sample_percent"))
            self.sample_seed = normalize_sample_seed(
                config["sample_seed"] if "sample_seed" in config
                else secrets.randbits(64))
            config["sample_seed"] = self.sample_seed
            for key, check in self.review_reason_checks.items():
                check.setChecked(key in reasons)
            self.show_roi()
            self.set_classes(validate_classes(config["classes"]))
            self.count_combo.setCurrentIndex(
                self.count_combo.findData(config["count_mode"]))
            self.count_spin.setValue(int(config["target_count"]))
            self.selection_combo.setCurrentIndex(
                self.selection_combo.findData(config["selection_rule"]))
            self.sample_percent_spin.setValue(config["sample_percent"])
            self.update_sample_summary()
            self.confidence_spin.setValue(float(config["confidence"]))
            self.review_score_spin.setValue(config["review_score"])
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
            self.set_config_controls(True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.status.setText(f"读取旧项目失败: {exc}")

    def ensure_project(self, allow_reconfigure: bool = False) -> dict:
        config = self.config()
        source, output = self.source_dir(), self.output_dir()
        if not source.is_dir():
            raise ValueError(f"图片目录不存在: {source}")
        if output == source or output.is_relative_to(source):
            raise ValueError("结果目录不能位于图片目录内部")
        path = output / "project.json"
        if path.exists():
            previous = json.loads(path.read_text(encoding="utf-8"))
            previous["roi"] = normalize_region(previous.get("roi"))
            previous["review_reasons"] = normalize_review_reasons(
                previous.get("review_reasons"))
            missing_review_score = "review_score" not in previous
            previous["review_score"] = normalize_review_score(
                previous.get("review_score"))
            previous["selection_rule"] = normalize_selection_rule(
                previous.get("selection_rule"))
            previous["sample_percent"] = normalize_sample_percent(
                previous.get("sample_percent"))
            missing_seed = "sample_seed" not in previous
            previous["sample_seed"] = normalize_sample_seed(
                previous.get("sample_seed", config["sample_seed"]))
            needs_random_migration = missing_seed and previous["sample_percent"] < 100
            score_only_change = ((previous != config or missing_review_score) and
                                 not needs_random_migration and
                                 {key: value for key, value in previous.items()
                                  if key != "review_score"} ==
                                 {key: value for key, value in config.items()
                                  if key != "review_score"})
            if score_only_change:
                if not allow_reconfigure:
                    raise ValueError("配置已修改，请先按 F5 根据新配置重新检测，再导出。")
                backup = backup_project(output)
                updated = recheck_review_scores(
                    load_records(output / "review.jsonl"), config["review_score"],
                    "low_score" in config["review_reasons"])
                review_path = output / "review.jsonl"
                try:
                    replace_review_records(review_path, updated)
                    replace_project_config(path, config)
                except OSError:
                    old_review = backup / "review.jsonl"
                    if old_review.exists():
                        shutil.copy2(old_review, review_path)
                    else:
                        review_path.unlink(missing_ok=True)
                    raise
                self.records = updated
                self.clear_undo_history()
            elif previous != config or needs_random_migration:
                if not allow_reconfigure:
                    raise ValueError("配置已修改，请先按 F5 根据新配置重新检测，再导出。")
                if previous["sample_percent"] != config["sample_percent"]:
                    config["sample_seed"] = secrets.randbits(64)
                files = image_files(source)
                if not files:
                    raise ValueError("图片目录为空，无法按新配置继续检测")
                backup = backup_project(output)
                updated = self.records_after_config_change(files, previous, config)
                review_path = output / "review.jsonl"
                try:
                    replace_review_records(review_path, updated)
                    replace_project_config(path, config)
                except OSError:
                    old_review = backup / "review.jsonl"
                    if old_review.exists():
                        shutil.copy2(old_review, review_path)
                    else:
                        review_path.unlink(missing_ok=True)
                    raise
                self.records = updated
                self.clear_undo_history()
                self.sample_seed = config["sample_seed"]
                self.current_name = None
                self.clear_preview()
            else:
                if missing_seed:
                    replace_project_config(path, config)
                self.records = load_records(output / "review.jsonl")
        else:
            if (output / "review.jsonl").exists():
                raise ValueError("结果目录有 review.jsonl 但缺少 project.json，请换目录")
            output.mkdir(parents=True, exist_ok=True)
            replace_project_config(path, config)
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
        self.set_config_controls(not (self.thread and self.thread.isRunning()))
        return config

    def records_after_config_change(self, files, previous, config):
        return {
            image.name: {"name": image.name, "objects": [], "candidates": [],
                         "candidate_count": 0, "status": "unlabeled",
                         "reason": "项目配置已更新，等待重新自动检测"}
            for image in files
        }

    def apply_sampling(self, files: list[Path], sampled: list[Path], percent: int):
        """Persist exclusions so resume and review display the same image selection."""
        selected_names = {path.name for path in sampled}
        updated = dict(self.records)
        changed = False
        dropped_results = False
        for path in files:
            previous = updated.get(path.name)
            if path.name in selected_names:
                if previous is not None and previous.get("status") == "skipped":
                    updated[path.name] = {
                        "name": path.name, "objects": [], "candidates": [],
                        "candidate_count": 0, "status": "unlabeled",
                        "reason": "已纳入抽样，等待自动检测",
                    }
                    changed = True
            elif previous is None or previous.get("status") != "skipped":
                if previous is not None and previous.get("status") not in {"unlabeled", "skipped"}:
                    dropped_results = True
                updated[path.name] = {
                    "name": path.name, "objects": [], "candidates": [],
                    "candidate_count": 0, "status": "skipped",
                    "reason": self.skipped_sampling_reason(percent),
                }
                changed = True
        if changed:
            if dropped_results:
                backup_project(self.active_output)
            replace_review_records(self.active_output / "review.jsonl", updated)
            self.records = updated
            self.clear_undo_history()

    def skipped_sampling_reason(self, percent):
        return f"未抽中：自动标注图片比例 {percent}%"

    def validate_model_ready(self):
        if self.segmenter is None and not SAM3_MODEL.is_file():
            raise FileNotFoundError(f"找不到 SAM3 模型: {SAM3_MODEL}")

    def make_detection_thread(self, files, config, done_names):
        return GenericThread(files, config, done_names, self.segmenter)

    def sampling_candidates(self, files, config):
        """Return images eligible for random sampling in this workflow."""
        return files

    def start_detection(self):
        try:
            config = self.config()
            files = image_files(self.source_dir())
            if not files:
                raise ValueError("图片目录为空")
            if not self.sampling_candidates(files, config):
                raise ValueError("抽帧设置未选中任何图片")
            self.validate_model_ready()
            config = self.ensure_project(allow_reconfigure=True)
            candidates = self.sampling_candidates(files, config)
            sampled = sample_image_files(candidates, config["sample_percent"],
                                         config["sample_seed"])
            self.apply_sampling(files, sampled, config["sample_percent"])
            self.update_sample_summary()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法开始", str(exc))
            return
        self.table.setRowCount(0)
        self.row_for_name.clear()
        for record in sorted(self.records.values(), key=lambda item: item["name"]):
            self.update_table(record)
        # Keep manual decisions, and retry images without labels or held by the old rule.
        done_names = {
            name for name, record in self.records.items()
            if record.get("status") in {"confirmed", "empty_confirmed"} or
            (record.get("objects") and "candidates" in record and
             record.get("reason") != "高分候选超过设定数量，请人工选择" and
             not has_duplicate_selected_boxes(record))
        }
        self._deleted_during_run.clear()
        self.clear_undo_history()
        self.thread = self.make_detection_thread(sampled, config, done_names)
        self.thread.result_signal.connect(self.receive_result)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.message_signal.connect(self.status.setText)
        self.thread.finished.connect(self.detection_finished)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.output_edit.setEnabled(False)
        self.path_buttons[1].setEnabled(False)
        self.set_config_controls(False)
        self.thread.start()

    def stop_detection(self):
        if self.thread and self.thread.isRunning():
            self.thread.stop()
            self.status.setText("正在停止；当前图片完成后退出。")

    def detection_finished(self):
        if self.thread.segmenter is not None:
            self.segmenter = self.thread.segmenter
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.export_button.setEnabled(True)
        self.output_edit.setEnabled(True)
        self.path_buttons[1].setEnabled(True)
        self.set_config_controls(True)
        self.update_counts()
        self.thread = None

    def receive_result(self, record):
        if record["name"] in self._deleted_during_run:
            return
        if self.records.get(record["name"], {}).get("status") in {
                "confirmed", "empty_confirmed"}:
            return
        append_record(self.active_output / "review.jsonl", record)
        self._undo_stack = [entry for entry in self._undo_stack
                            if entry["name"] != record["name"]]
        self._update_undo_button()
        self.records[record["name"]] = record
        self.update_table(record)
        self.update_counts()
        if self.follow_check.isChecked():
            self.table.blockSignals(True)
            self.table.selectRow(self.row_for_name[record["name"]])
            self.table.blockSignals(False)
            self.show_record(record["name"])
        elif record["name"] == self.current_name:
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
            if column == 1 and record["status"] in {"review", "unlabeled"}:
                item.setForeground(QColor("#bf5b00"))
            elif column == 1 and record["status"] == "skipped":
                item.setForeground(QColor("#777777"))
            self.table.setItem(row, column, item)
        self.table.setRowHidden(row, self.review_filter.isChecked() and
                                record["status"] not in {"review", "unlabeled"})

    def apply_filter(self):
        if self.review_filter.isChecked():
            self.follow_check.setChecked(False)
        for name, row in self.row_for_name.items():
            self.table.setRowHidden(row, self.review_filter.isChecked() and
                                    self.records[name]["status"] not in {"review", "unlabeled"})

    def select_table_row(self):
        row = self.table.currentRow()
        if row < 0 or self.table.item(row, 0) is None:
            return
        name = self.table.item(row, 0).text()
        if name != self.current_name:
            self.follow_check.setChecked(False)
            self.show_record(name)

    def color_for_candidate(self, candidate: dict, record: dict) -> tuple[int, int, int]:
        return SELECTED_COLOR if candidate["selected"] else UNSELECTED_COLOR

    def candidate_is_visible(self, candidate: dict, record: dict) -> bool:
        return True

    def display_class_name(self, class_id: int, record: dict,
                           selected: bool = True) -> str:
        return self.active_config["classes"][class_id]["name"]

    def show_record(self, name):
        record = self.records.get(name)
        if record is None:
            return
        display_record = sync_candidates(record)
        self.current_name = name
        image = cv2.imread(str(self.active_source / name))
        if image is None:
            self.preview.setText("无法读取图片")
            return
        if isinstance(self.roi, dict):
            cx, cy, radius = circle_pixels(self.roi, image.shape[1], image.shape[0])
            cv2.circle(image, (round(cx), round(cy)), round(radius), (30, 30, 230), 2)
        else:
            x1, y1, x2, y2 = region_bounds(self.roi, image.shape[1], image.shape[0])
            cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (30, 30, 230), 2)
        self._loading_objects = True
        self.object_table.setRowCount(0)
        captions = []
        for candidate in sorted(display_record["candidates"],
                                key=lambda item: bool(item["selected"])):
            if not self.candidate_is_visible(candidate, record):
                continue
            class_id = int(candidate["class_id"])
            color = self.color_for_candidate(candidate, record)
            points = np.rint(candidate["polygon"]).astype(np.int32)
            cv2.polylines(image, [points], True, color, 3)
            label = self.display_class_name(class_id, record, candidate["selected"])
            anchor = points.min(axis=0)
            captions.append((max(2, int(anchor[0])), max(24, int(anchor[1]) - 5),
                             f"{label} {float(candidate.get('score', 0)):.2f}", color))
        for index, obj in enumerate(record.get("objects", [])):
            class_id = int(obj["class_id"])
            label = self.display_class_name(class_id, record)
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
            f"已选 {len(record.get('objects', []))} / 候选 {len(display_record['candidates'])}  |  "
            f"{record.get('reason', '')}"
        )

    def preview_clicked(self, position):
        if self.current_name is None or self.preview_pixmap is None:
            return
        shown = self.preview.pixmap()
        if shown is None or shown.isNull():
            return
        offset_x = (self.preview.width() - shown.width()) / 2
        offset_y = (self.preview.height() - shown.height()) / 2
        if not (offset_x <= position.x() < offset_x + shown.width() and
                offset_y <= position.y() < offset_y + shown.height()):
            return
        x = (position.x() - offset_x) * self.preview_pixmap.width() / shown.width()
        y = (position.y() - offset_y) * self.preview_pixmap.height() / shown.height()
        self.toggle_candidate_at(x, y)

    def toggle_candidate_at(self, x, y):
        record = sync_candidates(self.records[self.current_name])
        hits = []
        for index, candidate in enumerate(record["candidates"]):
            if not self.candidate_is_visible(candidate, record):
                continue
            polygon = np.asarray(candidate["polygon"], dtype=np.float32)
            if (polygon.shape == (4, 2) and
                    cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0):
                hits.append((not candidate["selected"], cv2.contourArea(polygon), index))
        if not hits:
            return
        _, _, candidate_id = min(hits)
        candidate = record["candidates"][candidate_id]
        objects = [obj for obj in record["objects"]
                   if obj["candidate_id"] != candidate_id]
        if not candidate["selected"]:
            objects.append({"class_id": candidate["class_id"],
                            "score": candidate.get("score", 1.0),
                            "polygon": candidate["polygon"],
                            "candidate_id": candidate_id})
        self.follow_check.setChecked(False)
        self.save_review({**record, "objects": objects,
                          "status": "confirmed" if objects else "review",
                          "reason": "人工选择检测框" if objects else
                                    "已取消全部标签框，请选择目标或确认无目标"})

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
        record = self.records[self.current_name]
        if record["status"] == "skipped":
            QMessageBox.information(
                self, "图片未抽中",
                "这张图片未被当前比例抽中，不会检测或导出。"
                "如需标注它，请提高自动标注图片比例后按 F5；设为 100% 可包含全部图片。")
            return None
        return record

    def _update_undo_button(self):
        if hasattr(self, "undo_button"):
            self.undo_button.setEnabled(bool(self._undo_stack))

    def clear_undo_history(self):
        """Undo is scoped to manual edits in the current project session."""
        self._undo_stack.clear()
        self._update_undo_button()

    def undo_last_review(self):
        if not self._undo_stack:
            self.status.setText("当前没有可撤销的人工标注操作。")
            return
        entry = self._undo_stack[-1]
        name = entry["name"]
        if (self.active_output != entry["output"] or self.active_source != entry["source"]
                or self.records.get(name) != entry["after"]):
            self.clear_undo_history()
            self.status.setText("记录或项目已改变，之前的撤销历史已失效。")
            return
        previous = copy.deepcopy(entry["before"])
        try:
            append_record(self.active_output / "review.jsonl", previous)
        except OSError as exc:
            QMessageBox.warning(self, "撤销失败", f"原标注未恢复，请重试：{exc}")
            return
        self._undo_stack.pop()
        self._update_undo_button()
        if self.thread is not None:
            # Queued results can arrive even after the worker stopped running.
            self._deleted_during_run.add(name)
        self.records[name] = previous
        self.follow_check.setChecked(False)
        self.update_table(previous)
        self.apply_filter()
        row = self.row_for_name[name]
        if self.table.isRowHidden(row):
            self.review_filter.setChecked(False)
            if hasattr(self, "search_edit"):
                self.search_edit.clear()
                self.status_filter.setCurrentIndex(self.status_filter.findData("all"))
            self.apply_filter()
        self.table.blockSignals(True)
        self.table.selectRow(row)
        self.table.scrollToItem(self.table.item(row, 0))
        self.table.blockSignals(False)
        self.show_record(name)
        self.update_counts()
        self.status.setText(f"已撤销 {name} 的上一步人工标注。")

    def save_review(self, record):
        record = sync_candidates(record)
        previous = copy.deepcopy(self.records.get(record["name"]))
        try:
            append_record(self.active_output / "review.jsonl", record)
        except OSError as exc:
            QMessageBox.warning(self, "保存失败", f"当前结果未保存：{exc}")
            return False
        if previous is not None and previous != record:
            self._undo_stack.append({"name": record["name"], "before": previous,
                                     "after": copy.deepcopy(record),
                                     "source": self.active_source, "output": self.active_output})
            self._update_undo_button()
        self.records[record["name"]] = record
        self.update_table(record)
        self.show_record(record["name"])
        self.update_counts()
        return True

    def add_object(self):
        record = self.current_record()
        if record is None:
            return
        self.follow_check.setChecked(False)
        path = self.active_source / record["name"]
        dialog = GenericBoxDialog(path, f"添加目标：{record['name']}", parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        image = cv2.imread(str(path))
        if not polygon_in_region(dialog.result_polygon, self.roi, image.shape[1], image.shape[0]):
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
        dialog = GenericBoxDialog(path, f"修正目标 {index + 1}：{record['name']}",
                                 objects[index]["polygon"], parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        image = cv2.imread(str(path))
        if not polygon_in_region(dialog.result_polygon, self.roi, image.shape[1], image.shape[0]):
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
        if record["status"] == "empty_confirmed":
            self.next_review()
            return
        if not record.get("objects"):
            QMessageBox.information(self, "尚无目标", "请先补画目标框，或确认本张无目标。")
            return
        if self.save_review({**record, "status": "confirmed", "reason": "人工确认所有目标"}):
            self.next_review()

    def mark_empty(self):
        record = self.current_record()
        if record is not None:
            self.follow_check.setChecked(False)
            if self.save_review({**record, "objects": [], "status": "empty_confirmed",
                                 "reason": "人工确认区域内无目标"}):
                self.next_review()

    def clear_current_labels(self):
        """Persist an empty pending record so the current image can be detected again."""
        record = self.current_record()
        if record is None:
            return
        self.follow_check.setChecked(False)
        cleared = {**record, "objects": [], "candidates": [], "candidate_count": 0,
                   "status": "unlabeled", "reason": "标签已删除，等待重新自动检测"}
        if self.save_review(cleared):
            self._deleted_during_run.add(record["name"])
            self.stop_detection()
            self.status.setText(f"{record['name']} 的标签已清空并保存；按 F5 重新自动检测。")

    def clear_all_labels(self):
        """Clear every image in the active source directory and queue redetection."""
        try:
            files = image_files(self.source_dir())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法读取图片目录", str(exc))
            return
        if not files:
            QMessageBox.information(self, "没有图片", "当前图片目录中没有可处理的图片。")
            return
        answer = QMessageBox.question(
            self, "确认清空目录标签",
            f"将清空当前项目图片目录中 {len(files)} 张图片的全部标签，"
            "并标记为待重新检测。确认继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if answer != QMessageBox.Yes:
            return
        names = {path.name for path in files}
        try:
            self.ensure_project(allow_reconfigure=True)
            updated = dict(self.records)
            for path in files:
                updated[path.name] = {
                    **updated.get(path.name, {}), "name": path.name,
                    "objects": [], "candidates": [],
                    "candidate_count": 0, "status": "unlabeled",
                    "reason": "目录标签已批量删除，等待重新自动检测",
                }
            replace_review_records(self.active_output / "review.jsonl", updated)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "清空失败", f"目录标签未全部清空：{exc}")
            return
        self.stop_detection()
        self.clear_undo_history()
        self._deleted_during_run.update(names)
        self.follow_check.setChecked(False)
        self.records = updated
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.row_for_name.clear()
        for record in sorted(updated.values(), key=lambda item: item["name"]):
            self.update_table(record)
        name = self.current_name if self.current_name in names else files[0].name
        self.table.selectRow(self.row_for_name[name])
        self.table.blockSignals(False)
        self.show_record(name)
        self.update_counts()
        self.status.setText(
            f"已清空当前图片目录中 {len(files)} 张图片的标签；按 F5 重新自动检测。")

    def _move_review(self, direction: int):
        self.follow_check.setChecked(False)
        if not self.row_for_name:
            self.status.setText("还没有可查看的图片。")
            return
        count = self.table.rowCount()
        start = self.row_for_name.get(self.current_name,
                                      -1 if direction > 0 else 0)
        for offset in range(1, count + 1):
            row = (start + direction * offset) % count
            if self.table.isRowHidden(row):
                continue
            name = self.table.item(row, 0).text()
            if self.records[name]["status"] == "review":
                self.table.blockSignals(True)
                self.table.selectRow(row)
                self.table.scrollToItem(self.table.item(row, 0))
                self.table.blockSignals(False)
                self.show_record(name)
                return
        self.status.setText("当前没有待复核图片，已保存的确认结果可导出。")

    def next_review(self):
        self._move_review(1)

    def previous_review(self):
        self._move_review(-1)

    def _move_image(self, direction: int):
        if not self.row_for_name:
            self.status.setText("还没有可查看的图片。")
            return
        start = self.row_for_name.get(
            self.current_name, -1 if direction > 0 else self.table.rowCount())
        end = self.table.rowCount() if direction > 0 else -1
        for row in range(start + direction, end, direction):
            if self.table.isRowHidden(row):
                continue
            self.follow_check.setChecked(False)
            self.table.blockSignals(True)
            self.table.selectRow(row)
            self.table.scrollToItem(self.table.item(row, 0))
            self.table.blockSignals(False)
            self.show_record(self.table.item(row, 0).text())
            return
        self.status.setText("已经是最后一张图片。" if direction > 0
                            else "已经是第一张图片。")

    def show_next_image(self):
        self._move_image(1)

    def show_previous_image(self):
        self._move_image(-1)

    def update_counts(self):
        counts = {status: 0 for status in STATUS_TEXT}
        for record in self.records.values():
            counts[record["status"]] = counts.get(record["status"], 0) + 1
        self.status.setText(
            f"共记录 {len(self.records)} 张；自动候选 {counts['auto']}，"
            f"待复核 {counts['review']}，人工确认 {counts['confirmed']}，"
            f"确认无目标 {counts['empty_confirmed']}，待重检 {counts['unlabeled']}，"
            f"未抽中 {counts['skipped']}。"
        )

    def export(self):
        try:
            config = self.ensure_project()
            candidates = self.sampling_candidates(image_files(self.active_source), config)
            allowed_names = {path.name for path in sample_image_files(
                candidates, config["sample_percent"], config["sample_seed"])}
            target, images, objects = export_generic(
                self.active_source, self.active_output, self.records,
                config["classes"], self.format_combo.currentData(), allowed_names)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        unresolved = sum(record["status"] in {"review", "unlabeled"}
                         for record in self.records.values())
        skipped = len(image_files(self.active_source)) - len(allowed_names)
        self.status.setText(
            f"已导出 {images} 张有目标图片、{objects} 个目标；"
            f"{unresolved} 张待复核/待重检、{skipped} 张未抽中未导出。目录：{target}"
        )
        QMessageBox.information(self, "导出完成",
                                f"数据集：{target}\n待复核/待重检未导出：{unresolved} 张"
                                f"\n未抽中未导出：{skipped} 张"
                                f"\n可视化预览：{target / 'visualizations'}")

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
