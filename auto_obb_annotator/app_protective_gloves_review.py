"""Protective-glove annotation with automatic detection and manual review.

The original protective-glove launcher is intentionally left untouched.  This
entry point adds a barcode-style project/review workflow, merges SAM3 prompt
candidates, and suggests labels using the chosen screen/mirror convention.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
import sys

import cv2
import numpy as np
from PyQt5.QtCore import QCoreApplication, QLibraryInfo, QObject, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QColor, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QProgressBar, QPushButton, QShortcut, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QSpinBox, QDoubleSpinBox,
    QAbstractButton, QAbstractSpinBox,
)

from . import app as base_app
from .app_barcode import SingleBoxDialog, _pixmap
from .protective_glove_core import ProtectiveGloveFilterConfig
from .protective_glove_review_core import (
    CLASS_NAMES,
    STATUS_TEXT,
    append_record,
    assign_left_right_classes,
    export_dataset,
    image_files,
    load_records,
    make_glove_record,
    normalize_record_sides,
    polygon_valid,
    resolved_sides,
    apply_temporal_side,
    select_prompted_glove_objects,
    temporal_seed_objects,
)
from .app_protective_gloves import ProtectiveGloveSAM3TextSegmenter
from .project_paths import REALSENSE_COLOR_DIR, SAM3_MODEL


DEFAULT_INPUT = REALSENSE_COLOR_DIR
DEFAULT_OUTPUT = REALSENSE_COLOR_DIR.parent / "protective_glove_annotations_review"
SIDE_PROMPTS = (
    "left hand wearing a protective work glove",
    "right hand wearing a protective work glove",
    "hand wearing a protective work glove",
)


def _objects_text(record: dict) -> str:
    objects = record.get("objects", [])
    if not objects:
        return "0"
    counts = {name: 0 for name in CLASS_NAMES}
    unknown = 0
    for obj in objects:
        class_id = obj.get("class_id")
        if class_id in (0, 1):
            counts[CLASS_NAMES[class_id]] += 1
        else:
            unknown += 1
    text = f"{counts['left']}左/{counts['right']}右"
    return f"{text}/{unknown}未定" if unknown else text


def _side_style(obj):
    class_id = obj.get("class_id")
    if class_id == 0:
        return "left", (0, 60, 255)
    if class_id == 1:
        return "right", (255, 150, 0)
    return "unknown (1=L / 2=R)", (0, 210, 255)


class ButtonShortcuts(QObject):
    """Bind the key shown on each button to its normal enabled click action."""

    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.bindings = []
        for button in window.findChildren(QAbstractButton):
            if button.window() is not window:
                continue
            match = re.search(r"\[([^\]]+)\]$", button.text())
            if match is None:
                continue
            key = match.group(1)
            shortcut = QShortcut(QKeySequence(key), window)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(button.click)
            button.setToolTip(button.toolTip() or f"快捷键：{key}")
            button.setAutoRepeat(False)
            if isinstance(button, QPushButton):
                # Enter must not trigger whichever dialog button last had focus.
                button.setAutoDefault(False)
            self.bindings.append((shortcut, key))
        QApplication.instance().focusChanged.connect(self.update)
        self.update(None, QApplication.focusWidget())

    def update(self, _old, current):
        modal = QApplication.activeModalWidget()
        in_this_window = current is not None and current.window() is self.window
        typing = in_this_window and isinstance(
            current, (QLineEdit, QAbstractSpinBox, QComboBox))
        for shortcut, key in self.bindings:
            # Function keys and modified shortcuts remain usable in path fields.
            text_key = len(key) == 1 or key in {"Space", "Delete", "Enter"}
            shortcut.setEnabled((modal is None or modal is self.window)
                                and not (typing and text_key))


class GloveBoxDialog(SingleBoxDialog):
    """Keep the existing canvas keys and add shortcuts for all dialog buttons."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        labels = {"清除框": "清除框 [Ctrl+Backspace]",
                  "保存这个框": "保存这个框 [Ctrl+S]", "取消": "取消 [Esc]"}
        for button in self.findChildren(QPushButton):
            button.setText(labels[button.text()])
        self.button_shortcuts = ButtonShortcuts(self)
        self.canvas.setFocus()


class GloveObjectsDialog(QDialog):
    """Review all glove boxes in one image and edit each side explicitly."""

    def __init__(self, image_path: Path, initial_objects: list[dict],
                 side_mapping: str = "screen", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"人工修正防护手套：{image_path.name}")
        self.image_path = Path(image_path)
        self.objects = [dict(obj) for obj in initial_objects]
        self.side_mapping = side_mapping
        image = cv2.imread(str(self.image_path))
        if image is None:
            raise ValueError(f"无法读取图片: {self.image_path}")
        self.image = image
        self.height, self.width = image.shape[:2]

        layout = QVBoxLayout(self)
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumHeight(260)
        layout.addWidget(self.preview)
        hint = QLabel(
            "按图中编号选择手套框，可重画并修改类别。画面左右是类别建议；"
            "单手或两手交叉时请核对实际左右手。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["类别", "置信度", "框中心"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.refresh_preview)
        self.table.setMinimumHeight(130)
        layout.addWidget(self.table)

        controls = QHBoxLayout()
        for text, callback in (
            ("编辑选中框 [E]", self.edit_selected),
            ("新增框 [A]", self.add_object),
            ("删除选中框 [Delete]", self.remove_selected),
            ("双手按画面分配 [L]", self.auto_assign),
            ("交换左右 [X]", self.swap_sides),
            ("选中框设为左手 [1]", lambda: self.set_selected_side(0)),
            ("选中框设为右手 [2]", lambda: self.set_selected_side(1)),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            controls.addWidget(button)
        layout.addLayout(controls)

        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("取消 [Esc]")
        cancel.clicked.connect(self.reject)
        save = QPushButton("保存人工修正 [Ctrl+S]")
        save.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout.addLayout(buttons)
        self.button_shortcuts = ButtonShortcuts(self)
        self.refresh_table()
        self.table.setFocus()

    def refresh_table(self):
        selected = max(0, self.table.currentRow())
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for index, obj in enumerate(self.objects):
            row = self.table.rowCount()
            self.table.insertRow(row)
            combo = QComboBox()
            combo.addItem("左右待定", None)
            combo.addItem("left", 0)
            combo.addItem("right", 1)
            class_id = obj.get("class_id")
            combo.setCurrentIndex(class_id + 1 if class_id in (0, 1) else 0)
            combo.currentIndexChanged.connect(
                lambda value, i=index, c=combo: self._set_class(i, c.itemData(value)))
            self.table.setCellWidget(row, 0, combo)
            self.table.setItem(row, 1, QTableWidgetItem(
                f"{float(obj.get('score', 0.0)):.2f}"))
            points = np.asarray(obj.get("polygon"), dtype=np.float32).reshape(4, 2)
            center = points.mean(axis=0)
            self.table.setItem(row, 2, QTableWidgetItem(
                f"({center[0]:.0f}, {center[1]:.0f})"))
        self.table.resizeColumnsToContents()
        if self.objects:
            self.table.selectRow(min(selected, len(self.objects) - 1))
        self.table.blockSignals(False)
        self.refresh_preview()

    def refresh_preview(self):
        image = self.image.copy()
        for index, obj in enumerate(self.objects):
            points = np.rint(obj["polygon"]).astype(np.int32)
            name, color = _side_style(obj)
            thickness = 4 if index == self.table.currentRow() else 2
            cv2.polylines(image, [points], True, color, thickness)
            x, y = points.min(axis=0)
            cv2.putText(image, f"{index + 1}: {name}",
                        (max(2, int(x)), max(20, int(y) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        self.preview.setPixmap(_pixmap(image).scaled(
            880, 420, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _set_class(self, index, class_id):
        if 0 <= index < len(self.objects):
            self.objects[index].update(
                class_id=class_id,
                side=CLASS_NAMES[class_id] if class_id in (0, 1) else "unknown",
                side_source="manual" if class_id in (0, 1) else "unresolved")
            self.refresh_preview()

    def set_selected_side(self, class_id):
        index = self.selected_index()
        if index is not None:
            self._set_class(index, class_id)
            self.refresh_table()
            self.table.setFocus()

    def selected_index(self):
        row = self.table.currentRow()
        return row if 0 <= row < len(self.objects) else None

    def edit_selected(self):
        index = self.selected_index()
        if index is None:
            QMessageBox.information(self, "先选框", "请先在表格中选择一个手套框。")
            return
        obj = self.objects[index]
        dialog = GloveBoxDialog(self.image_path, "重画手套框",
                                 obj.get("polygon"), parent=self)
        if dialog.exec_() == QDialog.Accepted:
            obj["polygon"] = dialog.result_polygon
            obj["score"] = 1.0
            self.refresh_table()
            self.table.selectRow(index)

    def add_object(self):
        if len(self.objects) >= 2:
            QMessageBox.warning(self, "最多两只", "防护手套任务最多保留 left/right 两个框。")
            return
        dialog = GloveBoxDialog(self.image_path, "新增手套框", parent=self)
        if dialog.exec_() != QDialog.Accepted:
            return
        existing_side = self.objects[0].get("class_id") if self.objects else None
        class_id = 1 - existing_side if existing_side in (0, 1) else None
        self.objects.append({"class_id": class_id, "score": 1.0,
                             "side_source": "manual" if class_id in (0, 1) else "unresolved",
                             "polygon": dialog.result_polygon})
        self.refresh_table()
        self.table.selectRow(len(self.objects) - 1)

    def remove_selected(self):
        index = self.selected_index()
        if index is not None:
            self.objects.pop(index)
            self.refresh_table()

    def auto_assign(self):
        if len(self.objects) != 2:
            QMessageBox.information(self, "单手不能按位置判断", "请按 1 指定左手、2 指定右手。")
            return
        assign_left_right_classes(self.objects, self.width, self.side_mapping)
        self.refresh_table()

    def swap_sides(self):
        for obj in self.objects:
            if obj.get("class_id") in (0, 1):
                class_id = 1 - obj["class_id"]
                obj.update(class_id=class_id, side=CLASS_NAMES[class_id], side_source="manual")
        self.refresh_table()

    def accept(self):
        if len(self.objects) > 2:
            QMessageBox.warning(self, "框数量过多", "最多保留两个手套框。")
            return
        if any(obj.get("class_id") not in (0, 1) for obj in self.objects):
            QMessageBox.warning(self, "左右待定", "请先为每个框指定左手或右手；单手不能按画面位置判断。")
            return
        if len({obj["class_id"] for obj in self.objects}) != len(self.objects):
            QMessageBox.warning(self, "类别重复", "两只手套须分别标为 left 和 right。")
            return
        for obj in self.objects:
            if not polygon_valid(obj.get("polygon"), self.width, self.height):
                QMessageBox.warning(self, "框无效", "存在无效或过小的手套框，请重新编辑。")
                return
        super().accept()


class ProtectiveGloveThread(QThread):
    result_signal = pyqtSignal(object)
    progress_signal = pyqtSignal(int, int)
    message_signal = pyqtSignal(str)

    def __init__(self, files: list[Path], done_names: set[str], config: dict,
                 seed_records: dict[str, dict] | None = None):
        super().__init__()
        self.files = files
        self.done_names = done_names
        self.config = config
        self.seed_records = seed_records or {}
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
            device = "cuda:0" if base_app.torch is not None and base_app.torch.cuda.is_available() else "cpu"
            self.message_signal.emit(f"正在加载 SAM3（{device}），本次检测 {len(todo)} 张...")
            ProtectiveGloveSAM3TextSegmenter.configure_filter(
                ProtectiveGloveFilterConfig(**self.config["filter"])
            )
            segmenter = ProtectiveGloveSAM3TextSegmenter(
                model_path=str(SAM3_MODEL), device=device,
                conf=float(self.config.get("confidence", 0.25)))
            previous_record = None
            for path in self.files:
                if not self.running:
                    break
                if path.name in self.done_names:
                    # A confirmed/automatic row is both skipped for inference
                    # and a valid temporal seed for the next unresolved frame.
                    previous_record = self.seed_records.get(path.name)
                    continue
                try:
                    image = cv2.imread(str(path))
                    if image is None:
                        raise ValueError("图片无法解码")
                    prompt_results = segmenter.detect_many(image, SIDE_PROMPTS)
                    payload = select_prompted_glove_objects(
                        image, prompt_results,
                        ProtectiveGloveFilterConfig(**self.config["filter"]),
                        self.config.get("side_mapping", "screen"),
                        auto_score=float(self.config.get("auto_score", 0.70)),
                    )
                    payload = apply_temporal_side(
                        payload, temporal_seed_objects(
                            previous_record, path.name, image.shape[1], image.shape[0]),
                        image.shape[1], image.shape[0],
                        auto_score=float(self.config.get("auto_score", 0.70)),
                    )
                    record = make_glove_record(
                        path.name, image.shape[1], image.shape[0],
                        payload["objects"], payload["status"], payload["reason"],
                        payload.get("candidate_count"),
                    )
                except Exception as exc:
                    record = make_glove_record(
                        path.name, 0, 0, [], "review", f"检测失败: {exc}", 0)
                self.result_signal.emit(record)
                previous_record = record
                completed += 1
                self.progress_signal.emit(completed, total)
            self.message_signal.emit(
                "检测已停止。" if not self.running else "检测完成，请复核待处理图片。")
        except Exception as exc:
            self.message_signal.emit(f"检测无法启动: {exc}")


class ProtectiveGloveReviewApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("防护手套 left/right 自动标注与人工复核")
        self.resize(1450, 920)
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
        settings = QGroupBox("数据、左右手逻辑与模型参数")
        grid = QGridLayout(settings)
        self.input_edit = QLineEdit(str(DEFAULT_INPUT))
        self.output_edit = QLineEdit(str(DEFAULT_OUTPUT))
        self.path_buttons = []
        for row, (label, edit, chooser, shortcut_text) in enumerate((
            ("图片目录", self.input_edit, self.choose_input, "Alt+I"),
            ("结果目录", self.output_edit, self.choose_output, "Alt+O"),
        )):
            button = QPushButton(f"浏览 [{shortcut_text}]")
            button.clicked.connect(chooser)
            self.path_buttons.append(button)
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1, 1, 3)
            grid.addWidget(button, row, 4)

        self.side_combo = QComboBox()
        self.side_combo.addItem("画面左右：画面左=left、画面右=right", "screen")
        self.side_combo.addItem("镜像画面：画面左=right、画面右=left", "mirror")
        self.side_combo.setToolTip(
            "双手按画面左右分配，镜像选项交换左右；单手优先使用 SAM3 左右提示词和相邻帧，证据不足时再按 1 或 2 指定。")
        grid.addWidget(QLabel("左右手规则"), 2, 0)
        grid.addWidget(self.side_combo, 2, 1, 1, 4)

        self.sp_conf = QDoubleSpinBox()
        self.sp_conf.setRange(0.05, 0.95)
        self.sp_conf.setSingleStep(0.05)
        self.sp_conf.setValue(0.25)
        self.sp_auto = QDoubleSpinBox()
        self.sp_auto.setRange(0.0, 1.0)
        self.sp_auto.setSingleStep(0.05)
        self.sp_auto.setValue(0.70)
        self.sp_max_value = QSpinBox()
        self.sp_max_value.setRange(40, 220)
        self.sp_max_value.setValue(170)
        self.sp_dark_ratio = QDoubleSpinBox()
        self.sp_dark_ratio.setRange(0.0, 1.0)
        self.sp_dark_ratio.setSingleStep(0.05)
        self.sp_dark_ratio.setValue(0.0)
        self.sp_area = QDoubleSpinBox()
        self.sp_area.setRange(0.0, 10.0)
        self.sp_area.setSingleStep(0.05)
        self.sp_area.setValue(0.20)
        self.sp_padding = QDoubleSpinBox()
        self.sp_padding.setRange(0.0, 30.0)
        self.sp_padding.setSingleStep(1.0)
        self.sp_padding.setValue(8.0)
        self.sp_gap = QDoubleSpinBox()
        self.sp_gap.setRange(0.0, 20.0)
        self.sp_gap.setSingleStep(1.0)
        self.sp_gap.setValue(4.0)
        for widget in (self.sp_dark_ratio, self.sp_area):
            widget.setDecimals(2)
        for widget in (self.sp_padding, self.sp_gap):
            widget.setSuffix(" %")
        grid.addWidget(QLabel("SAM3 框阈值"), 3, 0)
        grid.addWidget(self.sp_conf, 3, 1)
        grid.addWidget(QLabel("自动候选最低置信度"), 3, 2)
        grid.addWidget(self.sp_auto, 3, 3)
        grid.addWidget(QLabel("最小框面积 %"), 4, 0)
        grid.addWidget(self.sp_area, 4, 1)
        grid.addWidget(QLabel("最小中心 Y"), 4, 2)
        self.sp_center_y = QDoubleSpinBox()
        self.sp_center_y.setRange(0.0, 1.0)
        self.sp_center_y.setSingleStep(0.05)
        self.sp_center_y.setValue(0.15)
        grid.addWidget(self.sp_center_y, 4, 3)
        grid.addWidget(QLabel("深色占比（0=关闭）"), 5, 0)
        grid.addWidget(self.sp_dark_ratio, 5, 1)
        grid.addWidget(QLabel("框留边 % / 碎片连接 %"), 5, 2)
        padding_row = QHBoxLayout()
        padding_row.addWidget(self.sp_padding)
        padding_row.addWidget(self.sp_gap)
        grid.addLayout(padding_row, 5, 3)
        note = QLabel(
            "重复候选先合并；两个高分独立框可自动导出，贴边不单独触发复核。"
            "单手优先由 SAM3 左右语义和相邻帧自动判断，只有证据不足时才按 1/2 指定。Space 确认已明确类别的结果。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#FF9800;font-size:11px;")
        grid.addWidget(note, 6, 0, 1, 5)
        settings.setLayout(grid)
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
        self.table.setHorizontalHeaderLabels(["图片", "状态", "对象", "原因"])
        self.table.setColumnWidth(0, 170)
        self.table.setColumnWidth(1, 110)
        self.table.setColumnWidth(2, 90)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self.select_table_row)
        left_layout.addWidget(self.table)
        filter_row = QHBoxLayout()
        self.review_filter = QCheckBox("只看待复核 [R]")
        self.review_filter.toggled.connect(self.apply_filter)
        self.review_button = QPushButton("下一张待复核 [D]")
        self.review_button.clicked.connect(self.next_review)
        filter_row.addWidget(self.review_filter)
        filter_row.addWidget(self.review_button)
        left_layout.addLayout(filter_row)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.preview_title = QLabel("检测开始后，这里会逐张显示标注结果")
        self.preview_title.setWordWrap(True)
        self.preview = QLabel("等待图片")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(700, 440)
        self.preview.setStyleSheet("background:#202020;color:white;")
        right_layout.addWidget(self.preview_title)
        right_layout.addWidget(self.preview, 1)
        edit_row = QHBoxLayout()
        self.review_actions = {}
        for text, callback in (
            ("修改/补画手套框 [E]", self.edit_current),
            ("确认并下一张待复核 [Space]", self.confirm_current),
            ("下一张 [N]", self.show_next_image),
            ("确认无手套 [Q]", self.mark_empty),
        ):
            button = QPushButton(text)
            button.clicked.connect(callback)
            edit_row.addWidget(button)
            self.review_actions[callback.__name__] = button
        right_layout.addLayout(edit_row)
        single_row = QHBoxLayout()
        self.single_left_button = QPushButton("单手：左手并下一张 [1]")
        self.single_right_button = QPushButton("单手：右手并下一张 [2]")
        self.single_left_button.clicked.connect(lambda: self.confirm_single_side(0))
        self.single_right_button.clicked.connect(lambda: self.confirm_single_side(1))
        single_row.addWidget(self.single_left_button)
        single_row.addWidget(self.single_right_button)
        right_layout.addLayout(single_row)
        splitter.addWidget(right)
        splitter.setSizes([480, 970])
        outer.addWidget(splitter, 1)

        self.progress = QProgressBar()
        outer.addWidget(self.progress)
        self.status = QLabel("两个高分独立框自动保留；单手优先由 SAM3 语义和相邻帧判断，证据不足才人工复核。")
        outer.addWidget(self.status)
        self.load_project()

        self.button_shortcuts = ButtonShortcuts(self)
        self.table.setFocus()

    def source_dir(self):
        return Path(self.input_edit.text().strip()).expanduser().resolve()

    def output_dir(self):
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

    def load_project(self):
        config_path = self.output_dir() / "project.json"
        if not config_path.exists():
            self.records.clear()
            self.table.setRowCount(0)
            self.row_for_name.clear()
            self.current_name = None
            self.active_source = None
            self.active_output = None
            self.clear_preview()
            return
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            self.input_edit.setText(config["source_dir"])
            index = self.side_combo.findData(config.get("side_mapping", "screen"))
            if index >= 0:
                self.side_combo.setCurrentIndex(index)
            self.records = load_records(self.output_dir() / "review.jsonl")
            self.table.setRowCount(0)
            self.row_for_name.clear()
            self.current_name = None
            self.active_source = Path(config["source_dir"])
            self.active_output = self.output_dir()
            self.clear_preview()
            for record in sorted(self.records.values(), key=lambda item: item["name"]):
                self.update_table(record)
            self.update_counts()
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            self.status.setText(f"读取旧项目失败: {exc}")

    def ensure_project(self):
        source, output = self.source_dir(), self.output_dir()
        if not source.is_dir():
            raise ValueError(f"图片目录不存在: {source}")
        if output == source or output.is_relative_to(source):
            raise ValueError("结果目录不能位于图片目录内部")
        config = {"source_dir": str(source),
                  "side_mapping": self.side_combo.currentData(),
                  "prompts": list(SIDE_PROMPTS)}
        config_path = output / "project.json"
        if config_path.exists():
            previous = json.loads(config_path.read_text(encoding="utf-8"))
            if previous.get("source_dir") != config["source_dir"]:
                raise ValueError("该结果目录已有其他图片目录，请换一个新结果目录")
            if previous.get("side_mapping", "screen") != config["side_mapping"]:
                raise ValueError("左右手规则已改变，请换一个新结果目录，避免混用标签")
            self.records = load_records(output / "review.jsonl")
        else:
            if (output / "review.jsonl").exists():
                raise ValueError("结果目录有 review.jsonl 但缺少 project.json，请换目录")
            output.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
            self.records = {}
        self.active_source = source
        self.active_output = output

    def _filter_config(self):
        return {
            "max_value": self.sp_max_value.value(),
            "min_dark_ratio": self.sp_dark_ratio.value(),
            "min_box_area_ratio": self.sp_area.value() / 100.0,
            "min_center_y_ratio": self.sp_center_y.value(),
            "max_hands": 2,
            "box_padding_ratio": self.sp_padding.value() / 100.0,
            "fragment_gap_ratio": self.sp_gap.value() / 100.0,
            "complete_box_enabled": True,
        }

    def start_detection(self):
        if self.thread is not None:
            return
        try:
            self.ensure_project()
            files = image_files(self.source_dir())
            if not files:
                raise ValueError("图片目录为空")
            if not SAM3_MODEL.is_file():
                raise FileNotFoundError(f"找不到 SAM3 模型: {SAM3_MODEL}")
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "无法开始", str(exc))
            return
        self.table.setRowCount(0)
        self.row_for_name.clear()
        for record in sorted(self.records.values(), key=lambda item: item["name"]):
            self.update_table(record)
        config = {"side_mapping": self.side_combo.currentData(),
                  "confidence": self.sp_conf.value(),
                  "auto_score": self.sp_auto.value(),
                  "filter": self._filter_config()}
        # Re-run unresolved rows so a changed candidate-merging policy can
        # repair an earlier batch of overly conservative ``review`` records.
        # Manual confirmations and confirmed empty frames remain untouched.
        done_names = {
            name for name, record in self.records.items()
            if record.get("status") != "review"
        }
        self.thread = ProtectiveGloveThread(files, done_names, config,
                                            seed_records=dict(self.records))
        self.thread.result_signal.connect(self.receive_result)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.message_signal.connect(self.status.setText)
        self.thread.finished.connect(self.detection_finished)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.export_button.setEnabled(False)
        self.input_edit.setEnabled(False)
        self.output_edit.setEnabled(False)
        self.side_combo.setEnabled(False)
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
        self.input_edit.setEnabled(True)
        self.output_edit.setEnabled(True)
        self.side_combo.setEnabled(True)
        for button in self.path_buttons:
            button.setEnabled(True)
        self.update_counts()
        self.thread = None

    def receive_result(self, record):
        # A queued rerun result must not overwrite a confirmation made while
        # inference was running (including an explicitly confirmed empty frame).
        previous = self.records.get(record["name"], {})
        if previous.get("status") in {"confirmed", "empty_confirmed"}:
            return
        record = normalize_record_sides(record)
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
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)

    def update_table(self, record):
        blocked = self.table.blockSignals(True)
        name = record["name"]
        if name not in self.row_for_name:
            row = self.table.rowCount()
            self.table.insertRow(row)
            self.row_for_name[name] = row
        row = self.row_for_name[name]
        values = [name, STATUS_TEXT.get(record.get("status"), record.get("status", "")),
                  _objects_text(record), record.get("reason", "")]
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            if column == 1 and record.get("status") == "review":
                item.setForeground(QColor("#bf5b00"))
            self.table.setItem(row, column, item)
        self.table.setRowHidden(row, self.review_filter.isChecked() and
                                record.get("status") != "review")
        self.table.blockSignals(blocked)

    def apply_filter(self):
        if self.review_filter.isChecked():
            self.follow_check.setChecked(False)
        for name, row in self.row_for_name.items():
            self.table.setRowHidden(row, self.review_filter.isChecked() and
                                    self.records[name].get("status") != "review")

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
        record = normalize_record_sides(record)
        self.records[name] = record
        self.current_name = name
        is_single = len(record.get("objects", [])) == 1
        unresolved_single = is_single and record["objects"][0].get("class_id") not in (0, 1)
        self.single_left_button.setEnabled(unresolved_single)
        self.single_right_button.setEnabled(unresolved_single)
        source = (self.active_source or self.source_dir()) / name
        image = cv2.imread(str(source))
        if image is None:
            self.preview.setText("无法读取图片")
            return
        for obj in record.get("objects", []):
            points = np.rint(obj["polygon"]).astype(np.int32)
            label, color = _side_style(obj)
            cv2.polylines(image, [points], True, color, 3)
            anchor_values = np.maximum(points.min(axis=0) + [3, -5], [2, 18])
            anchor = (int(anchor_values[0]), int(anchor_values[1]))
            cv2.putText(image, label, anchor, cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, color, 2, cv2.LINE_AA)
        self.preview_pixmap = _pixmap(image)
        self.refresh_preview()
        self.preview_title.setText(
            f"{name} | {STATUS_TEXT.get(record.get('status'), record.get('status'))} | "
            f"{_objects_text(record)} | {record.get('reason', '')}")

    def clear_preview(self):
        self.single_left_button.setEnabled(False)
        self.single_right_button.setEnabled(False)
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
        record = normalize_record_sides(self.records[self.current_name])
        self.records[self.current_name] = record
        return record

    def save_review(self, record):
        self.follow_check.setChecked(False)
        if record.get("status") == "confirmed":
            if not resolved_sides(record.get("objects", [])):
                QMessageBox.warning(self, "左右待定", "请明确每个框的左右类别后再确认。")
                return False
            record = normalize_record_sides(record)
        try:
            append_record(self.active_output / "review.jsonl", record)
        except OSError as exc:
            QMessageBox.warning(self, "保存失败", f"当前结果未保存：{exc}")
            return False
        self.records[record["name"]] = record
        self.update_table(record)
        self.show_record(record["name"])
        self.update_counts()
        return True

    def edit_current(self):
        record = self.current_record()
        if record is None:
            return
        self.follow_check.setChecked(False)
        dialog = GloveObjectsDialog(
            self.active_source / record["name"], record.get("objects", []),
            self.side_combo.currentData(), self)
        if dialog.exec_() != QDialog.Accepted:
            return
        image = cv2.imread(str(self.active_source / record["name"]))
        objects = dialog.objects
        if not objects:
            QMessageBox.information(self, "没有框", "请使用“确认无手套”保存空标注。")
            return
        updated = make_glove_record(
            record["name"], image.shape[1], image.shape[0], objects,
            "confirmed", "人工修正手套框和 left/right 类别", len(objects))
        self.save_review(updated)

    def confirm_current(self):
        record = self.current_record()
        if record is None:
            return
        if not record.get("objects"):
            QMessageBox.information(self, "尚无框", "请先补画手套框，或确认无手套。")
            return
        if not resolved_sides(record["objects"]):
            self.follow_check.setChecked(False)
            self.status.setText("左右待定，尚未确认。单手按 1 指定左手、2 指定右手；多框按 E 修改。")
            return
        if self.save_review({**record, "status": "confirmed",
                             "reason": "人工确认手套框及 left/right 类别"}):
            self.next_review()

    def confirm_single_side(self, class_id):
        """Explicit anatomical identity; never infer it from the single box x."""
        record = self.current_record()
        if record is None or len(record.get("objects", [])) != 1:
            return
        obj = {**record["objects"][0], "class_id": class_id,
               "side": CLASS_NAMES[class_id], "side_source": "manual"}
        if self.save_review({**record, "objects": [obj], "status": "confirmed",
                             "reason": f"人工指定单手为 {CLASS_NAMES[class_id]}"}):
            self.next_review()

    def mark_empty(self):
        record = self.current_record()
        if record is not None:
            self.save_review({**record, "objects": [], "status": "empty_confirmed",
                              "reason": "人工确认图片中无可见防护手套"})

    def show_next_image(self):
        if not self.row_for_name:
            self.status.setText("还没有可查看的图片。")
            return
        start = self.row_for_name.get(self.current_name, -1)
        for row in range(start + 1, self.table.rowCount()):
            if self.table.isRowHidden(row):
                continue
            self.follow_check.setChecked(False)
            self.table.selectRow(row)
            self.show_record(self.table.item(row, 0).text())
            return
        self.status.setText("已经是最后一张图片。")

    def next_review(self):
        self.follow_check.setChecked(False)
        if not self.row_for_name:
            return
        count = self.table.rowCount()
        start = self.row_for_name.get(self.current_name, -1)
        for offset in range(1, count + 1):
            row = (start + offset) % count
            name = self.table.item(row, 0).text()
            if self.records[name].get("status") == "review":
                self.table.blockSignals(True)
                self.table.selectRow(row)
                self.table.scrollToItem(self.table.item(row, 0))
                self.table.blockSignals(False)
                self.show_record(name)
                return
        self.status.setText("当前没有待复核图片，已保存的确认结果可导出。")

    def update_counts(self):
        counts = {status: 0 for status in STATUS_TEXT}
        for record in self.records.values():
            counts[record.get("status")] = counts.get(record.get("status"), 0) + 1
        self.status.setText(
            f"已处理 {len(self.records)} 张；自动候选 {counts['auto']}，"
            f"待复核 {counts['review']}，人工确认 {counts['confirmed']}，"
            f"确认无目标 {counts['empty_confirmed']}。")

    def export(self):
        try:
            self.ensure_project()
            target, positive, object_count = export_dataset(
                self.source_dir(), self.output_dir(), self.records,
                self.format_combo.currentData())
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        unresolved = sum(record.get("status") == "review"
                         for record in self.records.values())
        empty = sum(record.get("status") == "empty_confirmed"
                    for record in self.records.values())
        self.status.setText(
            f"已导出 {positive} 张有手套图片、{object_count} 个框、{empty} 张空标注；"
            f"{unresolved} 张待复核未导出。目录：{target}")
        QMessageBox.information(
            self, "导出完成",
            f"数据集：{target}\n有手套图片：{positive}\n手套框：{object_count}\n"
            f"待复核未导出：{unresolved}")

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
    window = ProtectiveGloveReviewApp()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
