"""Separate SAM3 gray glove annotator with video import and manual review."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PyQt5.QtCore import QCoreApplication, QLibraryInfo, QThread, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog,
    QDoubleSpinBox, QFileDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QMessageBox, QPushButton, QScrollArea, QShortcut, QSpinBox,
    QVBoxLayout,
)

from .app import torch
from .app_generic import GenericAnnotatorApp, GenericSAM3Segmenter, GenericThread
from .generic_core import detect_generic
from .gray_glove_video import collect_videos, extract_video_batch, frame_camera_ids
from .gray_glove_mcap import CAMERA_TOPICS
from .orbbec_undistort import OrbbecUndistorter
from .project_paths import ROOT_DIR, SAM3_MODEL


DEFAULT_VIDEO = Path(
    "/home/p3/data_sda1/ego_hand_system/recordings/20261027/"
    "EGO-4B_AZLCA610091_20261027_110819/"
    "EGO-4B_AZLCA610091_20261027_110819_camera_left.mp4")
DEFAULT_OUTPUT = ROOT_DIR / "data" / "gray_glove_lr_annotations"
DEFAULT_ROI = [0.0, 0.42, 1.0, 1.0]
GRAY_GLOVE_CLASSES = [
    {"name": "left", "prompt": "left hand wearing a light gray glove"},
    {"name": "right", "prompt": "right hand wearing a light gray glove"},
]
GRAY_GLOVE_PROMPTS = (
    "hand wearing a light gray glove",
    "hand wearing a gray protective work glove",
    "hand wearing a protective work glove",
)
WORKFLOW = "gray_glove_lr_v2"


def _center_x(polygon) -> float:
    return float(np.asarray(polygon, dtype=np.float32).reshape(4, 2)[:, 0].mean())


def assign_glove_sides(record: dict, width: int, mapping: str = "screen") -> dict:
    """Apply the prior screen/mirror convention to a pair of glove boxes.

    A lone glove gets a visual suggestion but remains pending until a person
    explicitly assigns left or right. This avoids positional guesses for a
    hand that crosses the image center.
    """
    if mapping not in {"screen", "mirror"}:
        raise ValueError(f"未知左右手映射方式: {mapping}")
    result = {**record, "objects": [dict(obj) for obj in record.get("objects", [])],
              "candidates": [dict(item) for item in record.get("candidates", [])]}
    objects = result["objects"]
    if len(objects) == 2:
        ordered = sorted(objects, key=lambda obj: _center_x(obj["polygon"]))
        for index, obj in enumerate(ordered):
            obj["class_id"] = index if mapping == "screen" else 1 - index
        result["side_pending"] = False
        if abs(_center_x(ordered[0]["polygon"]) -
               _center_x(ordered[1]["polygon"])) < width * 0.05:
            result.update(status="review", reason="两只手套位置靠近，请人工核对左右类别")
    elif len(objects) == 1:
        suggested = int(_center_x(objects[0]["polygon"]) >= width / 2)
        objects[0]["class_id"] = suggested if mapping == "screen" else 1 - suggested
        result.update(side_pending=True, status="review",
                      reason="单手左右未定：请按 Alt+1 或 Alt+2 明确指定")
    elif len(objects) > 2:
        result.update(side_pending=True, status="review",
                      reason="超过两只手套，请删除误检并核对左右类别")
    else:
        result["side_pending"] = False
    _update_candidate_sides(result, width, mapping)
    return result


def _update_candidate_sides(record: dict, width: int, mapping: str):
    objects = record["objects"]
    for candidate in record.get("candidates", []):
        x = _center_x(candidate["polygon"])
        if objects:
            nearest = min(objects, key=lambda obj: abs(_center_x(obj["polygon"]) - x))
            candidate["class_id"] = nearest["class_id"]
        else:
            side = int(x >= width / 2)
            candidate["class_id"] = side if mapping == "screen" else 1 - side


def flag_side_camera_review(record: dict, camera_id: str | None) -> dict:
    """Side views need a person to confirm physical left/right identity."""
    if camera_id is None:
        return record
    result = {**record, "camera_id": camera_id}
    if (camera_id in {"cam_2", "cam_3"} and result.get("objects")
            and result.get("status") == "auto"):
        result["status"] = "review"
        result["reason"] = "侧视相机的左右手仅按画面位置建议，请人工确认"
    return result


class LargePreviewDialog(QDialog):
    """Inspect the full-resolution image with its current selection colors."""

    def __init__(self, pixmap, name, parent=None):
        super().__init__(parent)
        self.source_pixmap = pixmap
        self.setWindowTitle(f"放大查看：{name}")
        screen = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1600, max(600, screen.width() - 80)),
                    min(900, max(450, screen.height() - 80)))
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
        self.scroll = QScrollArea()
        self.scroll.setWidget(self.image_label)
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.scroll)
        self.update_zoom()
        QTimer.singleShot(0, self.fit_to_window)

    def update_zoom(self):
        target = self.source_pixmap.size() * (self.zoom_spin.value() / 100)
        scaled = self.source_pixmap.scaled(
            target, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())

    def fit_to_window(self):
        viewport = self.scroll.viewport().size()
        if self.source_pixmap.width() and self.source_pixmap.height():
            scale = min(viewport.width() / self.source_pixmap.width(),
                        viewport.height() / self.source_pixmap.height())
            self.zoom_spin.setValue(max(25, min(400, int(scale * 100))))


class GrayGloveSegmenter(GenericSAM3Segmenter):
    """Merge several glove descriptions before assigning left/right labels."""

    def detect_many(self, image_bgr, _prompts):
        detections = super().detect_many(image_bgr, list(GRAY_GLOVE_PROMPTS))
        return [{**detection, "class_id": 0} for detection in detections]


class GrayGloveThread(GenericThread):
    def run(self):
        try:
            todo = [path for path in self.files if path.name not in self.done_names]
            camera_ids = frame_camera_ids(Path(self.config["source_dir"]))
            total = len(self.files)
            completed = total - len(todo)
            self.progress_signal.emit(completed, total)
            if not todo:
                self.message_signal.emit("抽中图片已有结果；可复核或导出。")
                return
            device = "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"
            if self.segmenter is None or self.segmenter.device != device:
                self.message_signal.emit(f"正在加载 SAM3（{device}），待检测 {len(todo)} 张...")
                self.segmenter = GrayGloveSegmenter(
                    model_path=str(SAM3_MODEL), device=device,
                    conf=self.config["confidence"])
            else:
                self.segmenter.conf = float(self.config["confidence"])
                self.message_signal.emit(f"复用 SAM3（{device}），待检测 {len(todo)} 张...")
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
                        self.config.get("selection_rule"))
                    result = assign_glove_sides(
                        result, image.shape[1],
                        self.config.get("side_mapping", "screen"))
                    result = flag_side_camera_review(
                        result, camera_ids.get(path.name))
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
                "检测已停止。" if not self.running else "检测完成，请检查待复核图片。")
        except Exception as exc:
            self.message_signal.emit(f"灰色手套检测无法启动: {exc}")


class VideoFrameThread(QThread):
    progress_signal = pyqtSignal(int, int)
    complete_signal = pyqtSignal(str, int, int, str)
    error_signal = pyqtSignal(str)

    def __init__(self, videos: list[Path], output: Path, interval_mode: str,
                 interval_value: float, start_seconds: float,
                 end_seconds: float | None, target_dir: Path | None,
                 undistort: bool = False, calibration: Path | None = None,
                 balance: float = 0.0, mcap_camera_ids=None):
        super().__init__()
        self.videos = videos
        self.output = output
        self.interval_mode = interval_mode
        self.interval_value = interval_value
        self.start_seconds = start_seconds
        self.end_seconds = end_seconds
        self.target_dir = target_dir
        self.undistort = undistort
        self.calibration = calibration
        self.balance = balance
        self.mcap_camera_ids = mcap_camera_ids
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        try:
            rectifier = (OrbbecUndistorter(self.videos, self.calibration, self.balance,
                                          self.mcap_camera_ids)
                         if self.undistort else None)
            folder, count, reused = extract_video_batch(
                self.videos, self.output, self.interval_mode,
                self.interval_value, self.start_seconds, self.end_seconds,
                progress=lambda done, total: self.progress_signal.emit(done, total),
                should_continue=lambda: self.running, target_dir=self.target_dir,
                frame_transform=rectifier,
                transform_info=rectifier.metadata if rectifier is not None else None,
                mcap_camera_ids=self.mcap_camera_ids)
            self.complete_signal.emit(str(folder), count, reused, str(self.output))
        except Exception as exc:
            self.error_signal.emit(str(exc))


class GrayGloveAnnotatorApp(GenericAnnotatorApp):
    """Keep generic review tools while applying gray glove defaults."""

    def __init__(self):
        self._building_gray = True
        self.video_thread = None
        self._visible_rows = 0
        self._run_started_at = None
        self._run_start_done = None
        self._focus_view = False
        super().__init__()
        self._building_gray = False
        self.setWindowTitle("灰色手套 · 视频抽帧 / SAM3 标注 / 人工复核")
        self.output_edit.setText(str(DEFAULT_OUTPUT))
        self.set_classes(GRAY_GLOVE_CLASSES)
        self.class_table.setHorizontalHeaderLabels(["导出类别", "SAM3 提示词"])
        self.class_table.setToolTip(
            "固定输出 left/right 两类；模型检索这三条手套提示词：\n" +
            "\n".join(GRAY_GLOVE_PROMPTS))
        self.settings_group = self.centralWidget().layout().itemAt(0).widget()
        self.side_combo = QComboBox()
        self.side_combo.addItem("画面左侧=left，右侧=right", "screen")
        self.side_combo.addItem("镜像：画面左侧=right，右侧=left", "mirror")
        self.side_combo.setToolTip("双手自动分配类别的规则；单手须人工指定。")
        settings_grid = self.settings_group.layout()
        settings_grid.addWidget(QLabel("左右手类别规则"), 8, 0)
        settings_grid.addWidget(self.side_combo, 8, 1, 1, 4)
        self.roi = DEFAULT_ROI.copy()
        self.show_roi()
        self.count_combo.setCurrentIndex(self.count_combo.findData("fixed"))
        self.count_spin.setValue(2)
        self.selection_combo.setCurrentIndex(self.selection_combo.findData("area"))
        self.confidence_spin.setValue(0.20)
        self.review_reason_checks["class_conflict"].setChecked(False)
        self.review_reason_checks["near_edge"].setChecked(False)
        self.format_combo.setCurrentIndex(self.format_combo.findData("hbb"))

        self.video_group = QGroupBox("视频 / MCAP 抽帧：可添加录像或文件夹，完成后直接载入标注")
        video_layout = QVBoxLayout(self.video_group)
        video_row = QHBoxLayout()
        self.video_edit = QLineEdit(str(DEFAULT_VIDEO) if DEFAULT_VIDEO.is_file() else "")
        self.video_edit.setPlaceholderText("输入一个视频路径，点“加入列表”；也可直接批量选择")
        self.video_button = QPushButton("添加视频 [Alt+V]")
        self.video_button.clicked.connect(self.choose_video)
        self.folder_button = QPushButton("添加文件夹")
        self.folder_button.clicked.connect(self.choose_video_folder)
        self.add_path_button = QPushButton("加入列表")
        self.add_path_button.clicked.connect(self.add_manual_video)
        video_row.addWidget(self.video_edit, 1)
        video_row.addWidget(self.video_button)
        video_row.addWidget(self.folder_button)
        video_row.addWidget(self.add_path_button)
        video_layout.addLayout(video_row)
        queue_row = QHBoxLayout()
        self.video_list = QListWidget()
        self.video_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.video_list.setMaximumHeight(90)
        self.video_list.setToolTip("每个视频的抽帧图片会使用唯一文件名；可以多选并移除。")
        self.remove_video_button = QPushButton("移除选中")
        self.remove_video_button.clicked.connect(self.remove_selected_videos)
        self.clear_video_button = QPushButton("清空列表")
        self.clear_video_button.clicked.connect(self.video_list.clear)
        queue_row.addWidget(self.video_list, 1)
        queue_buttons = QVBoxLayout()
        queue_buttons.addWidget(self.remove_video_button)
        queue_buttons.addWidget(self.clear_video_button)
        queue_row.addLayout(queue_buttons)
        video_layout.addLayout(queue_row)
        mcap_row = QHBoxLayout()
        mcap_row.addWidget(QLabel("MCAP 相机"))
        self.mcap_camera_checks = {}
        for camera_id, label in (("cam_0", "左主"), ("cam_1", "右主"),
                                 ("cam_2", "右侧"), ("cam_3", "左侧")):
            check = QCheckBox(f"{label} ({camera_id})")
            check.setChecked(True)
            self.mcap_camera_checks[camera_id] = check
            mcap_row.addWidget(check)
        mcap_row.addStretch(1)
        video_layout.addLayout(mcap_row)
        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("抽帧间隔"))
        self.interval_mode = QComboBox()
        self.interval_mode.addItem("每隔秒", "seconds")
        self.interval_mode.addItem("每隔帧", "frames")
        interval_row.addWidget(self.interval_mode)
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.1, 60.0)
        self.interval_spin.setSingleStep(0.25)
        self.interval_spin.setDecimals(2)
        self.interval_spin.setValue(1.0)
        self.interval_spin.setSuffix(" 秒")
        self.interval_spin.setToolTip("例如 1 秒约取 1 张；载入后还能用随机百分比抽样。")
        interval_row.addWidget(self.interval_spin)
        self._interval_values = {"seconds": 1.0, "frames": 30.0}
        self._last_interval_mode = "seconds"
        self.interval_mode.currentIndexChanged.connect(self.update_interval_mode)
        interval_row.addWidget(QLabel("开始于"))
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0, 864000)
        self.start_spin.setDecimals(2)
        self.start_spin.setSuffix(" 秒")
        interval_row.addWidget(self.start_spin)
        self.end_check = QCheckBox("结束于")
        interval_row.addWidget(self.end_check)
        self.end_spin = QDoubleSpinBox()
        self.end_spin.setRange(0, 864000)
        self.end_spin.setDecimals(2)
        self.end_spin.setValue(60)
        self.end_spin.setSuffix(" 秒")
        self.end_spin.setEnabled(False)
        self.end_check.toggled.connect(self.end_spin.setEnabled)
        interval_row.addWidget(self.end_spin)
        self.import_button = QPushButton("抽帧并载入 [Alt+G]")
        self.import_button.clicked.connect(self.import_video)
        interval_row.addWidget(self.import_button)
        self.video_progress = QLabel("尚未抽帧；也可直接选择已有图片目录")
        interval_row.addWidget(self.video_progress, 1)
        video_layout.addLayout(interval_row)
        undistort_row = QHBoxLayout()
        undistort_row.addWidget(QLabel("Orbbec 标定"))
        self.calibration_edit = QLineEdit()
        self.calibration_edit.setPlaceholderText("留空：自动查找视频目录或 MCAP 上级目录的四目标定 YAML")
        self.calibration_edit.setToolTip(
            "手动指定仅适合所选视频共用同一标定；留空可按每段视频自动匹配。")
        undistort_row.addWidget(self.calibration_edit, 1)
        self.calibration_button = QPushButton("选择标定")
        self.calibration_button.clicked.connect(self.choose_calibration)
        undistort_row.addWidget(self.calibration_button)
        undistort_row.addWidget(QLabel("保留视野"))
        self.balance_spin = QDoubleSpinBox()
        self.balance_spin.setRange(0.0, 1.0)
        self.balance_spin.setSingleStep(0.1)
        self.balance_spin.setValue(0.0)
        self.balance_spin.setToolTip("0 减少黑边并让手更大；1 保留更宽视野，可能有大片黑边。")
        undistort_row.addWidget(self.balance_spin)
        self.undistort_button = QPushButton("按当前设置去畸变抽帧并载入")
        self.undistort_button.clicked.connect(
            lambda: self.import_video(undistort=True))
        undistort_row.addWidget(self.undistort_button)
        video_layout.addLayout(undistort_row)
        outer = self.centralWidget().layout()
        outer.insertWidget(1, self.video_group)
        self.focus_button = QPushButton("进入大图复核 [Alt+S]")
        self.focus_button.clicked.connect(self.toggle_focus_view)
        outer.insertWidget(0, self.focus_button)
        self.video_shortcuts = []
        for key, button in (("Alt+V", self.video_button),
                            ("Alt+G", self.import_button)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(button.click)
            self.video_shortcuts.append(shortcut)

        left_layout = self.table.parentWidget().layout()
        search_row = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索图片名")
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
        self.visible_count = QLabel("显示 0/0 张")
        left_layout.insertWidget(2, self.visible_count)

        self.large_preview_button = QPushButton("放大查看检测图 [Alt+Z]")
        self.large_preview_button.clicked.connect(self.open_large_preview)
        self.preview.parentWidget().layout().insertWidget(3, self.large_preview_button)
        self.large_preview_shortcut = QShortcut(QKeySequence("Alt+Z"), self)
        self.large_preview_shortcut.setContext(Qt.WindowShortcut)
        self.large_preview_shortcut.activated.connect(self.large_preview_button.click)
        side_row = QHBoxLayout()
        side_row.addWidget(QLabel("单手类别确认 / 双手交换："))
        self.side_left_button = QPushButton("设为 left [Alt+1]")
        self.side_right_button = QPushButton("设为 right [Alt+2]")
        self.swap_button = QPushButton("交换左右 [Alt+X]")
        self.side_left_button.clicked.connect(lambda: self.assign_current_side(0))
        self.side_right_button.clicked.connect(lambda: self.assign_current_side(1))
        self.swap_button.clicked.connect(self.swap_current_sides)
        for button in (self.side_left_button, self.side_right_button, self.swap_button):
            side_row.addWidget(button)
        self.preview.parentWidget().layout().insertLayout(5, side_row)
        for key, button in (("Alt+1", self.side_left_button),
                            ("Alt+2", self.side_right_button),
                            ("Alt+X", self.swap_button),
                            ("Alt+S", self.focus_button)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(button.click)
            self.video_shortcuts.append(shortcut)
        self.progress_detail = QLabel("检测尚未开始")
        self.summary = QLabel()
        outer.insertWidget(outer.count() - 1, self.progress_detail)
        outer.insertWidget(outer.count() - 1, self.summary)

        self.load_project()
        self.set_config_controls(True)
        self.apply_filter()
        self.update_counts()
        content = self.takeCentralWidget()
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setWidget(content)
        self.setCentralWidget(scroller)
        self.resize(1500, 950)

    def config(self):
        config = super().config()
        config["workflow"] = WORKFLOW
        config["model_path"] = str(SAM3_MODEL.expanduser().resolve())
        config["prompt_ensemble"] = list(GRAY_GLOVE_PROMPTS)
        config["side_mapping"] = self.side_combo.currentData()
        return config

    def load_project(self):
        if self._building_gray:
            return
        self._visible_rows = 0
        path = self.output_dir() / "project.json"
        if path.is_file():
            try:
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous.get("workflow") != WORKFLOW:
                    raise ValueError("结果目录属于其他标注任务，请另选目录")
            except (OSError, ValueError) as exc:
                self.status.setText(f"读取灰色手套项目失败: {exc}")
                return
        super().load_project()
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            self.side_combo.setCurrentIndex(
                self.side_combo.findData(previous.get("side_mapping", "screen")))
        self.set_classes(GRAY_GLOVE_CLASSES)
        self.set_config_controls(True)

    def ensure_project(self, allow_reconfigure=False):
        path = self.output_dir() / "project.json"
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("workflow") != WORKFLOW:
                raise ValueError("结果目录属于其他标注任务，请另选目录")
        return super().ensure_project(allow_reconfigure=allow_reconfigure)

    def set_config_controls(self, enabled):
        super().set_config_controls(enabled)
        self.class_table.setEnabled(False)
        for button in self.class_buttons:
            button.setEnabled(False)
        if hasattr(self, "side_combo"):
            self.side_combo.setEnabled(enabled)
        if hasattr(self, "video_edit"):
            available = enabled and not (self.video_thread and self.video_thread.isRunning())
            for widget in (self.video_edit, self.video_button,
                           self.folder_button, self.add_path_button,
                           self.video_list, self.remove_video_button,
                           self.clear_video_button, self.interval_mode,
                           self.interval_spin, self.start_spin, self.end_check,
                           self.import_button, self.calibration_edit,
                           self.calibration_button, self.balance_spin,
                           self.undistort_button):
                widget.setEnabled(available)
            for check in self.mcap_camera_checks.values():
                check.setEnabled(available)
            self.end_spin.setEnabled(available and self.end_check.isChecked())

    def make_detection_thread(self, files, config, done_names):
        return GrayGloveThread(files, config, done_names, self.segmenter)

    def apply_filter(self, *_args):
        if not hasattr(self, "search_edit"):
            return super().apply_filter()
        query = self.search_edit.text().strip().casefold()
        wanted = self.status_filter.currentData()
        review_only = self.review_filter.isChecked()
        if query or wanted != "all" or review_only:
            self.follow_check.setChecked(False)
        visible = 0
        for name, row in self.row_for_name.items():
            status = self.records[name]["status"]
            show = (query in name.casefold() and
                    wanted in {"all", status} and
                    (not review_only or status in {"review", "unlabeled"}))
            self.table.setRowHidden(row, not show)
            visible += show
        self._visible_rows = visible
        self.visible_count.setText(f"显示 {visible}/{len(self.row_for_name)} 张")

    def update_table(self, record):
        tracked = hasattr(self, "search_edit")
        old_row = self.row_for_name.get(record["name"])
        was_visible = (tracked and old_row is not None and
                       not self.table.isRowHidden(old_row))
        super().update_table(record)
        if tracked:
            name, status = record["name"], record["status"]
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
        if self.video_thread is not None and self.video_thread.isRunning():
            self.status.setText("抽帧完成后才能开始自动检测。")
            return
        self._run_started_at = time.monotonic()
        self._run_start_done = None
        self._visible_rows = 0
        super().start_detection()
        self.apply_filter()
        if self.thread is not None and self.thread.isRunning():
            self.progress_detail.setText("正在准备检测...")
            if not self._focus_view:
                self.toggle_focus_view()

    def update_progress(self, done, total):
        super().update_progress(done, total)
        if self._run_started_at is None:
            self._run_started_at = time.monotonic()
        if self._run_start_done is None or done < self._run_start_done:
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

    def clear_all_labels(self):
        super().clear_all_labels()
        self.apply_filter()

    def show_record(self, name):
        super().show_record(name)
        record = self.records.get(name)
        if record and record.get("camera_id"):
            self.preview_title.setText(
                self.preview_title.text() + f"  |  {record['camera_id']}")
        if record and record.get("side_pending") and record.get("objects"):
            self.preview_title.setText(
                self.preview_title.text() + "  |  左右未确认：按 Alt+1 / Alt+2")

    def save_review(self, record):
        """Keep unresolved single-hand frames out of the exported dataset."""
        name = record["name"]
        previous_count = len(self.records.get(name, {}).get("objects", []))
        objects = record.get("objects", [])
        width = int(record.get("width") or
                    (self.preview_pixmap.width() if self.preview_pixmap else 1))
        mapping = self.side_combo.currentData()
        if len(objects) == 2 and previous_count != 2:
            record = assign_glove_sides(record, width, mapping)
        elif len(objects) == 1:
            record = {**record, "objects": [dict(objects[0])]}
            pending = bool(record.get("side_pending")) or previous_count == 0
            if record.get("reason") in {"人工指定左右类别", "人工修正类别"}:
                pending = False
            record["side_pending"] = pending
            if pending and record.get("status") not in {"unlabeled", "empty_confirmed"}:
                record.update(status="review",
                              reason="单手左右未定：请按 Alt+1 或 Alt+2 明确指定")
        elif len(objects) > 2:
            record.update(status="review", side_pending=True,
                          reason="超过两只手套，请删除误检并核对左右类别")
        else:
            record["side_pending"] = False
        if len(record.get("objects", [])) == 2:
            classes = [obj.get("class_id") for obj in record["objects"]]
            if set(classes) != {0, 1}:
                record.update(status="review", reason="两只手套须分别标为 left 和 right")
        return super().save_review(record)

    def confirm_current(self):
        record = self.current_record()
        if record is None:
            return
        if record.get("side_pending") and record.get("objects"):
            QMessageBox.information(
                self, "左右待确认", "请先按 Alt+1 或 Alt+2 明确指定单手类别。")
            return
        objects = record.get("objects", [])
        if len(objects) > 2 or (len(objects) == 2 and
                                {obj.get("class_id") for obj in objects} != {0, 1}):
            QMessageBox.information(
                self, "左右类别有冲突", "每张最多两只手套，且 left/right 各最多一个。")
            return
        super().confirm_current()

    def assign_current_side(self, class_id: int):
        record = self.current_record()
        if record is None:
            return
        objects = [dict(obj) for obj in record.get("objects", [])]
        row = self.object_table.currentRow()
        if len(objects) == 1:
            row = 0
        if not 0 <= row < len(objects):
            QMessageBox.information(self, "先选手套框", "请在图片下方选择一个手套框。")
            return
        objects[row]["class_id"] = class_id
        self.follow_check.setChecked(False)
        if self.save_review({**record, "objects": objects, "side_pending": False,
                             "status": "confirmed", "reason": "人工指定左右类别"}):
            self.object_table.selectRow(row)

    def swap_current_sides(self):
        record = self.current_record()
        if record is None:
            return
        objects = [dict(obj) for obj in record.get("objects", [])]
        if len(objects) != 2 or {obj.get("class_id") for obj in objects} != {0, 1}:
            QMessageBox.information(self, "无法交换", "请先保留两只手套，且类别分别为 left/right。")
            return
        for obj in objects:
            obj["class_id"] = 1 - obj["class_id"]
        self.follow_check.setChecked(False)
        self.save_review({**record, "objects": objects, "side_pending": False,
                          "status": "confirmed", "reason": "人工交换左右类别"})

    def toggle_focus_view(self):
        self._focus_view = not self._focus_view
        self.settings_group.setVisible(not self._focus_view)
        self.video_group.setVisible(not self._focus_view)
        if self._focus_view:
            self.preview.setMinimumSize(900, 620)
        else:
            self.preview.setMinimumSize(650, 340)
        self.focus_button.setText(
            "展开项目设置 [Alt+S]" if self._focus_view else "进入大图复核 [Alt+S]")
        if self._focus_view:
            self.showMaximized()
            self.centralWidget().verticalScrollBar().setValue(0)
        else:
            self.showNormal()
            self.resize(1500, 950)

    def export(self):
        for record in self.records.values():
            if record.get("status") not in {"auto", "confirmed"}:
                continue
            objects = record.get("objects", [])
            classes = [obj.get("class_id") for obj in objects]
            if (record.get("side_pending") or len(objects) > 2 or
                    any(class_id not in (0, 1) for class_id in classes) or
                    len(set(classes)) != len(classes)):
                QMessageBox.warning(
                    self, "导出前需复核", f"{record['name']} 的左右类别未确认或重复。")
                return
        super().export()

    def open_large_preview(self):
        if self.current_name is None or self.preview_pixmap is None:
            self.status.setText("请先在左侧选择一张已检测的图片。")
            return
        LargePreviewDialog(self.preview_pixmap, self.current_name, self).exec_()

    def choose_video(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择戴灰色手套的视频", self.video_edit.text(),
            "录像 (*.mp4 *.mov *.avi *.mkv *.webm *.m4v *.mcap)")
        if paths:
            self.add_video_paths(paths)

    def choose_video_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "添加视频文件夹（递归查找）", self.video_edit.text())
        if folder:
            self.add_video_paths([folder])

    def choose_calibration(self):
        start = self.calibration_edit.text().strip()
        if not start and self.video_list.count():
            start = str(Path(self.video_list.item(0).text()).parent)
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 Orbbec 四目标定 YAML", start,
            "相机标定 (*.yaml *.yml)")
        if path:
            self.calibration_edit.setText(path)

    def add_video_paths(self, paths):
        try:
            videos = collect_videos(paths)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法添加视频", str(exc))
            return
        existing = {self.video_list.item(index).text()
                    for index in range(self.video_list.count())}
        for video in videos:
            if str(video) not in existing:
                self.video_list.addItem(str(video))
        self.video_edit.clear()
        self.video_progress.setText(f"已加入 {self.video_list.count()} 个录像文件；可设时间范围再抽帧")

    def add_manual_video(self):
        raw = self.video_edit.text().strip()
        if raw:
            self.add_video_paths([raw])

    def remove_selected_videos(self):
        for item in self.video_list.selectedItems():
            self.video_list.takeItem(self.video_list.row(item))
        self.video_progress.setText(f"列表中还有 {self.video_list.count()} 个录像文件")

    def update_interval_mode(self):
        mode = self.interval_mode.currentData()
        self._interval_values[self._last_interval_mode] = self.interval_spin.value()
        if mode == "frames":
            self.interval_spin.setDecimals(0)
            self.interval_spin.setRange(1, 1000000)
            self.interval_spin.setSingleStep(1)
            self.interval_spin.setSuffix(" 帧")
        else:
            self.interval_spin.setDecimals(2)
            self.interval_spin.setRange(0.1, 3600)
            self.interval_spin.setSingleStep(0.25)
            self.interval_spin.setSuffix(" 秒")
        self.interval_spin.setValue(self._interval_values[mode])
        self._last_interval_mode = mode

    def video_target_dir(self, output: Path) -> Path:
        """Append to an existing video project so its reviewed labels remain visible."""
        source = self.active_source
        if (source is not None and self.active_output == output and source.is_dir()
                and source.is_relative_to(output / "video_frames")):
            return source
        project_path = output / "project.json"
        if project_path.is_file():
            try:
                previous = json.loads(project_path.read_text(encoding="utf-8"))
                source = Path(previous["source_dir"]).expanduser().resolve()
                if source.is_dir() and source.is_relative_to(output / "video_frames"):
                    return source
            except (OSError, ValueError, KeyError, TypeError):
                pass
        return output / "video_frames" / "collection"

    def output_for_video_mode(self, undistort: bool) -> Path:
        """Use separate annotation projects for raw and rectified coordinates."""
        output = self.output_dir()
        suffix = "_undistorted"
        if undistort and not output.name.endswith(suffix):
            return output.with_name(output.name + suffix)
        if not undistort and output.name.endswith(suffix):
            return output.with_name(output.name[:-len(suffix)])
        return output

    def import_video(self, undistort: bool = False):
        if self.thread is not None and self.thread.isRunning():
            self.status.setText("请先停止自动检测，再切换视频。")
            return
        if self.video_thread is not None and self.video_thread.isRunning():
            return
        try:
            sources = [self.video_list.item(index).text()
                       for index in range(self.video_list.count())]
            if not sources and self.video_edit.text().strip():
                sources = [self.video_edit.text().strip()]
            videos = collect_videos(sources)
            mcap_camera_ids = tuple(camera_id for camera_id in CAMERA_TOPICS
                                    if self.mcap_camera_checks[camera_id].isChecked())
            if any(video.suffix.lower() == ".mcap" for video in videos) and not mcap_camera_ids:
                raise ValueError("请至少勾选一路 MCAP 相机")
            if self.end_check.isChecked() and self.end_spin.value() <= self.start_spin.value():
                raise ValueError("结束时间必须大于开始时间")
            output = self.output_for_video_mode(undistort)
            target = self.video_target_dir(output)
            calibration = (Path(self.calibration_edit.text().strip()).expanduser().resolve()
                           if undistort and self.calibration_edit.text().strip() else None)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "无法抽帧", str(exc))
            return
        self.video_thread = VideoFrameThread(
            videos, output, self.interval_mode.currentData(),
            self.interval_spin.value(), self.start_spin.value(),
            self.end_spin.value() if self.end_check.isChecked() else None,
            target, undistort, calibration, self.balance_spin.value(), mcap_camera_ids)
        self.video_thread.progress_signal.connect(self.update_video_progress)
        self.video_thread.complete_signal.connect(self.video_imported)
        self.video_thread.error_signal.connect(self.video_import_error)
        self.video_thread.finished.connect(self.video_import_finished)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.set_config_controls(False)
        action = "去畸变并抽帧" if undistort else "抽帧"
        self.video_progress.setText(f"正在{action}：{len(videos)} 个录像文件...")
        self.video_thread.start()

    def update_video_progress(self, done, total):
        self.video_progress.setText(f"抽帧 {done}/{total} 张")

    def video_imported(self, folder, count, reused, output):
        action = "去畸变抽帧" if self.video_thread and self.video_thread.undistort else "抽帧"
        if self.output_dir() != Path(output):
            self.output_edit.setText(output)
            self.load_project()
        old_source = self.input_edit.text().strip()
        self.input_edit.setText(folder)
        if old_source != folder:
            self.roi = DEFAULT_ROI.copy()
            self.show_roi()
        self.update_sample_summary()
        self.video_progress.setText(
            f"{action}完成 {count} 张（复用 {reused} 张）；按 F5 开始标注")
        self.status.setText(f"图片目录已切换到 {folder}；可调抽样比例后按 F5。")

    def video_import_error(self, message):
        self.video_progress.setText(message)
        if not message.startswith("已停止抽帧"):
            action = ("去畸变抽帧失败" if self.video_thread and self.video_thread.undistort
                      else "抽帧失败")
            QMessageBox.warning(self, action, message)

    def video_import_finished(self):
        self.video_thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.set_config_controls(True)

    def stop_detection(self):
        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_progress.setText("正在停止抽帧...")
            return
        super().stop_detection()

    def closeEvent(self, event):
        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        super().closeEvent(event)


def main() -> int:
    plugin_dir = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_dir
    QCoreApplication.setLibraryPaths([plugin_dir])
    app = QApplication(sys.argv)
    window = GrayGloveAnnotatorApp()
    window.show()
    return app.exec_()
