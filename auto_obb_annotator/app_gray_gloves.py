"""Separate SAM3 gray glove annotator with video import and manual review."""

from __future__ import annotations

import copy
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
from .app_generic import (
    GenericAnnotatorApp, GenericSAM3Segmenter, GenericThread, backup_project,
    replace_project_config,
    SELECTED_COLOR, UNSELECTED_COLOR,
)
from .barcode_core import image_files, load_records
from .generic_core import (
    detect_generic, export_current_annotations, normalize_review_reasons, normalize_review_score,
    normalize_sample_percent, normalize_sample_seed, normalize_selection_rule,
    sample_image_files, validate_classes,
)
from .generic_region import normalize_region, region_bounds
from .gray_glove_video import collect_videos, extract_video_batch, frame_camera_ids
from .gray_glove_mcap import CAMERA_TOPICS, validate_mcap_files
from .gray_glove_projection import (
    PROJECTION_WORKFLOW, ProjectedGloveSegmenter, RawFrameReader, frame_projection_sources,
)
from .orbbec_undistort import OrbbecUndistorter, camera_id_for_video
from .project_paths import ROOT_DIR, SAM3_MODEL


DEFAULT_OUTPUT = ROOT_DIR / "data" / "gray_glove_lr_annotations"
SESSION_STATE_PATH = ROOT_DIR / "data" / ".gray_glove_review_session.json"
DEFAULT_ROI = [0.0, 0.0, 1.0, 1.0]
GRAY_GLOVE_CLASSES = [
    {"name": "left", "prompt": "left hand wearing a light gray glove"},
    {"name": "right", "prompt": "right hand wearing a light gray glove"},
]
GRAY_GLOVE_PROMPTS = (
    "hand wearing a light gray glove",
    "hand wearing a gray protective work glove",
    "hand wearing a protective work glove",
)
SIDE_GLOVE_PROMPTS = ("left gloved hand", "right gloved hand")
SIDE_CAMERA_CLASS = {"cam_2": 1, "cam_3": 0}
LEFT_LABEL_COLOR = SELECTED_COLOR
RIGHT_LABEL_COLOR = (255, 180, 0)  # cyan blue in BGR
PENDING_SIDE_COLOR = (180, 90, 210)  # purple in BGR
WORKFLOW = "gray_glove_lr_v2"
GLOVE_SELECTION_VERSION = 2
DEFAULT_CONFIDENCE = 0.20


def normalize_glove_config(config: dict) -> dict:
    """Keep low-score proposals for review while limiting selected gloves."""
    result = {**config, "count_mode": "fixed",
              "target_count": (2 if config.get("count_mode") == "all" else
                               max(1, min(2, int(config.get("target_count", 2))))),
              "glove_selection_version": GLOVE_SELECTION_VERSION}
    if config.get("glove_selection_version", 0) < GLOVE_SELECTION_VERSION:
        confidence = float(config.get("confidence", DEFAULT_CONFIDENCE))
        # Version 1 raised the detection cutoff to 0.50 and could hide real
        # hands. Restore recall once, without changing other custom cutoffs.
        result["confidence"] = DEFAULT_CONFIDENCE if confidence == 0.50 else confidence
        result["selection_rule"] = "confidence"
    return result


def _center_x(polygon) -> float:
    return float(np.asarray(polygon, dtype=np.float32).reshape(4, 2)[:, 0].mean())


def _box_iou(first, second) -> float:
    xa, ya = max(first[0], second[0]), max(first[1], second[1])
    xb, yb = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, xb - xa) * max(0.0, yb - ya)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def attach_side_scores(record: dict, side_detections: list[dict],
                       offset: tuple[int, int]) -> dict:
    """Attach directional SAM3 scores to each selected glove by box overlap."""
    result = {**record, "objects": [dict(obj) for obj in record.get("objects", [])]}
    for obj in result["objects"]:
        polygon = np.asarray(obj["polygon"], dtype=np.float32)
        lower, upper = polygon.min(axis=0), polygon.max(axis=0)
        box = (lower[0], lower[1], upper[0], upper[1])
        scores = [0.0, 0.0]
        for detection in side_detections:
            class_id = int(detection["class_id"])
            if class_id not in (0, 1):
                continue
            x1, y1, x2, y2 = detection["box"]
            shifted = (x1 + offset[0], y1 + offset[1],
                       x2 + offset[0], y2 + offset[1])
            if _box_iou(box, shifted) >= 0.35:
                scores[class_id] = max(scores[class_id], float(detection["score"]))
        obj["side_scores"] = [round(score, 4) for score in scores]
    return result


def assign_glove_sides(record: dict, width: int, mapping: str = "screen",
                       camera_id: str | None = None) -> dict:
    """Assign sides from image position, corroborated by SAM3 on side cameras."""
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
        if (camera_id in SIDE_CAMERA_CLASS and
                abs(_center_x(ordered[0]["polygon"]) -
                    _center_x(ordered[1]["polygon"])) >= width * 0.08):
            votes_agree = all(
                obj.get("side_scores", [0, 0])[obj["class_id"]] >= 0.70 and
                obj.get("side_scores", [0, 0])[obj["class_id"]] -
                obj.get("side_scores", [0, 0])[1 - obj["class_id"]] >= 0.08
                for obj in objects)
            result["side_autoclassified"] = votes_agree
    elif len(objects) == 1:
        suggested = int(_center_x(objects[0]["polygon"]) >= width / 2)
        objects[0]["class_id"] = suggested if mapping == "screen" else 1 - suggested
        prior = SIDE_CAMERA_CLASS.get(camera_id)
        scores = objects[0].get("side_scores", [0.0, 0.0])
        if (prior is not None and scores[prior] >= 0.65 and
                scores[prior] - scores[1 - prior] >= 0.01):
            objects[0]["class_id"] = prior
            result["side_pending"] = False
            result["side_autoclassified"] = True
            if result.get("status") == "auto":
                result["reason"] = (f"侧视单手：SAM3 与 {camera_id} 相机方向一致，"
                                    f"自动标为 {'left' if prior == 0 else 'right'}")
        else:
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
    """Review side views when directional evidence did not settle handedness."""
    if camera_id is None:
        return record
    result = {**record, "camera_id": camera_id}
    if (camera_id in SIDE_CAMERA_CLASS and result.get("objects")
            and not result.get("side_autoclassified")
            and result.get("status") == "auto"):
        result["status"] = "review"
        result["reason"] = "侧视相机的左右提示词证据不足或冲突，请人工确认"
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
    """Locate gloves and optionally query SAM3 for left/right evidence."""

    def detect_many(self, image_bgr, _prompts):
        prompts = list(GRAY_GLOVE_PROMPTS)
        if getattr(self, "side_prompts_enabled", False):
            prompts.extend(SIDE_GLOVE_PROMPTS)
        detections = super().detect_many(image_bgr, prompts)
        self.last_side_detections = [
            {**item, "class_id": int(item["class_id"]) - len(GRAY_GLOVE_PROMPTS)}
            for item in detections if int(item["class_id"]) >= len(GRAY_GLOVE_PROMPTS)]
        return [{**item, "class_id": 0} for item in detections
                if int(item["class_id"]) < len(GRAY_GLOVE_PROMPTS)
                and float(item["score"]) >= self.conf]


class GrayGloveThread(GenericThread):
    def run(self):
        raw_reader = None
        try:
            self.config = normalize_glove_config(self.config)
            todo = [path for path in self.files if path.name not in self.done_names]
            camera_ids = frame_camera_ids(Path(self.config["source_dir"]))
            total = len(self.files)
            completed = total - len(todo)
            self.progress_signal.emit(completed, total)
            if not todo:
                self.message_signal.emit("抽中图片已有结果；可复核或导出。")
                return
            raw_reader = RawFrameReader(Path(self.config["source_dir"]), lambda: self.running)
            # MCAP H264 needs sequential decoding from a keyframe. Sorting the
            # sampled subset avoids restarting the decoder for every image.
            if raw_reader.entries:
                todo.sort(key=raw_reader.sort_key)
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
                    camera_id = camera_ids.get(path.name)
                    auto_side = (self.config.get("auto_side", True)
                                 and camera_id in SIDE_CAMERA_CLASS)
                    self.segmenter.side_prompts_enabled = auto_side
                    detector = self.segmenter
                    if path.name in raw_reader.entries:
                        raw_image, rectifier, source_key = raw_reader.load(path.name)
                        detector = ProjectedGloveSegmenter(
                            self.segmenter, raw_image, rectifier, source_key,
                            self.config["roi"], image.shape)
                    review_reasons = normalize_review_reasons(
                        self.config.get("review_reasons"))
                    if auto_side:
                        review_reasons = [reason for reason in review_reasons
                                          if reason != "short_count"]
                    result = detect_generic(
                        image, self.config["roi"], detector,
                        self.config["classes"], self.config["count_mode"],
                        self.config["target_count"],
                        review_reasons,
                        self.config.get("selection_rule"),
                        self.config.get("review_score"))
                    if auto_side:
                        x1, y1, _x2, _y2 = region_bounds(
                            self.config["roi"], image.shape[1], image.shape[0])
                        result = attach_side_scores(
                            result, detector.last_side_detections, (x1, y1))
                        result["side_auto_attempted"] = True
                    result = assign_glove_sides(
                        result, image.shape[1],
                        self.config.get("side_mapping", "screen"),
                        camera_id if auto_side else None)
                    result = flag_side_camera_review(
                        result, camera_id)
                    result.update(name=path.name, width=image.shape[1],
                                  height=image.shape[0])
                    if path.name in raw_reader.entries:
                        source = raw_reader.entries[path.name]
                        result.update(detection_space=PROJECTION_WORKFLOW,
                                      source_key=source["source_key"],
                                      source_frame=source["source_frame"])
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
        finally:
            if raw_reader is not None:
                raw_reader.close()


class DatasetExportThread(QThread):
    """Export a snapshot of the current boxes without blocking Qt's event loop."""

    progress_signal = pyqtSignal(int, int, str)

    def __init__(self, source, output, records, classes, box_format):
        super().__init__()
        self.source, self.output = source, output
        # Unselected proposals are unnecessary for exporting or previewing.
        self.records = {
            name: copy.deepcopy({key: record[key] for key in ("name", "objects", "camera_id")
                                 if key in record}) for name, record in records.items()
        }
        self.classes = copy.deepcopy(classes)
        self.box_format = box_format
        self.running = True
        self.result = None
        self.error = None
        self.cancelled = False
        self._last_progress = (None, 0.0)

    def stop(self):
        self.running = False

    def report_progress(self, done, total, stage):
        last_stage, last_time = self._last_progress
        now = time.monotonic()
        if stage != last_stage or done in {0, total} or now - last_time >= 0.1:
            self.progress_signal.emit(done, total, stage)
            self._last_progress = (stage, now)

    def run(self):
        try:
            target, images, objects = export_current_annotations(
                self.source, self.output, self.records, self.classes, self.box_format,
                progress=self.report_progress, should_continue=lambda: self.running)
            total = sum(1 for _ in (target / "images").iterdir())
            self.result = (target, images, objects, total)
        except InterruptedError:
            self.cancelled = True
        except Exception as exc:
            self.error = str(exc)


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

    previous_image_shortcut = "B"

    def __init__(self):
        self._building_gray = True
        self.video_thread = None
        self.export_thread = None
        self._close_after_export = False
        self._detect_after_video_import = False
        self._video_import_succeeded = False
        self._loaded_video_folder = None
        self._review_recheck_names = None
        self._review_recheck_failures = []
        self._resume_warning = None
        self._loaded_review_ui_config = None
        self._loaded_review_project_config = None
        self._visible_rows = 0
        self._run_started_at = None
        self._run_start_done = None
        self._focus_view = False
        super().__init__()
        self._building_gray = False
        outer = self.centralWidget().layout()
        self.recheck_review_button = QPushButton("仅重检待人工复核")
        self.recheck_review_button.setToolTip(
            "使用当前设置重新检测所有待人工复核图片，保留其他状态和全部列表。")
        self.recheck_review_button.clicked.connect(self.recheck_review_images)
        self.export_all_button = QPushButton("导出全部数据集 [Ctrl+Shift+E]")
        self.export_all_button.setToolTip(
            "导出当前目录全部图片及此刻已选框，不按复核状态筛选，不划分 train/val；无框生成空标签。")
        self.export_all_button.clicked.connect(self.export_all_dataset)
        self.export_all_shortcut = QShortcut(QKeySequence("Ctrl+Shift+E"), self)
        self.export_all_shortcut.setContext(Qt.WindowShortcut)
        self.export_all_shortcut.setAutoRepeat(False)
        self.export_all_shortcut.activated.connect(self.export_all_button.click)
        for index in range(outer.count()):
            controls = outer.itemAt(index).layout()
            if controls is not None and controls.indexOf(self.start_button) >= 0:
                controls.insertWidget(controls.indexOf(self.start_button) + 1,
                                      self.recheck_review_button)
                controls.insertWidget(controls.indexOf(self.export_button) + 1,
                                      self.export_all_button)
                break
        self.setWindowTitle("灰色手套 · 视频抽帧 / SAM3 标注 / 人工复核")
        self.output_edit.setText(str(DEFAULT_OUTPUT))
        self.set_classes(GRAY_GLOVE_CLASSES)
        self.class_table.setHorizontalHeaderLabels(["导出类别", "历史提示词"])
        self.class_table.setColumnHidden(1, True)
        self.class_table.setToolTip(
            "固定输出 left/right 两类；基础定位使用三条不区分左右的手套提示词。"
            "勾选侧视自动分类时，另使用 left gloved hand / right gloved hand 辅助判断。")
        self.settings_group = self.centralWidget().layout().itemAt(0).widget()
        self.side_combo = QComboBox()
        self.side_combo.addItem("画面左侧=left，右侧=right", "screen")
        self.side_combo.addItem("镜像：画面左侧=right，右侧=left", "mirror")
        self.side_combo.setToolTip("双手自动分配类别的规则；单手须人工指定。")
        settings_grid = self.settings_group.layout()
        settings_grid.itemAtPosition(3, 0).widget().setText("导出类别")
        settings_grid.addWidget(QLabel("左右手类别规则"), 8, 0)
        settings_grid.addWidget(self.side_combo, 8, 1, 1, 4)
        settings_grid.addWidget(QLabel(
            "SAM3 定位手套；侧视单手在左右提示词与相机方向一致时自动定类，否则待复核。"),
            9, 0, 1, 5)
        self.auto_side_check = QCheckBox(
            "侧视相机自动区分左右（cam_2 默认 right，cam_3 默认 left；SAM3 辅助核对）")
        self.auto_side_check.setChecked(True)
        self.auto_side_check.setToolTip(
            "只在 SAM3 左右提示词与相机方向一致且分数足够时自动标注；"
            "不同意或看不清时仍待人工复核。")
        settings_grid.addWidget(self.auto_side_check, 10, 0, 1, 5)
        self.roi = DEFAULT_ROI.copy()
        self.show_roi()
        self.count_combo.removeItem(self.count_combo.findData("all"))
        self.count_combo.setCurrentIndex(self.count_combo.findData("fixed"))
        self.count_combo.setItemText(0, "每张最多选取（允许单手）")
        self.count_combo.setToolTip("每张自动选取最多 1 或 2 只手套；不会补足数量。")
        self.count_spin.setRange(1, 2)
        self.count_spin.setValue(2)
        self.selection_combo.setCurrentIndex(self.selection_combo.findData("confidence"))
        self.selection_combo.setToolTip(
            "默认保留去重后置信度最高的两个框；低于复核阈值的框仍保留，供人工选择。")
        self.confidence_spin.setValue(DEFAULT_CONFIDENCE)
        self.confidence_spin.setToolTip(
            "候选检测门槛，默认 0.20，保留可能的真实手；低分框由复核阈值控制是否待人工确认。")
        self.review_reason_checks["class_conflict"].setChecked(False)
        self.review_reason_checks["near_edge"].setChecked(False)
        self.format_combo.setCurrentIndex(self.format_combo.findData("hbb"))
        self.preview_legend.setText(
            "绿色：left；青蓝色：right；紫色：左右待定。勾选显示未选候选框后，暗黄色框可点击补选。")

        self.video_group = QGroupBox("输入与抽帧：图片目录或视频 / MCAP 任选一种")
        video_layout = QVBoxLayout(self.video_group)
        video_row = QHBoxLayout()
        self.video_edit = QLineEdit()
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
        self.change_video_button = QPushButton("更换录像")
        self.change_video_button.setToolTip("解锁录像列表，选择新的输入；已有抽帧和标注保留。")
        self.change_video_button.clicked.connect(self.change_video_input)
        queue_row.addWidget(self.video_list, 1)
        queue_buttons = QVBoxLayout()
        queue_buttons.addWidget(self.remove_video_button)
        queue_buttons.addWidget(self.clear_video_button)
        queue_buttons.addWidget(self.change_video_button)
        queue_row.addLayout(queue_buttons)
        video_layout.addLayout(queue_row)
        self.input_edit.editingFinished.connect(self.sync_video_sources)
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
        interval_row.addWidget(QLabel("录像抽帧间隔"))
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
        interval_row.addWidget(QLabel("图片抽样间隔"))
        self.image_interval_spin = QSpinBox()
        self.image_interval_spin.setRange(1, 1000000)
        self.image_interval_spin.setValue(1)
        self.image_interval_spin.setSuffix(" 张")
        self.image_interval_spin.setToolTip("图片按文件名排序，从第一张开始每隔 N 张取一张；不使用录像的秒数设置。")
        interval_row.addWidget(self.image_interval_spin)
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
        # clicked(bool) must not be mistaken for the detect-after flag.
        self.import_button.clicked.connect(lambda: self.import_video())
        self.import_button.setToolTip(
            "四目录像默认去畸变，并保留原图供 SAM3 分割；完成后按 F5 检测。")
        interval_row.addWidget(self.import_button)
        self.video_progress = QLabel("尚未抽帧；也可直接选择已有图片目录")
        interval_row.addWidget(self.video_progress, 1)
        video_layout.addLayout(interval_row)
        self.image_interval_spin.valueChanged.connect(self.update_sample_summary)
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
        video_layout.addLayout(undistort_row)
        outer = self.centralWidget().layout()
        outer.insertWidget(1, self.video_group)
        self.open_project_button = QPushButton("打开已有复核项目 [Alt+P]")
        self.open_project_button.setToolTip("选择包含 project.json 和 review.jsonl 的结果目录。")
        self.open_project_button.clicked.connect(self.open_review_project)
        self.focus_button = QPushButton("进入大图复核 [Alt+S]")
        self.focus_button.clicked.connect(self.toggle_focus_view)
        focus_row = QHBoxLayout()
        focus_row.addWidget(self.open_project_button)
        focus_row.addWidget(self.focus_button)
        outer.insertLayout(0, focus_row)
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
        self.show_candidates_check = QCheckBox("显示未选候选框（暗黄色，可点击补选）")
        self.show_candidates_check.setToolTip("仅影响预览和点击，不改变标签或候选记录。")
        self.show_candidates_check.toggled.connect(self.redraw_current_record)
        self.preview.parentWidget().layout().insertWidget(4, self.show_candidates_check)
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
        self.preview.parentWidget().layout().insertLayout(6, side_row)
        for key, button in (("Alt+1", self.side_left_button),
                            ("Alt+2", self.side_right_button),
                            ("Alt+X", self.swap_button),
                            ("Alt+S", self.focus_button),
                            ("Alt+P", self.open_project_button)):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.activated.connect(button.click)
            self.video_shortcuts.append(shortcut)
        self.progress_detail = QLabel("检测尚未开始")
        self.summary = QLabel()
        outer.insertWidget(outer.count() - 1, self.progress_detail)
        outer.insertWidget(outer.count() - 1, self.summary)

        resume_output, resume_name = self.previous_review_project()
        if resume_output is not None:
            self.output_edit.setText(str(resume_output))
        self.load_project()
        self.set_config_controls(True)
        self.apply_filter()
        self.update_counts()
        if self.active_output is not None:
            self.resume_review_image(resume_name)
        if self._resume_warning:
            self.status.setText(self._resume_warning)
        content = self.takeCentralWidget()
        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setWidget(content)
        self.setCentralWidget(scroller)
        self.resize(1500, 950)

    @staticmethod
    def review_project_error(output, validate_records=False):
        """Check that an existing review project can be opened without inference."""
        try:
            output = Path(output).expanduser().resolve()
            path = output / "project.json"
            if not path.is_file() or not (output / "review.jsonl").is_file():
                return "结果目录须包含 project.json 和 review.jsonl。"
            config = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(config, dict) or config.get("workflow") != WORKFLOW:
                return "该结果目录不是灰色手套标注项目。"
            raw_source = config["source_dir"]
            if not isinstance(raw_source, str) or not raw_source.strip():
                return "项目中的图片目录为空。"
            source = Path(raw_source).expanduser().resolve()
            if not source.is_dir():
                return f"项目中的图片目录不存在：{source}"
            if validate_records:
                if normalize_region(config["roi"]) is None:
                    return "项目中的检测区域无效。"
                validate_classes(config["classes"])
                normalize_review_reasons(config.get("review_reasons"))
                normalize_review_score(config.get("review_score"))
                normalize_selection_rule(config.get("selection_rule"))
                normalize_sample_percent(config.get("sample_percent"))
                if "sample_seed" in config:
                    normalize_sample_seed(config["sample_seed"])
                if config["count_mode"] not in {"fixed", "all"} or int(config["target_count"]) < 1:
                    return "项目中的目标数量配置无效。"
                if not np.isfinite(float(config["confidence"])):
                    return "项目中的检测置信度无效。"
                if config.get("side_mapping", "screen") not in {"screen", "mirror"}:
                    return "项目中的左右手映射方式无效。"
                settings = config.get("image_sampling")
                if settings is not None and (not isinstance(settings, dict) or
                        isinstance(settings.get("interval"), bool) or
                        not isinstance(settings.get("interval"), int) or settings["interval"] < 1):
                    return "项目中的图片抽样间隔无效。"
                normalize_glove_config(config)
                records = load_records(output / "review.jsonl")
                if any(not isinstance(name, str) or not isinstance(record.get("status"), str)
                       or not isinstance(record.get("objects", []), list)
                       or not isinstance(record.get("candidates", []), list)
                       for name, record in records.items()):
                    return "复核记录格式无效，请检查 review.jsonl。"
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
            return f"无法读取项目：{exc}"
        return None

    def previous_review_project(self):
        """Restore the last opened review project, or an existing rectified project."""
        try:
            state = json.loads(SESSION_STATE_PATH.read_text(encoding="utf-8"))
            if not isinstance(state, dict):
                raise ValueError("复核会话记录无效")
            output = Path(state["output_dir"]).expanduser().resolve()
            error = self.review_project_error(output, validate_records=True)
            if error is None:
                return output, state.get("current_name")
            self._resume_warning = f"上次复核项目无法恢复：{error} 请打开已有复核项目。"
        except (OSError, ValueError, KeyError, TypeError):
            pass
        default = Path(DEFAULT_OUTPUT).expanduser().resolve()
        rectified = default.with_name(default.name + "_undistorted")
        if (not (default / "project.json").exists()
                and self.review_project_error(rectified, validate_records=True) is None):
            return rectified, None
        return None, None

    def resume_review_image(self, preferred=None):
        """Select the last visited image, or the first available review image."""
        if self.active_source is None:
            return
        names = ([preferred] if isinstance(preferred, str) else [])
        names += sorted(name for name, record in self.records.items()
                        if record.get("status") == "review")
        names += sorted(self.records)
        for name in names:
            if (name in self.row_for_name and Path(name).name == name
                    and (self.active_source / name).is_file()):
                self.follow_check.setChecked(False)
                self.table.selectRow(self.row_for_name[name])
                if self.current_name != name:
                    self.show_record(name)
                self.status.setText("已载入已有结果，可直接复核和微调标签。")
                return

    def remember_review_project(self):
        """Persist only review navigation state; annotations remain in review.jsonl."""
        if self.active_output is None or self.active_source is None:
            return
        try:
            output = self.active_output.resolve()
            if self.review_project_error(output) is not None:
                return
            SESSION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
            replace_project_config(SESSION_STATE_PATH, {
                "output_dir": str(output), "current_name": self.current_name,
            })
        except OSError:
            # This optional shortcut must never prevent saving or closing a project.
            pass

    def open_review_project(self):
        if ((self.thread is not None and self.thread.isRunning()) or
                (self.video_thread is not None and self.video_thread.isRunning())):
            self.status.setText("请先停止当前任务，再打开已有项目。")
            return
        folder = QFileDialog.getExistingDirectory(
            self, "打开已有复核项目（选择结果目录）", self.output_edit.text().strip())
        if not folder:
            return
        output = Path(folder).expanduser().resolve()
        error = self.review_project_error(output, validate_records=True)
        if error:
            QMessageBox.warning(self, "无法打开复核项目", error)
            return
        self.output_edit.setText(str(output))
        self.load_project()
        if self.active_output == output:
            self.resume_review_image()
            self.remember_review_project()

    def config(self):
        config = super().config()
        config["workflow"] = WORKFLOW
        config["model_path"] = str(SAM3_MODEL.expanduser().resolve())
        config["prompt_ensemble"] = list(GRAY_GLOVE_PROMPTS)
        config["side_mapping"] = self.side_combo.currentData()
        if (not self.is_extracted_video_source(Path(config["source_dir"]))
                and self.image_interval_spin.value() != 1):
            config["image_sampling"] = self.image_sampling_settings()
        config["glove_selection_version"] = GLOVE_SELECTION_VERSION
        if frame_projection_sources(Path(config["source_dir"])):
            config["undistorted_detection"] = PROJECTION_WORKFLOW
        return normalize_glove_config(config)

    def image_sampling_settings(self):
        return {"interval": self.image_interval_spin.value()}

    def is_extracted_video_source(self, source):
        return (source.is_relative_to(self.output_dir() / "video_frames")
                and any(source.glob("extract_*.json")))

    def sampling_candidates(self, files, config):
        settings = config.get("image_sampling")
        if settings is None:
            return files
        interval = settings["interval"]
        if isinstance(interval, bool) or not isinstance(interval, int) or interval < 1:
            raise ValueError("图片抽样间隔必须为正整数")
        return files[::interval]

    def skipped_sampling_reason(self, percent):
        return f"未抽中：不符合图片间隔或自动标注图片比例 {percent}%"

    def update_sample_summary(self, *_args):
        if not hasattr(self, "image_interval_spin"):
            return super().update_sample_summary()
        try:
            source = self.source_dir()
            files = image_files(source)
            settings = None if self.is_extracted_video_source(source) else self.image_sampling_settings()
            candidates = self.sampling_candidates(files, {"image_sampling": settings})
            count = len(sample_image_files(candidates, self.sample_percent_spin.value(), self.sample_seed))
            self.sample_summary.setText(f"抽帧 {len(candidates)}/{len(files)} 张；随机抽取 {count} 张")
        except (OSError, ValueError):
            self.sample_summary.setText("选择图片目录后显示预计抽取数量")

    def load_project(self):
        if self._building_gray:
            return
        self._loaded_review_ui_config = None
        self._loaded_review_project_config = None
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
        if self.video_thread is None:
            collection = self.output_dir() / "video_frames" / "collection"
            if not path.exists() and not self.input_edit.text().strip() and collection.is_dir():
                self.input_edit.setText(str(collection))
            self.sync_video_sources()
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            normalized = normalize_glove_config(previous)
            self.count_combo.setCurrentIndex(self.count_combo.findData("fixed"))
            self.count_spin.setValue(normalized["target_count"])
            self.confidence_spin.setValue(normalized["confidence"])
            self.selection_combo.setCurrentIndex(
                self.selection_combo.findData(normalized["selection_rule"]))
            self.side_combo.setCurrentIndex(
                self.side_combo.findData(previous.get("side_mapping", "screen")))
            settings = previous.get("image_sampling")
            self.image_interval_spin.setValue(settings["interval"] if settings else 1)
            self.update_sample_summary()
            if normalized != previous:
                self.status.setText(
                    f"旧项目已调整为最多两只手套、按置信度选框，检测门槛 {normalized['confidence']:.2f}；"
                    "可直接复核或导出已有结果；按 F5 将备份并重检自动结果。")
        self.set_classes(GRAY_GLOVE_CLASSES)
        self.set_config_controls(True)
        if (path.is_file() and self.active_output == self.output_dir()
                and self.review_project_error(self.active_output, validate_records=True) is None):
            self._loaded_review_project_config = copy.deepcopy(previous)
            self._loaded_review_ui_config = copy.deepcopy(self.config())

    def ensure_project(self, allow_reconfigure=False):
        output = self.output_dir()
        path = output / "project.json"
        if path.is_file():
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("workflow") != WORKFLOW:
                raise ValueError("结果目录属于其他标注任务，请另选目录")
            if not allow_reconfigure and self._loaded_review_ui_config is not None:
                if self.active_output != output:
                    raise ValueError("结果目录已更改，请先载入对应项目。")
                if self._loaded_review_project_config != previous:
                    raise ValueError("项目配置文件已改变，请重新载入项目后再导出。")
                if self._loaded_review_ui_config != self.config():
                    raise ValueError("配置已修改，请先按 F5 根据新配置重新检测，再导出。")
            # Loading an older project can normalize the displayed settings.  If
            # the user has not changed them, export the already-saved results
            # using the old project's sampling configuration without migration.
            if (not allow_reconfigure and self._loaded_review_ui_config is not None
                    and self._loaded_review_project_config == previous
                    and self._loaded_review_ui_config == self.config()
                    and self.active_output == self.output_dir()
                    and self.active_source is not None
                    and self.active_source.resolve() == self.source_dir()):
                if not self.active_source.is_dir():
                    raise ValueError(f"图片目录不存在: {self.active_source}")
                if ("sample_seed" not in previous and
                        normalize_sample_percent(previous.get("sample_percent")) < 100):
                    raise ValueError("旧项目缺少随机抽样种子，无法确定原有抽样结果；请先按 F5 更新项目。")
                self.records = load_records(self.active_output / "review.jsonl")
                return self.active_config
        config = super().ensure_project(allow_reconfigure=allow_reconfigure)
        if allow_reconfigure and path.is_file():
            self._loaded_review_project_config = json.loads(path.read_text(encoding="utf-8"))
            self._loaded_review_ui_config = copy.deepcopy(self.config())
        return config

    def records_after_config_change(self, files, previous, config):
        updated = super().records_after_config_change(files, previous, config)
        if Path(previous["source_dir"]).resolve() == Path(config["source_dir"]).resolve():
            for name, record in load_records(self.output_dir() / "review.jsonl").items():
                if name in updated and record.get("status") in {"confirmed", "empty_confirmed"}:
                    updated[name] = record
        return updated

    def apply_sampling(self, files, sampled, percent):
        # A new sampling fraction must not erase a manual review decision.
        selected_names = {path.name for path in sampled}
        manual = [path for path in files if path.name not in selected_names and
                  self.records.get(path.name, {}).get("status") in
                  {"confirmed", "empty_confirmed"}]
        super().apply_sampling(files, [*sampled, *manual], percent)

    def set_config_controls(self, enabled):
        enabled = enabled and self.export_thread is None
        super().set_config_controls(enabled)
        if hasattr(self, "export_all_button"):
            available = enabled and not (self.video_thread and self.video_thread.isRunning())
            self.export_all_button.setEnabled(available)
        if hasattr(self, "open_project_button"):
            available = enabled and not (self.video_thread and self.video_thread.isRunning())
            self.open_project_button.setEnabled(available)
        if hasattr(self, "recheck_review_button"):
            available = enabled and not (self.video_thread and self.video_thread.isRunning())
            self.recheck_review_button.setEnabled(available)
        self.class_table.setEnabled(False)
        for button in self.class_buttons:
            button.setEnabled(False)
        if hasattr(self, "side_combo"):
            self.side_combo.setEnabled(enabled)
        if hasattr(self, "auto_side_check"):
            self.auto_side_check.setEnabled(enabled)
        if hasattr(self, "video_edit"):
            available = enabled and not (self.video_thread and self.video_thread.isRunning())
            for widget in (self.video_edit, self.video_button,
                           self.folder_button, self.add_path_button,
                           self.video_list, self.remove_video_button,
                           self.clear_video_button, self.change_video_button,
                           self.interval_mode, self.image_interval_spin,
                           self.interval_spin, self.start_spin, self.end_check,
                           self.import_button, self.calibration_edit,
                           self.calibration_button, self.balance_spin):
                widget.setEnabled(available)
            for widget in (self.video_edit, self.video_button, self.folder_button,
                           self.add_path_button, self.remove_video_button,
                           self.clear_video_button):
                widget.setEnabled(available and self._loaded_video_folder is None)
            self.change_video_button.setEnabled(available and self._loaded_video_folder is not None)
            for check in self.mcap_camera_checks.values():
                check.setEnabled(available)
            self.end_spin.setEnabled(available and self.end_check.isChecked())

    def make_detection_thread(self, files, config, done_names):
        config = {**config, "auto_side": self.auto_side_check.isChecked()}
        if config.get("undistorted_detection") == PROJECTION_WORKFLOW:
            projected_names = frame_projection_sources(Path(config["source_dir"]))
            done_names = set(done_names) - {
                name for name, record in self.records.items()
                if name in projected_names and
                record.get("status") not in {"confirmed", "empty_confirmed"} and
                record.get("detection_space") != PROJECTION_WORKFLOW
            }
        done_names = set(done_names) - {
            name for name, record in self.records.items()
            if len(record.get("objects", [])) > 2 and
            record.get("status") not in {"confirmed", "empty_confirmed"}
        }
        if config["auto_side"]:
            done_names = set(done_names) - {
                name for name, record in self.records.items()
                if (record.get("camera_id") in SIDE_CAMERA_CLASS and
                    record.get("status") == "review" and
                    not record.get("side_auto_attempted"))
            }
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

    def record_image_available(self, name):
        return (self.active_source is None or
                (Path(name).name == name and (self.active_source / name).is_file()))

    def update_table(self, record):
        # review.jsonl is a history log. Missing source images must not become
        # phantom rows, but their saved annotations remain available on disk.
        if not self.record_image_available(record["name"]):
            row = self.row_for_name.pop(record["name"], None)
            if row is not None:
                was_blocked = self.table.blockSignals(True)
                try:
                    self.table.removeRow(row)
                    self.row_for_name = {name: index - int(index > row)
                                         for name, index in self.row_for_name.items()}
                finally:
                    self.table.blockSignals(was_blocked)
                self.apply_filter()
            return
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
        current_records = [self.records[name] for name in self.row_for_name if name in self.records]
        for record in current_records:
            status = record.get("status")
            counts[status] = counts.get(status, 0) + 1
        self.summary.setText(
            f"图片 {len(current_records)} 张  ·  待复核 {counts['review']}  ·  待重检 {counts['unlabeled']}"
            f"  ·  已确认 {counts['confirmed']}  ·  无目标 {counts['empty_confirmed']}"
            f"  ·  自动候选 {counts['auto']}  ·  未抽中 {counts['skipped']}")
        missing = len(self.records) - len(current_records)
        if missing:
            self.summary.setText(self.summary.text() + f"  ·  已隐藏 {missing} 条缺失图片记录（历史标签保留）")

    def recheck_review_images(self):
        """Reprocess only saved review records without resetting other statuses."""
        if self.export_thread is not None:
            return
        if ((self.thread is not None and self.thread.isRunning()) or
                (self.video_thread is not None and self.video_thread.isRunning())):
            self.status.setText("请先停止当前任务，再重检待复核图片。")
            return
        try:
            source, output = self.source_dir(), self.output_dir()
            project_path = output / "project.json"
            if not project_path.is_file():
                raise ValueError("请先载入包含已有检测记录的灰色手套项目。")
            previous = json.loads(project_path.read_text(encoding="utf-8"))
            if previous.get("workflow") != WORKFLOW:
                raise ValueError("结果目录属于其他标注任务，请选择灰色手套项目。")
            if Path(previous["source_dir"]).expanduser().resolve() != source:
                raise ValueError("图片目录与已保存项目不一致，请先载入原项目。")
            if not source.is_dir():
                raise ValueError(f"图片目录不存在: {source}")
            records = load_records(output / "review.jsonl")
            names = sorted(name for name, record in records.items()
                           if record.get("status") == "review")
            if not names:
                self.status.setText("当前项目没有待人工复核图片。")
                return
            missing = [name for name in names
                       if Path(name).name != name or not (source / name).is_file()]
            if missing:
                raise ValueError(f"有 {len(missing)} 张待复核图片不存在或名称无效：\n{missing[0]}")
            config = self.config()
            self.validate_model_ready()
            backup_project(output)
            replace_project_config(project_path, config)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "无法重检待复核图片", str(exc))
            return

        self.records = records
        self.active_source, self.active_output, self.active_config = source, output, config
        self._loaded_review_project_config = copy.deepcopy(config)
        self._loaded_review_ui_config = copy.deepcopy(config)
        self.search_edit.clear()
        self.status_filter.setCurrentIndex(self.status_filter.findData("all"))
        self.review_filter.setChecked(False)
        self.table.setRowCount(0)
        self.row_for_name.clear()
        self._visible_rows = 0
        for record in sorted(records.values(), key=lambda item: item["name"]):
            self.update_table(record)
        self.update_counts()
        self.apply_filter()
        self._review_recheck_names = set(names)
        self._review_recheck_failures = []
        self.clear_undo_history()
        self._run_started_at = time.monotonic()
        self._run_start_done = None
        self._deleted_during_run.clear()
        self.thread = self.make_detection_thread([source / name for name in names], config, set())
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
        self.progress_detail.setText(f"正在准备重检 {len(names)} 张待复核图片...")
        self.thread.start()

    def receive_result(self, record):
        if self._review_recheck_names is not None:
            name = record.get("name")
            if (name not in self._review_recheck_names or
                    self.records.get(name, {}).get("status") != "review"):
                return
            self._review_recheck_names.discard(name)
            if str(record.get("reason", "")).startswith("检测失败:"):
                self._review_recheck_failures.append(name)
                return
        super().receive_result(record)

    def start_detection(self):
        if self.export_thread is not None:
            return
        if self.video_thread is not None and self.video_thread.isRunning():
            self.status.setText("抽帧完成后才能开始自动检测。")
            return
        has_images = bool(self.input_edit.text().strip())
        has_video = (self._loaded_video_folder is None
                     and bool(self.video_list.count() or self.video_edit.text().strip()))
        if has_images and has_video:
            QMessageBox.warning(
                self, "无法开始", "请选择一种输入：图片目录或视频 / MCAP。"
                "如需追加录像，请先点击抽帧并载入。")
            return
        if has_video:
            self.import_video(detect_after=True)
            return
        if not has_images:
            QMessageBox.warning(self, "无法开始", "请选择图片目录或添加视频 / MCAP。")
            return
        if (self._loaded_video_folder is not None and
                not frame_projection_sources(self.source_dir())):
            # Upgrade raw extractions made through the old ordinary button.
            # Keep their images and labels in the original project directory.
            videos = [Path(self.video_list.item(index).text())
                      for index in range(self.video_list.count())]
            try:
                rectify = self.should_undistort_videos(videos)
            except ValueError as exc:
                QMessageBox.warning(self, "无法开始", str(exc))
                return
            if rectify:
                self.import_video(detect_after=True)
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
        review_only = self._review_recheck_names is not None
        failures = len(self._review_recheck_failures)
        super().detection_finished()
        self.progress_detail.setText(
            f"{'待复核重检' if review_only else '检测'}{'已停止' if stopped else '已完成'}："
            f"{self.progress.value()}/{self.progress.maximum()} 张")
        if review_only and failures:
            self.status.setText(f"{failures} 张重检失败，已保留原复核记录。")
        self._review_recheck_names = None
        self._review_recheck_failures = []
        self._run_started_at = None

    def clear_all_labels(self):
        super().clear_all_labels()
        self.apply_filter()

    def color_for_candidate(self, candidate: dict, record: dict) -> tuple[int, int, int]:
        if not candidate["selected"]:
            return UNSELECTED_COLOR
        if record.get("side_pending"):
            return PENDING_SIDE_COLOR
        return (RIGHT_LABEL_COLOR if int(candidate["class_id"]) == 1
                else LEFT_LABEL_COLOR)

    def candidate_is_visible(self, candidate: dict, record: dict) -> bool:
        return bool(candidate["selected"] or self.show_candidates_check.isChecked())

    def redraw_current_record(self):
        if self.current_name is not None:
            self.show_record(self.current_name)

    def display_class_name(self, class_id: int, record: dict,
                           selected: bool = True) -> str:
        if record.get("side_pending"):
            return "左右待定" if selected else "手套候选"
        return super().display_class_name(class_id, record, selected)

    def show_record(self, name):
        if not super().show_record(name):
            return False
        record = self.records.get(name)
        if record and record.get("camera_id"):
            self.preview_title.setText(
                self.preview_title.text() + f"  |  {record['camera_id']}")
        if record and record.get("side_pending") and record.get("objects"):
            self.preview_title.setText(
                self.preview_title.text() + "  |  左右未确认：按 Alt+1 / Alt+2")
        return True

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
        if self.export_thread is not None:
            return
        for record in self.records.values():
            if not self.record_image_available(record["name"]):
                continue
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

    def export_all_dataset(self):
        """Save the current objects without rechecking, relabeling or inference."""
        if self.export_thread is not None:
            return
        if ((self.thread is not None and self.thread.isRunning()) or
                (self.video_thread is not None and self.video_thread.isRunning())):
            self.status.setText("请先停止抽帧或检测，再导出当前标注。")
            return
        try:
            if self.active_source is None or self.active_output is None or self.active_config is None:
                raise ValueError("请先载入已有标注项目或完成一次检测。")
            if self.source_dir() != self.active_source.resolve() or self.output_dir() != self.active_output:
                raise ValueError("图片或结果目录已更改，请先载入对应项目再导出。")
            worker = DatasetExportThread(
                self.active_source, self.active_output, self.records,
                self.active_config["classes"], self.format_combo.currentData())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "导出失败", str(exc))
            return
        self.export_thread = worker
        self._export_control_states = [
            (widget, widget.isEnabled()) for widget in
            (self.start_button, self.export_button, self.output_edit,
             self.path_buttons[1], self.format_combo)
        ]
        self._export_previous_stop_text = self.stop_button.text()
        for widget, _ in self._export_control_states:
            widget.setEnabled(False)
        self.set_config_controls(False)
        self.stop_button.setText("取消导出 [Esc]")
        self.stop_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.progress_detail.setText("正在准备导出当前标注...")
        self.status.setText("后台导出中，可取消；使用点击导出时的标注框。")
        worker.progress_signal.connect(self.export_progress)
        worker.finished.connect(self.export_finished)
        worker.start()

    def export_progress(self, done, total, stage):
        self.progress.setRange(0, max(1, total))
        self.progress.setValue(done)
        self.progress_detail.setText(f"{stage}：{done}/{total}")

    def export_finished(self):
        worker = self.export_thread
        if worker is None:
            return
        self.export_thread = None
        self.stop_button.setText(self._export_previous_stop_text)
        self.stop_button.setEnabled(False)
        self.set_config_controls(True)
        for widget, enabled in self._export_control_states:
            widget.setEnabled(enabled)
        worker.deleteLater()
        if self._close_after_export:
            self._close_after_export = False
            self.close()
            return
        if worker.cancelled:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.progress_detail.setText("导出已取消")
            self.status.setText("已取消导出并清理未完成文件，原项目和已有导出保持不变。")
            return
        if worker.error is not None:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.progress_detail.setText("导出失败")
            self.status.setText(f"导出失败：{worker.error}")
            QMessageBox.warning(self, "导出失败", worker.error)
            return
        target, images, objects, total = worker.result
        self.progress.setRange(0, total)
        self.progress.setValue(total)
        self.progress_detail.setText(f"导出完成：{total}/{total} 张")
        self.status.setText(f"已导出当前标注：{total} 张图片、{objects} 个框；未划分数据集。目录：{target}")
        QMessageBox.information(
            self, "全部数据集导出完成",
            f"目录：{target}\n图片：{total} 张，有框：{images} 张，标注框：{objects} 个"
            f"\nimages/ 与 labels/ 已按当前标注保存，未划分 train/val。"
            f"\n可视化预览：{target / 'visualizations'}")

    def open_large_preview(self):
        if self.current_name is None or self.preview_pixmap is None:
            self.status.setText("请先在左侧选择一张已检测的图片。")
            return
        LargePreviewDialog(self.preview_pixmap, self.current_name, self).exec_()

    def choose_input(self):
        super().choose_input()
        if hasattr(self, "video_list"):
            self.sync_video_sources()

    def sync_video_sources(self):
        """Show completed extraction sources without treating them as pending input."""
        if not hasattr(self, "video_list"):
            return
        try:
            source = self.source_dir()
            manifests = sorted(source.glob("extract_*.json")) if self.is_extracted_video_source(source) else []
            sources = set()
            for manifest in manifests:
                try:
                    data = json.loads(manifest.read_text(encoding="utf-8"))
                    if data.get("complete"):
                        sources.update(item["video"] for item in data.get("sources", [])
                                       if isinstance(item, dict) and isinstance(item.get("video"), str))
                except (OSError, ValueError, TypeError):
                    continue
        except (OSError, ValueError):
            source, sources = None, set()
        if sources:
            self.video_list.clear()
            self.video_list.addItems(sorted(sources))
            self.video_edit.clear()
            self._loaded_video_folder = source
        elif self._loaded_video_folder is not None:
            self.video_list.clear()
            self._loaded_video_folder = None
        self.set_config_controls(not (self.thread and self.thread.isRunning()))

    def change_video_input(self):
        if self._loaded_video_folder is None:
            return
        self._loaded_video_folder = None
        self.video_list.clear()
        self.video_edit.clear()
        self.input_edit.clear()
        self.update_sample_summary()
        self.set_config_controls(True)
        self.video_progress.setText("请选择新的录像；已有抽帧和标注仍保留。")

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

    @staticmethod
    def should_undistort_videos(videos) -> bool:
        """Default known four-camera recordings to the synchronized workflow."""
        supported = []
        for video in videos:
            if video.suffix.lower() == ".mcap":
                supported.append(True)
            else:
                try:
                    camera_id_for_video(video)
                    supported.append(True)
                except ValueError:
                    supported.append(False)
        if any(supported) and not all(supported):
            raise ValueError("四目录像与普通视频请分批抽帧。")
        return bool(supported) and all(supported)

    def import_video(self, detect_after: bool = False):
        if self.export_thread is not None:
            return
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
            undistort = self.should_undistort_videos(videos)
            validate_mcap_files(video for video in videos if video.suffix.lower() == ".mcap")
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
        self._detect_after_video_import = detect_after
        self._video_import_succeeded = False
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
        action = "去畸变抽帧" if self.video_thread and self.video_thread.undistort else "原图抽帧"
        self.video_progress.setText(f"{action} {done}/{total} 张")

    def video_imported(self, folder, count, reused, output):
        self._video_import_succeeded = True
        action = "去畸变抽帧" if self.video_thread and self.video_thread.undistort else "抽帧"
        if self.output_dir() != Path(output):
            self.output_edit.setText(output)
            self.load_project()
        old_source = self.input_edit.text().strip()
        self.input_edit.setText(folder)
        self.video_edit.clear()
        self.sync_video_sources()
        if old_source != folder:
            self.roi = DEFAULT_ROI.copy()
            self.show_roi()
        self.update_sample_summary()
        next_step = "即将开始检测" if self._detect_after_video_import else "按 F5 开始标注"
        self.video_progress.setText(f"{action}完成 {count} 张（复用 {reused} 张）；{next_step}")
        mode = ("原图分割 → 掩码去畸变 → 去畸变图复核"
                if self.video_thread and self.video_thread.undistort else "原图检测与复核")
        self.status.setText(f"图片目录已切换到 {folder}；{mode}；{next_step}。")

    def video_import_error(self, message):
        self.video_progress.setText(message)
        if not message.startswith("已停止抽帧"):
            action = ("去畸变抽帧失败" if self.video_thread and self.video_thread.undistort
                      else "抽帧失败")
            QMessageBox.warning(self, action, message)

    def video_import_finished(self):
        detect_after = self._detect_after_video_import and self._video_import_succeeded
        self._detect_after_video_import = False
        self._video_import_succeeded = False
        self.video_thread = None
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.set_config_controls(True)
        if detect_after:
            self.start_detection()

    def stop_detection(self):
        if self.export_thread is not None:
            self.export_thread.stop()
            self.stop_button.setEnabled(False)
            self.status.setText("正在取消导出并清理未完成文件...")
            return
        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_progress.setText("正在停止抽帧...")
            return
        super().stop_detection()

    def closeEvent(self, event):
        if self.export_thread is not None:
            self._close_after_export = True
            self.stop_detection()
            event.ignore()
            return
        if self.video_thread is not None and self.video_thread.isRunning():
            self.video_thread.stop()
            self.video_thread.wait()
        super().closeEvent(event)
        if event.isAccepted():
            self.remember_review_project()


def main() -> int:
    plugin_dir = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_dir
    QCoreApplication.setLibraryPaths([plugin_dir])
    app = QApplication(sys.argv)
    window = GrayGloveAnnotatorApp()
    window.show()
    return app.exec_()
