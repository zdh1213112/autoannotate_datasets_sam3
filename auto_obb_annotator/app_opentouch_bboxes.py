"""Annotate decoded OpenTouch images into one right-hand bbox JSONL file."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import time

import cv2
from PyQt5.QtCore import (
    QCoreApplication,
    QLibraryInfo,
    QSettings,
    QThread,
    Qt,
    pyqtSignal,
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from . import app as base_app
from .app_black_gloves import (
    BlackGloveFilterConfig,
    BlackGloveSAM3TextSegmenter,
)
from .opentouch_bbox_core import (
    expand_bbox_xyxy,
    make_bbox_record,
    prepare_detections,
    select_verified_right_hand,
)
from .project_paths import ROOT_DIR


GENERIC_PROMPT = "hand wearing black tactile glove"
RIGHT_PROMPT = "right hand wearing black tactile glove"
LEFT_PROMPT = "left hand wearing black tactile glove"
PREVIEW_EVERY_FRAMES = 5
DEFAULT_INPUT_DIR = ROOT_DIR / "datasets"
DEFAULT_OUTPUT_PATH = (
    ROOT_DIR / "datasets" / "opentouch_annotations" / "bboxes.jsonl"
)


@dataclass(frozen=True)
class OpenTouchBBoxConfig:
    input_dir: Path
    output_path: Path
    confidence: float = 0.25
    max_black_value: int = 130
    min_dark_ratio: float = 0.55
    min_box_area_ratio: float = 0.002
    min_center_y_ratio: float = 0.0
    match_iou: float = 0.45
    side_score_margin: float = 0.05
    border_margin_ratio: float = 0.002
    bbox_padding_ratio: float = 0.10


REASON_DISPLAY = {
    "useful": "有效右手 bbox",
    "no_right_hand": "未检测到可确认的右手",
    "left_right_ambiguous": "左右手无法区分",
    "right_hand_not_fully_visible": "右手未完整可见（触边）",
    "right_hand_unconfirmed": "右手未通过多提示确认",
    "multiple_right_candidates": "存在多个右手候选",
    "decode_error": "JPEG 解码失败",
}


def _draw_bbox_preview(image, bbox_xyxy, reason: str):
    """Return an in-memory preview with the bbox or NULL decision overlaid."""

    preview = image.copy()
    image_height, image_width = preview.shape[:2]
    scale = max(0.55, min(image_width, image_height) / 720.0)
    thickness = max(2, int(round(scale * 3)))
    if bbox_xyxy is not None:
        x1, y1, x2, y2 = [int(round(value)) for value in bbox_xyxy]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 220, 0), thickness)
        label = "RIGHT HAND | BBOX"
        color = (0, 220, 0)
    else:
        label = f"NULL | {reason}"
        color = (20, 20, 240)

    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness
    )
    text_x = 10
    text_y = max(text_height + baseline + 10, 28)
    cv2.rectangle(
        preview,
        (text_x - 5, text_y - text_height - baseline - 5),
        (min(image_width - 1, text_x + text_width + 5), text_y + baseline + 5),
        (0, 0, 0),
        -1,
    )
    cv2.putText(
        preview,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return preview


class OpenTouchBBoxThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    preview_signal = pyqtSignal(object, str)
    completed_signal = pyqtSignal(bool, str, object)

    def __init__(self, config: OpenTouchBBoxConfig):
        super().__init__()
        self.config = config
        self.is_running = True

    def stop(self):
        self.is_running = False

    @staticmethod
    def _count_manifest_rows(manifest_path: Path) -> int:
        with manifest_path.open("r", encoding="utf-8") as stream:
            return sum(1 for line in stream if line.strip())

    def run(self):
        partial_path: Path | None = None
        try:
            manifest_path = self.config.input_dir / "frames.jsonl"
            if not manifest_path.is_file():
                raise FileNotFoundError(
                    f"已解码数据目录缺少 frames.jsonl: {manifest_path}"
                )
            total_frames = self._count_manifest_rows(manifest_path)
            if total_frames == 0:
                raise ValueError(f"frames.jsonl 为空: {manifest_path}")

            if not os.path.isfile(base_app.SAM3_MODEL_PATH):
                raise FileNotFoundError(
                    f"找不到 SAM3 权重: {base_app.SAM3_MODEL_PATH}"
                )

            device = (
                "cuda:0"
                if base_app.torch is not None and base_app.torch.cuda.is_available()
                else "cpu"
            )
            if device.startswith("cuda"):
                gpu_name = base_app.torch.cuda.get_device_name(0)
                base_app.torch.backends.cudnn.benchmark = True
                self.log_signal.emit(f"🖥️ 推理设备: {device} ({gpu_name})")
            else:
                self.log_signal.emit("⚠️ 推理设备: CPU（未检测到可用 CUDA）")

            BlackGloveSAM3TextSegmenter.configure_filter(
                BlackGloveFilterConfig(
                    max_value=self.config.max_black_value,
                    min_dark_ratio=self.config.min_dark_ratio,
                    min_box_area_ratio=self.config.min_box_area_ratio,
                    min_center_y_ratio=self.config.min_center_y_ratio,
                    # Do not hide ambiguity before the side-verification step.
                    max_hands=4,
                )
            )
            self.log_signal.emit("🚀 正在加载 SAM3 黑手套模型...")
            segmenter = BlackGloveSAM3TextSegmenter(
                model_path=base_app.SAM3_MODEL_PATH,
                device=device,
                conf=self.config.confidence,
            )
            self.log_signal.emit(
                f"✅ 模型已加载；将按 frames.jsonl 处理 {total_frames} 张已解码图片。"
            )
            self.log_signal.emit(
                "⚡ 已启用单次三提示推理：每帧只编码一次，同时获取通用/右手/左手候选。"
            )
            self.log_signal.emit(
                "规则：仅输出可由通用/右手提示共同确认、且不与左手提示冲突的完整右手。"
            )
            # Reset GUI ETA timing after model loading, immediately before the
            # first source frame is processed.
            self.progress_signal.emit(0, total_frames)

            self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
            partial_path = self.config.output_path.with_name(
                self.config.output_path.name + ".partial"
            )
            reason_counts: Counter[str] = Counter()
            written = 0
            seen_sample_ids: set[str] = set()
            current_source_file = None

            with (
                manifest_path.open("r", encoding="utf-8") as manifest_file,
                partial_path.open("w", encoding="utf-8") as output_file,
            ):
                for line_number, line in enumerate(manifest_file, start=1):
                    if not self.is_running:
                        break
                    if not line.strip():
                        continue
                    try:
                        source_row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"frames.jsonl 第 {line_number} 行不是有效 JSON: {exc}"
                        ) from exc

                    required_fields = (
                        "sample_id",
                        "source_file",
                        "clip_id",
                        "source_frame_index",
                        "image",
                    )
                    missing = [
                        field for field in required_fields if field not in source_row
                    ]
                    if missing:
                        raise ValueError(
                            f"frames.jsonl 第 {line_number} 行缺少字段: "
                            + ", ".join(missing)
                        )

                    source_file = str(source_row["source_file"])
                    clip_id = str(source_row["clip_id"])
                    frame_index = int(source_row["source_frame_index"])
                    record = make_bbox_record(
                        source_file,
                        clip_id,
                        frame_index,
                        None,
                    )
                    if str(source_row["sample_id"]) != record["sample_id"]:
                        raise ValueError(
                            f"frames.jsonl 第 {line_number} 行 sample_id 不一致: "
                            f"{source_row['sample_id']!r} != {record['sample_id']!r}"
                        )
                    if record["sample_id"] in seen_sample_ids:
                        raise ValueError(
                            f"frames.jsonl 存在重复 sample_id: {record['sample_id']}"
                        )
                    seen_sample_ids.add(record["sample_id"])

                    relative_image = Path(str(source_row["image"]))
                    image_path = (
                        self.config.input_dir / relative_image
                    ).resolve()
                    if not image_path.is_relative_to(self.config.input_dir):
                        raise ValueError(
                            f"frames.jsonl 第 {line_number} 行图片路径越界: "
                            f"{relative_image}"
                        )
                    if source_file != current_source_file:
                        current_source_file = source_file
                        self.log_signal.emit(f"🖼️ 读取已解码图片: {source_file}")

                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    bbox = None
                    reason = "decode_error"
                    if image is not None and image.size > 0:
                        image_height, image_width = image.shape[:2]
                        prompt_results = segmenter.detect_many(
                            image,
                            [GENERIC_PROMPT, RIGHT_PROMPT, LEFT_PROMPT],
                        )
                        _, generic_boxes, generic_scores = (
                            prompt_results[GENERIC_PROMPT]
                        )
                        generic = prepare_detections(
                            generic_boxes,
                            generic_scores,
                            image_width,
                            image_height,
                        )

                        if generic:
                            _, right_boxes, right_scores = (
                                prompt_results[RIGHT_PROMPT]
                            )
                            right = prepare_detections(
                                right_boxes,
                                right_scores,
                                image_width,
                                image_height,
                            )
                            _, left_boxes, left_scores = (
                                prompt_results[LEFT_PROMPT]
                            )
                            left = prepare_detections(
                                left_boxes,
                                left_scores,
                                image_width,
                                image_height,
                            )
                            selection = select_verified_right_hand(
                                generic,
                                right,
                                left,
                                image_width,
                                image_height,
                                match_iou=self.config.match_iou,
                                side_score_margin=(
                                    self.config.side_score_margin
                                ),
                                border_margin_ratio=(
                                    self.config.border_margin_ratio
                                ),
                            )
                            bbox = selection.bbox_xyxy
                            if bbox is not None:
                                bbox = expand_bbox_xyxy(
                                    bbox,
                                    image_width,
                                    image_height,
                                    self.config.bbox_padding_ratio,
                                )
                            reason = selection.reason
                        else:
                            reason = "no_right_hand"

                    record = make_bbox_record(
                        source_file,
                        clip_id,
                        frame_index,
                        bbox,
                    )
                    next_written = written + 1
                    should_preview = (
                        next_written == 1
                        or next_written % PREVIEW_EVERY_FRAMES == 0
                    )
                    if (
                        image is not None
                        and image.size > 0
                        and should_preview
                    ):
                        preview = _draw_bbox_preview(image, bbox, reason)
                        caption = (
                            f"{record['sample_id']}  |  "
                            f"{REASON_DISPLAY.get(reason, reason)}"
                        )
                        self.preview_signal.emit(preview, caption)
                    output_file.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    written += 1
                    reason_counts[reason] += 1
                    if written % 50 == 0:
                        output_file.flush()
                    self.progress_signal.emit(written, total_frames)

                output_file.flush()
                os.fsync(output_file.fileno())

            if not self.is_running:
                message = (
                    f"已停止；完成 {written}/{total_frames} 帧。部分结果保留在: "
                    f"{partial_path}"
                )
                self.log_signal.emit("⏹ " + message)
                self.completed_signal.emit(False, message, dict(reason_counts))
                return

            if written != total_frames:
                raise RuntimeError(
                    f"行数校验失败: 应写 {total_frames} 行，实际 {written} 行。"
                )

            os.replace(partial_path, self.config.output_path)
            useful = reason_counts.get("useful", 0)
            message = (
                f"完成：{written} 行，有用样本 {useful}，无用样本 "
                f"{written - useful}。输出: {self.config.output_path}"
            )
            self.log_signal.emit("🎉 " + message)
            self.completed_signal.emit(True, message, dict(reason_counts))
        except Exception as exc:
            message = f"处理失败: {type(exc).__name__}: {exc}"
            if partial_path is not None and partial_path.exists():
                message += f"\n已写入的部分结果保留在: {partial_path}"
            self.log_signal.emit("❌ " + message)
            self.completed_signal.emit(False, message, {})
        finally:
            base_app.clear_torch_cache()


class OpenTouchBBoxApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OpenTouch 右手 BBox JSONL 标注器")
        self.resize(1180, 820)
        self.settings = QSettings("AutoOBBAnnotator", "OpenTouchBBoxAnnotator")
        self.thread: OpenTouchBBoxThread | None = None
        self._processing_started_at: float | None = None
        self._preview_pixmap: QPixmap | None = None

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        path_group = QGroupBox("📁 已解码 OpenTouch 输入与交付文件")
        path_grid = QGridLayout(path_group)
        saved_input = self.settings.value(
            "input_dir", str(DEFAULT_INPUT_DIR), type=str
        )
        saved_output = self.settings.value(
            "output_path", str(DEFAULT_OUTPUT_PATH), type=str
        )
        self.lbl_input = QLabel(saved_input)
        self.lbl_output = QLabel(saved_output)
        btn_input = QPushButton("选择 decoded_data 目录")
        btn_output = QPushButton("设置 bboxes.jsonl")
        btn_input.clicked.connect(self.select_input_dir)
        btn_output.clicked.connect(self.select_output_file)
        path_grid.addWidget(btn_input, 0, 0)
        path_grid.addWidget(self.lbl_input, 0, 1)
        path_grid.addWidget(btn_output, 1, 0)
        path_grid.addWidget(self.lbl_output, 1, 1)
        path_hint = QLabel(
            "程序按 decoded_data/frames.jsonl 直接读取 images/ 中已有 JPG；"
            "不打开 HDF5、不提取或生成任何图片，只交付一个 bboxes.jsonl。"
        )
        path_hint.setWordWrap(True)
        path_hint.setStyleSheet("color:#64B5F6;font-size:11px;")
        path_grid.addWidget(path_hint, 2, 0, 1, 2)
        layout.addWidget(path_group)

        params_group = QGroupBox("🧤 保守右手判定参数")
        params = QGridLayout(params_group)
        self.sp_confidence = self._double_spin(0.05, 0.95, 0.25, 0.05, 2)
        self.sp_black_value = self._double_spin(40, 220, 130, 5, 0)
        self.sp_dark_ratio = self._double_spin(0.05, 1.0, 0.55, 0.05, 2)
        self.sp_min_area = self._double_spin(0.0, 10.0, 0.20, 0.05, 2)
        self.sp_match_iou = self._double_spin(0.10, 0.90, 0.45, 0.05, 2)
        self.sp_side_margin = self._double_spin(0.0, 0.50, 0.05, 0.01, 2)
        self.sp_border_margin = self._double_spin(0.0, 5.0, 0.20, 0.10, 2)
        self.sp_bbox_padding = self._double_spin(0.0, 100.0, 10.0, 1.0, 1)
        self.sp_min_area.setSuffix(" %")
        self.sp_border_margin.setSuffix(" %")
        self.sp_bbox_padding.setSuffix(" %")
        self.sp_bbox_padding.setToolTip(
            "最终右手框向四周扩展的比例。10% 表示左右各增加原框宽度的 10%，"
            "上下各增加原框高度的 10%；扩展后自动限制在图像范围内。"
        )

        rows = [
            ("SAM3 最小置信度:", self.sp_confidence,
             "黑色亮度上限 V:", self.sp_black_value),
            ("mask 最小黑色占比:", self.sp_dark_ratio,
             "最小框面积:", self.sp_min_area),
            ("多提示匹配 IoU:", self.sp_match_iou,
             "右手领先左手分差:", self.sp_side_margin),
            ("完整可见边界余量:", self.sp_border_margin,
             "BBox 每侧扩边:", self.sp_bbox_padding),
        ]
        for row, (left_name, left_widget, right_name, right_widget) in enumerate(rows):
            params.addWidget(QLabel(left_name), row, 0)
            params.addWidget(left_widget, row, 1)
            if right_name:
                params.addWidget(QLabel(right_name), row, 2)
                params.addWidget(right_widget, row, 3)

        rule_hint = QLabel(
            "判定规则：右手完整可见才输出 bbox；左手入镜会忽略；只有左手、"
            "左右证据冲突、多个右手候选或目标触边均输出 null。有效框确认后再按扩边比例"
            "向四周放大，确保整只手包含在框内。"
        )
        rule_hint.setWordWrap(True)
        rule_hint.setStyleSheet("color:#FF9800;font-size:11px;")
        params.addWidget(rule_hint, len(rows), 0, 1, 4)
        layout.addWidget(params_group)

        action_group = QGroupBox("🚀 运行")
        action_layout = QVBoxLayout(action_group)
        buttons = QHBoxLayout()
        self.btn_start = QPushButton("▶️ 生成 bboxes.jsonl")
        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_start.setStyleSheet(
            "background-color:#4CAF50;color:white;padding:10px;font-weight:bold;"
        )
        self.btn_stop.setStyleSheet(
            "background-color:#F44336;color:white;padding:10px;font-weight:bold;"
        )
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start_processing)
        self.btn_stop.clicked.connect(self.stop_processing)
        buttons.addWidget(self.btn_start)
        buttons.addWidget(self.btn_stop)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("0 / ?（0.00%）  预计剩余 --")

        action_layout.addLayout(buttons)
        action_layout.addWidget(self.progress)
        layout.addWidget(action_group)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet(
            "background-color:#1E1E1E;color:#00FF00;font-family:Consolas;"
        )
        preview_group = QGroupBox("👁️ 实时结果可视化（每 5 帧刷新）")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_caption = QLabel("等待处理...")
        self.preview_caption.setWordWrap(True)
        self.preview_caption.setStyleSheet("font-weight:bold;color:#1565C0;")
        self.preview_image = QLabel("等待首个可视化结果")
        self.preview_image.setAlignment(Qt.AlignCenter)
        self.preview_image.setMinimumSize(480, 300)
        self.preview_image.setStyleSheet(
            "background-color:#111;color:#AAA;border:1px solid #555;"
        )
        preview_layout.addWidget(self.preview_caption)
        preview_layout.addWidget(self.preview_image, 1)

        result_splitter = QSplitter(Qt.Horizontal)
        result_splitter.addWidget(self.log_console)
        result_splitter.addWidget(preview_group)
        result_splitter.setSizes([500, 650])
        layout.addWidget(result_splitter, 1)
        self.append_log(
            "准备就绪。每个源帧固定输出一行；有明确、完整的右手框为有用样本，"
            "否则 bbox_xyxy 为 null。"
        )

    @staticmethod
    def _double_spin(minimum, maximum, value, step, decimals):
        widget = QDoubleSpinBox()
        widget.setRange(float(minimum), float(maximum))
        widget.setValue(float(value))
        widget.setSingleStep(float(step))
        widget.setDecimals(int(decimals))
        return widget

    def append_log(self, text: str):
        self.log_console.append(str(text))
        self.log_console.verticalScrollBar().setValue(
            self.log_console.verticalScrollBar().maximum()
        )

    def select_input_dir(self):
        current = self.lbl_input.text().strip()
        directory = QFileDialog.getExistingDirectory(
            self, "选择包含 frames.jsonl 和 images/ 的 decoded_data 目录", current
        )
        if directory:
            selected = Path(directory)
            if selected.name == "images" and (selected.parent / "frames.jsonl").is_file():
                selected = selected.parent
            self.lbl_input.setText(str(selected))

    def select_output_file(self):
        current = self.lbl_output.text().strip()
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "设置 bbox 交付文件",
            current,
            "JSON Lines (*.jsonl)",
        )
        if output_path:
            if not output_path.lower().endswith(".jsonl"):
                output_path += ".jsonl"
            self.lbl_output.setText(output_path)

    def start_processing(self):
        input_dir = Path(self.lbl_input.text().strip()).expanduser().resolve()
        output_path = Path(self.lbl_output.text().strip()).expanduser().resolve()
        if not input_dir.is_dir():
            QMessageBox.warning(self, "输入无效", f"已解码数据目录不存在:\n{input_dir}")
            return
        if not (input_dir / "frames.jsonl").is_file():
            QMessageBox.warning(
                self,
                "输入无效",
                f"目录中没有 frames.jsonl:\n{input_dir}",
            )
            return
        if output_path.suffix.lower() != ".jsonl":
            QMessageBox.warning(self, "输出无效", "交付文件必须以 .jsonl 结尾。")
            return
        if output_path == (input_dir / "frames.jsonl").resolve():
            QMessageBox.critical(
                self,
                "输出无效",
                "bboxes.jsonl 不能覆盖输入清单 frames.jsonl。",
            )
            return

        config = OpenTouchBBoxConfig(
            input_dir=input_dir,
            output_path=output_path,
            confidence=self.sp_confidence.value(),
            max_black_value=int(round(self.sp_black_value.value())),
            min_dark_ratio=self.sp_dark_ratio.value(),
            min_box_area_ratio=self.sp_min_area.value() / 100.0,
            match_iou=self.sp_match_iou.value(),
            side_score_margin=self.sp_side_margin.value(),
            border_margin_ratio=self.sp_border_margin.value() / 100.0,
            bbox_padding_ratio=self.sp_bbox_padding.value() / 100.0,
        )
        self.settings.setValue("input_dir", str(input_dir))
        self.settings.setValue("output_path", str(output_path))
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("0 / ?（0.00%）  预计剩余 --")
        self._processing_started_at = time.monotonic()
        self._preview_pixmap = None
        self.preview_image.setPixmap(QPixmap())
        self.preview_image.setText("等待首个可视化结果")
        self.preview_caption.setText("等待处理...")
        self.log_console.clear()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.append_log(f"输入: {input_dir}")
        self.append_log(f"输出: {output_path}")
        self.thread = OpenTouchBBoxThread(config)
        self.thread.log_signal.connect(self.append_log)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.preview_signal.connect(self.update_preview)
        self.thread.completed_signal.connect(self.processing_completed)
        self.thread.start()

    def stop_processing(self):
        if self.thread is not None and self.thread.isRunning():
            self.thread.stop()
            self.btn_stop.setEnabled(False)
            self.append_log("正在完成当前帧并安全停止...")

    def update_progress(self, current: int, total: int):
        total = max(1, int(total))
        current = max(0, min(int(current), total))
        self.progress.setRange(0, total)
        self.progress.setValue(current)
        percent = current * 100.0 / total
        eta_text = "--"
        if current == 0:
            self._processing_started_at = time.monotonic()
        if self._processing_started_at is not None and current > 0:
            elapsed = max(0.001, time.monotonic() - self._processing_started_at)
            frames_per_second = current / elapsed
            if frames_per_second > 0:
                eta_seconds = (total - current) / frames_per_second
                eta_text = self._format_duration(eta_seconds)
        self.progress.setFormat(
            f"{current} / {total}（{percent:.2f}%）  预计剩余 {eta_text}"
        )

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        if seconds < 60:
            return f"{seconds} 秒"
        minutes = seconds // 60
        if minutes < 60:
            return f"{minutes} 分钟"
        hours, remaining_minutes = divmod(minutes, 60)
        if hours < 48:
            return f"{hours} 小时 {remaining_minutes} 分钟"
        days, remaining_hours = divmod(hours, 24)
        return f"{days} 天 {remaining_hours} 小时"

    def update_preview(self, image_bgr, caption: str):
        if image_bgr is None or image_bgr.size == 0:
            return
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        bytes_per_line = int(image_rgb.strides[0])
        qimage = QImage(
            image_rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_RGB888,
        ).copy()
        self._preview_pixmap = QPixmap.fromImage(qimage)
        self.preview_caption.setText(caption)
        self._refresh_preview_pixmap()

    def _refresh_preview_pixmap(self):
        if self._preview_pixmap is None or self._preview_pixmap.isNull():
            return
        self.preview_image.setPixmap(
            self._preview_pixmap.scaled(
                self.preview_image.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview_pixmap()

    def processing_completed(self, success: bool, message: str, reason_counts):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        if reason_counts:
            self.append_log("判定统计: " + json.dumps(
                reason_counts, ensure_ascii=False, sort_keys=True
            ))
        if success:
            maximum = self.progress.maximum()
            self.progress.setValue(maximum)
            self.progress.setFormat(
                f"{maximum} / {maximum}（100.00%）  已完成"
            )
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.warning(self, "未完成", message)

    def closeEvent(self, event):
        if self.thread is not None and self.thread.isRunning():
            reply = QMessageBox.question(
                self,
                "任务仍在运行",
                "确定停止任务并退出吗？已完成内容会保留为 .partial 文件。",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                event.ignore()
                return
            self.thread.stop()
            self.thread.wait()
        event.accept()


def main():
    # The non-headless opencv-python wheel rewrites this environment variable
    # to cv2/qt/plugins while it is imported.  Those bundled Qt libraries are
    # not ABI-compatible with PyQt5 and make the xcb plugin fail at startup.
    # Restore PyQt5's own plugin root immediately before QApplication is made.
    pyqt_plugin_root = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyqt_plugin_root
    QCoreApplication.setLibraryPaths([pyqt_plugin_root])
    qt_app = QApplication(sys.argv)
    window = OpenTouchBBoxApp()
    window.show()
    return qt_app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
