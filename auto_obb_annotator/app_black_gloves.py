"""SAM3 annotator variant for egocentric black-gloved hands only.

This module intentionally leaves :mod:`auto_obb_annotator.app` unchanged.  It
reuses the existing UI/export pipeline and replaces only the SAM3 text
segmenter with a conservative black-glove selector.
"""

from dataclasses import dataclass
import math
import os
import re
import sys

import cv2
import numpy as np
from PyQt5.QtCore import QCoreApplication, QLibraryInfo
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QSpinBox,
)

from . import app as base_app


_HAND_PROMPT_RE = re.compile(r"\bhand(s)?\b", re.IGNORECASE)
_GLOVE_PROMPT_RE = re.compile(r"\bglove(s|d)?\b", re.IGNORECASE)


@dataclass(frozen=True)
class BlackGloveFilterConfig:
    """Thresholds used after SAM3 semantic segmentation."""

    max_value: int = 130
    min_dark_ratio: float = 0.55
    min_box_area_ratio: float = 0.002
    min_center_y_ratio: float = 0.35
    max_hands: int = 2


def _mask_for_image(mask, img_w, img_h):
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        mask = np.squeeze(mask)
    if mask.shape != (img_h, img_w):
        mask = cv2.resize(
            mask, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def _mask_iou(mask_a, mask_b):
    intersection = np.count_nonzero(mask_a & mask_b)
    if intersection == 0:
        return 0.0
    union = np.count_nonzero(mask_a | mask_b)
    return float(intersection / union) if union else 0.0


def _box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if intersection <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.0


def select_black_gloved_hands(
    image_bgr,
    masks,
    boxes,
    scores,
    config=BlackGloveFilterConfig(),
):
    """Return at most two large, dark, lower-frame SAM3 hand detections.

    Color is measured only inside each SAM mask, rather than in the bounding
    rectangle, so skin/background pixels around an otherwise good mask do not
    distort the result.  Candidates are ranked by semantic confidence, black
    ratio, size and proximity to the camera wearer.
    """
    if image_bgr is None or image_bgr.size == 0:
        return [], [], []

    img_h, img_w = image_bgr.shape[:2]
    image_area = float(max(1, img_w * img_h))
    value_channel = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    candidates = []

    score_values = list(scores) if scores else [1.0] * len(boxes)
    for index, (mask, box, score) in enumerate(zip(masks, boxes, score_values)):
        if len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        x1 = float(np.clip(x1, 0.0, img_w))
        y1 = float(np.clip(y1, 0.0, img_h))
        x2 = float(np.clip(x2, 0.0, img_w))
        y2 = float(np.clip(y2, 0.0, img_h))
        box_area_ratio = max(0.0, x2 - x1) * max(0.0, y2 - y1) / image_area
        center_y_ratio = (y1 + y2) / (2.0 * img_h)

        mask_bool = _mask_for_image(mask, img_w, img_h)
        mask_pixels = int(np.count_nonzero(mask_bool))
        if mask_pixels == 0:
            continue

        # Erosion avoids counting a thin halo of background introduced by mask
        # resizing.  Very small masks use the original mask to avoid vanishing.
        mask_u8 = mask_bool.astype(np.uint8)
        inner = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=1) > 0
        if np.count_nonzero(inner) < max(20, int(mask_pixels * 0.35)):
            inner = mask_bool
        dark_ratio = float(np.mean(value_channel[inner] <= config.max_value))

        if dark_ratio < config.min_dark_ratio:
            continue
        if box_area_ratio < config.min_box_area_ratio:
            continue
        if center_y_ratio < config.min_center_y_ratio:
            continue

        area_quality = min(1.0, math.sqrt(box_area_ratio / 0.03))
        quality = (
            float(score)
            + 0.75 * dark_ratio
            + 0.35 * area_quality
            + 0.25 * center_y_ratio
        )
        candidates.append({
            "index": index,
            "box": [x1, y1, x2, y2],
            "mask": mask_bool,
            "quality": quality,
        })

    # Suppress repeated SAM masks before enforcing the two-hand limit.
    selected = []
    for candidate in sorted(candidates, key=lambda item: item["quality"], reverse=True):
        duplicate = any(
            _box_iou(candidate["box"], previous["box"]) >= 0.65
            or _mask_iou(candidate["mask"], previous["mask"]) >= 0.60
            for previous in selected
        )
        if duplicate:
            continue
        selected.append(candidate)
        if len(selected) >= max(1, int(config.max_hands)):
            break

    # Restore SAM's original ordering so masks, boxes and scores stay aligned.
    selected.sort(key=lambda item: item["index"])
    indices = [item["index"] for item in selected]
    return (
        [masks[index] for index in indices],
        [boxes[index] for index in indices],
        [score_values[index] for index in indices],
    )


_BaseSAM3TextSegmenter = base_app.SAM3TextSegmenter


class BlackGloveSAM3TextSegmenter(_BaseSAM3TextSegmenter):
    """Original SAM3 segmenter plus the purpose-built candidate filter."""

    filter_config = BlackGloveFilterConfig()

    @classmethod
    def configure_filter(cls, config):
        cls.filter_config = config

    def detect(self, image_bgr, text_prompt):
        masks, boxes, scores = super().detect(image_bgr, text_prompt)
        prompt = str(text_prompt)
        if not (_HAND_PROMPT_RE.search(prompt) or _GLOVE_PROMPT_RE.search(prompt)):
            return masks, boxes, scores
        return select_black_gloved_hands(
            image_bgr, masks, boxes, scores, self.filter_config)

    def detect_many(self, image_bgr, text_prompts):
        """Detect several semantic classes with one shared image encoding.

        SAM3 supports multiple text classes in one forward pass.  Keeping the
        class split here avoids encoding the same frame once per prompt while
        preserving the exact per-prompt black-glove filter used by ``detect``.
        The return value maps every input prompt to ``(masks, boxes, scores)``.
        """

        prompts = [str(prompt) for prompt in text_prompts]
        if not prompts:
            return {}
        results = base_app.UltralyticsModel.predict(
            self.model,
            source=image_bgr,
            predictor=base_app.SAM3SemanticPredictor,
            prompts={"text": prompts},
            device=self.device,
            conf=self.conf,
            verbose=False,
        )
        empty = {prompt: ([], [], []) for prompt in prompts}
        if not results:
            return empty

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return empty
        boxes = result.boxes.xyxy.detach().cpu().numpy().tolist()
        scores = (
            result.boxes.conf.detach().cpu().numpy().tolist()
            if result.boxes.conf is not None
            else [1.0] * len(boxes)
        )
        classes = (
            result.boxes.cls.detach().cpu().numpy().astype(int).tolist()
            if result.boxes.cls is not None
            else [0] * len(boxes)
        )
        masks = []
        if result.masks is not None:
            masks = result.masks.data.detach().cpu().numpy().astype(np.uint8)
            masks = list(masks)

        output = {}
        for class_id, prompt in enumerate(prompts):
            indices = [
                index
                for index, detected_class in enumerate(classes)
                if detected_class == class_id
                and index < len(boxes)
                and index < len(masks)
            ]
            prompt_masks = [masks[index] for index in indices]
            prompt_boxes = [boxes[index] for index in indices]
            prompt_scores = [scores[index] for index in indices]
            if _HAND_PROMPT_RE.search(prompt) or _GLOVE_PROMPT_RE.search(prompt):
                prompt_masks, prompt_boxes, prompt_scores = (
                    select_black_gloved_hands(
                        image_bgr,
                        prompt_masks,
                        prompt_boxes,
                        prompt_scores,
                        self.filter_config,
                    )
                )
            output[prompt] = (prompt_masks, prompt_boxes, prompt_scores)
        return output


class BlackGloveAnnotatorApp(base_app.AutoAnnotatorApp):
    """Specialized UI that keeps the original annotator available separately."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM3 黑手套左右手专用标注器")
        self.resize(940, 820)
        self.ed_text_prompt.setText("hand wearing black glove")
        self.ed_text_prompt.setToolTip(
            "专用版推荐保持为 hand wearing black glove。\n"
            "提示词必须包含 hand，程序才会导出 left/right 两类。"
        )
        self.sp_text_box_thresh.setValue(0.25)

        group = QGroupBox("🧤 黑手套专用过滤")
        grid = QGridLayout()

        self.sp_black_value = QSpinBox()
        self.sp_black_value.setRange(40, 220)
        self.sp_black_value.setValue(130)
        self.sp_black_value.setToolTip(
            "HSV 亮度 V 不高于该值的像素视为黑色。光线很暗可降低，手套反光严重可提高。"
        )

        self.sp_black_ratio = QDoubleSpinBox()
        self.sp_black_ratio.setRange(0.05, 1.0)
        self.sp_black_ratio.setSingleStep(0.05)
        self.sp_black_ratio.setDecimals(2)
        self.sp_black_ratio.setValue(0.55)
        self.sp_black_ratio.setToolTip(
            "SAM mask 内至少有多少比例的像素必须是黑色；裸手通常会在这里被排除。"
        )

        self.sp_min_area_pct = QDoubleSpinBox()
        self.sp_min_area_pct.setRange(0.0, 10.0)
        self.sp_min_area_pct.setSingleStep(0.05)
        self.sp_min_area_pct.setDecimals(2)
        self.sp_min_area_pct.setSuffix(" %")
        self.sp_min_area_pct.setValue(0.20)
        self.sp_min_area_pct.setToolTip(
            "候选框占整张图的最小面积；用于排除远处人物的小手。"
        )

        self.sp_min_center_y = QDoubleSpinBox()
        self.sp_min_center_y.setRange(0.0, 1.0)
        self.sp_min_center_y.setSingleStep(0.05)
        self.sp_min_center_y.setDecimals(2)
        self.sp_min_center_y.setValue(0.35)
        self.sp_min_center_y.setToolTip(
            "手框中心至少位于画面高度的这个比例以下；0.35 表示排除上方 35% 区域。"
        )

        self.sp_max_hands = QSpinBox()
        self.sp_max_hands.setRange(1, 4)
        self.sp_max_hands.setValue(2)
        self.sp_max_hands.setToolTip("你的任务是左右手，建议固定为 2。")

        grid.addWidget(QLabel("黑色亮度上限 V:"), 0, 0)
        grid.addWidget(self.sp_black_value, 0, 1)
        grid.addWidget(QLabel("mask 最小黑色占比:"), 0, 2)
        grid.addWidget(self.sp_black_ratio, 0, 3)
        grid.addWidget(QLabel("最小框面积:"), 1, 0)
        grid.addWidget(self.sp_min_area_pct, 1, 1)
        grid.addWidget(QLabel("最小中心高度比例:"), 1, 2)
        grid.addWidget(self.sp_min_center_y, 1, 3)
        grid.addWidget(QLabel("每帧最多保留:"), 2, 0)
        grid.addWidget(self.sp_max_hands, 2, 1)
        hint = QLabel(
            "默认值按当前鱼眼数据设置：先用精确提示词找黑手套，再按颜色、大小、位置筛选，最后只留质量最高的两只。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #FF9800; font-size: 11px;")
        grid.addWidget(hint, 2, 2, 1, 2)
        group.setLayout(grid)

        # Base layout order is paths, parameters, actions, log.  Insert the
        # specialized controls without changing the original UI source file.
        self.centralWidget().layout().insertWidget(2, group)
        self.append_log(
            "\n🧤 当前是黑手套专用版：SAM3 精确提示 + 黑色占比 + 第一视角位置/尺寸 + 最多两手过滤已启用。"
        )

    def start_processing(self):
        prompt = self.ed_text_prompt.text().strip()
        if not _HAND_PROMPT_RE.search(prompt):
            QMessageBox.warning(
                self,
                "提示词不正确",
                "黑手套专用版的提示词必须包含英文 hand，才能正确导出 left/right。\n"
                "建议直接使用：hand wearing black glove",
            )
            return

        BlackGloveSAM3TextSegmenter.configure_filter(BlackGloveFilterConfig(
            max_value=self.sp_black_value.value(),
            min_dark_ratio=self.sp_black_ratio.value(),
            min_box_area_ratio=self.sp_min_area_pct.value() / 100.0,
            min_center_y_ratio=self.sp_min_center_y.value(),
            max_hands=self.sp_max_hands.value(),
        ))
        self.append_log(
            "🧤 本次黑手套过滤参数: "
            f"V≤{self.sp_black_value.value()}, "
            f"黑色占比≥{self.sp_black_ratio.value():.2f}, "
            f"框面积≥{self.sp_min_area_pct.value():.2f}%, "
            f"中心Y≥{self.sp_min_center_y.value():.2f}, "
            f"最多{self.sp_max_hands.value()}只"
        )
        super().start_processing()


def main():
    # AnnotationThread resolves this symbol from base_app at runtime.  Replacing
    # it here scopes the behavior to this dedicated launcher; run.py still uses
    # the untouched original class in a separate process.
    base_app.SAM3TextSegmenter = BlackGloveSAM3TextSegmenter

    # Importing the non-headless OpenCV wheel points Qt at cv2's bundled plugin
    # directory.  That Qt build is not ABI-compatible with PyQt5, so make sure
    # QApplication loads PyQt5's own xcb platform plugin instead.
    pyqt_plugin_root = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyqt_plugin_root
    QCoreApplication.setLibraryPaths([pyqt_plugin_root])

    qt_app = QApplication(sys.argv)
    window = BlackGloveAnnotatorApp()
    window.show()
    return qt_app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
