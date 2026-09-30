"""SAM3 annotator variant for egocentric protective gloves.

This module intentionally leaves :mod:`auto_obb_annotator.app` unchanged.  It
reuses the existing UI/export pipeline and replaces only the SAM3 text
segmenter with a conservative whole-glove selector.
"""

import os
import re
import sys
import weakref

import cv2
import numpy as np
from PyQt5.QtCore import QCoreApplication, QLibraryInfo, pyqtSignal
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
)

from . import app as base_app
from .protective_glove_core import (
    ProtectiveGloveFilterConfig,
    select_protective_gloved_hands,
)


_HAND_PROMPT_RE = re.compile(r"\bhand(s)?\b", re.IGNORECASE)
_GLOVE_PROMPT_RE = re.compile(r"\bglove(s|d)?\b", re.IGNORECASE)


_BaseSAM3TextSegmenter = base_app.SAM3TextSegmenter
_ORIGINAL_MAKE_DATASET_RECORD = base_app.make_dataset_record
_ACTIVE_PREVIEW_WINDOW = None


def build_protective_glove_preview(img_path, image_labels, max_size=(960, 640)):
    """Draw one frame's completed labels for immediate display in the UI."""
    image = cv2.imread(img_path)
    if image is None:
        return None
    img_h, img_w = image.shape[:2]
    colors = [(0, 60, 255), (0, 210, 255)]  # left=red, right=yellow
    names = ["left", "right"]

    for class_id, label in image_labels:
        class_id = int(class_id)
        annotation_format = "hbb" if len(label) == 4 else "obb"
        color = colors[class_id % len(colors)]
        base_app.draw_yolo_label(
            image, label, annotation_format, color=color, thickness=3)
        if annotation_format == "hbb":
            cx, cy, width, height = label
            anchor_x = int(round((cx - width / 2.0) * img_w))
            anchor_y = int(round((cy - height / 2.0) * img_h))
        else:
            points = np.asarray(label, dtype=np.float32).reshape(4, 2)
            anchor_x = int(round(float(points[:, 0].min()) * img_w))
            anchor_y = int(round(float(points[:, 1].min()) * img_h))
        name = names[class_id] if 0 <= class_id < len(names) else str(class_id)
        text_y = max(22, anchor_y - 7)
        cv2.putText(
            image, name, (max(2, anchor_x + 3), text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

    status = f"complete glove boxes: {len(image_labels)}"
    cv2.rectangle(image, (8, 8), (300, 40), (20, 20, 20), -1)
    cv2.putText(
        image, status, (16, 31), cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (80, 255, 120), 2, cv2.LINE_AA)

    max_width, max_height = max_size
    scale = min(1.0, max_width / img_w, max_height / img_h)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(round(img_w * scale))),
             max(1, int(round(img_h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return image


def _make_dataset_record_with_live_preview(
    img_path, img_name, labels, img_w, img_h
):
    """Keep the base record contract and emit a preview from the worker."""
    record = _ORIGINAL_MAKE_DATASET_RECORD(
        img_path, img_name, labels, img_w, img_h)
    window = _ACTIVE_PREVIEW_WINDOW() if _ACTIVE_PREVIEW_WINDOW else None
    if window is not None:
        try:
            preview = build_protective_glove_preview(img_path, labels)
            if preview is not None:
                window.live_preview_signal.emit(preview, img_name)
        except Exception as exc:
            # Preview must never interrupt annotation/export.
            window.preview_error_signal.emit(str(exc))
    return record


def install_live_preview(window):
    """Install a process-local hook used only by the protective-glove entry."""
    global _ACTIVE_PREVIEW_WINDOW
    _ACTIVE_PREVIEW_WINDOW = weakref.ref(window)
    base_app.make_dataset_record = _make_dataset_record_with_live_preview


class ProtectiveGloveSAM3TextSegmenter(_BaseSAM3TextSegmenter):
    """Original SAM3 segmenter plus the purpose-built candidate filter."""

    filter_config = ProtectiveGloveFilterConfig()

    @classmethod
    def configure_filter(cls, config):
        cls.filter_config = config

    def detect(self, image_bgr, text_prompt):
        masks, boxes, scores = super().detect(image_bgr, text_prompt)
        prompt = str(text_prompt)
        if not (_HAND_PROMPT_RE.search(prompt) or _GLOVE_PROMPT_RE.search(prompt)):
            return masks, boxes, scores
        return select_protective_gloved_hands(
            image_bgr, masks, boxes, scores, self.filter_config)

    def detect_many(self, image_bgr, text_prompts):
        """Detect several semantic classes with one shared image encoding.

        SAM3 supports multiple text classes in one forward pass.  Keeping the
        class split here avoids encoding the same frame once per prompt while
        preserving the exact per-prompt glove filter used by ``detect``.
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
                    select_protective_gloved_hands(
                        image_bgr,
                        prompt_masks,
                        prompt_boxes,
                        prompt_scores,
                        self.filter_config,
                    )
                )
            output[prompt] = (prompt_masks, prompt_boxes, prompt_scores)
        return output


class ProtectiveGloveAnnotatorApp(base_app.AutoAnnotatorApp):
    """Specialized UI that keeps the original annotator available separately."""

    live_preview_signal = pyqtSignal(object, str)
    preview_error_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("SAM3 灰白/黑色防护手套完整框标注器")
        self.resize(1080, 960)
        self.ed_text_prompt.setText("hand wearing a protective work glove")
        self.ed_text_prompt.setToolTip(
            "专用版推荐保持为 hand wearing a protective work glove。\n"
            "提示词必须包含 hand，程序才会导出 left/right 两类。"
        )
        self.sp_text_box_thresh.setValue(0.25)
        hbb_index = self.cmb_annotation_format.findData("hbb")
        if hbb_index >= 0:
            self.cmb_annotation_format.setCurrentIndex(hbb_index)
        # The generic fixed-table ROI does not fit a moving egocentric camera.
        self.chk_workspace_roi.blockSignals(True)
        self.chk_workspace_roi.setChecked(False)
        self.chk_workspace_roi.blockSignals(False)
        self.live_preview_signal.connect(self.update_annotation_preview)
        self.preview_error_signal.connect(
            lambda message: self.append_log(f"⚠️ 实时预览失败: {message}"))

        group = QGroupBox("🧤 防护手套完整框参数")
        grid = QGridLayout()

        self.sp_black_value = QSpinBox()
        self.sp_black_value.setRange(40, 220)
        self.sp_black_value.setValue(170)
        self.sp_black_value.setToolTip(
            "仅在启用深色部件过滤时使用；V 不高于该值视为深色。"
        )

        self.sp_black_ratio = QDoubleSpinBox()
        self.sp_black_ratio.setRange(0.0, 1.0)
        self.sp_black_ratio.setSingleStep(0.05)
        self.sp_black_ratio.setDecimals(2)
        self.sp_black_ratio.setValue(0.0)
        self.sp_black_ratio.setToolTip(
            "0 表示关闭颜色过滤（推荐）。目标手套是灰白护具与黑色手指混合，"
            "不应再用纯黑占比作为硬条件。"
        )

        self.sp_box_padding = QDoubleSpinBox()
        self.sp_box_padding.setRange(0.0, 30.0)
        self.sp_box_padding.setSingleStep(1.0)
        self.sp_box_padding.setDecimals(1)
        self.sp_box_padding.setSuffix(" %")
        self.sp_box_padding.setValue(8.0)
        self.sp_box_padding.setToolTip(
            "在 SAM 整体手套框四周增加留边，防止切掉指尖、护手板或腕部。")

        self.sp_fragment_gap = QDoubleSpinBox()
        self.sp_fragment_gap.setRange(0.0, 20.0)
        self.sp_fragment_gap.setSingleStep(1.0)
        self.sp_fragment_gap.setDecimals(1)
        self.sp_fragment_gap.setSuffix(" %")
        self.sp_fragment_gap.setValue(4.0)
        self.sp_fragment_gap.setToolTip(
            "连接同一只手套中被 SAM 分开的手指/掌面小区域；过大可能粘连两只手。")

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
        self.sp_min_center_y.setValue(0.15)
        self.sp_min_center_y.setToolTip(
            "手框中心至少位于画面高度的这个比例以下；当前第一视角样例中的手套"
            "会出现在画面中上部，因此默认仅排除最上方 15%。"
        )

        self.sp_max_hands = QSpinBox()
        self.sp_max_hands.setRange(1, 4)
        self.sp_max_hands.setValue(2)
        self.sp_max_hands.setToolTip("你的任务是左右手，建议固定为 2。")

        grid.addWidget(QLabel("深色亮度上限 V:"), 0, 0)
        grid.addWidget(self.sp_black_value, 0, 1)
        grid.addWidget(QLabel("最小深色占比（0=关闭）:"), 0, 2)
        grid.addWidget(self.sp_black_ratio, 0, 3)
        grid.addWidget(QLabel("最小框面积:"), 1, 0)
        grid.addWidget(self.sp_min_area_pct, 1, 1)
        grid.addWidget(QLabel("最小中心高度比例:"), 1, 2)
        grid.addWidget(self.sp_min_center_y, 1, 3)
        grid.addWidget(QLabel("每帧最多保留:"), 2, 0)
        grid.addWidget(self.sp_max_hands, 2, 1)
        grid.addWidget(QLabel("完整框四周留边:"), 2, 2)
        grid.addWidget(self.sp_box_padding, 2, 3)
        grid.addWidget(QLabel("mask 断裂连接距离:"), 3, 0)
        grid.addWidget(self.sp_fragment_gap, 3, 1)
        hint = QLabel(
            "目标是灰白护具 + 黑色手指的整只防护手套。程序合并手指碎片，"
            "保留 SAM 整体语义范围并增加留边；颜色过滤默认关闭。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #FF9800; font-size: 11px;")
        grid.addWidget(hint, 3, 2, 1, 2)
        group.setLayout(grid)

        # Base layout order is paths, parameters, actions, log.  Insert the
        # specialized controls without changing the original UI source file.
        self.centralWidget().layout().insertWidget(2, group)

        preview_group = QGroupBox("👁️ 实时标注结果预览")
        preview_layout = QGridLayout(preview_group)
        self.lbl_preview_title = QLabel("开始后，每完成一张就会立即显示")
        self.lbl_preview_title.setStyleSheet(
            "color:#64B5F6;font-size:12px;font-weight:bold;")
        self.lbl_preview = QLabel("运行后将在这里实时显示完整手套标注框")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setMinimumSize(640, 260)
        self.lbl_preview.setStyleSheet(
            "QLabel{background:#181818;color:#999;border:1px solid #444;}")
        preview_layout.addWidget(self.lbl_preview_title, 0, 0)
        preview_layout.addWidget(self.lbl_preview, 1, 0)
        nav_layout = QHBoxLayout()
        self.btn_preview_prev = QPushButton("◀ 上一张")
        self.btn_preview_next = QPushButton("下一张 ▶")
        self.btn_preview_prev.setEnabled(False)
        self.btn_preview_next.setEnabled(False)
        self.btn_preview_prev.clicked.connect(self.show_previous_preview)
        self.btn_preview_next.clicked.connect(self.show_next_preview)
        nav_layout.addStretch(1)
        nav_layout.addWidget(self.btn_preview_prev)
        nav_layout.addWidget(self.btn_preview_next)
        nav_layout.addStretch(1)
        preview_layout.addLayout(nav_layout, 2, 0)
        layout = self.centralWidget().layout()
        layout.insertWidget(layout.count() - 1, preview_group)
        self.log_console.setMaximumHeight(170)
        self._preview_pixmap = None
        self._preview_files = []
        self._preview_index = -1
        self._processing_image_names = []
        self._last_live_preview_name = ""
        self.append_log(
            "\n🧤 当前是混合色防护手套专用版：整只手套语义框 + mask 碎片合并 + "
            "完整框留边 + 逐张实时预览已启用。"
        )

    def update_annotation_preview(self, image_bgr, image_name):
        """Display the newest annotated frame delivered by the worker."""
        if image_bgr is None or image_bgr.size == 0:
            return
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        qt_image = QImage(
            rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()
        self._preview_pixmap = QPixmap.fromImage(qt_image)
        self._last_live_preview_name = image_name
        self.lbl_preview_title.setText(f"实时结果：{image_name}")
        self._refresh_preview_pixmap()

    def _refresh_preview_pixmap(self):
        if self._preview_pixmap is None:
            return
        self.lbl_preview.setPixmap(self._preview_pixmap.scaled(
            self.lbl_preview.size(), Qt.KeepAspectRatio,
            Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview_pixmap()

    def _load_exported_previews(self):
        out_dir = str(base_app.resolve_path(self.lbl_out.text()))
        vis_dir = os.path.join(out_dir, "visualizations")
        if not os.path.isdir(vis_dir):
            self.lbl_preview_title.setText("没有找到可视化输出目录")
            return
        self._preview_files = sorted(
            os.path.join(vis_dir, name)
            for name in os.listdir(vis_dir)
            if name.lower().endswith((".jpg", ".jpeg", ".png"))
            and name.startswith("vis_")
        )
        if not self._preview_files:
            self._preview_index = -1
            self.btn_preview_prev.setEnabled(False)
            self.btn_preview_next.setEnabled(False)
            self.lbl_preview_title.setText("本次没有生成可预览的有效标注")
            self.lbl_preview.setText("请查看运行日志中的过滤统计和参数提示")
            return
        self._show_preview_at(0)

    def _show_preview_at(self, index):
        if not self._preview_files:
            return
        self._preview_index = int(index) % len(self._preview_files)
        path = self._preview_files[self._preview_index]
        image = cv2.imread(path)
        if image is None:
            self.lbl_preview_title.setText(f"无法读取预览：{os.path.basename(path)}")
            return
        self.update_annotation_preview(image, os.path.basename(path))
        total = len(self._preview_files)
        self.lbl_preview_title.setText(
            f"标注结果 {self._preview_index + 1}/{total}：{os.path.basename(path)}")
        self.btn_preview_prev.setEnabled(total > 1)
        self.btn_preview_next.setEnabled(total > 1)

    def show_previous_preview(self):
        self._show_preview_at(self._preview_index - 1)

    def show_next_preview(self):
        self._show_preview_at(self._preview_index + 1)

    def start_processing(self):
        prompt = self.ed_text_prompt.text().strip()
        if not _HAND_PROMPT_RE.search(prompt):
            QMessageBox.warning(
                self,
                "提示词不正确",
                "防护手套专用版的提示词必须包含英文 hand，才能正确导出 left/right。\n"
                "建议直接使用：hand wearing a protective work glove",
            )
            return

        ProtectiveGloveSAM3TextSegmenter.configure_filter(ProtectiveGloveFilterConfig(
            max_value=self.sp_black_value.value(),
            min_dark_ratio=self.sp_black_ratio.value(),
            min_box_area_ratio=self.sp_min_area_pct.value() / 100.0,
            min_center_y_ratio=self.sp_min_center_y.value(),
            max_hands=self.sp_max_hands.value(),
            box_padding_ratio=self.sp_box_padding.value() / 100.0,
            fragment_gap_ratio=self.sp_fragment_gap.value() / 100.0,
            complete_box_enabled=True,
        ))
        self.append_log(
            "🧤 本次防护手套参数: "
            f"V≤{self.sp_black_value.value()}, "
            f"深色占比≥{self.sp_black_ratio.value():.2f}, "
            f"框面积≥{self.sp_min_area_pct.value():.2f}%, "
            f"中心Y≥{self.sp_min_center_y.value():.2f}, "
            f"留边={self.sp_box_padding.value():.1f}%, "
            f"断裂连接={self.sp_fragment_gap.value():.1f}%, "
            f"最多{self.sp_max_hands.value()}只"
        )
        self._preview_pixmap = None
        input_dir = str(base_app.resolve_path(self.lbl_input.text()))
        self._processing_image_names = sorted(
            name for name in os.listdir(input_dir)
            if name.lower().endswith((".png", ".jpg"))
        ) if os.path.isdir(input_dir) else []
        self._last_live_preview_name = ""
        self.lbl_preview.clear()
        self.lbl_preview.setText("正在等待第一张有效标注结果…")
        self.lbl_preview_title.setText("正在加载模型，首张完成后立即显示…")
        super().start_processing()

    def update_progress(self, current, total):
        """Also show processed frames with zero boxes so misses are visible."""
        super().update_progress(current, total)
        index = int(current) - 1
        if not (0 <= index < len(self._processing_image_names)):
            return
        image_name = self._processing_image_names[index]
        if image_name == self._last_live_preview_name:
            return
        input_dir = str(base_app.resolve_path(self.lbl_input.text()))
        preview = build_protective_glove_preview(
            os.path.join(input_dir, image_name), [])
        if preview is not None:
            self.update_annotation_preview(preview, image_name)

    def refilter_with_new_threshold(self):
        super().refilter_with_new_threshold()
        self._load_exported_previews()

    def on_processing_finished(self):
        super().on_processing_finished()
        self._load_exported_previews()


def main():
    # AnnotationThread resolves this symbol from base_app at runtime.  Replacing
    # it here scopes the behavior to this dedicated launcher; run.py still uses
    # the untouched original class in a separate process.
    base_app.SAM3TextSegmenter = ProtectiveGloveSAM3TextSegmenter

    # Importing the non-headless OpenCV wheel points Qt at cv2's bundled plugin
    # directory.  That Qt build is not ABI-compatible with PyQt5, so make sure
    # QApplication loads PyQt5's own xcb platform plugin instead.
    pyqt_plugin_root = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyqt_plugin_root
    QCoreApplication.setLibraryPaths([pyqt_plugin_root])

    qt_app = QApplication(sys.argv)
    window = ProtectiveGloveAnnotatorApp()
    install_live_preview(window)
    window.show()
    return qt_app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
