import sys
import os
import cv2
import math
import numpy as np
import shutil
import random
import gc
import re
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QLabel, QSpinBox,
                             QDoubleSpinBox, QFileDialog, QMessageBox, QGroupBox,
                             QTextEdit, QProgressBar, QGridLayout, QInputDialog,
                             QDialog, QSlider, QFrame, QCheckBox, QComboBox,
                             QLineEdit)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QPoint, QRect, QSettings
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QTransform
from ultralytics import SAM
from ultralytics.engine.model import Model as UltralyticsModel
try:
    from ultralytics.models.sam.predict import SAM3SemanticPredictor
except Exception:
    SAM3SemanticPredictor = None
try:
    from ultralytics.models.sam.build_sam3 import build_sam3_image_model
except Exception:
    build_sam3_image_model = None
from .project_paths import (
    GROUNDING_DINO_MODEL,
    GROUNDING_DINO_TOKENIZER,
    HBB_DATASET_DIR,
    MOBILE_SAM_MODEL,
    OBB_DATASET_DIR,
    REALSENSE_BG_DIR,
    REALSENSE_COLOR_DIR,
    SAM3_MODEL,
    display_path,
    resolve_path,
)

try:
    import torch
except Exception:
    torch = None


SAM3_MODEL_PATH = str(SAM3_MODEL)

# 该默认值对应固定相机画面中的灰色台面（用户示例中的红框）。坐标按图片
# 宽高归一化，所以同一机位的不同分辨率可以共用；也可在界面中重新框选。
DEFAULT_WORKSPACE_ROI = (0.23, 0.02, 0.74, 0.66)


class GroundingDINOTextDetector:
    """使用本地 GroundingDINO ONNX，根据文字提示输出 xyxy 候选框。"""

    SPECIAL_TOKENS = [101, 102, 1012, 1029]
    IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path=str(GROUNDING_DINO_MODEL),
                 tokenizer_path=str(GROUNDING_DINO_TOKENIZER)):
        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise RuntimeError(
                "文本提示模式需要 onnxruntime-gpu 和 tokenizers。"
            ) from exc
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"GroundingDINO 模型不存在: {model_path}")
        if not os.path.isfile(tokenizer_path):
            raise FileNotFoundError(f"GroundingDINO tokenizer 不存在: {tokenizer_path}")

        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            providers.insert(0, "CUDAExecutionProvider")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            model_path, providers=providers, sess_options=options)
        self.provider = self.session.get_providers()[0]
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_text_len = 256
        self.input_size = (1200, 800)

    @staticmethod
    def _caption(text_prompt):
        caption = str(text_prompt).lower().strip()
        return caption if caption.endswith(".") else caption + "."

    def _preprocess_image(self, image_bgr):
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size, interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        image = (image - self.IMAGE_MEAN) / self.IMAGE_STD
        return np.expand_dims(np.transpose(image, (2, 0, 1)), 0).astype(np.float32)

    def _encode_text(self, text_prompt):
        caption = self._caption(text_prompt)
        raw = self.tokenizer.encode(caption)
        input_ids = np.array([raw.ids], dtype=np.int64)
        token_type_ids = np.array([raw.type_ids], dtype=np.int64)
        attention_mask = np.array([raw.attention_mask], dtype=bool)
        _, num_token = input_ids.shape
        special_mask = np.zeros_like(input_ids, dtype=bool)
        for token in self.SPECIAL_TOKENS:
            special_mask |= input_ids == token
        text_mask = np.eye(num_token, dtype=bool)[None, ...]
        position_ids = np.zeros((1, num_token), dtype=np.int64)
        previous_col = 0
        for row, col in np.argwhere(special_mask):
            if col == 0 or col == num_token - 1:
                text_mask[row, col, col] = True
            else:
                text_mask[row, previous_col + 1:col + 1,
                          previous_col + 1:col + 1] = True
                position_ids[row, previous_col + 1:col + 1] = np.arange(
                    col - previous_col, dtype=np.int64)
            previous_col = col
        limit = min(num_token, self.max_text_len)
        tokenized = {
            "input_ids": input_ids[:, :limit],
            "attention_mask": attention_mask[:, :limit],
            "token_type_ids": token_type_ids[:, :limit],
        }
        return tokenized, text_mask[:, :limit, :limit], position_ids[:, :limit]

    def detect(self, image_bgr, text_prompt, box_threshold=0.25):
        img_h, img_w = image_bgr.shape[:2]
        blob = self._preprocess_image(image_bgr)
        tokenized, text_mask, position_ids = self._encode_text(text_prompt)
        logits, boxes = self.session.run(None, {
            "img": blob,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "token_type_ids": tokenized["token_type_ids"],
            "position_ids": position_ids,
            "text_token_mask": text_mask,
        })
        scores = 1.0 / (1.0 + np.exp(-np.squeeze(logits, 0)))
        boxes = np.squeeze(boxes, 0)
        keep = scores.max(axis=1) > float(box_threshold)
        scores = scores.max(axis=1)[keep]
        boxes = boxes[keep]
        prompts = []
        prompt_scores = []
        for box, score in zip(boxes, scores):
            cx, cy, bw, bh = box * np.array([img_w, img_h, img_w, img_h])
            x1 = int(np.clip(round(cx - bw / 2), 0, img_w - 1))
            y1 = int(np.clip(round(cy - bh / 2), 0, img_h - 1))
            x2 = int(np.clip(round(cx + bw / 2), 1, img_w))
            y2 = int(np.clip(round(cy + bh / 2), 1, img_h))
            if x2 > x1 and y2 > y1:
                prompts.append([x1, y1, x2, y2])
                prompt_scores.append(float(score))
        return prompts, prompt_scores


class SAM3TextSegmenter:
    """SAM3 图像语义预测：直接用文本提示返回目标 mask 和框。"""

    def __init__(self, model_path=SAM3_MODEL_PATH, device="cpu", conf=0.25):
        if SAM3SemanticPredictor is None:
            raise RuntimeError("当前 Ultralytics 版本没有 SAM3SemanticPredictor。")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"SAM3 权重不存在: {model_path}")
        self.model = SAM(model_path)
        if build_sam3_image_model is None:
            raise RuntimeError("当前 Ultralytics 版本没有 SAM3 图像语义模型构造器。")
        # SAM(model_path) 默认加载交互式 SAM3Model；文本图像预测器需要
        # 带 backbone 的 SAM3SemanticModel，因此替换为 image model。
        old_model = self.model.model
        self.model.model = None
        del old_model
        clear_torch_cache()
        self.model.model = build_sam3_image_model(model_path)
        self.device = device
        self.conf = float(conf)
        self.model.to(device)

    def detect(self, image_bgr, text_prompt):
        # SAM3SemanticPredictor 从 prompts['text'] 读取文本提示。
        # SAM.predict() 会自动构造 prompts={bboxes,points,labels}，与 SAM3
        # 文本 prompts 重复。直接调用基础 Model.predict，避免参数冲突。
        results = UltralyticsModel.predict(
            self.model,
            source=image_bgr,
            predictor=SAM3SemanticPredictor,
            prompts={"text": [str(text_prompt)]},
            device=self.device,
            conf=self.conf,
            verbose=False,
        )
        if not results:
            return [], [], []
        result = results[0]
        masks = []
        boxes = []
        scores = []
        if result.masks is not None:
            masks = result.masks.data.detach().cpu().numpy().astype(np.uint8)
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.detach().cpu().numpy().tolist()
            if result.boxes.conf is not None:
                scores = result.boxes.conf.detach().cpu().numpy().tolist()
        # 以 mask 数量为准；SAM3 的结果应当 mask、box 一一对应。
        return list(masks), boxes, scores


def write_project_dataset_yaml(out_dir: str, class_names=None) -> None:
    """在导出目录写入可随数据集移动的 Ultralytics 配置。"""
    yaml_path = Path(out_dir).resolve() / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    class_names = list(class_names or ["object"])
    names_yaml = "\n".join(
        f"  {idx}: {name}" for idx, name in enumerate(class_names))
    yaml_path.write_text(
        "path: .\n"
        "train: images/train\n"
        "val: images/val\n\n"
        f"nc: {len(class_names)}\n"
        f"names:\n{names_yaml}\n",
        encoding="utf-8",
    )
# ================= 特征校验器（HOG + 颜色直方图，无需训练） =================
#
# 为什么替换 CNN：
#   随机初始化的 CNN 特征在余弦空间没有语义，效果等同于随机。
#   HOG（方向梯度直方图）+ 颜色直方图是无需训练的经典局部特征，
#   对纹理/形状/颜色都有稳定描述能力，适合小样本模板匹配场景。

class FeatureVerifier:
    """
    基于多通道颜色直方图 + 多尺度灰度直方图 的特征校验器。

    设计原则：
    ─ 使用「直方图交叉距离」作为相似度，范围天然 [0,1]，可直接与阈值比较
    ─ 不依赖向量维度/归一化方式，对不同大小的 patch 都稳定
    ─ 多尺度分块（全图 + 4宫格）捕捉空间布局，防止颜色相近但形态不同被误通过
    ─ 默认阈值 0.45（直方图交叉距离的合理范围），用户可在 UI 调节

    sim = 0 表示完全不同，sim = 1 表示完全一致
    """
    def __init__(self, sim_threshold=0.45):
        self.threshold = sim_threshold
        self.template_feats = []
        self.bg_feats = []
        # 某些 opencv-python-headless/精简构建不包含 objdetect 的 HOGDescriptor。
        # 文本提示模式通常没有模板特征，此时 HOG 不是必需项，应允许继续运行。
        self._hog = None
        if hasattr(cv2, "HOGDescriptor"):
            self._hog = cv2.HOGDescriptor(
                _winSize=(64, 64),
                _blockSize=(16, 16),
                _blockStride=(8, 8),
                _cellSize=(8, 8),
                _nbins=9,
            )

    # ── 特征提取 ────────────────────────────────────────────────────────
    def _extract(self, bgr_patch):
        """
        多通道颜色直方图 + 多尺度分块。
        输出：已归一化的一维 float32 向量（总和 = 1.0）。
        """
        img = cv2.resize(bgr_patch, (64, 64)).astype(np.float32)
        hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)

        parts = []

        # ── 全图 HSV 直方图 ──────────────────────────────────────────
        for ch, bins, rng in [(0, 18, (0, 180)),   # H
                               (1, 8,  (0, 256)),   # S
                               (2, 8,  (0, 256))]:  # V
            h = cv2.calcHist([hsv], [ch], None, [bins], rng).flatten()
            parts.append(h)

        # ── 全图灰度直方图（捕捉亮度分布）──────────────────────────
        gh = cv2.calcHist([gray], [0], None, [16], [0, 256]).flatten()
        parts.append(gh)

        # ── HOG 形状特征（补充边缘/纹理方向信息）────────────────────
        if self._hog is not None:
            hog = self._hog.compute(gray).flatten()
            hog = np.clip(hog, 0.0, None)
            parts.append(hog)

        # ── 4宫格空间分块（捕捉空间布局差异）────────────────────────
        h2, w2 = 32, 32
        for r in range(2):
            for c in range(2):
                cell_hsv  = hsv[r*h2:(r+1)*h2, c*w2:(c+1)*w2]
                cell_gray = gray[r*h2:(r+1)*h2, c*w2:(c+1)*w2]
                ch_hist = cv2.calcHist([cell_hsv], [0], None, [12], (0, 180)).flatten()
                cv_hist = cv2.calcHist([cell_gray], [0], None, [8],  (0, 256)).flatten()
                parts.append(ch_hist)
                parts.append(cv_hist)

        feat = np.concatenate(parts).astype(np.float32)
        s = feat.sum()
        if s > 0:
            feat /= s          # L1 归一化 → 概率分布，适合直方图交叉
        return feat

    # ── 相似度：直方图交叉（0~1）────────────────────────────────────────
    @staticmethod
    def _hist_intersect(a, b):
        """直方图交叉距离：sum(min(a_i, b_i))，两个L1归一化向量结果在 [0,1]"""
        return float(np.minimum(a, b).sum())

    # ── 公开接口 ────────────────────────────────────────────────────────
    def add_template(self, bgr_patch):
        if bgr_patch is not None and bgr_patch.size > 0:
            self.template_feats.append(self._extract(bgr_patch))

    def add_background(self, bgr_patch):
        if bgr_patch is not None and bgr_patch.size > 0:
            self.bg_feats.append(self._extract(bgr_patch))

    def verify(self, bgr_patch):
        """
        返回 (is_valid: bool, best_sim: float)
        best_sim 是与所有模板中最高的直方图交叉相似度。
        """
        if not self.template_feats:
            return True, 1.0   # 未设模板 → 全放行

        if bgr_patch is None or bgr_patch.size == 0:
            return False, 0.0

        q = self._extract(bgr_patch)
        sims = [self._hist_intersect(q, t) for t in self.template_feats]
        best_sim = max(sims)

        if best_sim < self.threshold:
            return False, best_sim

        # 背景排斥：与模板相似度须明显高于背景
        if self.bg_feats:
            bg_sims = [self._hist_intersect(q, b) for b in self.bg_feats]
            best_bg = max(bg_sims)
            if best_sim - best_bg < 0.02:
                return False, best_sim

        return True, best_sim

    def calibrate_threshold(self, sample_patches, neg_patches=None):
        """
        自动校准阈值。
        策略：
          1. 计算所有模板 patch 互相之间的相似度 → 正样本分布
          2. 若有 neg_patches → 计算模板与负样本的相似度 → 负样本分布
             阈值 = (正样本最低值 + 负样本最高值) / 2  (取中间分界)
          3. 若无负样本 → 阈值 = 正样本最低值 × 0.90  (留10%余量)
        """
        if not self.template_feats or not sample_patches:
            return

        # 正样本相似度：每个模板 patch 与特征库中最高相似度
        pos_sims = []
        for p in sample_patches:
            if p is not None and p.size > 0:
                q = self._extract(p)
                sims = [self._hist_intersect(q, t) for t in self.template_feats]
                pos_sims.append(max(sims))

        if not pos_sims:
            return

        pos_min = float(np.min(pos_sims))
        if neg_patches:
            neg_sims = []
            for p in neg_patches:
                if p is not None and p.size > 0:
                    q = self._extract(p)
                    sims = [self._hist_intersect(q, t) for t in self.template_feats]
                    neg_sims.append(max(sims))
            if neg_sims:
                neg_max = float(np.max(neg_sims))
                # 取正负分布中间，偏向正样本保留更多（×0.85 比原 ×0.95 更宽松）
                thresh = (pos_min + neg_max) / 2 * 0.85
                # 上限不超过 pos_min，防止阈值反而把正样本挡在外面
                thresh = min(thresh, pos_min * 0.9)
                self.threshold = round(max(0.1, thresh), 3)
                return

        # 无负样本：用正样本最低值的 75%
        self.threshold = round(max(0.10, pos_min * 0.75), 3)


# ================= ROI 选框（旋转坐标系控制点 + 快捷键 + 复制） =================
#
# 快捷键一览（在弹窗图片区域获得焦点时生效）：
#   Space / Enter  确认当前框
#   Z              大步左旋 (-15°)
#   V              大步右旋 (+15°)
#   X              微调左旋 (-1°)
#   C              微调右旋 (+1°)
#   D              复制最后一个已确认框为新的 pending
#   Ctrl+D         复制并向右偏移 (便于批量同类目标)
#   Delete/BackSpace  删除最后一个已确认框
#   Escape         放弃当前 pending
#   方向键 ↑↓←→  平移 pending 框 (每次 2px)
#   W / S          纵向缩放 pending (+2px / -2px)
#   A / Q          横向缩放 pending (+2px / -2px)
#   R              重置旋转为 0°
#   Tab            依次激活（重新编辑）上一个已确认框

HANDLE_R      = 5
HANDLE_HIT    = 11
ROTATE_OFFSET = 30
ROTATE_R      = 8
ROTATE_HIT    = 13


def _rot_pt(px, py, cx, cy, angle_deg):
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    dx, dy = px - cx, py - cy
    return QPoint(int(round(cx + dx*c - dy*s)),
                  int(round(cy + dx*s + dy*c)))


def _rot_pt_f(px, py, cx, cy, angle_deg):
    rad = math.radians(angle_deg)
    c, s = math.cos(rad), math.sin(rad)
    dx, dy = px - cx, py - cy
    return cx + dx*c - dy*s, cy + dx*s + dy*c


class RotatableROILabel(QLabel):
    """
    Figma 风格 ROI 编辑器（含快捷键 + 框复制）
    状态: (cx, cy, w, h, angle_deg)
    """
    _ST_IDLE   = 0
    _ST_DRAW   = 1
    _ST_RESIZE = 2
    _ST_ROTATE = 3
    _ST_MOVE   = 4   # 框内拖拽平移

    # 快捷键信号：用于同步外部滑条
    rotation_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)   # 接收键盘事件
        self.rois = []

        self._state = self._ST_IDLE
        self._cx = 0.0; self._cy = 0.0
        self._w  = 0.0; self._h  = 0.0
        self._angle = 0.0
        self._has_pending = False

        self._draw_start = QPoint()
        self._drag_handle = -1
        self._move_last = (0.0, 0.0)   # 上一帧鼠标位置（用于平移增量）

    # ── 控制点 ──────────────────────────────────────────────────────────
    def _local_handles(self):
        w2, h2 = self._w/2, self._h/2
        return [
            (-w2,-h2),(0,-h2),(w2,-h2),
            (-w2, 0),         (w2, 0),
            (-w2, h2),(0, h2),(w2, h2),
            (0, -h2 - ROTATE_OFFSET),
        ]

    def _world_handles(self):
        if not self._has_pending:
            return []
        return [_rot_pt(self._cx+lx, self._cy+ly, self._cx, self._cy, self._angle)
                for lx,ly in self._local_handles()]

    def _hit_handle(self, pos):
        if not self._has_pending:
            return -1
        handles = self._world_handles()
        rh = handles[8]
        if math.hypot(pos.x()-rh.x(), pos.y()-rh.y()) <= ROTATE_HIT:
            return 8
        for i, h in enumerate(handles[:8]):
            if math.hypot(pos.x()-h.x(), pos.y()-h.y()) <= HANDLE_HIT:
                return i
        return -1

    def _hit_inside(self, pos):
        """判断鼠标是否在 pending 框内部（旋转坐标系点测试）"""
        if not self._has_pending:
            return False
        # 将鼠标点变换到框的本地坐标系
        rad = math.radians(-self._angle)
        c, s = math.cos(rad), math.sin(rad)
        dx, dy = pos.x() - self._cx, pos.y() - self._cy
        lx = dx * c - dy * s
        ly = dx * s + dy * c
        return abs(lx) <= self._w / 2 and abs(ly) <= self._h / 2

    # ── 鼠标 ────────────────────────────────────────────────────────────
    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() != Qt.LeftButton:
            return
        hit = self._hit_handle(event.pos())
        if hit == 8:
            self._state = self._ST_ROTATE
        elif hit >= 0:
            self._state = self._ST_RESIZE
            self._drag_handle = hit
        elif self._hit_inside(event.pos()):
            # 框内按下 → 进入平移模式
            self._state = self._ST_MOVE
            self._move_last = (float(event.pos().x()), float(event.pos().y()))
        else:
            self._state = self._ST_DRAW
            self._has_pending = False
            self._angle = 0.0
            self._draw_start = event.pos()
        self.update()

    def mouseMoveEvent(self, event):
        mx, my = float(event.pos().x()), float(event.pos().y())
        if self._state == self._ST_DRAW:
            s = self._draw_start
            x1=min(s.x(),event.pos().x()); y1=min(s.y(),event.pos().y())
            x2=max(s.x(),event.pos().x()); y2=max(s.y(),event.pos().y())
            self._cx,self._cy=(x1+x2)/2.0,(y1+y2)/2.0
            self._w,self._h=float(x2-x1),float(y2-y1)
            self._has_pending=(self._w>5 and self._h>5)
            self.update()
        elif self._state == self._ST_RESIZE:
            self._apply_resize(mx, my); self.update()
        elif self._state == self._ST_ROTATE:
            dx,dy=mx-self._cx,my-self._cy
            self._angle=math.degrees(math.atan2(dx,-dy))
            self.rotation_changed.emit(self._angle)
            self.update()
        elif self._state == self._ST_MOVE:
            # 平移：直接加上鼠标增量
            dx = mx - self._move_last[0]
            dy = my - self._move_last[1]
            self._cx += dx
            self._cy += dy
            self._move_last = (mx, my)
            self.update()
        else:
            # 悬停：根据位置设置光标
            hit = self._hit_handle(event.pos())
            if hit != -1:
                cursors={8:Qt.PointingHandCursor,0:Qt.SizeFDiagCursor,
                         7:Qt.SizeFDiagCursor,2:Qt.SizeBDiagCursor,
                         5:Qt.SizeBDiagCursor,1:Qt.SizeVerCursor,
                         6:Qt.SizeVerCursor,3:Qt.SizeHorCursor,4:Qt.SizeHorCursor}
                self.setCursor(cursors.get(hit, Qt.CrossCursor))
            elif self._hit_inside(event.pos()):
                self.setCursor(Qt.SizeAllCursor)   # 四向箭头
            else:
                self.setCursor(Qt.CrossCursor)

    def mouseReleaseEvent(self, event):
        if event.button()==Qt.LeftButton:
            if self._state==self._ST_DRAW:
                s=self._draw_start; p=event.pos()
                x1=min(s.x(),p.x()); y1=min(s.y(),p.y())
                x2=max(s.x(),p.x()); y2=max(s.y(),p.y())
                self._cx,self._cy=(x1+x2)/2.0,(y1+y2)/2.0
                self._w,self._h=float(x2-x1),float(y2-y1)
                self._has_pending=(self._w>5 and self._h>5)
            self._state=self._ST_IDLE; self._drag_handle=-1
            self.update()

    def _apply_resize(self, mx, my):
        h=self._drag_handle
        rad=math.radians(-self._angle)
        c,s=math.cos(rad),math.sin(rad)
        dx_w,dy_w=mx-self._cx,my-self._cy
        lx=dx_w*c-dy_w*s; ly=dx_w*s+dy_w*c
        MIN=10.0
        ang=self._angle

        def push_right(v):
            nw=max(MIN,v*2); sh=(nw-self._w)/2
            self._cx+=sh*math.cos(math.radians(ang))
            self._cy+=sh*math.sin(math.radians(ang)); self._w=nw
        def push_left(v):
            nw=max(MIN,-v*2); sh=-(nw-self._w)/2
            self._cx+=sh*math.cos(math.radians(ang))
            self._cy+=sh*math.sin(math.radians(ang)); self._w=nw
        def push_bottom(v):
            nh=max(MIN,v*2); sh=(nh-self._h)/2
            self._cx+=-sh*math.sin(math.radians(ang))
            self._cy+=sh*math.cos(math.radians(ang)); self._h=nh
        def push_top(v):
            nh=max(MIN,-v*2); sh=-(nh-self._h)/2
            self._cx+=-sh*math.sin(math.radians(ang))
            self._cy+=sh*math.cos(math.radians(ang)); self._h=nh

        if h==0: push_left(lx); push_top(ly)
        elif h==1: push_top(ly)
        elif h==2: push_right(lx); push_top(ly)
        elif h==3: push_left(lx)
        elif h==4: push_right(lx)
        elif h==5: push_left(lx); push_bottom(ly)
        elif h==6: push_bottom(ly)
        elif h==7: push_right(lx); push_bottom(ly)

    # ── 快捷键 ──────────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        key = event.key()
        ctrl = bool(event.modifiers() & Qt.ControlModifier)

        # ---- 确认当前框 ----
        if key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            self.confirm_pending()

        # ---- 旋转（大步 ±15° / 微调 ±1°）----
        elif key == Qt.Key_Z:
            self._rotate_by(-15.0)
        elif key == Qt.Key_V:
            self._rotate_by(+15.0)
        elif key == Qt.Key_X:
            self._rotate_by(-1.0)
        elif key == Qt.Key_C:
            self._rotate_by(+1.0)
        elif key == Qt.Key_R:
            self._set_angle(0.0)  # 重置旋转

        # ---- 复制框 ----
        elif key == Qt.Key_D:
            offset = (30, 30) if ctrl else (0, 0)
            self._duplicate_last(offset)

        # ---- 删除最后一个已确认框 ----
        elif key in (Qt.Key_Delete, Qt.Key_Backspace):
            if self._has_pending:
                self._has_pending = False
            elif self.rois:
                self.rois.pop()
            self.update()

        # ---- 放弃 pending ----
        elif key == Qt.Key_Escape:
            self._has_pending = False
            self.update()

        # ---- 平移（方向键，步长 2px）----
        elif key == Qt.Key_Left:
            self._move_pending(-2, 0)
        elif key == Qt.Key_Right:
            self._move_pending(+2, 0)
        elif key == Qt.Key_Up:
            self._move_pending(0, -2)
        elif key == Qt.Key_Down:
            self._move_pending(0, +2)

        # ---- 缩放（W/S 纵向, A/Q 横向，步长 2px）----
        elif key == Qt.Key_W:
            self._scale_pending(0, +2)
        elif key == Qt.Key_S:
            self._scale_pending(0, -2)
        elif key == Qt.Key_A:
            self._scale_pending(+2, 0)
        elif key == Qt.Key_Q:
            self._scale_pending(-2, 0)

        # ---- Tab：将最后一个已确认框取回为 pending 重新编辑 ----
        elif key == Qt.Key_Tab:
            if self.rois:
                cx,cy,w,h,a = self.rois.pop()
                self._cx,self._cy,self._w,self._h,self._angle = cx,cy,w,h,a
                self._has_pending = True
                self.rotation_changed.emit(self._angle)
            self.update()

        else:
            super().keyPressEvent(event)

    # ── 辅助操作 ────────────────────────────────────────────────────────
    def _rotate_by(self, delta):
        if self._has_pending:
            self._angle = (self._angle + delta) % 360
            if self._angle > 180:
                self._angle -= 360
            self.rotation_changed.emit(self._angle)
            self.update()

    def _set_angle(self, a):
        if self._has_pending:
            self._angle = a
            self.rotation_changed.emit(self._angle)
            self.update()

    def _move_pending(self, dx, dy):
        if self._has_pending:
            self._cx += dx; self._cy += dy
            self.update()

    def _scale_pending(self, dw, dh):
        if self._has_pending:
            self._w = max(10.0, self._w + dw)
            self._h = max(10.0, self._h + dh)
            self.update()

    def _duplicate_last(self, offset=(0, 0)):
        """复制最后一个已确认框（或当前 pending）作为新 pending"""
        src = None
        if self.rois:
            src = self.rois[-1]
        elif self._has_pending:
            src = (self._cx, self._cy, self._w, self._h, self._angle)

        if src:
            cx,cy,w,h,a = src
            # 先把当前 pending 确认掉（如果有）
            if self._has_pending:
                self.rois.append((self._cx,self._cy,self._w,self._h,self._angle))
            self._cx = cx + offset[0]
            self._cy = cy + offset[1]
            self._w, self._h, self._angle = w, h, a
            self._has_pending = True
            self.rotation_changed.emit(self._angle)
            self.update()

    # ── 外部接口 ────────────────────────────────────────────────────────
    def set_rotation(self, angle_deg):
        self._angle = float(angle_deg)
        self.update()

    def get_angle(self):
        return self._angle

    def confirm_pending(self):
        if self._has_pending and self._w > 5 and self._h > 5:
            self.rois.append((self._cx,self._cy,self._w,self._h,self._angle))
            self._has_pending = False
            self._angle = 0.0
            self.rotation_changed.emit(0.0)
            self.update()
            return True
        return False

    def clear_rois(self):
        self.rois.clear()
        self._has_pending = False
        self._angle = 0.0
        self.update()

    # ── 绘制 ────────────────────────────────────────────────────────────
    def _draw_box(self, painter, cx, cy, w, h, angle, color, lw=2):
        pen = QPen(color, lw, Qt.SolidLine)
        painter.setPen(pen); painter.setBrush(Qt.NoBrush)
        w2,h2=w/2,h/2
        corners=[(-w2,-h2),(w2,-h2),(w2,h2),(-w2,h2)]
        pts=[_rot_pt(cx+lx,cy+ly,cx,cy,angle) for lx,ly in corners]
        for i in range(4):
            painter.drawLine(pts[i],pts[(i+1)%4])

    def _draw_handles(self, painter):
        handles = self._world_handles()
        if not handles: return
        painter.setPen(QPen(QColor(255,200,60,160),1,Qt.DashLine))
        painter.drawLine(handles[1], handles[8])
        for h in handles[:8]:
            painter.setPen(QPen(QColor(255,210,0),1.5))
            painter.setBrush(QColor(255,255,255,230))
            painter.drawEllipse(h, HANDLE_R, HANDLE_R)
        rh=handles[8]
        painter.setPen(QPen(QColor(200,100,0),2))
        painter.setBrush(QColor(255,165,0,230))
        painter.drawEllipse(rh, ROTATE_R, ROTATE_R)
        painter.setPen(QPen(QColor(80,30,0),1.5))
        painter.drawArc(rh.x()-5,rh.y()-5,10,10,45*16,270*16)
        painter.setPen(QColor(255,230,80))
        t=handles[1]
        painter.drawText(t.x()+10,t.y()-6,
                         f"{int(round(self._w))} × {int(round(self._h))}  {self._angle:.1f}°")

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        for i,(cx,cy,w,h,angle) in enumerate(self.rois):
            self._draw_box(painter,cx,cy,w,h,angle,QColor(0,220,80),2)
            w2,h2=w/2,h/2
            tl=_rot_pt(cx-w2,cy-h2,cx,cy,angle)
            painter.setPen(QColor(80,200,100,200))
            painter.drawText(tl.x()+3,tl.y()-4,
                             f"#{i+1} {int(round(w))}×{int(round(h))} {angle:.1f}°")
        if self._state==self._ST_DRAW and self._has_pending:
            painter.setPen(QPen(Qt.red,1,Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(int(self._cx-self._w/2),int(self._cy-self._h/2),
                             int(self._w),int(self._h))
        if self._has_pending:
            self._draw_box(painter,self._cx,self._cy,
                           self._w,self._h,self._angle,QColor(255,220,0),2)
            self._draw_handles(painter)


class ROISelectorDialog(QDialog):
    def __init__(self, img_path, title="提取模板", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        layout = QVBoxLayout(self)

        cv_img = cv2.imread(img_path)
        self.cv_img = cv_img
        self.img_h,self.img_w = cv_img.shape[:2]
        rgb = cv2.cvtColor(cv_img,cv2.COLOR_BGR2RGB)
        qt_img = QImage(rgb.data,self.img_w,self.img_h,
                        3*self.img_w,QImage.Format_RGB888)

        self.image_label = RotatableROILabel()
        self.image_label.setPixmap(QPixmap.fromImage(qt_img))
        self.image_label.setFixedSize(self.img_w,self.img_h)
        self.image_label.rotation_changed.connect(self._on_rot_changed)
        layout.addWidget(self.image_label)

        # 旋转滑条行
        rot_row = QHBoxLayout()
        rot_row.addWidget(QLabel("精确旋转:"))
        self.rot_slider = QSlider(Qt.Horizontal)
        self.rot_slider.setRange(-180,180); self.rot_slider.setValue(0)
        self.rot_slider.setTickInterval(15)
        self.rot_val_lbl = QLabel("0.0°")
        self.rot_val_lbl.setMinimumWidth(52)
        self.rot_slider.valueChanged.connect(self._on_slider)

        self.btn_confirm = QPushButton("✚ 确认当前框  [Space]")
        self.btn_confirm.setStyleSheet(
            "background-color:#FF9800;color:white;font-weight:bold;padding:5px 14px;")
        self.btn_confirm.clicked.connect(self._confirm)

        rot_row.addWidget(self.rot_slider,3)
        rot_row.addWidget(self.rot_val_lbl)
        rot_row.addWidget(self.btn_confirm)
        layout.addLayout(rot_row)

        # 快捷键说明面板（折叠展示）
        from PyQt5.QtWidgets import QFrame
        kb_frame = QFrame()
        kb_frame.setStyleSheet(
            "QFrame{background:#1a1a2e;border-radius:6px;padding:4px;}"
            "QLabel{color:#8888cc;font-size:10px;}")
        kb_layout = QGridLayout(kb_frame)
        kb_layout.setSpacing(2)
        shortcuts = [
            ("Space/Enter","确认当前框"),  ("D","复制框(原位)"),   ("Ctrl+D","复制框(偏移)"),
            ("Z / V","大旋转 ±15°"),       ("X / C","微旋转 ±1°"), ("R","重置旋转"),
            ("方向键","平移框 2px"),        ("W/S","纵向缩放"),     ("A/Q","横向缩放"),
            ("Tab","取回最后确认框"),       ("Delete","删除最后框"),("Esc","放弃当前框"),
        ]
        for i,(k,v) in enumerate(shortcuts):
            col = (i % 3) * 2
            row = i // 3
            lbl_k = QLabel(f"[{k}]")
            lbl_k.setStyleSheet("color:#ffcc44;font-size:10px;font-weight:bold;")
            lbl_v = QLabel(v)
            kb_layout.addWidget(lbl_k, row, col)
            kb_layout.addWidget(lbl_v, row, col+1)
        layout.addWidget(kb_frame)

        hint = QLabel(
            "① 拖拽画框  ② 拖 8 白点调大小  ③ 拖橙色手柄/滑条旋转  "
            "④ Space 确认→绿  ⑤ D 复制框  ⑥ 点击图片区域后键盘快捷键生效"
        )
        hint.setStyleSheet("color:#888;font-size:11px;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        bot = QHBoxLayout()
        btn_clear = QPushButton("🗑 清除全部  [Delete]")
        btn_clear.clicked.connect(self.image_label.clear_rois)
        btn_ok = QPushButton("✅ 完成，继续下一张")
        btn_ok.setStyleSheet(
            "background-color:#4CAF50;color:white;font-weight:bold;padding:5px;")
        btn_ok.clicked.connect(self.accept)
        bot.addWidget(btn_clear); bot.addWidget(btn_ok)
        layout.addLayout(bot)

    def _on_rot_changed(self, angle):
        """接收来自 label 的旋转信号（键盘/手柄拖拽）→ 同步滑条"""
        v = int(round(angle))
        v = max(-180, min(180, v))
        self.rot_slider.blockSignals(True)
        self.rot_slider.setValue(v)
        self.rot_slider.blockSignals(False)
        self.rot_val_lbl.setText(f"{angle:.1f}°")

    def _on_slider(self, val):
        self.rot_val_lbl.setText(f"{val}.0°")
        self.image_label.set_rotation(float(val))

    def _confirm(self):
        ok = self.image_label.confirm_pending()
        if ok:
            self.rot_slider.blockSignals(True)
            self.rot_slider.setValue(0)
            self.rot_slider.blockSignals(False)
            self.rot_val_lbl.setText("0.0°")

    def get_rois_with_angle(self):
        return list(self.image_label.rois)


# ================= 核心算法函数 =================

def get_template_prompts(img_gray, templates_info, thresh):
    """模板匹配：十字形 5 变体采样，兼顾召回与速度。
    变体 = 3 尺度(原角度) + 2 旋转(原尺度)。
    跳过"缩放+旋转同时变化"的对角变体（实际场景罕见，性价比低）。
    """
    variants = (
        (0.85, 0.0),
        (1.0,  0.0),
        (1.15, 0.0),
        (1.0, -10.0),
        (1.0,  10.0),
    )
    all_boxes, all_scores = [], []
    img_h, img_w = img_gray.shape[:2]
    for template_gray, tw, th in templates_info:
        for scale, angle in variants:
            new_w = int(round(tw * scale))
            new_h = int(round(th * scale))
            if new_w < 8 or new_h < 8 or new_w >= img_w or new_h >= img_h:
                continue
            if abs(scale - 1.0) < 1e-6:
                tpl = template_gray
            else:
                tpl = cv2.resize(template_gray, (new_w, new_h))
            if abs(angle) > 0.01:
                M = cv2.getRotationMatrix2D((new_w/2.0, new_h/2.0), angle, 1.0)
                tpl = cv2.warpAffine(tpl, M, (new_w, new_h),
                                     borderMode=cv2.BORDER_REPLICATE)
            res = cv2.matchTemplate(img_gray, tpl, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= thresh)
            for pt in zip(*loc[::-1]):
                all_boxes.append([int(pt[0]), int(pt[1]), new_w, new_h])
                all_scores.append(float(res[pt[1], pt[0]]))
    return all_boxes, all_scores


def get_multi_bg_diff_prompts(img_gray, bg_images_blur):
    """
    背景差分引擎。
    背景图片本身不含目标物，所以差分出来的亮区就是潜在目标。
    同时反过来：背景区域出现的亮区可能是噪声/光照，多张背景取最小差分来抑制。
    """
    if not bg_images_blur:
        return [], []
    blur_img = cv2.GaussianBlur(img_gray, (5, 5), 0)
    min_diff = np.full(img_gray.shape, 255, dtype=np.uint8)
    for bg_blur in bg_images_blur:
        diff = cv2.absdiff(blur_img, bg_blur)
        min_diff = cv2.min(min_diff, diff)
    _, thresh = cv2.threshold(min_diff, 30, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes, scores = [], []
    img_h, img_w = img_gray.shape
    for cnt in contours:
        if cv2.contourArea(cnt) < 300:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        pad = 10
        x1, y1 = max(0, x - pad), max(0, y - pad)
        w_pad = min(img_w - x1, w + pad * 2)
        h_pad = min(img_h - y1, h + pad * 2)
        boxes.append([x1, y1, w_pad, h_pad])
        scores.append(1.0)
    return boxes, scores


def build_bg_exclusion_mask(img_gray, bg_images_blur):
    """
    构建背景排斥掩码：在背景上"不存在目标"的区域打标，
    用于后续屏蔽在这些区域出现的 SAM proposal。
    返回 uint8 掩码，255=背景一致区域（要排斥），0=可能有目标。
    """
    if not bg_images_blur:
        return None
    blur_img = cv2.GaussianBlur(img_gray, (5, 5), 0)
    max_sim = np.zeros(img_gray.shape, dtype=np.uint8)
    for bg_blur in bg_images_blur:
        diff = cv2.absdiff(blur_img, bg_blur)
        sim = 255 - diff
        max_sim = cv2.max(max_sim, sim)
    _, bg_mask = cv2.threshold(max_sim, 225, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    bg_mask = cv2.erode(bg_mask, kernel)
    return bg_mask


def mask_overlaps_bg(mask_bin, bg_exclusion_mask, overlap_thresh=0.85):
    """检查 mask 是否大部分落在背景排斥区域内"""
    if bg_exclusion_mask is None:
        return False
    mask_area = float(np.count_nonzero(mask_bin))
    if mask_area == 0:
        return True
    overlap = float(np.count_nonzero(cv2.bitwise_and(mask_bin, bg_exclusion_mask)))
    return (overlap / mask_area) > overlap_thresh


def merge_prompts(boxes_A, scores_A, boxes_B, scores_B):
    all_boxes = boxes_A + boxes_B
    all_scores = scores_A + scores_B
    if not all_boxes:
        return []
    indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, 0.5, 0.3)
    merged_prompts = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = all_boxes[i]
            merged_prompts.append([x, y, x + w, y + h])
    return merged_prompts


def _bbox_iou_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _bbox_area_xyxy(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _bbox_intersection_area_xyxy(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    return max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)


def normalize_workspace_roi(roi):
    """校验并规范化工作区域，格式为归一化的 (x1, y1, x2, y2)。"""
    try:
        valid_length = roi is not None and len(roi) == 4
    except TypeError:
        valid_length = False
    if not valid_length:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in roi]
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((float(np.clip(x1, 0.0, 1.0)),
                     float(np.clip(x2, 0.0, 1.0))))
    y1, y2 = sorted((float(np.clip(y1, 0.0, 1.0)),
                     float(np.clip(y2, 0.0, 1.0))))
    if x2 - x1 < 0.01 or y2 - y1 < 0.01:
        return None
    return x1, y1, x2, y2


def bbox_passes_workspace_roi(box_xyxy, img_w, img_h, roi,
                              min_overlap=0.5):
    """框中心须在 ROI 内，且至少一半框面积与 ROI 相交。"""
    roi = normalize_workspace_roi(roi)
    if roi is None:
        return True
    if img_w <= 0 or img_h <= 0 or box_xyxy is None or len(box_xyxy) != 4:
        return False

    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    box_area = _bbox_area_xyxy(box)
    if box_area <= 0:
        return False

    rx1, ry1, rx2, ry2 = roi
    roi_pixels = [rx1 * img_w, ry1 * img_h, rx2 * img_w, ry2 * img_h]
    center_x, center_y = _bbox_center(box)
    center_inside = (
        roi_pixels[0] <= center_x <= roi_pixels[2]
        and roi_pixels[1] <= center_y <= roi_pixels[3]
    )
    overlap = _bbox_intersection_area_xyxy(box, roi_pixels) / box_area
    return center_inside and overlap >= float(min_overlap)


def workspace_roi_text(roi):
    roi = normalize_workspace_roi(roi)
    if roi is None:
        return "未设置"
    x1, y1, x2, y2 = roi
    return f"x={x1:.3f}~{x2:.3f}, y={y1:.3f}~{y2:.3f}"


def draw_workspace_roi(vis_img, roi, color=(0, 0, 255), thickness=2):
    """在诊断可视化中画出工作区域；不会改动训练图片。"""
    roi = normalize_workspace_roi(roi)
    if roi is None or vis_img is None:
        return
    img_h, img_w = vis_img.shape[:2]
    x1, y1, x2, y2 = roi
    cv2.rectangle(
        vis_img,
        (int(round(x1 * img_w)), int(round(y1 * img_h))),
        (int(round(x2 * img_w)), int(round(y2 * img_h))),
        color,
        thickness,
    )


def _bbox_containment_ratio(inner_box, outer_box):
    inner_area = _bbox_area_xyxy(inner_box)
    if inner_area <= 0:
        return 0.0
    inter = _bbox_intersection_area_xyxy(inner_box, outer_box)
    return inter / inner_area


def _bbox_center(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def _center_distance_ratio(box_a, box_b):
    ax, ay = _bbox_center(box_a)
    bx, by = _bbox_center(box_b)
    dist = math.hypot(ax - bx, ay - by)
    diag_a = math.hypot(max(1, box_a[2] - box_a[0]), max(1, box_a[3] - box_a[1]))
    diag_b = math.hypot(max(1, box_b[2] - box_b[0]), max(1, box_b[3] - box_b[1]))
    norm = max(1.0, min(diag_a, diag_b))
    return dist / norm


def _mask_iou(mask_a, mask_b):
    a = mask_a > 0
    b = mask_b > 0
    inter = np.count_nonzero(a & b)
    if inter == 0:
        return 0.0
    union = np.count_nonzero(a | b)
    return inter / union if union > 0 else 0.0


def _mask_containment_ratio(mask_small, mask_big):
    small = mask_small > 0
    big = mask_big > 0
    small_area = np.count_nonzero(small)
    if small_area == 0:
        return 0.0
    inter = np.count_nonzero(small & big)
    return inter / small_area


def _order_quad_points(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = np.argmin(pts[:, 0] + pts[:, 1])
    return np.roll(pts, -start, axis=0)


def mask_to_obb(mask, img_w, img_h):
    """将二值 mask 转为归一化 OBB 四点坐标"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return None
    rect = cv2.minAreaRect(cnt)
    (cx, cy), (w, h), angle = rect
    if w == 0 or h == 0:
        return None
    box = cv2.boxPoints(rect)
    box = _order_quad_points(box)
    box_norm = box.copy()
    box_norm[:, 0] = np.clip(box_norm[:, 0] / img_w, 0.0, 1.0)
    box_norm[:, 1] = np.clip(box_norm[:, 1] / img_h, 0.0, 1.0)
    return box_norm.flatten().tolist()


def mask_to_hbb(mask, img_w, img_h):
    """将二值 mask 转为标准 YOLO HBB: [cx, cy, w, h]（归一化）。"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return None
    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 0 or h <= 0:
        return None
    cx = (x + w / 2.0) / img_w
    cy = (y + h / 2.0) / img_h
    return [
        float(np.clip(cx, 0.0, 1.0)),
        float(np.clip(cy, 0.0, 1.0)),
        float(np.clip(w / img_w, 0.0, 1.0)),
        float(np.clip(h / img_h, 0.0, 1.0)),
    ]


def xyxy_to_hbb(box_xyxy, img_w, img_h):
    """将像素 xyxy 框转为 YOLO HBB 归一化 cx,cy,w,h。"""
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    x1 = np.clip(x1, 0.0, float(img_w))
    y1 = np.clip(y1, 0.0, float(img_h))
    x2 = np.clip(x2, 0.0, float(img_w))
    y2 = np.clip(y2, 0.0, float(img_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return [
        float(np.clip((x1 + x2) / 2.0 / img_w, 0.0, 1.0)),
        float(np.clip((y1 + y2) / 2.0 / img_h, 0.0, 1.0)),
        float(np.clip((x2 - x1) / img_w, 0.0, 1.0)),
        float(np.clip((y2 - y1) / img_h, 0.0, 1.0)),
    ]


def draw_yolo_label(vis_img, label, annotation_format, color=(0, 255, 0), thickness=3):
    """在图片上绘制归一化的 YOLO OBB 或 HBB 标签。"""
    img_h, img_w = vis_img.shape[:2]
    if annotation_format == 'hbb':
        cx, cy, w, h = label
        x1 = int(round((cx - w / 2.0) * img_w))
        y1 = int(round((cy - h / 2.0) * img_h))
        x2 = int(round((cx + w / 2.0) * img_w))
        y2 = int(round((cy + h / 2.0) * img_h))
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, thickness)
        return

    pts_pixel = (np.array(label).reshape(4, 2) * [img_w, img_h]).astype(np.int32)
    cv2.polylines(vis_img, [pts_pixel], True, color, thickness)


def assign_left_right_classes(candidates, img_w):
    """按画面左右位置给手候选分配类别：0=left，1=right。"""
    if not candidates:
        return
    centers = []
    for idx, cand in enumerate(candidates):
        x1, _, x2, _ = cand['bbox_xyxy']
        centers.append((idx, (float(x1) + float(x2)) / 2.0))

    if len(centers) == 1:
        # 单手时仍按它位于画面左/右侧决定类别。
        candidates[centers[0][0]]['class_id'] = int(centers[0][1] >= img_w / 2.0)
        return

    # 两手是主要场景：x 更小的一只为 left(0)，x 更大的一只为 right(1)。
    ordered = sorted(centers, key=lambda item: item[1])
    if len(ordered) == 2:
        candidates[ordered[0][0]]['class_id'] = 0
        candidates[ordered[1][0]]['class_id'] = 1
        return

    # 多于两框时，以左右两端中心的中点为分界，避免按候选生成顺序赋类。
    split_x = (ordered[0][1] + ordered[-1][1]) / 2.0
    for idx, center_x in centers:
        candidates[idx]['class_id'] = int(center_x >= split_x)


def make_refine_points_from_box(x1, y1, x2, y2):
    """从候选框中生成 1 个正点 + 4 个负点，用于提示分割精修。"""
    cx = int(round((x1 + x2) / 2.0))
    cy = int(round((y1 + y2) / 2.0))
    w = max(4, x2 - x1)
    h = max(4, y2 - y1)
    margin_x = max(3, int(round(w * 0.12)))
    margin_y = max(3, int(round(h * 0.12)))

    points = np.array([
        [cx, cy],
        [x1 + margin_x, y1 + margin_y],
        [x2 - margin_x, y1 + margin_y],
        [x1 + margin_x, y2 - margin_y],
        [x2 - margin_x, y2 - margin_y],
    ], dtype=np.float32)
    labels = np.array([1, 0, 0, 0, 0], dtype=np.int32)
    return points, labels


def refine_masks_with_points(sam_model, img_path, prompts_xyxy, device=None):
    """
    对 bbox prompt 再追加点提示精修。
    若当前 ultralytics 版本不支持 points/labels，则回退为 None。
    """
    all_masks = []
    for x1, y1, x2, y2 in prompts_xyxy:
        points, labels = make_refine_points_from_box(x1, y1, x2, y2)
        try:
            predict_args = {
                'bboxes': [[x1, y1, x2, y2]],
                'points': points.tolist(),
                'labels': labels.tolist(),
                'verbose': False,
            }
            if device:
                predict_args['device'] = device
                predict_args['half'] = device.startswith('cuda')
            results = sam_model(img_path, **predict_args)
        except TypeError:
            return None
        except RuntimeError as exc:
            if is_cuda_oom_error(exc):
                clear_torch_cache()
                continue
            continue
        except Exception:
            continue
        if not results or results[0].masks is None:
            continue
        masks = results[0].masks.data.cpu().numpy()
        if masks.shape[0] > 0:
            all_masks.append(masks[0])
        # 成功推理后不做全量 gc.collect()；频繁 GC 会让 CPU 成为瓶颈。
    return all_masks


def deduplicate_mask_candidates(candidates, mask_iou_thresh=0.7, box_iou_thresh=0.85):
    """对候选 mask 做去重，优先保留相似度更高、面积更大的结果。"""
    if not candidates:
        return []
    ordered = sorted(
        candidates,
        key=lambda x: (x.get('sim', 0.0), x.get('mask_area', 0.0)),
        reverse=True,
    )
    kept = []
    for cand in ordered:
        duplicate = False
        for prev in kept:
            if _bbox_iou_xyxy(cand['bbox_xyxy'], prev['bbox_xyxy']) >= box_iou_thresh:
                duplicate = True
                break
            if _mask_iou(cand['mask'], prev['mask']) >= mask_iou_thresh:
                duplicate = True
                break
        if not duplicate:
            kept.append(cand)
    return kept


def suppress_contained_candidates(
    candidates,
    box_contain_thresh=0.76,
    mask_contain_thresh=0.72,
    center_ratio_thresh=0.72,
    area_ratio_thresh=0.86,
    sim_tolerance=0.03,
):
    """
    清理“同一物体的大框套小残框”。
    规则：
      1. 小框大部分被大框包含
      2. 小 mask 大部分被大 mask 覆盖
      3. 两者中心接近
      4. 小框面积明显更小
    满足后优先保留面积更大且相似度不更差的候选。
    """
    if len(candidates) < 2:
        return candidates

    keep_flags = [True] * len(candidates)
    for i, cand_i in enumerate(candidates):
        if not keep_flags[i]:
            continue
        area_i = max(1.0, float(cand_i.get('mask_area', 0.0)))
        sim_i = float(cand_i.get('sim', 0.0))
        for j, cand_j in enumerate(candidates):
            if i == j or not keep_flags[j]:
                continue
            area_j = max(1.0, float(cand_j.get('mask_area', 0.0)))
            if area_i >= area_j:
                big, small = cand_i, cand_j
                big_idx, small_idx = i, j
                big_area, small_area = area_i, area_j
                big_sim, small_sim = sim_i, float(cand_j.get('sim', 0.0))
            else:
                continue

            area_ratio = small_area / max(big_area, 1.0)
            if area_ratio > area_ratio_thresh:
                continue

            box_contain = _bbox_containment_ratio(small['bbox_xyxy'], big['bbox_xyxy'])
            if box_contain < box_contain_thresh:
                continue

            mask_contain = _mask_containment_ratio(small['mask'], big['mask'])
            if mask_contain < mask_contain_thresh:
                continue

            center_ratio = _center_distance_ratio(small['bbox_xyxy'], big['bbox_xyxy'])
            if center_ratio > center_ratio_thresh:
                continue

            if big_sim + sim_tolerance < small_sim:
                continue

            keep_flags[small_idx] = False

    return [cand for cand, keep in zip(candidates, keep_flags) if keep]


def suppress_neighbor_duplicates(
    candidates,
    center_ratio_thresh=0.38,
    area_ratio_min=0.72,
    area_ratio_max=1.38,
    box_iou_thresh=0.38,
    mask_iou_thresh=0.33,
    sim_tolerance=0.03,
):
    """
    清理“同一物体的近邻双框”。
    特征是：
      1. 两框中心很近
      2. 面积接近
      3. bbox 或 mask 有中高重叠
    满足后保留相似度更高、面积更大的那个。
    """
    if len(candidates) < 2:
        return candidates

    ordered = sorted(
        candidates,
        key=lambda x: (float(x.get('sim', 0.0)), float(x.get('mask_area', 0.0))),
        reverse=True,
    )
    kept = []
    for cand in ordered:
        cand_area = max(1.0, float(cand.get('mask_area', 0.0)))
        cand_sim = float(cand.get('sim', 0.0))
        duplicate = False
        for prev in kept:
            prev_area = max(1.0, float(prev.get('mask_area', 0.0)))
            prev_sim = float(prev.get('sim', 0.0))

            center_ratio = _center_distance_ratio(cand['bbox_xyxy'], prev['bbox_xyxy'])
            if center_ratio > center_ratio_thresh:
                continue

            area_ratio = cand_area / prev_area
            if area_ratio < area_ratio_min or area_ratio > area_ratio_max:
                continue

            box_iou = _bbox_iou_xyxy(cand['bbox_xyxy'], prev['bbox_xyxy'])
            mask_iou = _mask_iou(cand['mask'], prev['mask'])
            if box_iou < box_iou_thresh and mask_iou < mask_iou_thresh:
                continue

            if cand_sim + sim_tolerance < prev_sim:
                duplicate = True
                break
            if abs(cand_sim - prev_sim) <= sim_tolerance and cand_area <= prev_area:
                duplicate = True
                break

        if not duplicate:
            kept.append(cand)

    return kept


def is_cuda_oom_error(exc):
    text = str(exc).lower()
    return "out of memory" in text and "cuda" in text


def clear_torch_cache():
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
        except Exception:
            pass


def run_sam_in_batches(sam_model, img_path, prompts_xyxy, batch_size, device=None, log_fn=None):
    """
    分批执行 SAM bbox prompt，避免一次性送入过多 proposal 导致 CUDA OOM。
    返回 list[np.ndarray]，每个元素是一张 mask。
    """
    masks = []
    if not prompts_xyxy:
        return masks

    current_batch = max(1, int(batch_size))
    index = 0
    while index < len(prompts_xyxy):
        batch = prompts_xyxy[index:index + current_batch]
        try:
            predict_args = {'bboxes': batch, 'verbose': False}
            if device:
                predict_args['device'] = device
                predict_args['half'] = device.startswith('cuda')
            results = sam_model(img_path, **predict_args)
            if results and results[0].masks is not None:
                batch_masks = results[0].masks.data.cpu().numpy()
                for m in batch_masks:
                    masks.append(m)
            index += len(batch)
            # 成功推理后不做全量 gc.collect()；仅在 CUDA OOM 时清理缓存。
        except RuntimeError as exc:
            if not is_cuda_oom_error(exc):
                raise
            # 只有 CUDA OOM 时才清理缓存，避免正常路径频繁触发 GC。
            clear_torch_cache()
            if current_batch == 1:
                if log_fn:
                    log_fn(f"⚠️ 单框 SAM 仍然显存不足，跳过该候选: {batch[0]}")
                index += 1
                continue
            next_batch = max(1, current_batch // 2)
            if log_fn:
                log_fn(f"⚠️ SAM 显存不足，批大小从 {current_batch} 降到 {next_batch} 后重试")
            current_batch = next_batch
    return masks



# ================= 后台处理线程 =================

class AnnotationThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal()
    raw_records_signal = pyqtSignal(object)  # 发送原始结果缓存供重新过滤

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_running = True

    def run(self):
        try:
            # 显式选择推理设备，避免 Ultralytics 在环境异常时静默回落到 CPU。
            device = "cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu"
            if device.startswith("cuda"):
                gpu_name = torch.cuda.get_device_name(0)
                torch.backends.cudnn.benchmark = True
                self.log_signal.emit(f"🖥️ 推理设备: {device} ({gpu_name})")
            else:
                self.log_signal.emit("⚠️ 推理设备: CPU（未检测到可用 CUDA）")
            text_prompt = str(self.config.get('text_prompt', '')).strip()
            use_sam3_text = bool(text_prompt and os.path.isfile(SAM3_MODEL_PATH))
            if text_prompt and not use_sam3_text:
                raise FileNotFoundError(
                    f"已填写文本提示，但找不到 SAM3 权重: {SAM3_MODEL_PATH}\n"
                    "请将 sam3.pt 放入项目 models/，或设置 SAM3_MODEL_PATH。"
                )
            self.log_signal.emit(
                "🚀 [后台] 正在加载 SAM3 文本模型..." if use_sam3_text
                else "🚀 [后台] 正在加载 MobileSAM 模型...")
            model_path = SAM3_MODEL_PATH if use_sam3_text else str(MOBILE_SAM_MODEL)
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"找不到分割模型权重: {model_path}")
            # SAM3 文本模式由 SAM3TextSegmenter 独占模型实例，避免同时加载两份 3.3GB 权重。
            sam_model = None if use_sam3_text else SAM(model_path)
            # Ultralytics 通常会在第一次 predict 时自动选设备；这里提前迁移模型，
            # 防止模型仍停留在 CPU，导致 GPU 只偶尔有很低的利用率。
            if sam_model is not None:
                try:
                    sam_model.to(device)
                except Exception:
                    pass
                try:
                    model_device = next(sam_model.model.parameters()).device
                    self.log_signal.emit(f"🔎 SAM 模型参数设备: {model_device}")
                except Exception:
                    pass
            self.log_signal.emit("✅ 模型加载完毕！")
            use_point_refine = bool(self.config.get('use_point_refine', True))
            sam_batch_size = max(1, int(self.config.get('sam_batch_size', 8)))
            annotation_format = str(self.config.get('annotation_format', 'obb')).lower()
            if annotation_format not in ('obb', 'hbb'):
                annotation_format = 'obb'
            format_name = "YOLO-HBB 普通框" if annotation_format == 'hbb' else "YOLO-OBB 旋转框"
            self.log_signal.emit(f"🏷️ 当前标签格式: {format_name}")
            workspace_roi = normalize_workspace_roi(
                self.config.get('workspace_roi'))
            if workspace_roi is not None:
                self.log_signal.emit(
                    f"🟥 只保留灰色台面工作区域内的目标: "
                    f"{workspace_roi_text(workspace_roi)}")
            else:
                self.log_signal.emit("⬜ 工作区域过滤未启用，将检测整张图片。")
            if use_point_refine:
                self.log_signal.emit("🎯 已启用 SAM 点提示精修模式（框 + 正负点提示）。")
            self.log_signal.emit(f"🧱 SAM 分批推理已启用 (batch={sam_batch_size}, device={device})")
            self.log_signal.emit("🧹 已启用重复去重（包含型小残框 + 近邻双框）")

            # ---- 可选：文本提示检测器（文字找框 → SAM 分割） ----
            text_detector = None
            sam3_segmenter = None
            if use_sam3_text:
                try:
                    sam3_segmenter = SAM3TextSegmenter(
                        model_path=SAM3_MODEL_PATH,
                        device=device,
                        conf=self.config.get('text_box_threshold', 0.25),
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"SAM3 加载失败: {exc}。请在 d415 环境安装 timm 和 Ultralytics CLIP 依赖。"
                    ) from exc
                self.log_signal.emit(
                    f"✅ SAM3 文本模型已加载 ({device})，提示词: {text_prompt!r}")
            elif text_prompt:
                self.log_signal.emit(f"🔤 正在加载文本定位模型，提示词: {text_prompt!r}")
                text_detector = GroundingDINOTextDetector()
                self.log_signal.emit(
                    f"✅ GroundingDINO 已加载 ({text_detector.provider})，"
                    "将跳过模板/背景候选生成。")

            # ---- 加载背景图 ----
            bg_images_blur = []
            bg_images_bgr_patches = []
            bg_dir = self.config['bg_dir']
            if bg_dir and os.path.exists(bg_dir):
                bg_files = [f for f in os.listdir(bg_dir)
                            if f.lower().endswith(('.png', '.jpg'))]
                for f in bg_files:
                    bg = cv2.imread(os.path.join(bg_dir, f))
                    if bg is not None:
                        bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
                        bg_images_blur.append(cv2.GaussianBlur(bg_gray, (5, 5), 0))
                        # 采样若干 patch 作为背景特征
                        bh, bw = bg.shape[:2]
                        if bw >= 32 and bh >= 32:
                            for _ in range(10):
                                rx = random.randint(0, bw - 32)
                                ry = random.randint(0, bh - 32)
                                bg_images_bgr_patches.append(bg[ry:ry + 32, rx:rx + 32])
                        else:
                            self.log_signal.emit(
                                f"⚠️ 背景图尺寸过小，跳过背景特征采样: {f} ({bw}x{bh})")
                self.log_signal.emit(
                    f"✅ 成功加载 {len(bg_images_blur)} 张背景图！(双引擎 + 背景排斥模式)")
            else:
                self.log_signal.emit("⚠️ 未选择背景目录，跳过引擎B和背景排斥。")

            # ---- 初始化特征校验器 ----
            verifier = FeatureVerifier(
                sim_threshold=self.config['sim_threshold']
            )
            self.log_signal.emit(f"🧠 特征校验器(HOG+颜色)已启动 (threshold={self.config['sim_threshold']:.2f})")

            # 注入模板特征
            for patch in self.config['template_patches']:
                verifier.add_template(patch)
            self.log_signal.emit(f"  → 已注入 {len(self.config['template_patches'])} 个模板特征")

            # 注入背景特征（排斥）
            for patch in bg_images_bgr_patches:
                verifier.add_background(patch)
            if bg_images_bgr_patches:
                self.log_signal.emit(f"  → 已注入 {len(bg_images_bgr_patches)} 个背景排斥特征")

            # ---- 准备输出目录 ----
            out_dir = self.config['out_dir']
            os.makedirs(out_dir, exist_ok=True)
            for d in ["images", "labels", "visualizations"]:
                target_path = os.path.join(out_dir, d)
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
            for d in ["images/train", "images/val", "labels/train", "labels/val", "visualizations"]:
                os.makedirs(os.path.join(out_dir, d), exist_ok=True)

            # ---- 遍历图片 ----
            input_dir = self.config['input_dir']
            all_images = sorted([f for f in os.listdir(input_dir)
                                  if f.lower().endswith(('.png', '.jpg'))])
            total_imgs = len(all_images)
            if total_imgs == 0:
                self.log_signal.emit("❌ 输入目录为空！")
                return

            # 🔧 自动校准阈值：负样本仅从 bg_dir 采样。
            #    从输入图随机采样会混入目标物，把 neg_max 推高 → 阈值被错误拉高 → 漏标。
            neg_samples_for_calib = []
            if bg_dir and os.path.exists(bg_dir):
                bg_files_neg = [f for f in os.listdir(bg_dir)
                                if f.lower().endswith(('.png', '.jpg'))]
                random.shuffle(bg_files_neg)
                for cname in bg_files_neg[:5]:
                    cimg = cv2.imread(os.path.join(bg_dir, cname))
                    if cimg is not None:
                        ch, cw = cimg.shape[:2]
                        if cw < 65 or ch < 65:
                            continue
                        for _ in range(8):
                            rx = random.randint(0, cw - 65)
                            ry = random.randint(0, ch - 65)
                            neg_samples_for_calib.append(cimg[ry:ry+64, rx:rx+64])
            verifier.calibrate_threshold(
                self.config['template_patches'],
                neg_patches=neg_samples_for_calib
            )
            self.log_signal.emit(
                f"  → 自动校准阈值: {verifier.threshold:.3f}  "
                f"(负样本仅用 bg_dir，正负中间×85%，上限=pos_min×0.9)"
            )
            # 仅当用户把 UI 滑条调离默认值 0.45 时才手动覆盖
            ui_thresh = self.config['sim_threshold']
            if abs(ui_thresh - 0.45) > 0.001:
                verifier.threshold = ui_thresh
                self.log_signal.emit(f"  → UI 手动覆盖阈值: {ui_thresh:.3f}")

            dataset_records = []
            stats = {
                "total": 0, "pass": 0, "rej_roi": 0,
                "rej_bg": 0, "rej_feat": 0,
            }
            sim_samples = []        # 所有相似度（用于最终诊断）
            rejected_vis = []       # 被拦截样本的可视化（最多保存20个）
            stage_stats = {
                'text_boxes': 0, 'sam_masks': 0, 'mask_contours': 0,
                'valid_labels': 0, 'hbb_direct': 0,
            }
            # 原始结果缓存：每个候选区域的(img_path,img,mask,patch,sim,img_w,img_h,img_name)
            raw_candidates = []

            self.log_signal.emit(f"\n🔥 开始全自动标注 (共 {total_imgs} 张)...")

            for idx, img_name in enumerate(all_images):
                if not self.is_running:
                    break

                img_path = os.path.join(input_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    self.log_signal.emit(f"⚠️ 跳过无法读取的图片: {img_name}")
                    self.progress_signal.emit(idx + 1, total_imgs)
                    continue
                img_h, img_w = img.shape[:2]
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # 构建背景排斥掩码
                bg_excl_mask = build_bg_exclusion_mask(img_gray, bg_images_blur)

                if sam3_segmenter is not None:
                    sam3_masks, sam3_boxes, sam3_scores = sam3_segmenter.detect(
                        img, text_prompt)
                    stats["total"] += len(sam3_boxes)
                    stage_stats['text_boxes'] += len(sam3_boxes)
                    if workspace_roi is not None:
                        keep_indices = [
                            i for i, box in enumerate(sam3_boxes)
                            if bbox_passes_workspace_roi(
                                box, img_w, img_h, workspace_roi)
                        ]
                        stats["rej_roi"] += len(sam3_boxes) - len(keep_indices)
                        score_values = sam3_scores or [1.0] * len(sam3_boxes)
                        sam3_masks = [
                            sam3_masks[i] for i in keep_indices
                            if i < len(sam3_masks)
                        ]
                        sam3_boxes = [sam3_boxes[i] for i in keep_indices]
                        sam3_scores = [
                            score_values[i] for i in keep_indices
                            if i < len(score_values)
                        ]
                    merged_sam_prompts = [
                        [int(round(v)) for v in box] for box in sam3_boxes]
                    text_scores = sam3_scores or [1.0] * len(merged_sam_prompts)
                elif text_detector is not None:
                    text_boxes, text_scores = text_detector.detect(
                        img, text_prompt,
                        box_threshold=self.config.get('text_box_threshold', 0.25))
                    stage_stats['text_boxes'] += len(text_boxes)
                    merged_sam_prompts = merge_prompts(
                        text_boxes, text_scores, [], [])
                else:
                    prompts_A, scores_A = get_template_prompts(
                        img_gray, self.config['templates'], self.config['match_thresh'])
                    prompts_B, scores_B = get_multi_bg_diff_prompts(img_gray, bg_images_blur)
                    merged_sam_prompts = merge_prompts(
                        prompts_A, scores_A, prompts_B, scores_B)

                if sam3_segmenter is None:
                    stats["total"] += len(merged_sam_prompts)
                    if workspace_roi is not None:
                        prompts_before_roi = len(merged_sam_prompts)
                        merged_sam_prompts = [
                            box for box in merged_sam_prompts
                            if bbox_passes_workspace_roi(
                                box, img_w, img_h, workspace_roi)
                        ]
                        stats["rej_roi"] += (
                            prompts_before_roi - len(merged_sam_prompts))

                if merged_sam_prompts:
                    image_labels = []
                    img_candidates = []
                    masks_data = []
                    # HBB 文本模式直接使用 GroundingDINO 框，不再依赖 SAM mask。
                    # 这样即使 MobileSAM 没返回 mask，也能稳定输出 hand 框。
                    if sam3_segmenter is not None and sam3_masks:
                        for mask, prompt, sim in zip(
                                sam3_masks, merged_sam_prompts, text_scores):
                            mask = cv2.resize(mask, (img_w, img_h),
                                              interpolation=cv2.INTER_NEAREST) * 255
                            if annotation_format == 'hbb':
                                yolo_label = mask_to_hbb(mask, img_w, img_h)
                            else:
                                yolo_label = mask_to_obb(mask, img_w, img_h)
                            if yolo_label is None:
                                continue
                            x1, y1, x2, y2 = prompt
                            cnt_area = max(1.0, float((x2 - x1) * (y2 - y1)))
                            img_candidates.append({
                                'img_name': img_name, 'img_path': img_path,
                                'label': yolo_label, 'sim': float(sim),
                                'img_w': img_w, 'img_h': img_h,
                                'mask': (mask > 0).astype(np.uint8),
                                'mask_area': cnt_area,
                                'bbox_xyxy': [x1, y1, x2, y2],
                            })
                            stage_stats['sam_masks'] += 1
                    elif text_detector is not None and annotation_format == 'hbb':
                        for prompt, sim in zip(merged_sam_prompts,
                                               [1.0] * len(merged_sam_prompts)):
                            yolo_label = xyxy_to_hbb(prompt, img_w, img_h)
                            if yolo_label is None:
                                continue
                            x1, y1, x2, y2 = prompt
                            img_candidates.append({
                                'img_name': img_name,
                                'img_path': img_path,
                                'label': yolo_label,
                                'sim': sim,
                                'img_w': img_w,
                                'img_h': img_h,
                                'mask': np.zeros((img_h, img_w), dtype=np.uint8),
                                'mask_area': float(max(1, (x2 - x1) * (y2 - y1))),
                                'bbox_xyxy': [x1, y1, x2, y2],
                            })
                            stage_stats['hbb_direct'] += 1
                    else:
                        masks_data = run_sam_in_batches(
                            sam_model,
                            img_path,
                            merged_sam_prompts,
                            batch_size=sam_batch_size,
                            device=device,
                            log_fn=self.log_signal.emit,
                        )

                    if (not img_candidates) and sam3_segmenter is None and use_point_refine and merged_sam_prompts:
                        refined_masks = refine_masks_with_points(
                            sam_model, img_path, merged_sam_prompts, device=device)
                        if refined_masks is None:
                            use_point_refine = False
                            self.log_signal.emit("⚠️ 当前 ultralytics 版本不支持 points/labels，已自动回退为框提示。")
                        elif refined_masks:
                            masks_data = refined_masks

                    if not img_candidates and masks_data:
                        stage_stats['sam_masks'] += len(masks_data)
                        for i in range(len(masks_data)):
                            mask = cv2.resize(
                                masks_data[i].astype(np.uint8), (img_w, img_h),
                                interpolation=cv2.INTER_NEAREST) * 255

                            if mask_overlaps_bg(mask, bg_excl_mask):
                                stats["rej_bg"] += 1
                                continue

                            cnts_for_bbox, _ = cv2.findContours(
                                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            if not cnts_for_bbox:
                                continue
                            stage_stats['mask_contours'] += 1
                            main_cnt = max(cnts_for_bbox, key=cv2.contourArea)
                            mask_area = float(cv2.contourArea(main_cnt))
                            if mask_area < 80:
                                continue
                            x_m, y_m, w_m, h_m = cv2.boundingRect(main_cnt)
                            pad_x = max(4, int(w_m * 0.1))
                            pad_y = max(4, int(h_m * 0.1))
                            x1p = max(0, x_m - pad_x)
                            y1p = max(0, y_m - pad_y)
                            x2p = min(img_w, x_m + w_m + pad_x)
                            y2p = min(img_h, y_m + h_m + pad_y)
                            patch = img[y1p:y2p, x1p:x2p]

                            valid = True
                            sim = 1.0
                            if patch.size > 0:
                                valid, sim = verifier.verify(patch)
                                sim_samples.append(round(sim, 4))
                                if not valid:
                                    stats["rej_feat"] += 1
                                    if len(rejected_vis) < 20:
                                        rej_thumb = cv2.resize(patch, (80, 80))
                                        cv2.putText(rej_thumb, f"{sim:.2f}",
                                                    (2, 14), cv2.FONT_HERSHEY_SIMPLEX,
                                                    0.45, (0, 80, 255), 1)
                                        rejected_vis.append(rej_thumb)
                                    continue

                            if annotation_format == 'hbb':
                                yolo_label = mask_to_hbb(mask, img_w, img_h)
                            else:
                                yolo_label = mask_to_obb(mask, img_w, img_h)
                            if not yolo_label:
                                continue

                            img_candidates.append({
                                'img_name': img_name,
                                'img_path': img_path,
                                'label': yolo_label,
                                'sim': sim,
                                'img_w': img_w,
                                'img_h': img_h,
                                'mask': (mask > 0).astype(np.uint8),
                                'mask_area': mask_area,
                                'bbox_xyxy': [x_m, y_m, x_m + w_m, y_m + h_m],
                            })
                            stage_stats['valid_labels'] += 1

                    if workspace_roi is not None:
                        candidates_before_roi = len(img_candidates)
                        img_candidates = [
                            cand for cand in img_candidates
                            if bbox_passes_workspace_roi(
                                cand.get('bbox_xyxy'), img_w, img_h,
                                workspace_roi)
                        ]
                        stats["rej_roi"] += (
                            candidates_before_roi - len(img_candidates))

                    img_candidates = deduplicate_mask_candidates(img_candidates)
                    img_candidates = suppress_contained_candidates(img_candidates)
                    img_candidates = suppress_neighbor_duplicates(img_candidates)
                    if self.config.get('left_right_mode', False):
                        assign_left_right_classes(img_candidates, img_w)
                    else:
                        for cand in img_candidates:
                            cand['class_id'] = 0
                    for cand in img_candidates:
                        raw_candidates.append({
                            'img_name': cand['img_name'],
                            'img_path': cand['img_path'],
                            'label': cand['label'],
                            'sim': cand['sim'],
                            'img_w': cand['img_w'],
                            'img_h': cand['img_h'],
                            'class_id': cand['class_id'],
                            'left_right_mode': self.config.get('left_right_mode', False),
                        })
                        image_labels.append((cand['class_id'], cand['label']))
                        stats["pass"] += 1

                    if image_labels:
                        dataset_records.append({
                            'path': img_path, 'name': img_name,
                            'labels': image_labels, 'img': img,
                            'img_w': img_w, 'img_h': img_h
                        })

                self.progress_signal.emit(idx + 1, total_imgs)

            if self.is_running:
                self.log_signal.emit("\n📦 正在划分并导出数据集格式...")
                random.shuffle(dataset_records)
                train_count = int(len(dataset_records) * self.config['split_ratio'])

                for idx, record in enumerate(dataset_records):
                    subset = "train" if idx < train_count else "val"
                    shutil.copy(record['path'],
                                os.path.join(out_dir, f"images/{subset}", record['name']))
                    txt_path = os.path.join(
                        out_dir, f"labels/{subset}",
                        f"{os.path.splitext(record['name'])[0]}.txt")
                    with open(txt_path, 'w') as f:
                        for class_id, label in record['labels']:
                            f.write(str(class_id) + " " +
                                    " ".join([f"{p:.6f}" for p in label]) + "\n")

                    if idx < self.config['num_vis']:
                        vis_img = record['img'].copy()
                        draw_workspace_roi(vis_img, workspace_roi)
                        for _, label in record['labels']:
                            draw_yolo_label(vis_img, label, annotation_format)
                        cv2.imwrite(
                            os.path.join(out_dir, f"visualizations/vis_{record['name']}"),
                            vis_img)

                class_names = ["left", "right"] if self.config.get(
                    'left_right_mode', False) else [self.config.get('class_name', 'object')]
                write_project_dataset_yaml(out_dir, class_names)
                yaml_path = os.path.join(out_dir, "dataset.yaml")

                # 导出被拦截样本拼图（方便人工判断阈值是否合适）
                if rejected_vis:
                    cols = min(10, len(rejected_vis))
                    rows = math.ceil(len(rejected_vis) / cols)
                    mosaic_h = rows * 80 + (rows + 1) * 4
                    mosaic_w = cols * 80 + (cols + 1) * 4
                    mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
                    mosaic[:] = (30, 30, 30)
                    for ri, thumb in enumerate(rejected_vis):
                        row, col = ri // cols, ri % cols
                        y0 = row * 84 + 4
                        x0 = col * 84 + 4
                        mosaic[y0:y0+80, x0:x0+80] = thumb
                    rej_path = os.path.join(out_dir, "visualizations", "rejected_samples.jpg")
                    cv2.imwrite(rej_path, mosaic)
                    self.log_signal.emit(f"  🔍 被拦截样本拼图已保存: visualizations/rejected_samples.jpg")
                    self.log_signal.emit(f"     → 若图中有目标物，说明阈值偏高，请适当调低")

                self.log_signal.emit("📊 =================【校验报告】=================")
                self.log_signal.emit(f"  总计提取区域:           {stats['total']} 个")
                self.log_signal.emit(f"  ✅ 生成标签:            {stats['pass']} 个")
                self.log_signal.emit(f"  🟥 工作区域外拦截:      {stats['rej_roi']} 个")
                self.log_signal.emit(f"  ❌ 背景排斥拦截:        {stats['rej_bg']} 个")
                self.log_signal.emit(f"  ❌ 特征相似度拦截:      {stats['rej_feat']} 个")
                self.log_signal.emit(
                    f"  🔎 文本框/SAM/轮廓/直接HBB: "
                    f"{stage_stats['text_boxes']}/{stage_stats['sam_masks']}/"
                    f"{stage_stats['mask_contours']}/{stage_stats['hbb_direct']}")
                if sim_samples:
                    arr = np.array(sim_samples)
                    pct25 = float(np.percentile(arr, 25))
                    pct75 = float(np.percentile(arr, 75))
                    self.log_signal.emit(
                        f"  📐 相似度分布: min={arr.min():.3f}  p25={pct25:.3f}  "
                        f"mean={arr.mean():.3f}  p75={pct75:.3f}  max={arr.max():.3f}"
                    )
                    self.log_signal.emit(
                        f"     当前阈值={verifier.threshold:.3f}  "
                        f"→ 阈值建议范围 [{arr.min():.3f}, {pct25:.3f}]"
                    )
                self.log_signal.emit(f"  📁 训练集: {train_count} 张 | 验证集: {len(dataset_records) - train_count} 张")
                self.log_signal.emit(f"  📂 结果根目录: {os.path.abspath(out_dir)}")
                self.log_signal.emit(f"  🖼️ 图片目录: {os.path.abspath(os.path.join(out_dir, 'images'))}")
                self.log_signal.emit(f"  🏷️ 标签目录: {os.path.abspath(os.path.join(out_dir, 'labels'))}")
                self.log_signal.emit(f"  👁️ 可视化目录: {os.path.abspath(os.path.join(out_dir, 'visualizations'))}")
                self.log_signal.emit(f"  📄 数据集配置: {os.path.abspath(yaml_path)}")
                self.log_signal.emit("🎉 全自动标注完成！")
                # 发送原始缓存供「重新过滤」功能使用
                self.raw_records_signal.emit({
                    'raw_candidates': raw_candidates,
                    'out_dir': out_dir,
                    'annotation_format': annotation_format,
                    'class_names': class_names,
                    'left_right_mode': self.config.get('left_right_mode', False),
                    'workspace_roi': workspace_roi,
                    'split_ratio': self.config['split_ratio'],
                    'num_vis': self.config['num_vis'],
                })

        except Exception as e:
            import traceback
            self.log_signal.emit(f"❌ 发生致命错误: {str(e)}\n{traceback.format_exc()}")
        finally:
            self.finished_signal.emit()

    def stop(self):
        self.is_running = False


# ================= 主窗口 GUI =================

class AutoAnnotatorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO OBB/HBB 全自动标注系统 v2 (旋转模板 + 特征校验)")
        self.resize(900, 720)
        self.templates_info = []      # list of (gray_rotated, w, h)
        self.template_patches = []    # list of BGR patch (for feature verifier)
        self._raw_records = []        # 缓存原始 SAM 结果（含相似度），供重新过滤
        self.settings = QSettings("AutoOBBAnnotator", "AutoOBBAnnotator")
        saved_roi = str(self.settings.value(
            "workspace_roi", ",".join(map(str, DEFAULT_WORKSPACE_ROI))))
        self.workspace_roi = normalize_workspace_roi(saved_roi.split(","))
        if self.workspace_roi is None:
            self.workspace_roi = DEFAULT_WORKSPACE_ROI

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # ---- 路径配置 ----
        group_path = QGroupBox("📁 路径配置")
        grid_path = QGridLayout()

        self.lbl_input = QLabel(display_path(REALSENSE_COLOR_DIR))
        self.lbl_bg = QLabel(display_path(REALSENSE_BG_DIR))
        self.lbl_bg.setStyleSheet("color: gray;")
        self.lbl_out = QLabel(display_path(OBB_DATASET_DIR))

        btn_input = QPushButton("选择原始图片目录")
        btn_bg = QPushButton("选择背景图目录 (可选)")
        btn_out = QPushButton("设置导出目录")

        btn_input.clicked.connect(lambda: self.select_dir(self.lbl_input))
        btn_bg.clicked.connect(lambda: self.select_dir(self.lbl_bg))
        btn_out.clicked.connect(lambda: self.select_dir(self.lbl_out))

        grid_path.addWidget(btn_input, 0, 0); grid_path.addWidget(self.lbl_input, 0, 1)
        grid_path.addWidget(btn_bg, 1, 0);    grid_path.addWidget(self.lbl_bg, 1, 1)
        grid_path.addWidget(btn_out, 2, 0);   grid_path.addWidget(self.lbl_out, 2, 1)

        bg_hint = QLabel(
            "💡 背景图说明：背景图中不应含目标物，仅含纯背景/环境。"
            "程序会利用背景做差分检测，并训练特征排斥器防止误标背景区域。"
        )
        bg_hint.setStyleSheet("color: #64B5F6; font-size: 11px;")
        bg_hint.setWordWrap(True)
        grid_path.addWidget(bg_hint, 3, 0, 1, 2)

        group_path.setLayout(grid_path)
        layout.addWidget(group_path)

        # ---- 算法参数 ----
        group_params = QGroupBox("⚙️ 算法参数")
        grid_params = QGridLayout()

        self.sp_match_thresh = QDoubleSpinBox()
        self.sp_match_thresh.setRange(0.1, 1.0)
        self.sp_match_thresh.setSingleStep(0.05)
        self.sp_match_thresh.setValue(0.65)

        self.sp_sim_thresh = QDoubleSpinBox()
        self.sp_sim_thresh.setRange(0.0, 1.0)
        self.sp_sim_thresh.setSingleStep(0.05)
        self.sp_sim_thresh.setValue(0.45)
        self.sp_sim_thresh.setToolTip(
            "特征相似度阈值（直方图交叉距离，范围0~1）：\n"
            "程序启动后会基于模板自相似度自动校准，此处为手动覆盖值。\n"
            "若标注结果为0，可在报告中查看'相似度诊断'，\n"
            "将阈值调到 min值 以下即可全部通过。\n"
            "建议范围：0.30~0.55"
        )

        self.sp_sam_batch = QSpinBox()
        self.sp_sam_batch.setRange(1, 128)
        self.sp_sam_batch.setValue(64)
        self.sp_sam_batch.setToolTip(
            "每批送入 SAM 的候选框数量。\n"
            "显存不足时建议降到 4 或 2；显存充足可升高以提速。"
        )

        self.cmb_annotation_format = QComboBox()
        self.cmb_annotation_format.addItem("YOLO-OBB（旋转框，8个坐标）", "obb")
        self.cmb_annotation_format.addItem("YOLO-HBB（普通框，cx cy w h）", "hbb")
        self.cmb_annotation_format.setToolTip(
            "OBB 每行: class x1 y1 x2 y2 x3 y3 x4 y4\n"
            "HBB 每行: class center_x center_y width height\n"
            "坐标均按图片宽高归一化到 0~1。"
        )
        self.cmb_annotation_format.currentIndexChanged.connect(
            self.on_annotation_format_changed)

        grid_params.addWidget(QLabel("模板匹配置信度:"), 0, 0)
        grid_params.addWidget(self.sp_match_thresh, 0, 1)
        grid_params.addWidget(QLabel("特征相似度阈值 (0~1):"), 0, 2)
        grid_params.addWidget(self.sp_sim_thresh, 0, 3)
        grid_params.addWidget(QLabel("SAM批大小:"), 1, 0)
        grid_params.addWidget(self.sp_sam_batch, 1, 1)
        grid_params.addWidget(QLabel("标签框格式:"), 1, 2)
        grid_params.addWidget(self.cmb_annotation_format, 1, 3)

        self.ed_text_prompt = QLineEdit()
        self.ed_text_prompt.setPlaceholderText("例如: hand（留空则使用模板/背景模式）")
        self.ed_text_prompt.setToolTip(
            "填写英文目标名称后，直接使用 SAM3 文本提示检测并分割。\n"
            "例如: hand、cup、bottle。留空则使用模板/背景 + MobileSAM 模式。"
        )
        self.sp_text_box_thresh = QDoubleSpinBox()
        self.sp_text_box_thresh.setRange(0.05, 0.95)
        self.sp_text_box_thresh.setSingleStep(0.05)
        self.sp_text_box_thresh.setValue(0.25)
        grid_params.addWidget(QLabel("文本提示（可选）:"), 2, 0)
        grid_params.addWidget(self.ed_text_prompt, 2, 1, 1, 2)
        grid_params.addWidget(self.sp_text_box_thresh, 2, 3)

        self.ed_class_name = QLineEdit("object")
        self.ed_class_name.setToolTip(
            "其他物体的 YOLO 类别名，例如 cup、bottle、box。\n"
            "文本提示为 hand 时会自动使用 left/right 两类。"
        )
        grid_params.addWidget(QLabel("类别名称:"), 3, 0)
        grid_params.addWidget(self.ed_class_name, 3, 1, 1, 3)

        sim_hint = QLabel("↑ 程序会自动校准阈值（模板自相似度×75%）。若标注为0，查看日志中「相似度诊断」，将此值调到 min值 以下。")
        sim_hint.setStyleSheet("color: #FF9800; font-size: 11px;")
        sim_hint.setWordWrap(True)
        grid_params.addWidget(sim_hint, 4, 0, 1, 4)

        self.chk_point_refine = QCheckBox("启用 SAM 点提示精修")
        self.chk_point_refine.setChecked(False)
        self.chk_point_refine.setToolTip(
            "对每个候选框追加中心正点 + 角落负点提示。\n"
            "通常能减少粘连背景、边缘外扩，但会略微变慢。\n"
            "默认关闭以避免每个候选框单独调用一次 SAM；需要更精细边界时再开启。\n"
            "若当前 ultralytics 版本不支持点提示，会自动回退。"
        )
        grid_params.addWidget(self.chk_point_refine, 5, 0, 1, 2)

        refine_hint = QLabel("建议默认开启：对细长物体、贴近背景的目标，分割边界通常更稳。")
        refine_hint.setStyleSheet("color: #4CAF50; font-size: 11px;")
        refine_hint.setWordWrap(True)
        grid_params.addWidget(refine_hint, 5, 2, 1, 2)

        self.chk_workspace_roi = QCheckBox("只标注灰色台面区域")
        self.chk_workspace_roi.setChecked(
            self.settings.value("workspace_roi_enabled", True, type=bool))
        self.chk_workspace_roi.setToolTip(
            "启用后，候选框中心必须位于所选区域内，且至少一半框面积在区域内。\n"
            "固定相机画面建议开启，可排除右侧货架等区域中的同类物体。")
        self.chk_workspace_roi.toggled.connect(
            lambda checked: self.settings.setValue(
                "workspace_roi_enabled", checked))

        self.btn_workspace_roi = QPushButton("🟥 重新框选台面区域")
        self.btn_workspace_roi.clicked.connect(self.select_workspace_roi)
        self.lbl_workspace_roi = QLabel(workspace_roi_text(self.workspace_roi))
        self.lbl_workspace_roi.setStyleSheet("color:#EF5350;font-size:11px;")
        self.lbl_workspace_roi.setToolTip("归一化区域坐标，不受图片分辨率变化影响。")

        grid_params.addWidget(self.chk_workspace_roi, 6, 0)
        grid_params.addWidget(self.btn_workspace_roi, 6, 1)
        grid_params.addWidget(self.lbl_workspace_roi, 6, 2, 1, 2)

        group_params.setLayout(grid_params)
        layout.addWidget(group_params)

        # ---- 运行控制 ----
        group_action = QGroupBox("🚀 运行控制")
        layout_action = QVBoxLayout()

        box_btns = QHBoxLayout()
        self.btn_extract = QPushButton("🎯 1. 提取/旋转模板")
        self.btn_extract.setStyleSheet(
            "background-color: #2196F3; color: white; padding: 10px; font-weight: bold;")
        self.btn_extract.clicked.connect(self.interactive_extract)

        self.btn_start = QPushButton("▶️ 2. 一键开始自动标注")
        self.btn_start.setStyleSheet(
            "background-color: #4CAF50; color: white; padding: 10px; font-weight: bold;")
        self.btn_start.clicked.connect(self.start_processing)

        self.btn_stop = QPushButton("⏹ 停止")
        self.btn_stop.setStyleSheet(
            "background-color: #F44336; color: white; padding: 10px; font-weight: bold;")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_processing)

        box_btns.addWidget(self.btn_extract)
        box_btns.addWidget(self.btn_start)
        box_btns.addWidget(self.btn_stop)

        # 重新校准按钮行
        box_btns2 = QHBoxLayout()
        self.btn_refilter = QPushButton("🔧 3. 调整阈值后重新过滤（无需重跑SAM）")
        self.btn_refilter.setStyleSheet(
            "background-color: #9C27B0; color: white; padding: 8px; font-weight: bold;")
        self.btn_refilter.setEnabled(False)
        self.btn_refilter.setToolTip(
            "上一次标注完成后，若想调整「特征相似度阈值」，\n"
            "点此按钮可直接用缓存的原始相似度数据重新过滤，\n"
            "无需重新运行耗时的 SAM 推理。")
        self.btn_refilter.clicked.connect(self.refilter_with_new_threshold)
        box_btns2.addWidget(self.btn_refilter)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        layout_action.addLayout(box_btns)
        layout_action.addLayout(box_btns2)
        layout_action.addWidget(self.progress_bar)
        group_action.setLayout(layout_action)
        layout.addWidget(group_action)

        self.log_console = QTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setStyleSheet(
            "background-color: #1E1E1E; color: #00FF00; font-family: Consolas;")
        layout.addWidget(self.log_console)

        self.append_log(
            "初始化完毕。\n"
            "1. 选择图片目录，背景图目录可选填（选了可大幅减少误标）。\n"
            "2. 点击【提取/旋转模板】→ 在弹窗中框选目标 → 旋转黄框贴合目标 → 确认。\n"
            "3. 点击【一键开始自动标注】。\n\n"
            "✨ 新功能：\n"
            "  • SAM3 文本提示模式：输入 hand/cup 等英文名称，无需模板即可自动找框并分割\n"
            "  • hand 自动按画面左右分为 left/right；其他物体使用一个可自定义类别\n"
            "  • 灰色台面工作区域过滤：默认按示例红框过滤，也可在界面重新框选\n"
            "  • 标签格式可选 YOLO-OBB 旋转框或 YOLO-HBB 普通框\n"
            "  • ROI 8控制点拖拽调整大小，橙色手柄拖拽旋转，全部跟着框走\n"
            "  • 快捷键: Space确认, Z/V大旋转, X/C微旋转, D复制框, 方向键平移\n"
            "  • 特征校验升级: HOG+颜色直方图（无需训练，即开即用，效果更稳）\n"
            "  • 可选 SAM 点提示精修: 在框提示基础上追加正负点，提升边界贴合度\n"
            "  • 背景图自动参与特征排斥，防止将背景误识别为目标"
        )

    def on_annotation_format_changed(self):
        """切换 OBB/HBB 时，仅自动替换尚未自定义的默认导出目录。"""
        annotation_format = self.cmb_annotation_format.currentData()
        current_out = self.lbl_out.text().strip()
        default_dirs = {
            display_path(OBB_DATASET_DIR), display_path(HBB_DATASET_DIR),
            str(OBB_DATASET_DIR), str(HBB_DATASET_DIR),
        }
        if current_out in default_dirs:
            target = HBB_DATASET_DIR if annotation_format == 'hbb' else OBB_DATASET_DIR
            self.lbl_out.setText(display_path(target))
        format_name = "YOLO-HBB 普通框" if annotation_format == 'hbb' else "YOLO-OBB 旋转框"
        self.append_log(f"🏷️ 已切换标签格式: {format_name}")

    def select_dir(self, label_widget):
        current = label_widget.text()
        current_path = str(resolve_path(current))
        directory = QFileDialog.getExistingDirectory(
            self, "选择目录", current_path if os.path.exists(current_path) else "")
        if directory:
            label_widget.setText(display_path(directory))
            label_widget.setStyleSheet("color: black;")

    def select_workspace_roi(self):
        """在输入目录第一张图上选择固定机位的有效标注区域。"""
        input_dir = str(resolve_path(self.lbl_input.text()))
        if not os.path.isdir(input_dir):
            QMessageBox.warning(self, "错误", "请先选择正确的原始图片目录！")
            return
        image_names = sorted(
            name for name in os.listdir(input_dir)
            if name.lower().endswith(('.png', '.jpg', '.jpeg'))
        )
        if not image_names:
            QMessageBox.warning(self, "错误", "输入目录中没有图片！")
            return

        image_path = os.path.join(input_dir, image_names[0])
        dialog = ROISelectorDialog(
            image_path,
            title="只框选灰色台面区域（画框后按 Space 确认）",
            parent=self,
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        rois = dialog.get_rois_with_angle()
        if not rois:
            QMessageBox.warning(
                self, "未设置区域",
                "没有已确认的框。请拖出矩形后按 Space，再点击完成。")
            return

        cx, cy, width, height, angle = rois[-1]
        half_w, half_h = width / 2.0, height / 2.0
        corners = [
            _rot_pt_f(cx + dx, cy + dy, cx, cy, angle)
            for dx, dy in (
                (-half_w, -half_h), (half_w, -half_h),
                (half_w, half_h), (-half_w, half_h),
            )
        ]
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        roi = normalize_workspace_roi((
            min(xs) / dialog.img_w, min(ys) / dialog.img_h,
            max(xs) / dialog.img_w, max(ys) / dialog.img_h,
        ))
        if roi is None:
            QMessageBox.warning(self, "区域无效", "框选区域过小，请重新选择。")
            return

        self.workspace_roi = roi
        self.settings.setValue("workspace_roi", ",".join(map(str, roi)))
        self.chk_workspace_roi.setChecked(True)
        self.lbl_workspace_roi.setText(workspace_roi_text(roi))
        self.append_log(f"🟥 灰色台面区域已更新: {workspace_roi_text(roi)}")

    def append_log(self, text):
        self.log_console.append(text)
        self.log_console.verticalScrollBar().setValue(
            self.log_console.verticalScrollBar().maximum())

    def interactive_extract(self):
        input_dir = str(resolve_path(self.lbl_input.text()))
        if not os.path.exists(input_dir):
            QMessageBox.warning(self, "错误", "请先选择正确的原始图片目录！")
            return

        all_images = [f for f in os.listdir(input_dir)
                      if f.lower().endswith(('.png', '.jpg'))]
        if not all_images:
            QMessageBox.warning(self, "错误", "目录下没有找到图片！")
            return

        num, ok = QInputDialog.getInt(
            self, "抽样数量", "请输入要随机抽样提模板的图片数量:", 3, 1, 20, 1)
        if not ok:
            return

        sample_num = min(num, len(all_images))
        sampled = random.sample(all_images, sample_num)

        self.append_log(
            f"开始交互式提取，抽取了 {sample_num} 张图片。\n"
            "操作方式：拖拽画框 → 拖控制点调大小 → 橙色手柄旋转 → Space确认 → D复制框\n快捷键: Z/V大旋转, X/C微旋转, 方向键平移, W/S/A/Q缩放, Tab取回上一框")

        self.templates_info.clear()
        self.template_patches.clear()

        for i, name in enumerate(sampled):
            img_path = os.path.join(input_dir, name)
            dialog = ROISelectorDialog(
                img_path,
                title=f"框选并旋转特征 ({i + 1}/{sample_num})",
                parent=self)

            if dialog.exec_() == QDialog.Accepted:
                rois = dialog.get_rois_with_angle()
                img_bgr = cv2.imread(img_path)
                if img_bgr is None:
                    self.append_log(f"⚠️ 跳过无法读取的图片: {name}")
                    continue
                img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

                for (cx, cy, w, h, angle) in rois:
                    if w <= 0 or h <= 0:
                        continue

                    # 旋转整张图，然后裁剪正向矩形 —— 这样模板和检测图方向一致
                    rot_mat = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
                    rotated_gray = cv2.warpAffine(
                        img_gray, rot_mat, (img_gray.shape[1], img_gray.shape[0]))
                    rotated_bgr = cv2.warpAffine(
                        img_bgr, rot_mat, (img_bgr.shape[1], img_bgr.shape[0]))

                    x1 = max(0, int(cx - w / 2))
                    y1 = max(0, int(cy - h / 2))
                    x2 = min(img_gray.shape[1], int(cx + w / 2))
                    y2 = min(img_gray.shape[0], int(cy + h / 2))

                    crop_gray = rotated_gray[y1:y2, x1:x2]
                    crop_bgr = rotated_bgr[y1:y2, x1:x2]

                    if crop_gray.size > 0 and crop_bgr.size > 0:
                        self.templates_info.append((crop_gray, x2 - x1, y2 - y1))
                        self.template_patches.append(crop_bgr)
            else:
                self.append_log(f"⚠️ 跳过了图片: {name}")

        self.append_log(
            f"✅ 模板提取完成，共收集 {len(self.templates_info)} 个特征模板"
            f"（含旋转贴合）！")

    def start_processing(self):
        in_dir = str(resolve_path(self.lbl_input.text()))
        out_dir = str(resolve_path(self.lbl_out.text()))

        if not os.path.exists(in_dir):
            QMessageBox.warning(self, "错误", f"原始图片目录无效:\n{in_dir}")
            return

        abs_in_dir = os.path.abspath(in_dir)
        abs_out_dir = os.path.abspath(out_dir)

        input_path = Path(abs_in_dir)
        output_path = Path(abs_out_dir)
        if (
            input_path == output_path
            or output_path.is_relative_to(input_path)
            or input_path.is_relative_to(output_path)
        ):
            QMessageBox.critical(
                self, "危险操作阻止",
                "【严重错误】原始图片目录和导出目录存在包含关系！\n"
                "为避免清理旧结果时删除原图，请将导出目录设置为独立文件夹。")
            return

        text_prompt = self.ed_text_prompt.text().strip()
        left_right_mode = bool(re.search(r"\bhand(s)?\b", text_prompt.lower()))
        class_name = self.ed_class_name.text().strip() or "object"
        if not left_right_mode and class_name == "object" and text_prompt:
            inferred_name = re.sub(
                r"[^a-zA-Z0-9_]+", "_", text_prompt.split(",", 1)[0]
            ).strip("_")
            if inferred_name:
                class_name = inferred_name[:64]
        if not self.templates_info and not text_prompt:
            reply = QMessageBox.question(
                self, "警告",
                "未提取模板！程序将仅依赖背景差分（需选背景目录），特征校验将全部通过。是否继续？",
                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return

        bg_dir_text = str(resolve_path(self.lbl_bg.text()))
        if "未选择" in bg_dir_text or not os.path.exists(bg_dir_text):
            bg_dir_text = ""

        config = {
            'input_dir': in_dir,
            'bg_dir': bg_dir_text,
            'out_dir': out_dir,
            'templates': self.templates_info,
            'template_patches': self.template_patches,
            'match_thresh': self.sp_match_thresh.value(),
            'sim_threshold': self.sp_sim_thresh.value(),
            'sam_batch_size': self.sp_sam_batch.value(),
            'use_point_refine': self.chk_point_refine.isChecked(),
            'annotation_format': self.cmb_annotation_format.currentData(),
            'text_prompt': text_prompt,
            'class_name': class_name,
            'left_right_mode': left_right_mode,
            'text_box_threshold': self.sp_text_box_thresh.value(),
            'workspace_roi': (
                self.workspace_roi if self.chk_workspace_roi.isChecked()
                else None
            ),
            'split_ratio': 0.8,
            'num_vis': 10
        }

        self.btn_start.setEnabled(False)
        self.btn_extract.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_console.clear()
        self.append_log(f"📂 本次结果将保存到: {out_dir}")

        self.thread = AnnotationThread(config)
        self.thread.log_signal.connect(self.append_log)
        self.thread.progress_signal.connect(self.update_progress)
        self.thread.finished_signal.connect(self.on_processing_finished)
        self.thread.raw_records_signal.connect(self._on_raw_records)
        self.thread.start()

    def update_progress(self, current, total):
        self.progress_bar.setValue(int((current / total) * 100))

    def _on_raw_records(self, data):
        """接收并缓存原始候选结果，激活重新过滤按钮"""
        self._raw_records = data
        self.btn_refilter.setEnabled(True)
        self.append_log(
            f"💾 已缓存 {len(data.get('raw_candidates', []))} 个候选区域原始数据，"
            "可调整阈值后点击【重新过滤】无需重跑SAM。")

    def refilter_with_new_threshold(self):
        """用新阈值对缓存的原始结果重新过滤并导出"""
        if not self._raw_records:
            QMessageBox.warning(self, "提示", "没有缓存数据，请先运行一次标注。")
            return

        new_thresh = self.sp_sim_thresh.value()
        raw = self._raw_records
        candidates = raw['raw_candidates']
        out_dir = raw['out_dir']
        annotation_format = raw.get('annotation_format', 'obb')
        format_name = "HBB" if annotation_format == 'hbb' else "OBB"

        self.append_log(
            f"\n🔧 正在用新阈值 {new_thresh:.3f} 重新过滤 "
            f"{len(candidates)} 个 {format_name} 候选...")

        # 按阈值过滤
        from collections import defaultdict
        per_img = defaultdict(list)
        rej = 0
        for c in candidates:
            if c['sim'] >= new_thresh:
                per_img[c['img_name']].append(c)
            else:
                rej += 1

        # 清理旧结果，避免旧阈值下的数据残留
        for d in ["images", "labels", "visualizations"]:
            target_path = os.path.join(out_dir, d)
            if os.path.exists(target_path):
                shutil.rmtree(target_path)
        for d in ["images/train", "images/val", "labels/train", "labels/val", "visualizations"]:
            os.makedirs(os.path.join(out_dir, d), exist_ok=True)

        all_names = list(per_img.keys())
        random.shuffle(all_names)
        train_count = int(len(all_names) * raw['split_ratio'])
        passed = 0

        for i, name in enumerate(all_names):
            subset = "train" if i < train_count else "val"
            records = per_img[name]
            # 找原始路径（从任一 record 取）
            src_path = records[0]['img_path']
            dst_img = os.path.join(out_dir, f"images/{subset}", name)
            if not os.path.exists(dst_img):
                shutil.copy(src_path, dst_img)
            txt_path = os.path.join(out_dir, f"labels/{subset}",
                                    f"{os.path.splitext(name)[0]}.txt")
            with open(txt_path, 'w') as f:
                for c in records:
                    label = c.get('label', c.get('obb'))  # 兼容旧版缓存
                    if label:
                        class_id = int(c.get('class_id', 0))
                        f.write(str(class_id) + " " +
                                " ".join([f"{p:.6f}" for p in label]) + "\n")
                    passed += 1

            if i < raw['num_vis']:
                vis_img = cv2.imread(src_path)
                if vis_img is not None:
                    draw_workspace_roi(vis_img, raw.get('workspace_roi'))
                    for c in records:
                        label = c.get('label', c.get('obb'))
                        if label:
                            draw_yolo_label(vis_img, label, annotation_format)
                    cv2.imwrite(
                        os.path.join(out_dir, f"visualizations/vis_{name}"),
                        vis_img)

        self.append_log(f"✅ 重新过滤完成: 保留 {passed} 个标签，拦截 {rej} 个")
        self.append_log(f"   训练集: {train_count} 张 | 验证集: {len(all_names)-train_count} 张")
        self.append_log(f"   📂 结果根目录: {os.path.abspath(out_dir)}")
        self.append_log(f"   🏷️ 标签目录: {os.path.abspath(os.path.join(out_dir, 'labels'))}")
        class_names = raw.get('class_names')
        if not class_names:
            class_names = ["left", "right"] if raw.get(
                'left_right_mode', False) else ["object"]
        write_project_dataset_yaml(out_dir, class_names)

    def stop_processing(self):
        if hasattr(self, 'thread') and self.thread.isRunning():
            self.thread.stop()
            self.append_log("⚠️ 收到停止指令，正在安全退出...")
            self.btn_stop.setEnabled(False)

    def on_processing_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_extract.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.append_log("========== 处理线程已结束 ==========")


if __name__ == "__main__":
    from .__main__ import main

    raise SystemExit(main())
