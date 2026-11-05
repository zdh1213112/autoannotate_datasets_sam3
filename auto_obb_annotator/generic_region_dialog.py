"""Draw a rectangular or circular search region on a sample image."""

from pathlib import Path
import math

import cv2
from PyQt5.QtCore import QPoint, QPointF, Qt
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialog, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QVBoxLayout,
)

from .button_shortcuts import ButtonShortcuts
from .generic_region import normalize_region


class RegionCanvas(QLabel):
    def __init__(self, image_bgr, region, parent=None):
        super().__init__(parent)
        height, width = image_bgr.shape[:2]
        screen = QApplication.primaryScreen().availableGeometry()
        scale = min(1.0, min(1100, max(450, screen.width() - 100)) / width,
                    min(620, max(270, screen.height() - 210)) / height)
        self.image_width, self.image_height = width, height
        self.scale = scale
        display = cv2.resize(image_bgr, (round(width * scale), round(height * scale)),
                             interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.setPixmap(QPixmap.fromImage(QImage(
            rgb.data, w, h, w * 3, QImage.Format_RGB888).copy()))
        self.setFixedSize(w, h)
        self.setCursor(Qt.CrossCursor)
        self.shape = "circle" if isinstance(region, dict) else "rectangle"
        self.selection = None
        self.drag_start = None
        if isinstance(region, dict):
            cx, cy = region["center"]
            self.selection = (cx * w, cy * h,
                              region["radius"] * min(width, height) * scale)
        elif region is not None:
            self.selection = (region[0] * w, region[1] * h,
                              region[2] * w, region[3] * h)

    def set_shape(self, shape):
        if self.shape != shape:
            self.shape = shape
            self.selection = None
            self.update()

    def _clamped(self, pos):
        return QPoint(max(0, min(self.width(), pos.x())),
                      max(0, min(self.height(), pos.y())))

    def _select_to(self, pos):
        end = self._clamped(pos)
        start = self.drag_start
        if self.shape == "rectangle":
            self.selection = (min(start.x(), end.x()), min(start.y(), end.y()),
                              max(start.x(), end.x()), max(start.y(), end.y()))
        else:
            radius = math.hypot(end.x() - start.x(), end.y() - start.y())
            self.selection = (start.x(), start.y(), radius)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start = self._clamped(event.pos())
            self.selection = None
            self.update()

    def mouseMoveEvent(self, event):
        if self.drag_start is not None:
            self._select_to(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drag_start is not None:
            self._select_to(event.pos())
            self.drag_start = None

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.selection is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(QColor(255, 210, 0), 3))
        painter.setBrush(QColor(255, 210, 0, 35))
        if self.shape == "rectangle":
            x1, y1, x2, y2 = self.selection
            painter.drawRect(round(x1), round(y1), round(x2 - x1), round(y2 - y1))
        else:
            cx, cy, radius = self.selection
            painter.drawEllipse(QPointF(cx, cy), radius, radius)
        painter.end()

    def region(self):
        if self.selection is None:
            return None
        if self.shape == "rectangle":
            x1, y1, x2, y2 = self.selection
            return normalize_region([x1 / self.width(), y1 / self.height(),
                                     x2 / self.width(), y2 / self.height()])
        cx, cy, radius = self.selection
        return normalize_region({"shape": "circle",
                                 "center": [cx / self.width(), cy / self.height()],
                                 "radius": radius / (self.scale * min(
                                     self.image_width, self.image_height))})


class GenericRegionDialog(QDialog):
    def __init__(self, image_path: Path, region, parent=None):
        super().__init__(parent)
        self.setWindowTitle("在样图上选择检测区域")
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        self.result_region = None
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(QLabel("区域形状"))
        self.shape_combo = QComboBox()
        self.shape_combo.addItem("矩形：拖动选两角", "rectangle")
        self.shape_combo.addItem("圆形：从圆心向外拖动", "circle")
        row.addWidget(self.shape_combo)
        row.addStretch()
        layout.addLayout(row)
        self.canvas = RegionCanvas(image, region, self)
        layout.addWidget(self.canvas, alignment=Qt.AlignCenter)
        self.shape_combo.setCurrentIndex(1 if isinstance(region, dict) else 0)
        self.shape_combo.currentIndexChanged.connect(
            lambda: self.canvas.set_shape(self.shape_combo.currentData()))
        layout.addWidget(QLabel("在图片上拖动可重画区域；保存后，圆形外的候选不会进入标注。"))
        buttons = QHBoxLayout()
        buttons.addStretch()
        cancel = QPushButton("取消 [Esc]")
        cancel.clicked.connect(self.reject)
        save = QPushButton("保存区域 [Ctrl+S]")
        save.clicked.connect(self.accept)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        layout.addLayout(buttons)
        self.button_shortcuts = ButtonShortcuts(self)

    def accept(self):
        region = self.canvas.region()
        if region is None:
            QMessageBox.warning(self, "区域无效", "请在样图上绘制有效的矩形或圆形区域。")
            return
        self.result_region = region
        super().accept()
