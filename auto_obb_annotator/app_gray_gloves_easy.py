"""A compact, direct-manipulation UI on top of the existing glove workflow.

Inspired by make-sense's editor layout, implemented with native Qt widgets.
Inference, review records, undo and exports use the existing project engine.
"""
from __future__ import annotations

import copy
from datetime import datetime
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from PyQt5.QtCore import QCoreApplication, QLibraryInfo, QPointF, QRectF, Qt, QTimer
from PyQt5.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QDialog, QGraphicsScene, QGraphicsView, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSplitter, QTabWidget, QTableWidgetItem,
    QVBoxLayout, QWidget, QAbstractItemView, QShortcut, QMessageBox,
)

from .app_gray_gloves import GrayGloveAnnotatorApp
from .app_generic import STATUS_TEXT, _pixmap, replace_project_config
from .button_shortcuts import ButtonShortcuts
from .generic_core import sync_candidates
from .gray_glove_projection import frame_projection_sources
from .generic_region import polygon_in_region
from .gray_glove_dataset_ui import YoloDatasetMixin
from .project_paths import ROOT_DIR

EASY_SESSION = ROOT_DIR / "data" / ".gray_glove_easy_session.json"


class GloveCanvas(QGraphicsView):
    """Image coordinates remain unchanged while the view zooms and pans."""

    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QColor("#151b24"))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(360, 280)
        self.name = None
        self.picture = QPixmap()
        self.objects = []
        self.candidates = []
        self.selected = -1
        self.drawing = False
        self.gesture = None
        self.overlays = []
        self.auto_fit = True

    def load(self, name, path, record):
        changed = name != self.name
        self.gesture = None
        if changed:
            self.name = name
            self.picture = QPixmap(str(path))
            self.scene().clear()
            self.overlays = []
            self.scene().addPixmap(self.picture)
            self.setSceneRect(QRectF(self.picture.rect()))
            self.selected = -1
            self.auto_fit = True
        self.objects = copy.deepcopy(record.get("objects", []))
        self.candidates = sync_candidates(record)["candidates"]
        self.pending = record.get("side_pending", False)
        if self.selected >= len(self.objects):
            self.selected = -1
        self.paint_boxes()
        if changed:
            self.fit_image()

    def fit_image(self):
        if not self.picture.isNull():
            self.fitInView(QRectF(self.picture.rect()), Qt.KeepAspectRatio)
        self.auto_fit = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.auto_fit:
            self.fit_image()

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        scale = self.transform().m11() * factor
        if 0.03 <= scale <= 20:
            self.scale(factor, factor)
            self.auto_fit = False
            self.paint_boxes()
        event.accept()

    def paint_boxes(self):
        for item in self.overlays:
            self.scene().removeItem(item)
        self.overlays = []
        scale = max(0.01, self.transform().m11())
        rows = [(obj, i, False) for i, obj in enumerate(self.objects)]
        if self.owner.show_candidates_check.isChecked():
            rows += [(obj, -1, True) for obj in self.candidates if not obj["selected"]]
        for obj, index, candidate in rows:
            color = QColor("#c99e45" if candidate else "#c490ee" if self.pending
                           else "#45d6a0" if obj["class_id"] == 0 else "#49baff")
            pen = QPen(color, 3 if index == self.selected else 2)
            pen.setCosmetic(True)
            if candidate:
                pen.setStyle(Qt.DashLine)
            polygon = QPolygonF([QPointF(float(x), float(y)) for x, y in obj["polygon"]])
            item = self.scene().addPolygon(polygon, pen)
            self.overlays.append(item)
            label = "候选" if candidate else f"框#{index + 1} · " + ("待定" if self.pending else ("left" if obj["class_id"] == 0 else "right"))
            text = self.scene().addSimpleText(label)
            text.setBrush(color)
            text.setScale(1 / scale)
            corner = polygon.boundingRect().topLeft()
            text.setPos(corner.x(), max(0, corner.y() - 22 / scale))
            self.overlays.append(text)
            if index == self.selected:
                for point in polygon:
                    handle = self.scene().addRect(point.x() - 4 / scale, point.y() - 4 / scale,
                                                  8 / scale, 8 / scale, pen, color)
                    self.overlays.append(handle)

    def point(self, event):
        point = self.mapToScene(event.pos())
        return np.array([np.clip(point.x(), 0, max(0, self.picture.width() - 1)),
                         np.clip(point.y(), 0, max(0, self.picture.height() - 1))])

    def set_draw(self, enabled=True):
        self.cancel()
        self.drawing = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.setFocus()
        self.owner.status.setText("拖出一个框，松开自动保存；1 / 2 选择左右类别；Esc 取消。" if enabled
                                  else "点击选框，拖动移动，拖四角缩放；滚轮缩放图片，右键拖动平移。")

    def cancel(self):
        self.gesture = None
        if self.name in self.owner.records:
            self.objects = copy.deepcopy(self.owner.records[self.name].get("objects", []))
        self.drawing = False
        self.setCursor(Qt.ArrowCursor)
        self.paint_boxes()

    def mousePressEvent(self, event):
        self.setFocus()
        if event.button() in (Qt.MiddleButton, Qt.RightButton):
            self.gesture = ("pan", event.pos())
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() != Qt.LeftButton or self.picture.isNull():
            return
        self.owner.follow_check.setChecked(False)
        point = self.point(event)
        if self.drawing:
            self.gesture = ("new", point)
            return
        if 0 <= self.selected < len(self.objects):
            polygon = np.asarray(self.objects[self.selected]["polygon"], float)
            distances = np.linalg.norm(polygon - point, axis=1)
            corner = int(distances.argmin())
            if distances[corner] * self.transform().m11() < 12:
                self.gesture = ("resize", point, polygon.copy(), corner)
                return
        hits = []
        for i, obj in enumerate(self.objects):
            polygon = np.asarray(obj["polygon"], np.float32)
            if cv2.pointPolygonTest(polygon, tuple(map(float, point)), False) >= 0:
                hits.append((cv2.contourArea(polygon), i))
        if hits:
            self.selected = min(hits)[1]
            self.owner.object_table.selectRow(self.selected)
            polygon = np.asarray(self.objects[self.selected]["polygon"], float)
            self.gesture = ("move", point, polygon.copy())
        else:
            self.selected = -1
            self.owner.object_table.clearSelection()
            if self.owner.show_candidates_check.isChecked():
                self.owner.toggle_candidate_at(*point)
        self.paint_boxes()

    def mouseMoveEvent(self, event):
        if self.gesture is None:
            return
        mode, start, *rest = self.gesture
        if mode == "pan":
            delta = event.pos() - start
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.gesture = (mode, event.pos())
            return
        point = self.point(event)
        if mode == "new":
            low, high = np.minimum(start, point), np.maximum(start, point)
            polygon = [[low[0], low[1]], [high[0], low[1]], [high[0], high[1]], [low[0], high[1]]]
            self.objects = copy.deepcopy(self.owner.records[self.name].get("objects", []))
            self.objects.append({"polygon": polygon, "class_id": self.owner.object_class_combo.currentIndex()})
        elif mode == "move":
            original = rest[0]
            delta = point - start
            delta = np.maximum(-original.min(axis=0), np.minimum(delta,
                np.array([self.picture.width() - 1, self.picture.height() - 1]) - original.max(axis=0)))
            self.objects[self.selected]["polygon"] = (original + delta).tolist()
        elif mode == "resize":
            original, corner = rest
            opposite = (corner + 2) % 4
            fixed = original[opposite]
            u = original[(opposite + 1) % 4] - fixed
            v = original[(opposite - 1) % 4] - fixed
            u /= max(np.linalg.norm(u), 1e-6)
            v /= max(np.linalg.norm(v), 1e-6)
            a, b = max(2, np.dot(point - fixed, u)), max(2, np.dot(point - fixed, v))
            polygon = original.copy()
            polygon[(opposite + 1) % 4] = fixed + a * u
            polygon[(opposite - 1) % 4] = fixed + b * v
            polygon[corner] = fixed + a * u + b * v
            if (polygon >= 0).all() and (polygon <= [self.picture.width() - 1, self.picture.height() - 1]).all():
                self.objects[self.selected]["polygon"] = polygon.tolist()
        self.paint_boxes()

    def mouseReleaseEvent(self, event):
        if self.gesture is None:
            return
        mode, start, *rest = self.gesture
        self.gesture = None
        if mode == "pan":
            self.setCursor(Qt.CrossCursor if self.drawing else Qt.ArrowCursor)
            return
        if mode == "new":
            point = self.point(event)
            if np.min(np.abs(point - start)) >= 3:
                low, high = np.minimum(start, point), np.maximum(start, point)
                self.owner.commit_polygon(-1, [[low[0], low[1]], [high[0], low[1]],
                                               [high[0], high[1]], [low[0], high[1]]])
            self.set_draw(False)
        elif mode in ("move", "resize"):
            polygon = self.objects[self.selected]["polygon"]
            if not np.allclose(polygon, rest[0]):
                self.owner.commit_polygon(self.selected, polygon)


class EasyGrayGloveApp(YoloDatasetMixin, GrayGloveAnnotatorApp):
    def __init__(self):
        self.canvas = None
        self._yolo_dataset = None
        self._dataset_job = None
        self._resume_dataset = None
        self._video_geometry_mode = "auto"
        super().__init__()
        # Retain the legacy widget tree for engine references, with its shortcuts
        # disabled. Reparent only the useful controls into the compact editor.
        QApplication.instance().focusChanged.disconnect(self.button_shortcuts.update)
        for shortcut in self.findChildren(QShortcut):
            shortcut.setEnabled(False)
        self.legacy = self.takeCentralWidget()
        self.legacy.setParent(self, Qt.Tool)
        self.legacy.hide()
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(12, 10, 12, 10)
        top = QHBoxLayout()
        title = QLabel("灰手套 · 简洁标注")
        title.setStyleSheet("font-size:19px;font-weight:600;")
        top.addWidget(title)
        top.addStretch()
        self.open_project_button.setText("打开项目 / 数据集 [Alt+P]")
        self.open_project_button.setToolTip("打开原复核项目，或含 images / labels 的 YOLO 数据集；也可选 images 或 labels 目录。")
        top.addWidget(self.open_project_button)
        self.new_project_button = self.button("新建检测项目", self.new_detection_project)
        top.addWidget(self.new_project_button)
        self.settings_button = self.button("输入 / 设置 [Alt+S]", self.open_settings)
        top.addWidget(self.settings_button)
        self.start_button.setText("自动检测 [F5]")
        top.addWidget(self.start_button)
        self.stop_button.setText("停止任务")
        top.addWidget(self.stop_button)
        self.export_all_button.setText("导出全部 [Ctrl+Shift+E]")
        top.addWidget(self.export_all_button)
        outer.addLayout(top)
        self.dataset_location = QLabel()
        self.dataset_location.setWordWrap(True)
        self.dataset_location.hide()
        outer.addWidget(self.dataset_location)
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split, 1)
        left, middle, right = QWidget(), QWidget(), QWidget()
        for widget in (left, middle, right):
            split.addWidget(widget)
        left.setMinimumWidth(220)
        right.setMinimumWidth(235)
        ll, ml, rl = QVBoxLayout(left), QVBoxLayout(middle), QVBoxLayout(right)
        ll.addWidget(QLabel("图片列表"))
        ll.addWidget(self.search_edit)
        ll.addWidget(self.status_filter)
        ll.addWidget(self.review_filter)
        ll.addWidget(self.visible_count)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setTextElideMode(Qt.ElideMiddle)
        self.table.setColumnHidden(2, True)
        self.table.setColumnHidden(3, True)
        self.table.setColumnWidth(0, 165)
        self.table.setColumnWidth(1, 100)
        ll.addWidget(self.table, 1)
        nav = QHBoxLayout()
        nav.addWidget(self.button("上一张 [B]", self.show_previous_image))
        nav.addWidget(self.button("下一张 [N]", self.show_next_image))
        ll.addLayout(nav)
        ml.addWidget(self.preview_title)
        self.canvas = GloveCanvas(self)
        ml.addWidget(self.canvas, 1)
        tools = QHBoxLayout()
        tools.addWidget(self.button("选择 / 移动 [V]", lambda: self.canvas.set_draw(False)))
        tools.addWidget(self.button("画框 [A]", self.add_object))
        tools.addWidget(self.button("适合窗口 [F]", self.canvas.fit_image))
        tools.addWidget(self.button("取消 [Esc]", self.canvas.cancel))
        ml.addLayout(tools)
        hint = QLabel("滚轮缩放 · 右键拖动画面 · 拖动框 / 四角调整 · 松手保存")
        hint.setWordWrap(True)
        ml.addWidget(hint)
        rl.addWidget(QLabel("当前图片的手套"))
        self.object_table.setMaximumHeight(16777215)
        self.object_table.setMinimumHeight(110)
        self.object_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.object_table.setHorizontalHeaderLabels(["框号", "类别", "置信度"])
        self.object_table.verticalHeader().setVisible(False)
        self.object_table.setColumnWidth(0, 50)
        self.object_table.setColumnWidth(1, 80)
        rl.addWidget(self.object_table, 1)
        rl.addWidget(QLabel("画框类别 / 选中框类别"))
        rl.addWidget(self.object_class_combo)
        self.side_left_button.setText("左手 left [1]")
        self.side_right_button.setText("右手 right [2]")
        self.side_left_button.clicked.disconnect()
        self.side_right_button.clicked.disconnect()
        self.side_left_button.clicked.connect(lambda: self.pick_side(0))
        self.side_right_button.clicked.connect(lambda: self.pick_side(1))
        rl.addWidget(self.side_left_button)
        rl.addWidget(self.side_right_button)
        self.swap_button.setText("交换左右 [X]")
        rl.addWidget(self.swap_button)
        rl.addWidget(self.button("删除选中框 [Delete]", self.delete_object))
        self.undo_button.setText("撤销 [Ctrl+Z]")
        rl.addWidget(self.undo_button)
        rl.addWidget(self.button("精细旋转编辑 [E]", self.edit_object))
        self.show_candidates_check.setText("显示候选框（点击补选）")
        rl.addWidget(self.show_candidates_check)
        rl.addWidget(self.button("确认 → 下一张待复核 [Space]", self.confirm_current))
        rl.addWidget(self.button("本张无目标 [Q]", self.mark_empty))
        rl.addWidget(self.button("下一张待复核 [D]", self.next_review))
        self.format_combo.setMinimumWidth(0)
        rl.addWidget(self.format_combo)
        self.export_button.setText("导出训练集 [Ctrl+E]")
        rl.addWidget(self.export_button)
        split.setSizes([280, 900, 255])
        split.setStretchFactor(1, 1)
        self.progress.setMaximumHeight(12)
        outer.addWidget(self.progress)
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        self.settings_dialog = QDialog(self)
        self.settings_dialog.setWindowTitle("输入与检测设置")
        self.settings_dialog.resize(1120, 750)
        settings_layout = QVBoxLayout(self.settings_dialog)
        self.dataset_settings_banner = QWidget()
        banner_layout = QHBoxLayout(self.dataset_settings_banner)
        banner_layout.setContentsMargins(0, 0, 0, 0)
        notice = QLabel("当前是数据集编辑模式。要添加录像或重新检测，请新建检测项目；已保存的标签会保留。")
        notice.setWordWrap(True)
        banner_layout.addWidget(notice, 1)
        self.dataset_new_project_button = self.button("新建检测项目", self.new_detection_project)
        banner_layout.addWidget(self.dataset_new_project_button)
        self.dataset_settings_banner.hide()
        settings_layout.addWidget(self.dataset_settings_banner)
        geometry_row = QHBoxLayout()
        geometry_row.addWidget(QLabel("录像画面处理"))
        self.video_geometry_combo = QComboBox()
        self.video_geometry_combo.addItem("自动（四目 / MCAP 去畸变）", "auto")
        self.video_geometry_combo.addItem("原图（不去畸变）", "raw")
        self.video_geometry_combo.addItem("去畸变（需四目标定）", "rectified")
        self.video_geometry_combo.setCurrentIndex(
            self.video_geometry_combo.findData(self._video_geometry_mode))
        self.video_geometry_combo.setToolTip(
            "控制视频 / MCAP 抽帧和后续检测坐标；不会把已去畸变的普通图片还原成原图。")
        self.video_geometry_combo.currentIndexChanged.connect(self.change_video_geometry)
        geometry_row.addWidget(self.video_geometry_combo)
        geometry_row.addStretch()
        self.video_group.layout().insertLayout(0, geometry_row)
        self.video_geometry_hint = QLabel()
        self.video_geometry_hint.setWordWrap(True)
        self.video_group.layout().insertWidget(1, self.video_geometry_hint)
        self.update_video_geometry_controls()
        tabs = QTabWidget()
        self.settings_tabs = tabs
        for title, widget in (("图片 / 检测", self.settings_group), ("视频 / MCAP 抽帧", self.video_group)):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(widget)
            tabs.addTab(scroll, title)
        settings_layout.addWidget(tabs)
        self.follow_check.setText("检测时跟随新结果")
        settings_layout.addWidget(self.follow_check)
        settings_layout.addWidget(self.recheck_review_button)
        settings_layout.addWidget(self.progress_detail)
        settings_layout.addWidget(self.summary)
        settings_layout.addWidget(self.button("返回标注", self.settings_dialog.hide))
        self.settings_dialog.setModal(True)
        self.button_shortcuts = ButtonShortcuts(self)
        self.setWindowTitle("灰色手套 · 简洁版（独立入口）")
        self.resize(1480, 900)
        self.setStyleSheet("""
            QMainWindow, QDialog { background: #f3f5f8; }
            QPushButton { padding: 7px 9px; border: 1px solid #cbd4df; border-radius: 5px; background: white; }
            QPushButton:hover { background: #e4eefb; border-color: #6a9cd7; }
            QPushButton:disabled { color: #9ca5b0; }
            QLineEdit, QComboBox { padding: 5px; }
            QTableWidget { background: white; gridline-color: #eef1f5; border: 1px solid #d9e0e8; }
            QHeaderView::section { background: #e9eef5; padding: 5px; border: 0; }
        """)
        if self.current_name:
            self.show_record(self.current_name)
        else:
            self.status.setText("打开已有项目或 YOLO 数据集直接修改；新图片、视频从「输入 / 设置」导入。")
        if self._resume_dataset is not None:
            folder, name = self._resume_dataset
            QTimer.singleShot(0, lambda: self.open_path(folder, name))

    @staticmethod
    def button(text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        return button

    def previous_review_project(self):
        try:
            state = json.loads(EASY_SESSION.read_text(encoding="utf-8"))
            mode = state.get("video_geometry_mode", "auto")
            if mode in {"auto", "raw", "rectified"}:
                self._video_geometry_mode = mode
            output = Path(state["output_dir"])
            if state.get("kind") == "yolo" and output.is_dir():
                self._resume_dataset = (output, state.get("current_name"))
                return None, None
            if self.review_project_error(output, validate_records=True) is None:
                return output, state.get("current_name")
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return super().previous_review_project()

    def remember_review_project(self):
        if self.active_output is not None:
            try:
                replace_project_config(EASY_SESSION, {"output_dir": str(self.active_output),
                                                      "current_name": self.current_name,
                                                      "kind": "yolo" if self._yolo_dataset else "review",
                                                      "video_geometry_mode": self._video_geometry_mode})
            except OSError:
                pass

    def open_settings(self):
        self.dataset_settings_banner.setVisible(self._yolo_dataset is not None)
        self.settings_dialog.show()

    def new_detection_project(self):
        if self.dataset_busy():
            self.status.setText("请先停止当前任务，再新建检测项目。")
            return False
        self.remember_review_project()
        stem = "gray_glove_project_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        output = ROOT_DIR / "data" / stem
        index = 2
        while output.exists() or output.with_name(output.name + "_undistorted").exists():
            output = ROOT_DIR / "data" / f"{stem}_{index}"
            index += 1
        self._yolo_dataset = None
        self._resume_dataset = None
        self._loaded_video_folder = None
        self._review_recheck_names = None
        self._review_recheck_failures = []
        self._deleted_during_run.clear()
        self.video_list.clear()
        self.video_edit.clear()
        self.input_edit.clear()
        self.output_edit.setText(str(output))
        self.roi = [0., 0., 1., 1.]
        self.show_roi()
        self.load_project()
        self.search_edit.clear()
        self.review_filter.setChecked(False)
        self.status_filter.setCurrentIndex(self.status_filter.findData("all"))
        self.follow_check.setChecked(True)
        self.set_config_controls(True)
        self.output_edit.setEnabled(True)
        self.path_buttons[1].setEnabled(True)
        self.export_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.format_combo.setToolTip("导出框格式")
        self.dataset_location.hide()
        self.dataset_settings_banner.hide()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress_detail.setText("新项目尚未开始检测")
        self.video_progress.setText("请选择视频或 MCAP，也可在图片页选择图片目录。")
        self.apply_filter()
        self.update_counts()
        self.status.setText(f"已切换到新检测项目，结果保存到：{output}")
        self.settings_tabs.setCurrentIndex(1)
        self.open_settings()
        return True

    def toggle_focus_view(self):
        # The compact editor already gives the image most of the window.
        # Detection should return to it instead of opening the settings dialog.
        self._focus_view = True
        if hasattr(self, "settings_dialog"):
            self.settings_dialog.hide()
        if self.canvas is not None:
            self.canvas.setFocus()

    def change_video_geometry(self):
        self._video_geometry_mode = self.video_geometry_combo.currentData()
        self.update_video_geometry_controls()
        self.remember_review_project()

    def update_video_geometry_controls(self):
        if not hasattr(self, "video_geometry_combo"):
            return
        available = (not self.config_locked and self._yolo_dataset is None
                     and self.export_thread is None
                     and not (self.video_thread and self.video_thread.isRunning()))
        self.video_geometry_combo.setEnabled(available)
        needs_calibration = self._video_geometry_mode != "raw"
        for control in (self.calibration_edit, self.calibration_button, self.balance_spin):
            control.setEnabled(available and needs_calibration)
        if not needs_calibration:
            text = ("原图抽帧 → SAM3 原图检测 → 原图复核 / 导出；无需标定。"
                    "若当前为去畸变项目，按自动检测将从原录像另抽原图，保留原项目。")
        else:
            text = ("四目 / MCAP：原图分割 → 掩码去畸变 → 去畸变图复核 / 导出。"
                    "需要对应相机标定；原图与去畸变结果分开保存。")
        self.video_geometry_hint.setText(text)
        self.import_button.setToolTip(text)

    def set_config_controls(self, enabled):
        super().set_config_controls(enabled)
        self.update_video_geometry_controls()
        for name in ("new_project_button", "dataset_new_project_button"):
            if hasattr(self, name):
                getattr(self, name).setEnabled(enabled and not self.dataset_busy())

    def should_undistort_videos(self, videos):
        if self._video_geometry_mode == "raw":
            return False
        supported = super().should_undistort_videos(videos)
        if self._video_geometry_mode == "rectified" and not supported:
            raise ValueError("去畸变模式需要可识别的四目录像或 MCAP 及相机标定；普通视频请选择原图模式。")
        return supported

    def start_detection(self):
        if self._yolo_dataset is not None or self.dataset_busy():
            return
        if self._video_geometry_mode == "raw" and self.input_edit.text().strip():
            try:
                projected = frame_projection_sources(self.source_dir())
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "无法开始", str(exc))
                return
            if projected:
                if self._loaded_video_folder is None or not self.video_list.count():
                    QMessageBox.warning(self, "需要原始录像", "当前图片已经去畸变。请在视频设置中选择原始录像，重新按原图抽帧；不能直接沿用当前图片与框坐标。")
                    return
                self.import_video(detect_after=True)
                return
        return super().start_detection()

    def recheck_review_images(self):
        if self._yolo_dataset is not None:
            return
        if self._video_geometry_mode == "raw" and self.input_edit.text().strip():
            try:
                projected = frame_projection_sources(self.source_dir())
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "无法重检", str(exc))
                return
            if projected:
                QMessageBox.information(self, "先切换为原图项目", "当前仍是去畸变项目。请返回标注并点击自动检测，先从原录像另抽原图，再复核或重检。")
                return
        return super().recheck_review_images()

    def show_record(self, name):
        if self.canvas is None:
            return super().show_record(name)
        record = self.records.get(name)
        if record is None or self.active_source is None:
            return False
        path = self.active_source / name
        self.current_name = name
        self.canvas.load(name, path, record)
        if self.canvas.picture.isNull():
            self.status.setText(f"图片不存在或无法读取：{path}")
        self._loading_objects = True
        self.object_table.setRowCount(len(record.get("objects", [])))
        for index, obj in enumerate(record.get("objects", [])):
            for column, value in enumerate((str(index + 1), self.display_class_name(obj["class_id"], record),
                                             ("—" if self._yolo_dataset else f"{obj.get('score', 1):.2f}"))):
                self.object_table.setItem(index, column, QTableWidgetItem(value))
        self._loading_objects = False
        if self.canvas.selected >= 0:
            self.object_table.selectRow(self.canvas.selected)
        self.preview_title.setText(f"{name}\n{STATUS_TEXT.get(record['status'], record['status'])} · {record.get('reason', '')}")
        return not self.canvas.picture.isNull()

    def select_object(self):
        super().select_object()
        if self.canvas is not None and not self._loading_objects:
            self.canvas.selected = self.object_table.currentRow()
            self.canvas.paint_boxes()

    def clear_preview(self):
        super().clear_preview()
        if self.canvas is not None:
            self.canvas.scene().clear()
            self.canvas.overlays = []
            self.canvas.picture = QPixmap()
            self.canvas.name = None
            self.canvas.objects = []
            self.canvas.candidates = []
            self.canvas.gesture = None

    def add_object(self):
        if self.canvas is not None and self.current_record() is not None:
            self.canvas.set_draw()

    def pick_side(self, side):
        if self.canvas is not None and self.canvas.drawing:
            self._loading_objects = True
            self.object_class_combo.setCurrentIndex(side)
            self._loading_objects = False
        elif self.object_table.currentRow() >= 0:
            self.assign_current_side(side)
        else:
            self.object_class_combo.setCurrentIndex(side)

    def commit_polygon(self, index, polygon):
        record = self.current_record()
        if record is None:
            self.canvas.cancel()
            return False
        if not polygon_in_region(polygon, self.roi, self.canvas.picture.width(), self.canvas.picture.height()):
            self.status.setText("框须位于检测区域内；本次调整未保存。")
            self.canvas.cancel()
            return False
        objects = copy.deepcopy(record.get("objects", []))
        if index < 0:
            if len(objects) >= 2:
                self.status.setText("每张最多两只手套，请先删除多余框。")
                self.canvas.cancel()
                return False
            objects.append({"class_id": self.object_class_combo.currentIndex(), "polygon": polygon, "score": 1.0})
            index = len(objects) - 1
        else:
            objects[index]["polygon"] = polygon
        self.follow_check.setChecked(False)
        ok = self.save_review({**record, "objects": objects, "status": "confirmed",
                               "reason": "人工添加或修正目标"})
        if ok:
            self.object_table.selectRow(index)
            if self._yolo_dataset is None:
                self.status.setText("已自动保存 · Ctrl+Z 撤销；1 / 2 确认左右类别；Space 确认并继续。")
        else:
            self.canvas.cancel()
        return ok


def main():
    plugin_dir = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_dir
    QCoreApplication.setLibraryPaths([plugin_dir])
    app = QApplication(sys.argv)
    window = EasyGrayGloveApp()
    window.show()
    return app.exec_()
