"""Independent generic annotation window backed by YOLO detection weights."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import cv2
from PyQt5.QtCore import QCoreApplication, QLibraryInfo, Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QLabel, QLineEdit, QMessageBox, QPushButton,
    QShortcut,
)
from ultralytics import YOLO

from .app import torch
from .app_generic import GenericAnnotatorApp, GenericThread
from .generic_core import detect_generic
from .project_paths import ROOT_DIR, resolve_path


DEFAULT_YOLO_MODEL = ROOT_DIR / "models" / "best_top_barcode_0929.pt"
DEFAULT_YOLO_OUTPUT = ROOT_DIR / "data" / "generic_yolo_annotations"


def model_classes(model_path: Path) -> list[dict]:
    """Read the fixed class IDs in a YOLO detection checkpoint."""
    if not model_path.is_file():
        raise FileNotFoundError(f"找不到 YOLO 检测权重: {model_path}")
    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        raise ValueError(f"无法加载 YOLO 检测权重 {model_path}: {exc}") from exc
    if model.task != "detect":
        raise ValueError(f"仅支持 YOLO 普通目标检测权重；当前权重任务为 {model.task}")
    names = model.names
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    elif isinstance(names, dict):
        names = {int(index): name for index, name in names.items()}
    else:
        raise ValueError("YOLO 权重没有可读取的类别名称")
    if not names or sorted(names) != list(range(len(names))):
        raise ValueError("YOLO 权重的类别编号不是从 0 连续排列")
    return [{"name": str(names[index]), "prompt": str(names[index])}
            for index in range(len(names))]


class GenericYOLODetector:
    """Expose YOLO boxes through the existing generic selection pipeline."""

    def __init__(self, model_path: str, device: str, conf: float):
        self.model_path = str(Path(model_path).resolve())
        self.model = YOLO(self.model_path)
        if self.model.task != "detect":
            raise ValueError(f"仅支持 YOLO 普通目标检测权重；当前权重任务为 {self.model.task}")
        self.device = device
        self.conf = float(conf)

    def detect_many(self, image_bgr, _prompts: list[str]) -> list[dict]:
        results = self.model.predict(source=image_bgr, device=self.device,
                                     conf=self.conf, verbose=False)
        if not results or results[0].boxes is None:
            return []
        boxes = results[0].boxes
        coordinates = boxes.xyxy.detach().cpu().numpy().tolist()
        scores = boxes.conf.detach().cpu().numpy().tolist()
        class_ids = boxes.cls.detach().cpu().numpy().astype(int).tolist()
        return [{"class_id": class_id, "box": box, "score": score}
                for class_id, box, score in zip(class_ids, coordinates, scores)]


class GenericYOLOThread(GenericThread):
    def run(self):
        try:
            todo = [path for path in self.files if path.name not in self.done_names]
            total = len(self.files)
            completed = total - len(todo)
            self.progress_signal.emit(completed, total)
            if not todo:
                self.message_signal.emit("所有抽中图片已有检测结果；可继续复核或导出。")
                return
            device = "cuda:0" if torch is not None and torch.cuda.is_available() else "cpu"
            model_path = str(Path(self.config["model_path"]).resolve())
            if (self.segmenter is None or self.segmenter.device != device or
                    self.segmenter.model_path != model_path):
                self.message_signal.emit(f"正在加载 YOLO 检测权重（{device}），本次检测 {len(todo)} 张...")
                self.segmenter = GenericYOLODetector(
                    model_path, device, self.config["confidence"])
            else:
                self.segmenter.conf = float(self.config["confidence"])
                self.message_signal.emit(f"复用已加载的 YOLO 权重（{device}），本次检测 {len(todo)} 张...")
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
            self.message_signal.emit(
                "检测已停止。" if not self.running else "检测完成，请复核待处理图片。")
        except Exception as exc:
            self.message_signal.emit(f"YOLO 检测无法启动: {exc}")


class GenericYOLOAnnotatorApp(GenericAnnotatorApp):
    def __init__(self):
        self._building_yolo_window = True
        self._class_model_path = None
        self._class_model_mtime = None
        super().__init__()
        self._building_yolo_window = False
        self.setWindowTitle("通用 YOLO 检测标注与人工复核")
        self.model_path_edit = QLineEdit(str(DEFAULT_YOLO_MODEL))
        self.model_path_edit.editingFinished.connect(self.sync_model_classes)
        self.model_button = QPushButton("选择 YOLO 权重 [Alt+M]")
        self.model_button.clicked.connect(self.choose_model)
        self.model_shortcut = QShortcut(QKeySequence("Alt+M"), self)
        self.model_shortcut.setContext(Qt.WindowShortcut)
        self.model_shortcut.activated.connect(self.model_button.click)
        settings = self.centralWidget().layout().itemAt(0).widget()
        grid = settings.layout()
        grid.addWidget(QLabel("检测模型权重"), 8, 0)
        grid.addWidget(self.model_path_edit, 8, 1, 1, 3)
        grid.addWidget(self.model_button, 8, 4)
        self.class_table.setHorizontalHeaderLabels(["模型类别名", "模型类别"])
        self.class_table.setColumnHidden(1, True)
        self.class_table.setToolTip("YOLO 类别由权重文件决定；检测提示词不参与推理。")
        self.confidence_spin.setToolTip("YOLO 最低检测置信度；提高可减少误检，也可能漏检。")
        self.output_edit.setText(str(DEFAULT_YOLO_OUTPUT))
        self.load_project()

    def set_config_controls(self, enabled: bool):
        super().set_config_controls(enabled)
        self.class_table.setEnabled(False)
        for button in self.class_buttons:
            button.setEnabled(False)
        if hasattr(self, "model_path_edit"):
            self.model_path_edit.setEnabled(enabled)
            self.model_button.setEnabled(enabled)

    def choose_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 YOLO 检测权重", self.model_path_edit.text(),
            "PyTorch 权重 (*.pt)")
        if path:
            self.model_path_edit.setText(path)
            self.sync_model_classes()

    def sync_model_classes(self, show_warning: bool = True):
        try:
            path = resolve_path(self.model_path_edit.text().strip())
            mtime = path.stat().st_mtime_ns
            if (self._class_model_path != path or
                    self._class_model_mtime != mtime):
                classes = model_classes(path)
                self.set_classes(classes)
                self._class_model_path = path
                self._class_model_mtime = mtime
            self.model_path_edit.setText(str(path))
        except (OSError, ValueError, RuntimeError) as exc:
            if show_warning:
                QMessageBox.warning(self, "无法读取模型", str(exc))
            else:
                raise

    def config(self) -> dict:
        config = super().config()
        config.update(model_type="yolo_detect",
                      model_path=str(resolve_path(self.model_path_edit.text().strip())))
        return config

    def load_project(self):
        if self._building_yolo_window:
            return
        try:
            project_path = self.output_dir() / "project.json"
            if project_path.exists():
                saved = json.loads(project_path.read_text(encoding="utf-8"))
                if saved.get("model_type") != "yolo_detect":
                    raise ValueError("结果目录属于其他标注项目，请为 YOLO 选择单独的结果目录")
                self.model_path_edit.setText(saved["model_path"])
            super().load_project()
            self.sync_model_classes(show_warning=False)
            self.set_config_controls(True)
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            self.status.setText(f"读取 YOLO 项目失败: {exc}")

    def ensure_project(self, allow_reconfigure: bool = False) -> dict:
        path = self.output_dir() / "project.json"
        if path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("model_type") != "yolo_detect":
                raise ValueError("结果目录属于其他标注项目，请为 YOLO 选择单独的结果目录")
        self.sync_model_classes(show_warning=False)
        return super().ensure_project(allow_reconfigure=allow_reconfigure)

    def validate_model_ready(self):
        self.sync_model_classes(show_warning=False)

    def make_detection_thread(self, files, config, done_names):
        return GenericYOLOThread(files, config, done_names, self.segmenter)


def main() -> int:
    pyqt_plugin_root = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyqt_plugin_root
    QCoreApplication.setLibraryPaths([pyqt_plugin_root])
    app = QApplication(sys.argv)
    window = GenericYOLOAnnotatorApp()
    window.show()
    return app.exec_()
