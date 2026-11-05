"""YOLO checkpoint integration for the independent generic annotation entrypoint."""

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5.QtCore import QCoreApplication, QEvent, QLibraryInfo
from PyQt5.QtWidgets import QApplication

from auto_obb_annotator import app_generic_yolo as yolo_ui


class GenericYoloTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plugin_dir = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_dir
        QCoreApplication.setLibraryPaths([plugin_dir])
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "images"
        self.source.mkdir()
        for name in ("a.png", "b.png"):
            cv2.imwrite(str(self.source / name), np.zeros((120, 240, 3), np.uint8))
        with patch.object(yolo_ui, "DEFAULT_YOLO_OUTPUT", self.root / "output"):
            self.window = yolo_ui.GenericYOLOAnnotatorApp()
        self.window.input_edit.setText(str(self.source))

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.qt.sendPostedEvents(None, QEvent.DeferredDelete)
        self.qt.processEvents()

    def test_supplied_weight_is_a_barcode_detector(self):
        self.assertEqual(yolo_ui.model_classes(yolo_ui.DEFAULT_YOLO_MODEL),
                         [{"name": "barcode", "prompt": "barcode"}])
        detector = yolo_ui.GenericYOLODetector(
            str(yolo_ui.DEFAULT_YOLO_MODEL), "cpu", 0.25)
        result = detector.detect_many(np.zeros((96, 128, 3), np.uint8), [])
        self.assertIsInstance(result, list)
        self.assertEqual(detector.model.task, "detect")

    def test_project_uses_weight_classes_and_separate_output(self):
        config = self.window.ensure_project()
        self.assertEqual(config["model_type"], "yolo_detect")
        self.assertEqual(config["model_path"], str(yolo_ui.DEFAULT_YOLO_MODEL))
        self.assertEqual(config["classes"], [{"name": "barcode", "prompt": "barcode"}])
        self.assertFalse(self.window.class_table.isEnabled())
        self.assertEqual(json.loads((self.root / "output" / "project.json").read_text()),
                         config)

    def test_worker_uses_yolo_boxes_for_existing_selection(self):
        config = self.window.ensure_project()
        config["count_mode"] = "fixed"
        config["target_count"] = 1
        results = []

        class FakeDetector:
            def __init__(self, model_path, device, conf):
                self.model_path = model_path
                self.device = device
                self.conf = conf

            def detect_many(self, _image, _prompts):
                return [{"class_id": 0, "box": [20, 20, 60, 70], "score": 0.92}]

        with patch.object(yolo_ui, "torch", None), \
                patch.object(yolo_ui, "GenericYOLODetector", FakeDetector):
            worker = yolo_ui.GenericYOLOThread(
                [self.source / "a.png"], config, set())
            worker.result_signal.connect(results.append)
            worker.run()
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "auto")
        self.assertEqual(results[0]["objects"][0]["class_id"], 0)
        self.assertEqual(results[0]["objects"][0]["score"], 0.92)
        self.assertEqual(len(results[0]["candidates"]), 1)

    def test_selecting_another_yolo_weight_updates_project_config(self):
        self.window.ensure_project()
        alternate = self.root / "second.pt"
        shutil.copy2(yolo_ui.DEFAULT_YOLO_MODEL, alternate)
        self.window.model_path_edit.setText(str(alternate))
        config = self.window.ensure_project(allow_reconfigure=True)
        self.assertEqual(config["model_path"], str(alternate))
        self.assertEqual(self.window.classes()[0]["name"], "barcode")
        self.assertEqual(len(list((self.root / "output").glob("backup_before_config_*"))), 1)

    def test_entrypoint_restores_pyqt_plugin_path_after_cv2_import(self):
        plugin_dir = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        with patch.dict(os.environ, {"QT_QPA_PLATFORM_PLUGIN_PATH": "/tmp/cv2/qt/plugins"}), \
                patch.object(yolo_ui.QCoreApplication, "setLibraryPaths") as set_paths, \
                patch.object(yolo_ui, "QApplication") as application, \
                patch.object(yolo_ui, "GenericYOLOAnnotatorApp") as window:
            application.return_value.exec_.return_value = 0
            self.assertEqual(yolo_ui.main(), 0)
            self.assertEqual(os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"], plugin_dir)
            set_paths.assert_called_once_with([plugin_dir])
            window.return_value.show.assert_called_once_with()
