"""User-facing workflows in the separate optimized YOLO window."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QCoreApplication, QEvent, QLibraryInfo
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QScrollArea

from auto_obb_annotator import app_generic_yolo_optimized as ui
from auto_obb_annotator import app_generic_yolo as original


class OptimizedYoloUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plugin_dir = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_dir
        QCoreApplication.setLibraryPaths([plugin_dir])
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "optimized_output"
        with patch.object(ui, "DEFAULT_YOLO_OUTPUT", self.output):
            self.window = ui.OptimizedYOLOAnnotatorApp()
        self.window.show()
        self.qt.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.qt.sendPostedEvents(None, QEvent.DeferredDelete)
        self.qt.processEvents()

    def test_separate_output_hbb_default_and_scrollable_window(self):
        self.assertNotEqual(ui.DEFAULT_YOLO_OUTPUT, original.DEFAULT_YOLO_OUTPUT)
        self.assertEqual(self.window.output_edit.text(), str(self.output))
        self.assertEqual(self.window.format_combo.currentData(), "hbb")
        self.assertIsInstance(self.window.centralWidget(), QScrollArea)
        self.assertLessEqual(self.window.height(), 960)

    def test_search_status_and_review_filters_work_together(self):
        records = (("Barcode_A.png", "review"), ("barcode_B.png", "auto"),
                   ("other.png", "review"))
        for name, status in records:
            record = {"name": name, "status": status, "objects": [],
                      "candidates": [], "candidate_count": 0}
            self.window.records[name] = record
            self.window.update_table(record)
        self.window.search_edit.setText("BARCODE")
        self.assertEqual(self.window.visible_count.text(), "显示 2/3 张")
        self.window.status_filter.setCurrentIndex(
            self.window.status_filter.findData("review"))
        self.assertEqual(self.window.visible_count.text(), "显示 1/3 张")
        self.assertFalse(self.window.table.isRowHidden(
            self.window.row_for_name["Barcode_A.png"]))
        self.window.status_filter.setCurrentIndex(0)
        self.window.review_filter.setChecked(True)
        self.assertEqual(self.window.visible_count.text(), "显示 1/3 张")

    def test_progress_and_counts_do_not_erase_worker_message(self):
        self.window.status.setText("正在加载 YOLO 检测权重")
        self.window.update_progress(2, 10)
        self.window.update_progress(3, 10)
        self.window.update_counts()
        self.assertEqual(self.window.status.text(), "正在加载 YOLO 检测权重")
        self.assertIn("3/10", self.window.progress_detail.text())
        self.assertIn("预计剩余", self.window.progress_detail.text())

    def test_large_preview_supports_native_size_and_zoom(self):
        dialog = ui.LargePreviewDialog(QPixmap(200, 100), "sample.png", self.window)
        self.assertEqual(dialog.zoom_spin.value(), 100)
        self.assertEqual(dialog.image_label.pixmap().size(), QPixmap(200, 100).size())
        dialog.zoom_spin.setValue(200)
        self.assertEqual(dialog.image_label.pixmap().width(), 400)
        dialog.close()
