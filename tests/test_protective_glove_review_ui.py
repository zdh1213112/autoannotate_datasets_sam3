"""Exercise actual Qt key events, review persistence and background-result races."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5.QtCore import QCoreApplication, QEvent, QLibraryInfo, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QDialog, QPushButton

from auto_obb_annotator import app_protective_gloves_review as ui
from auto_obb_annotator.protective_glove_review_core import load_records


class GloveReviewUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plugin_dir = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_dir
        QCoreApplication.setLibraryPaths([plugin_dir])
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "source"
        self.output = Path(self.temp.name) / "output"
        self.source.mkdir()
        with patch.object(ui, "DEFAULT_INPUT", self.source), patch.object(
                ui, "DEFAULT_OUTPUT", self.output):
            self.window = ui.ProtectiveGloveReviewApp()
        self.window.ensure_project()
        self.obj = {"class_id": 0, "score": .95, "side_source": "semantic",
                    "polygon": [[20, 25], [60, 25], [60, 95], [20, 95]]}
        for name, status in (("a.png", "review"), ("b.png", "auto"),
                             ("c.png", "review"), ("d.png", "empty_confirmed")):
            cv2.imwrite(str(self.source / name), np.zeros((120, 240, 3), np.uint8))
            record = {"name": name, "status": status,
                      "objects": [] if status == "empty_confirmed" else [self.obj]}
            self.window.records[name] = record
            self.window.update_table(record)
        self.window.table.selectRow(0)
        self.window.show_record("a.png")
        self.activate(self.window, self.window.table)

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.qt.sendPostedEvents(None, QEvent.DeferredDelete)
        self.qt.processEvents()

    def activate(self, window, widget):
        window.show()
        window.activateWindow()
        widget.setFocus()
        self.qt.processEvents()
        QTest.qWait(10)

    def test_space_persists_then_skips_nonreview_rows_and_stops_at_end(self):
        self.window.follow_check.setChecked(True)
        QTest.keyClick(self.window.table, Qt.Key_Space)
        self.assertEqual(self.window.current_name, "c.png")
        self.assertFalse(self.window.follow_check.isChecked())
        saved = load_records(self.output / "review.jsonl")
        self.assertEqual(saved["a.png"]["status"], "confirmed")
        self.assertEqual(self.window.records["b.png"]["status"], "auto")
        QTest.keyClick(self.window.table, Qt.Key_Space)
        self.assertEqual(self.window.current_name, "c.png")
        self.assertIn("没有待复核", self.window.status.text())
        self.assertEqual(load_records(self.output / "review.jsonl")["c.png"]["status"],
                         "confirmed")

    def test_confirm_button_wraps_to_review_while_filtered(self):
        self.window.review_filter.setChecked(True)
        self.window.show_record("c.png")
        self.window.review_actions["confirm_current"].click()
        self.assertEqual(self.window.current_name, "a.png")
        self.assertTrue(self.window.table.isRowHidden(self.window.row_for_name["c.png"]))

    def test_failed_save_keeps_current_review(self):
        with patch.object(ui, "append_record", side_effect=OSError("disk full")), \
                patch.object(ui.QMessageBox, "warning"):
            QTest.keyClick(self.window.table, Qt.Key_Space)
        self.assertEqual(self.window.current_name, "a.png")
        self.assertEqual(self.window.records["a.png"]["status"], "review")

    def test_queued_inference_does_not_overwrite_manual_reviews(self):
        self.window.confirm_current()
        self.window.receive_result({"name": "a.png", "objects": [], "status": "review"})
        self.window.receive_result({"name": "d.png", "objects": [self.obj], "status": "auto"})
        self.assertEqual(self.window.records["a.png"]["status"], "confirmed")
        self.assertEqual(self.window.records["d.png"]["status"], "empty_confirmed")
        self.assertEqual(self.window.current_name, "c.png")

    def test_shortcut_respects_disabled_button_and_typing(self):
        start = Mock()
        self.window.start_button.clicked.disconnect()
        self.window.start_button.clicked.connect(start)
        self.window.start_button.setEnabled(False)
        QTest.keyClick(self.window.table, Qt.Key_F5)
        start.assert_not_called()
        self.window.start_button.setEnabled(True)
        self.window.input_edit.setFocus()
        self.window.input_edit.selectAll()
        QTest.keyClicks(self.window.input_edit, "e f r q n ")
        self.assertEqual(self.window.input_edit.text(), "e f r q n ")
        self.assertEqual(self.window.records["a.png"]["status"], "review")
        QTest.keyClick(self.window.input_edit, Qt.Key_F5)
        start.assert_called_once()

    def test_every_action_button_has_a_unique_nonrepeating_shortcut(self):
        dialog = ui.GloveObjectsDialog(self.source / "a.png", [self.obj], parent=self.window)
        box = ui.GloveBoxDialog(self.source / "a.png", "edit", self.obj["polygon"],
                               parent=dialog)
        for window in (self.window, dialog, box):
            buttons = [b for b in window.findChildren(QPushButton) if b.window() is window]
            for button in buttons:
                self.assertRegex(button.text(), r"\[[^\]]+\]$", button.text())
            bindings = window.button_shortcuts.bindings
            keys = [key for _, key in bindings]
            self.assertEqual(len(keys), len(set(keys)))
            self.assertTrue(all(not shortcut.autoRepeat() for shortcut, _ in bindings))
        box.deleteLater()
        dialog.deleteLater()

    def test_nested_editor_shortcuts_do_not_confirm_main_image(self):
        dialog = ui.GloveObjectsDialog(self.source / "a.png", [self.obj], parent=self.window)
        dialog.setModal(True)
        self.activate(dialog, dialog.table)
        QTest.keyClick(dialog.table, Qt.Key_X)
        self.assertEqual(dialog.objects[0]["class_id"], 1)
        box = ui.GloveBoxDialog(self.source / "a.png", "edit", self.obj["polygon"],
                               parent=dialog)
        box.setModal(True)
        self.activate(box, box.canvas)
        QTest.keyClick(box.canvas, Qt.Key_Space)
        self.assertEqual(self.window.records["a.png"]["status"], "review")
        self.assertEqual(len(box.canvas.rois), 1)
        QTest.keyClick(box.canvas, Qt.Key_S, Qt.ControlModifier)
        self.assertEqual(box.result(), QDialog.Accepted)
        self.activate(dialog, dialog.table)
        QTest.keyClick(dialog.table, Qt.Key_S, Qt.ControlModifier)
        self.assertEqual(dialog.result(), QDialog.Accepted)
        self.activate(self.window, self.window.table)
        QTest.keyClick(self.window.table, Qt.Key_Space)
        self.assertEqual(self.window.current_name, "c.png")
        box.deleteLater()
        dialog.deleteLater()


if __name__ == "__main__":
    unittest.main()
