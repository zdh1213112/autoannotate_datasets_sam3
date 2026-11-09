"""Exercise generic annotator shortcuts through real Qt key events."""

import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5.QtCore import QCoreApplication, QEvent, QLibraryInfo, QPoint, Qt
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QDialog, QPushButton

from auto_obb_annotator import app_generic as ui
from auto_obb_annotator.barcode_core import append_record, load_records


class GenericShortcutUITests(unittest.TestCase):
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
        with patch.object(ui, "DEFAULT_OUTPUT", self.output):
            self.window = ui.GenericAnnotatorApp()
        self.window.input_edit.setText(str(self.source))
        self.window.ensure_project()
        obj = {"class_id": 0, "score": 0.95, "candidate_id": 0,
               "polygon": [[20, 25], [60, 25], [60, 95], [20, 95]]}
        for name, status in (("a.png", "review"), ("b.png", "auto"),
                             ("c.png", "review")):
            cv2.imwrite(str(self.source / name), np.zeros((120, 240, 3), np.uint8))
            record = {"name": name, "status": status, "objects": [obj],
                      "candidates": [{**obj, "selected": True}], "candidate_count": 1}
            self.window.records[name] = record
            self.window.update_table(record)
        self.window.table.selectRow(0)
        self.window.show_record("a.png")
        self.window.show()
        self.window.activateWindow()
        self.window.table.setFocus()
        self.qt.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.qt.sendPostedEvents(None, QEvent.DeferredDelete)
        self.qt.processEvents()

    def test_space_saves_and_moves_to_next_review(self):
        QTest.keyClick(self.window.table, Qt.Key_Space)
        self.assertEqual(self.window.current_name, "c.png")
        self.assertEqual(load_records(self.output / "review.jsonl")["a.png"]["status"],
                         "confirmed")
        self.assertFalse(self.window.follow_check.isChecked())

    def test_space_after_confirming_empty_moves_to_next_review(self):
        QTest.keyClick(self.window.table, Qt.Key_Q)
        self.assertEqual(self.window.records["a.png"]["status"], "empty_confirmed")
        saved_before = (self.output / "review.jsonl").read_text()
        with patch.object(ui.QMessageBox, "information") as information:
            QTest.keyClick(self.window.table, Qt.Key_Space)
        information.assert_not_called()
        self.assertEqual(self.window.current_name, "c.png")
        self.assertEqual((self.output / "review.jsonl").read_text(), saved_before)

    def test_next_visible_and_typing_suppression(self):
        QTest.keyClick(self.window.table, Qt.Key_N)
        self.assertEqual(self.window.current_name, "b.png")
        self.window.input_edit.setEnabled(True)
        self.window.input_edit.setFocus()
        self.window.input_edit.selectAll()
        QTest.keyClicks(self.window.input_edit, "e f r q n ")
        QTest.keyClick(self.window.input_edit, Qt.Key_N, Qt.ShiftModifier)
        QTest.keyClick(self.window.input_edit, Qt.Key_D, Qt.ShiftModifier)
        self.assertEqual(self.window.input_edit.text().lower(), "e f r q n nd")
        self.assertEqual(self.window.records["b.png"]["status"], "auto")
        self.assertEqual(self.window.current_name, "b.png")

    def test_previous_review_wraps_and_skips_other_statuses(self):
        QTest.keyClick(self.window.table, Qt.Key_D, Qt.ShiftModifier)
        self.assertEqual(self.window.current_name, "c.png")
        QTest.keyClick(self.window.table, Qt.Key_D, Qt.ShiftModifier)
        self.assertEqual(self.window.current_name, "a.png")
        QTest.keyClick(self.window.table, Qt.Key_D)
        self.assertEqual(self.window.current_name, "c.png")

    def test_previous_image_stops_at_start_and_skips_hidden_rows(self):
        QTest.keyClick(self.window.table, Qt.Key_N, Qt.ShiftModifier)
        self.assertEqual(self.window.current_name, "a.png")
        self.assertEqual(self.window.status.text(), "已经是第一张图片。")
        QTest.keyClick(self.window.table, Qt.Key_N)
        QTest.keyClick(self.window.table, Qt.Key_N)
        self.assertEqual(self.window.current_name, "c.png")
        QTest.keyClick(self.window.table, Qt.Key_N, Qt.ShiftModifier)
        self.assertEqual(self.window.current_name, "b.png")
        self.window.review_filter.setChecked(True)
        self.window.table.selectRow(self.window.row_for_name["c.png"])
        self.window.show_record("c.png")
        QTest.keyClick(self.window.table, Qt.Key_N, Qt.ShiftModifier)
        self.assertEqual(self.window.current_name, "a.png")

    def test_shortcut_respects_disabled_button_and_class_table_focus(self):
        start = Mock()
        self.window.start_button.clicked.disconnect()
        self.window.start_button.clicked.connect(start)
        self.window.start_button.setEnabled(False)
        QTest.keyClick(self.window.table, Qt.Key_F5)
        start.assert_not_called()
        self.window.object_table.selectRow(0)
        self.window.class_table.setEnabled(True)
        self.window.class_table.setFocus()
        self.qt.processEvents()
        QTest.keyClick(self.window.class_table, Qt.Key_Delete)
        self.assertEqual(len(self.window.records["a.png"]["objects"]), 1)

    def test_every_button_has_unique_shortcut(self):
        buttons = [button for button in self.window.findChildren(QPushButton)
                   if button.window() is self.window]
        for button in buttons:
            self.assertRegex(button.text(), r"\[[^\]]+\]$")
        keys = [key for _, key in self.window.button_shortcuts.bindings]
        self.assertEqual(len(keys), len(set(keys)))

    def test_failed_save_does_not_advance(self):
        with patch.object(ui, "append_record", side_effect=OSError("disk full")), \
                patch.object(ui.QMessageBox, "warning"):
            QTest.keyClick(self.window.table, Qt.Key_Space)
        self.assertEqual(self.window.current_name, "a.png")
        self.assertEqual(self.window.records["a.png"]["status"], "review")

    def test_preview_click_toggles_labels_and_persists_all_candidates(self):
        record = self.window.records["a.png"]
        candidate = {"class_id": 0, "score": 0.88,
                     "polygon": [[130, 25], [170, 25], [170, 95], [130, 95]],
                     "selected": False}
        record["candidates"].append(candidate)
        record["candidate_count"] = 2
        self.window.show_record("a.png")

        def border_color(x):
            return self.window.preview_pixmap.toImage().pixelColor(x, 60).name()

        def click_source(x, y):
            shown = self.window.preview.pixmap()
            offset_x = (self.window.preview.width() - shown.width()) / 2
            offset_y = (self.window.preview.height() - shown.height()) / 2
            point = QPoint(round(offset_x + x * shown.width() / 240),
                           round(offset_y + y * shown.height() / 120))
            QTest.mouseClick(self.window.preview, Qt.LeftButton, pos=point)

        self.assertEqual(border_color(130), "#b8860b")
        self.assertEqual(border_color(20), "#00c800")
        click_source(150, 60)
        self.assertEqual(len(self.window.records["a.png"]["objects"]), 2)
        self.assertEqual(border_color(130), "#00c800")
        self.assertTrue(load_records(self.output / "review.jsonl")["a.png"]
                        ["candidates"][1]["selected"])
        click_source(150, 60)
        self.assertEqual(len(self.window.records["a.png"]["objects"]), 1)
        self.assertEqual(border_color(130), "#b8860b")
        self.assertFalse(load_records(self.output / "review.jsonl")["a.png"]
                         ["candidates"][1]["selected"])
        click_source(40, 60)
        self.assertEqual(self.window.records["a.png"]["objects"], [])
        self.assertEqual(self.window.records["a.png"]["status"], "review")
        click_source(40, 60)
        self.assertEqual(len(self.window.records["a.png"]["objects"]), 1)
        self.assertEqual(self.window.records["a.png"]["status"], "confirmed")
        self.assertEqual(border_color(20), "#00c800")

    def test_clear_current_labels_persists_and_queues_redetection(self):
        self.window.review_filter.setChecked(True)
        self.window.table.setFocus()
        QTest.keyClick(self.window.table, Qt.Key_Delete, Qt.ControlModifier)
        self.assertEqual(self.window.records["a.png"]["status"], "unlabeled")
        self.assertEqual(self.window.records["a.png"]["objects"], [])
        self.assertEqual(self.window.object_table.rowCount(), 0)
        self.assertFalse(self.window.table.isRowHidden(self.window.row_for_name["a.png"]))
        saved = load_records(self.output / "review.jsonl")
        self.assertEqual(saved["a.png"]["status"], "unlabeled")
        self.assertEqual(saved["a.png"]["objects"], [])
        self.assertTrue(self.window.roi_button.isEnabled())
        self.assertTrue(self.window.confidence_spin.isEnabled())
        with patch.object(ui, "append_record") as save:
            self.window.receive_result({**saved["a.png"], "status": "auto",
                                        "objects": [{"class_id": 0}]})
        save.assert_not_called()
        append_record(self.output / "review.jsonl", self.window.records["b.png"])
        with patch.object(ui, "SAM3_MODEL", self.source / "a.png"), \
                patch.object(ui, "GenericThread") as thread_class:
            self.window.start_detection()
        self.assertNotIn("a.png", thread_class.call_args.args[2])
        self.assertIn("b.png", thread_class.call_args.args[2])
        restored = {**saved["a.png"], "status": "auto", "objects": [
            {"class_id": 0, "score": 0.95,
             "polygon": [[20, 25], [60, 25], [60, 95], [20, 95]]}]}
        self.window.receive_result(restored)
        self.assertEqual(self.window.object_table.rowCount(), 1)
        self.assertEqual(load_records(self.output / "review.jsonl")["a.png"]["status"],
                         "auto")

    def test_clear_shortcut_does_not_run_while_editing_a_path(self):
        self.window.input_edit.setEnabled(True)
        self.window.input_edit.setFocus()
        self.qt.processEvents()
        QTest.keyClick(self.window.input_edit, Qt.Key_Delete, Qt.ControlModifier)
        self.assertEqual(self.window.records["a.png"]["status"], "review")

    def test_clear_all_labels_in_directory_and_redetect(self):
        self.window.records["b.png"] = {
            **self.window.records["b.png"], "objects": [],
            "status": "empty_confirmed"}
        for record in self.window.records.values():
            append_record(self.output / "review.jsonl", record)
        cv2.imwrite(str(self.source / "d.png"), np.zeros((120, 240, 3), np.uint8))
        self.window.review_filter.setChecked(True)
        with patch.object(ui.QMessageBox, "question", return_value=ui.QMessageBox.Yes):
            self.window.table.setFocus()
            QTest.keyClick(self.window.table, Qt.Key_R,
                           Qt.ControlModifier | Qt.ShiftModifier)
        saved = load_records(self.output / "review.jsonl")
        self.assertEqual(set(saved), {"a.png", "b.png", "c.png", "d.png"})
        self.assertTrue(all(record["status"] == "unlabeled" and
                            record["objects"] == [] for record in saved.values()))
        self.assertEqual(len((self.output / "review.jsonl").read_text().splitlines()), 4)
        self.assertTrue(self.window.roi_button.isEnabled())
        self.assertTrue(self.window.review_reason_checks["near_edge"].isEnabled())
        self.assertEqual(self.window.object_table.rowCount(), 0)
        self.assertTrue(all(not self.window.table.isRowHidden(row)
                            for row in range(self.window.table.rowCount())))
        with patch.object(ui, "append_record") as append:
            self.window.receive_result({"name": "a.png", "status": "auto",
                                        "objects": [{"class_id": 0}]})
        append.assert_not_called()
        with patch.object(ui, "SAM3_MODEL", self.source / "a.png"), \
                patch.object(ui, "GenericThread") as thread_class:
            self.window.start_detection()
        self.assertEqual(thread_class.call_args.args[2], set())

    def test_clear_all_can_be_cancelled(self):
        with patch.object(ui.QMessageBox, "question", return_value=ui.QMessageBox.No):
            self.window.clear_all_button.click()
        self.assertEqual(len(self.window.records["a.png"]["objects"]), 1)
        self.assertFalse((self.output / "review.jsonl").exists())

    def test_clear_all_write_failure_keeps_saved_labels(self):
        for record in self.window.records.values():
            append_record(self.output / "review.jsonl", record)
        with patch.object(ui.QMessageBox, "question", return_value=ui.QMessageBox.Yes), \
                patch.object(ui, "replace_review_records", side_effect=OSError("disk full")), \
                patch.object(ui.QMessageBox, "warning"):
            self.window.clear_all_button.click()
        self.assertEqual(load_records(self.output / "review.jsonl")["a.png"]["status"],
                         "review")
        self.assertEqual(len(self.window.records["a.png"]["objects"]), 1)

    def test_changed_config_restarts_all_images_and_preserves_backup(self):
        for record in self.window.records.values():
            append_record(self.output / "review.jsonl", record)
        self.assertTrue(self.window.roi_button.isEnabled())
        self.assertTrue(self.window.class_table.isEnabled())
        self.window.count_combo.setCurrentIndex(1)
        self.window.count_spin.setValue(2)
        self.window.selection_combo.setCurrentIndex(1)
        self.window.sample_percent_spin.setValue(75)
        self.window.confidence_spin.setValue(0.4)
        self.window.class_table.item(0, 1).setText("another prompt")
        self.window.review_reason_checks["near_edge"].setChecked(False)
        self.window.roi = {"shape": "circle", "center": [0.5, 0.5], "radius": 0.4}
        self.window.show_roi()
        with patch.object(ui, "SAM3_MODEL", self.source / "a.png"), \
                patch.object(ui, "GenericThread") as thread_class:
            self.window.start_detection()
        config = json.loads((self.output / "project.json").read_text())
        self.assertEqual(config["count_mode"], "fixed")
        self.assertEqual(config["target_count"], 2)
        self.assertEqual(config["selection_rule"], "area")
        self.assertEqual(config["sample_percent"], 75)
        self.assertEqual(config["confidence"], 0.4)
        self.assertEqual(config["classes"][0]["prompt"], "another prompt")
        self.assertEqual(config["roi"], self.window.roi)
        self.assertNotIn("near_edge", config["review_reasons"])
        saved = load_records(self.output / "review.jsonl")
        self.assertTrue(all(record["status"] == "unlabeled" and
                            record["objects"] == [] for record in saved.values()))
        self.assertEqual(thread_class.call_args.args[2], set())
        self.assertFalse(self.window.roi_button.isEnabled())
        backups = list(self.output.glob("backup_before_config_*"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(load_records(backups[0] / "review.jsonl")["a.png"]["status"],
                         "review")
        self.assertEqual(json.loads((backups[0] / "project.json").read_text())
                         ["count_mode"], "all")
        self.window.detection_finished()
        self.assertTrue(self.window.roi_button.isEnabled())
        self.assertTrue(self.window.selection_combo.isEnabled())
        self.assertTrue(self.window.sample_percent_spin.isEnabled())
        self.assertTrue(self.window.review_reason_checks["near_edge"].isEnabled())

    def test_review_score_is_visible_and_updates_existing_results(self):
        self.assertEqual(self.window.review_score_spin.value(), 0.50)
        self.window.records["a.png"]["status"] = "confirmed"
        self.window.records["b.png"]["objects"][0]["score"] = 0.65
        self.window.records["c.png"]["objects"][0]["score"] = 0.65
        self.window.records["c.png"]["reason"] = "存在低置信度目标，请人工检查"
        for record in self.window.records.values():
            append_record(self.output / "review.jsonl", record)
        self.window.review_score_spin.setValue(0.70)
        config = self.window.ensure_project(allow_reconfigure=True)
        self.assertEqual(config["review_score"], 0.70)
        self.assertEqual(self.window.records["a.png"]["status"], "confirmed")
        self.assertEqual(self.window.records["b.png"]["status"], "review")
        self.assertIn("0.70", self.window.records["b.png"]["reason"])
        self.window.review_score_spin.setValue(0.50)
        self.window.ensure_project(allow_reconfigure=True)
        self.assertEqual(self.window.records["a.png"]["status"], "confirmed")
        self.assertEqual(self.window.records["b.png"]["status"], "auto")
        self.assertEqual(self.window.records["c.png"]["status"], "auto")
        self.window.load_project()
        self.assertEqual(self.window.review_score_spin.value(), 0.50)

    def test_legacy_review_score_migrates_without_clearing_labels(self):
        path = self.output / "project.json"
        config = json.loads(path.read_text())
        del config["review_score"]
        path.write_text(json.dumps(config))
        self.window.records["c.png"]["objects"][0]["score"] = 0.65
        self.window.records["c.png"]["reason"] = "存在低置信度目标，请人工检查"
        append_record(self.output / "review.jsonl", self.window.records["c.png"])
        self.window.load_project()
        self.window.ensure_project(allow_reconfigure=True)
        self.assertEqual(self.window.records["c.png"]["status"], "auto")
        self.assertEqual(json.loads(path.read_text())["review_score"], 0.50)
        self.assertEqual(len(self.window.records["c.png"]["objects"]), 1)

    def test_sample_percentage_limits_detection_and_persists_skips(self):
        for index in (3, 4):
            cv2.imwrite(str(self.source / f"{index}.png"),
                        np.zeros((120, 240, 3), np.uint8))
        self.window.sample_percent_spin.setValue(40)
        self.assertIn("预计随机抽取 2/5 张", self.window.sample_summary.text())
        with patch.object(ui, "SAM3_MODEL", self.source / "a.png"), \
                patch.object(ui, "GenericThread") as thread_class:
            self.window.start_detection()
        sampled = [path.name for path in thread_class.call_args.args[0]]
        seed = json.loads((self.output / "project.json").read_text())["sample_seed"]
        self.assertEqual(sampled, [path.name for path in ui.sample_image_files(
            ui.image_files(self.source), 40, seed)])
        self.assertEqual(len(sampled), 2)
        skipped_names = {"3.png", "4.png", "a.png", "b.png", "c.png"} - set(sampled)
        self.assertTrue(all(self.window.records[name]["status"] == "skipped"
                            for name in skipped_names))
        skipped_name = sorted(skipped_names)[0]
        self.window.show_record(skipped_name)
        with patch.object(ui.QMessageBox, "information") as information:
            self.window.clear_current_labels()
        information.assert_called_once()
        self.assertEqual(self.window.records[skipped_name]["status"], "skipped")
        saved = load_records(self.output / "review.jsonl")
        self.assertEqual({name for name, record in saved.items()
                          if record["status"] == "skipped"}, skipped_names)
        self.window.detection_finished()
        with patch.object(ui.QMessageBox, "question", return_value=ui.QMessageBox.Yes):
            self.window.clear_all_labels()
        self.assertTrue(all(record["status"] == "unlabeled"
                            for record in self.window.records.values()))
        with patch.object(ui, "GenericThread") as thread_class:
            self.window.start_detection()
        self.assertEqual([path.name for path in thread_class.call_args.args[0]], sampled)
        self.assertEqual(self.window.records[skipped_name]["status"], "skipped")

    def test_config_write_failure_restores_previous_labels(self):
        for record in self.window.records.values():
            append_record(self.output / "review.jsonl", record)
        self.window.confidence_spin.setValue(0.4)
        with patch.object(ui, "replace_project_config", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.window.ensure_project(allow_reconfigure=True)
        saved = load_records(self.output / "review.jsonl")
        self.assertEqual(saved["a.png"]["status"], "review")
        self.assertEqual(len(saved["a.png"]["objects"]), 1)
        self.assertEqual(json.loads((self.output / "project.json").read_text())
                         ["confidence"], 0.25)

    def test_stop_reopens_config_and_dirty_export_is_blocked(self):
        worker = Mock()
        worker.isRunning.return_value = True
        self.window.thread = worker
        self.window.set_config_controls(False)
        self.window.stop_detection()
        self.assertFalse(self.window.roi_button.isEnabled())
        worker.isRunning.return_value = False
        self.window.detection_finished()
        self.assertTrue(self.window.roi_button.isEnabled())
        self.assertTrue(self.window.input_edit.isEnabled())
        self.window.confidence_spin.setValue(0.4)
        with patch.object(ui.QMessageBox, "warning") as warning:
            self.window.export()
        warning.assert_called_once()
        self.assertEqual(json.loads((self.output / "project.json").read_text())
                         ["confidence"], 0.25)

    def test_second_worker_run_reuses_loaded_segmenter(self):
        models = []

        class FakeSegmenter:
            def __init__(self, model_path, device, conf):
                self.device = device
                self.conf = conf
                models.append(self)

            def detect_many(self, _image, _prompts):
                return []

        config = self.window.config()
        image = self.source / "a.png"
        with patch.object(ui, "GenericSAM3Segmenter", FakeSegmenter):
            first = ui.GenericThread([image], config, set())
            first.run()
            second_config = {**config, "confidence": 0.4}
            second = ui.GenericThread([image], second_config, set(), first.segmenter)
            second.run()
        self.assertEqual(len(models), 1)
        self.assertIs(second.segmenter, first.segmenter)
        self.assertEqual(second.segmenter.conf, 0.4)
        self.window.thread = second
        self.window.detection_finished()
        self.assertIs(self.window.segmenter, first.segmenter)
        with patch.object(ui, "GenericThread") as thread_class:
            self.window.start_detection()
        self.assertIs(thread_class.call_args.args[3], first.segmenter)

    def test_restart_reruns_only_old_extra_candidate_reviews(self):
        old = {**self.window.records["a.png"],
               "reason": "高分候选超过设定数量，请人工选择"}
        append_record(self.output / "review.jsonl", old)
        append_record(self.output / "review.jsonl", self.window.records["b.png"])
        with patch.object(ui, "SAM3_MODEL", self.source / "a.png"), \
                patch.object(ui, "GenericThread") as thread_class:
            self.window.start_detection()
        done_names = thread_class.call_args.args[2]
        self.assertNotIn("a.png", done_names)
        self.assertIn("b.png", done_names)
        self.window.records["a.png"]["status"] = "confirmed"
        with patch.object(ui, "append_record") as save:
            self.window.receive_result({**old, "status": "auto"})
        save.assert_not_called()
        self.assertEqual(self.window.records["a.png"]["status"], "confirmed")

    def test_box_dialog_shortcuts_stay_in_dialog(self):
        box = ui.GenericBoxDialog(self.source / "a.png", "edit",
                                  self.window.records["a.png"]["objects"][0]["polygon"],
                                  parent=self.window)
        box.setModal(True)
        box.show()
        box.activateWindow()
        box.canvas.setFocus()
        self.qt.processEvents()
        QTest.keyClick(box.canvas, Qt.Key_Space)
        self.assertEqual(self.window.records["a.png"]["status"], "review")
        QTest.keyClick(box.canvas, Qt.Key_S, Qt.ControlModifier)
        self.assertEqual(box.result(), QDialog.Accepted)
        box.deleteLater()

    def test_region_dialog_draws_circle_and_project_restores_options(self):
        dialog = ui.GenericRegionDialog(self.source / "a.png", self.window.roi,
                                        parent=self.window)
        dialog.show()
        dialog.shape_combo.setCurrentIndex(1)
        self.qt.processEvents()
        center = QPoint(120, 60)
        edge = QPoint(155, 60)
        QTest.mousePress(dialog.canvas, Qt.LeftButton, pos=center)
        QTest.mouseMove(dialog.canvas, edge)
        QTest.mouseRelease(dialog.canvas, Qt.LeftButton, pos=edge)
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.Accepted)
        region = dialog.result_region
        self.assertEqual(region["shape"], "circle")
        self.assertAlmostEqual(region["center"][0], 0.5)
        self.assertAlmostEqual(region["center"][1], 0.5)
        self.assertGreater(region["radius"], 0.2)
        dialog.deleteLater()

        new_output = Path(self.temp.name) / "circle_output"
        self.window.output_edit.setText(str(new_output))
        self.window.load_project()
        self.window.roi = region
        self.window.review_reason_checks["near_edge"].setChecked(False)
        config = self.window.ensure_project()
        self.assertEqual(config["roi"], region)
        self.assertNotIn("near_edge", config["review_reasons"])
        self.assertEqual(json.loads((new_output / "project.json").read_text()), config)
        with patch.object(ui, "DEFAULT_OUTPUT", new_output):
            reopened = ui.GenericAnnotatorApp()
        self.assertEqual(reopened.roi, region)
        self.assertEqual(reopened.sample_seed, config["sample_seed"])
        self.assertFalse(reopened.review_reason_checks["near_edge"].isChecked())
        self.assertTrue(reopened.roi_button.isEnabled())
        self.assertTrue(reopened.class_table.isEnabled())
        reopened.close()
        reopened.deleteLater()

    def test_legacy_project_defaults_to_all_review_reasons(self):
        path = self.output / "project.json"
        config = json.loads(path.read_text())
        del config["review_reasons"]
        del config["selection_rule"]
        del config["sample_percent"]
        del config["sample_seed"]
        path.write_text(json.dumps(config))
        self.window.load_project()
        self.assertTrue(all(check.isChecked()
                            for check in self.window.review_reason_checks.values()))
        self.assertEqual(self.window.selection_combo.currentData(), "confidence")
        self.assertEqual(self.window.sample_percent_spin.value(), 100)
        self.window.ensure_project()
        self.assertIsInstance(json.loads(path.read_text())["sample_seed"], int)
        self.assertFalse(list(self.output.glob("backup_before_config_*")))

    def test_percentage_change_draws_and_saves_new_random_seed(self):
        self.window.sample_percent_spin.setValue(40)
        with patch.object(ui.secrets, "randbits", return_value=98765):
            config = self.window.ensure_project(allow_reconfigure=True)
        self.assertEqual(config["sample_seed"], 98765)
        self.assertEqual(self.window.sample_seed, 98765)
        self.assertEqual(json.loads((self.output / "project.json").read_text())
                         ["sample_seed"], 98765)
        self.assertEqual(self.window.ensure_project()["sample_seed"], 98765)

    def test_old_partial_sample_requires_random_resampling_before_export(self):
        path = self.output / "project.json"
        previous = json.loads(path.read_text())
        previous["sample_percent"] = 40
        del previous["sample_seed"]
        path.write_text(json.dumps(previous))
        self.window.load_project()
        with self.assertRaisesRegex(ValueError, "按 F5"):
            self.window.ensure_project()
        config = self.window.ensure_project(allow_reconfigure=True)
        self.assertIsInstance(config["sample_seed"], int)
        self.assertEqual(len(list(self.output.glob("backup_before_config_*"))), 1)

    def test_selection_rule_is_only_editable_for_fixed_count(self):
        self.assertEqual(self.window.selection_combo.currentData(), "confidence")
        self.assertFalse(self.window.selection_combo.isEnabled())
        self.window.count_combo.setCurrentIndex(1)
        self.assertTrue(self.window.selection_combo.isEnabled())
        self.window.selection_combo.setCurrentIndex(1)
        self.assertEqual(self.window.config()["selection_rule"], "area")
        self.window.set_config_controls(False)
        self.assertFalse(self.window.selection_combo.isEnabled())
        self.window.set_config_controls(True)
        self.assertTrue(self.window.selection_combo.isEnabled())

    def test_worker_passes_area_rule_to_detector(self):
        config = {**self.window.config(), "count_mode": "fixed",
                  "selection_rule": "area"}
        segmenter = Mock(device="cpu", conf=0.25)
        with patch.object(ui, "torch", None), patch.object(ui, "detect_generic") as detect:
            detect.return_value = {"objects": [], "candidates": [],
                                   "status": "review", "reason": "test",
                                   "candidate_count": 0}
            worker = ui.GenericThread([self.source / "a.png"], config,
                                      set(), segmenter)
            worker.run()
        self.assertEqual(detect.call_args.args[-2], "area")
        self.assertEqual(detect.call_args.args[-1], 0.50)

    def test_f5_retries_old_nested_boxes_and_keeps_confirmed_records(self):
        first = self.window.records["a.png"]
        first["status"] = "auto"
        first["objects"] = [
            {"class_id": 0, "score": 0.92,
             "polygon": [[80, 20], [180, 20], [180, 110], [80, 110]]},
            {"class_id": 0, "score": 0.38,
             "polygon": [[90, 30], [145, 30], [145, 95], [90, 95]]},
        ]
        first["candidates"] = [{**obj, "selected": True}
                               for obj in first["objects"]]
        self.window.records["b.png"]["status"] = "confirmed"
        for record in self.window.records.values():
            append_record(self.output / "review.jsonl", record)
        with patch.object(ui, "SAM3_MODEL", self.source / "a.png"), \
                patch.object(ui, "GenericThread") as thread_class:
            self.window.start_detection()
        done_names = thread_class.call_args.args[2]
        self.assertNotIn("a.png", done_names)
        self.assertIn("b.png", done_names)
        self.window.detection_finished()


if __name__ == "__main__":
    unittest.main()
