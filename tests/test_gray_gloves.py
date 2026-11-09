"""Gray glove video import and annotation entrypoint workflows."""

import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5.QtCore import QCoreApplication, QEvent, QLibraryInfo
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QScrollArea

from auto_obb_annotator import app_gray_gloves as ui
from auto_obb_annotator.barcode_core import append_record, load_records
from auto_obb_annotator.gray_glove_video import (
    collect_videos, extract_video_batch, extract_video_frames, video_frame_directory,
)


def make_video(path: Path, frames=20, fps=10):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (80, 60))
    if not writer.isOpened():
        raise RuntimeError("测试环境无法编码 MP4")
    for index in range(frames):
        image = np.full((60, 80, 3), index * 5, np.uint8)
        writer.write(image)
    writer.release()


class GrayGloveVideoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "gloves.mp4"
        make_video(self.video)

    def test_extract_interval_reuse_and_complete_manifest(self):
        target, count, reused = extract_video_frames(self.video, self.root / "out", 0.5)
        self.assertEqual((count, reused), (4, False))
        self.assertEqual(len(list(target.glob("*.jpg"))), 4)
        manifest = json.loads((target / "manifest.json").read_text())
        self.assertEqual(manifest["frames"], [
            "frame_000000000.jpg", "frame_000000005.jpg",
            "frame_000000010.jpg", "frame_000000015.jpg"])
        self.assertEqual(extract_video_frames(self.video, self.root / "out", 0.5),
                         (target, 4, True))
        self.assertNotEqual(target, video_frame_directory(self.video, self.root / "out", 1.0))

    def test_cancel_removes_partial_frames(self):
        with self.assertRaises(InterruptedError):
            extract_video_frames(
                self.video, self.root / "out", 0.5,
                should_continue=lambda: False)
        self.assertFalse(video_frame_directory(self.video, self.root / "out", 0.5).exists())
        self.assertEqual(list((self.root / "out" / "video_frames").glob("*.partial")), [])

    def test_batch_unique_names_time_window_and_resume(self):
        other_dir = self.root / "other"
        other_dir.mkdir()
        other = other_dir / "gloves.mp4"
        make_video(other)
        videos = collect_videos([self.video, other_dir])
        self.assertEqual(videos, sorted([self.video, other]))
        output = self.root / "batch_out"
        calls = []

        def still_running():
            return len(calls) < 2

        with self.assertRaisesRegex(InterruptedError, "再次点击可继续"):
            extract_video_batch(videos, output, "frames", 3, 0.2, 1.2,
                                progress=lambda done, total: calls.append((done, total)),
                                should_continue=still_running)
        target = output / "video_frames" / "collection"
        self.assertEqual(len(list(target.glob("*.jpg"))), 2)
        self.assertEqual(calls, [(1, 8), (2, 8)])
        directory, count, reused = extract_video_batch(
            videos, output, "frames", 3, 0.2, 1.2)
        self.assertEqual((directory, count, reused), (target, 8, 2))
        names = [path.name for path in target.glob("*.jpg")]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 8)
        manifest = json.loads(next(target.glob("extract_*.json")).read_text())
        self.assertTrue(manifest["complete"])
        self.assertEqual({item["source_frame"] for item in manifest["frames"]},
                         {2, 5, 8, 11})
        self.assertEqual({item["video"] for item in manifest["frames"]},
                         {str(self.video), str(other)})
        self.assertEqual(extract_video_batch(videos, output, "frames", 3, 0.2, 1.2),
                         (target, 8, 8))

    def test_batch_seconds_mode_keeps_existing_frames_when_adding_video(self):
        output = self.root / "batch_out"
        target, count, reused = extract_video_batch(
            [self.video], output, "seconds", 0.5)
        self.assertEqual((count, reused), (4, 0))
        original_names = {path.name for path in target.glob("*.jpg")}
        other = self.root / "other.mp4"
        make_video(other)
        _, count, reused = extract_video_batch(
            [self.video, other], output, "seconds", 0.5)
        self.assertEqual((count, reused), (8, 4))
        self.assertTrue(original_names.issubset({path.name for path in target.glob("*.jpg")}))

    def test_seconds_mode_uses_elapsed_time_without_frame_drift(self):
        target, count, _ = extract_video_batch(
            [self.video], self.root / "batch_out", "seconds", 0.35)
        self.assertEqual(count, 6)
        manifest = json.loads(next(target.glob("extract_*.json")).read_text())
        self.assertEqual([item["source_frame"] for item in manifest["frames"]],
                         [0, 4, 7, 11, 14, 18])

    def test_pair_assignment_and_single_hand_review(self):
        left = {"class_id": 0, "score": 0.9,
                "polygon": [[10, 50], [40, 50], [40, 90], [10, 90]]}
        right = {"class_id": 0, "score": 0.9,
                 "polygon": [[150, 50], [180, 50], [180, 90], [150, 90]]}
        record = {"status": "auto", "objects": [right, left], "candidates": [],
                  "candidate_count": 2}
        screen = ui.assign_glove_sides(record, 200, "screen")
        self.assertEqual([obj["class_id"] for obj in screen["objects"]], [1, 0])
        self.assertFalse(screen["side_pending"])
        mirror = ui.assign_glove_sides(record, 200, "mirror")
        self.assertEqual([obj["class_id"] for obj in mirror["objects"]], [0, 1])
        single = ui.assign_glove_sides({**record, "objects": [right]}, 200)
        self.assertEqual(single["status"], "review")
        self.assertTrue(single["side_pending"])


class GrayGloveUITests(unittest.TestCase):
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
        self.output = self.root / "out"
        with patch.object(ui, "DEFAULT_OUTPUT", self.output):
            self.window = ui.GrayGloveAnnotatorApp()
        self.window.show()
        self.qt.processEvents()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.qt.sendPostedEvents(None, QEvent.DeferredDelete)
        self.qt.processEvents()

    def test_gray_glove_defaults_and_independent_project(self):
        self.assertEqual(self.window.classes(), ui.GRAY_GLOVE_CLASSES)
        self.assertEqual(self.window.roi, ui.DEFAULT_ROI)
        self.assertEqual(self.window.count_combo.currentData(), "fixed")
        self.assertEqual(self.window.count_spin.value(), 2)
        self.assertEqual(self.window.selection_combo.currentData(), "area")
        self.assertEqual(self.window.format_combo.currentData(), "hbb")
        self.assertEqual(self.window.output_edit.text(), str(self.output))
        self.assertIsInstance(self.window.centralWidget(), QScrollArea)
        self.assertFalse(self.window.class_table.isEnabled())
        self.assertEqual(self.window.side_combo.currentData(), "screen")
        self.window.interval_mode.setCurrentIndex(
            self.window.interval_mode.findData("frames"))
        self.assertEqual(self.window.interval_spin.value(), 30)
        self.window.interval_mode.setCurrentIndex(
            self.window.interval_mode.findData("seconds"))
        self.assertEqual(self.window.interval_spin.value(), 1)

    def test_search_and_large_preview(self):
        for name, status in (("left_001.jpg", "review"),
                             ("right_002.jpg", "auto")):
            record = {"name": name, "status": status, "objects": [],
                      "candidates": [], "candidate_count": 0}
            self.window.records[name] = record
            self.window.update_table(record)
        self.window.search_edit.setText("LEFT")
        self.assertEqual(self.window.visible_count.text(), "显示 1/2 张")
        self.window.status_filter.setCurrentIndex(
            self.window.status_filter.findData("auto"))
        self.assertEqual(self.window.visible_count.text(), "显示 0/2 张")
        dialog = ui.LargePreviewDialog(QPixmap(200, 100), "sample.jpg", self.window)
        dialog.zoom_spin.setValue(200)
        self.assertEqual(dialog.image_label.pixmap().width(), 400)
        dialog.close()

    def test_project_config_tracks_prompts_and_model(self):
        source = self.root / "images"
        source.mkdir()
        self.window.input_edit.setText(str(source))
        config = self.window.config()
        self.assertEqual(config["workflow"], ui.WORKFLOW)
        self.assertEqual(config["prompt_ensemble"], list(ui.GRAY_GLOVE_PROMPTS))
        self.assertEqual(config["model_path"], str(ui.SAM3_MODEL.resolve()))
        self.assertEqual(config["side_mapping"], "screen")

    def test_video_import_updates_image_directory_and_random_sample(self):
        video = self.root / "gloves.mp4"
        make_video(video)
        self.window.video_edit.setText(str(video))
        self.window.interval_spin.setValue(0.5)
        self.window.import_video()
        deadline = time.monotonic() + 10
        while self.window.video_thread is not None and time.monotonic() < deadline:
            self.qt.processEvents()
            time.sleep(0.01)
        self.qt.processEvents()
        self.assertIsNone(self.window.video_thread)
        self.assertTrue(Path(self.window.input_edit.text()).is_dir())
        self.assertIn("4/4", self.window.sample_summary.text())
        self.assertTrue(self.window.start_button.isEnabled())

    def test_batch_import_appends_to_existing_video_project(self):
        legacy = self.output / "video_frames" / "legacy"
        legacy.mkdir(parents=True)
        cv2.imwrite(str(legacy / "old.jpg"), np.zeros((60, 80, 3), np.uint8))
        self.window.input_edit.setText(str(legacy))
        self.window.ensure_project()
        old_record = {"name": "old.jpg", "status": "confirmed", "objects": [],
                      "candidates": [], "candidate_count": 0}
        append_record(self.output / "review.jsonl", old_record)
        self.window.records["old.jpg"] = old_record
        video = self.root / "gloves.mp4"
        make_video(video)
        self.window.add_video_paths([video])
        self.window.interval_mode.setCurrentIndex(
            self.window.interval_mode.findData("frames"))
        self.window.interval_spin.setValue(5)
        self.window.end_check.setChecked(True)
        self.window.end_spin.setValue(1.0)
        self.window.import_video()
        deadline = time.monotonic() + 10
        while self.window.video_thread is not None and time.monotonic() < deadline:
            self.qt.processEvents()
            time.sleep(0.01)
        self.qt.processEvents()
        self.assertEqual(self.window.input_edit.text(), str(legacy))
        self.assertTrue((legacy / "old.jpg").is_file())
        self.assertEqual(len(list(legacy.glob("*.jpg"))), 3)
        self.assertEqual(load_records(self.output / "review.jsonl")["old.jpg"], old_record)

    def test_prompt_ensemble_merges_candidates_before_side_assignment(self):
        fake = [{"class_id": index, "box": [0, 0, 20, 20], "score": 0.9}
                for index in range(3)]
        with patch.object(ui.GenericSAM3Segmenter, "detect_many", return_value=fake) as run:
            segmenter = ui.GrayGloveSegmenter.__new__(ui.GrayGloveSegmenter)
            result = segmenter.detect_many(None, [])
        self.assertEqual(run.call_args.args[-1], list(ui.GRAY_GLOVE_PROMPTS))
        self.assertEqual([item["class_id"] for item in result], [0, 0, 0])

    def test_single_hand_requires_explicit_side_and_focus_view_is_larger(self):
        source = self.root / "images"
        source.mkdir()
        image = np.zeros((100, 200, 3), np.uint8)
        cv2.imwrite(str(source / "one.jpg"), image)
        self.window.input_edit.setText(str(source))
        self.window.ensure_project()
        record = ui.assign_glove_sides({
            "name": "one.jpg", "status": "auto", "width": 200, "height": 100,
            "objects": [{"class_id": 0, "score": 0.9,
                         "polygon": [[140, 50], [180, 50], [180, 90], [140, 90]]}],
            "candidates": [], "candidate_count": 1,
        }, 200)
        self.window.records["one.jpg"] = record
        self.window.update_table(record)
        self.window.show_record("one.jpg")
        with patch.object(ui.QMessageBox, "information"):
            self.window.confirm_current()
        self.assertEqual(self.window.records["one.jpg"]["status"], "review")
        self.window.assign_current_side(1)
        saved = load_records(self.output / "review.jsonl")["one.jpg"]
        self.assertEqual(saved["status"], "confirmed")
        self.assertEqual(saved["objects"][0]["class_id"], 1)
        self.window.toggle_focus_view()
        self.assertFalse(self.window.settings_group.isVisible())
        self.assertGreaterEqual(self.window.preview.minimumHeight(), 620)
        self.window.toggle_focus_view()
        self.assertTrue(self.window.settings_group.isVisible())

    def test_export_yaml_has_left_right_classes(self):
        source = self.root / "images"
        source.mkdir()
        for name in ("both.jpg", "empty.jpg"):
            cv2.imwrite(str(source / name), np.zeros((100, 200, 3), np.uint8))
        self.window.input_edit.setText(str(source))
        self.window.ensure_project()
        pair = ui.assign_glove_sides({
            "name": "both.jpg", "status": "auto", "width": 200, "height": 100,
            "objects": [
                {"class_id": 0, "score": 0.9,
                 "polygon": [[120, 50], [160, 50], [160, 90], [120, 90]]},
                {"class_id": 0, "score": 0.9,
                 "polygon": [[20, 50], [60, 50], [60, 90], [20, 90]]},
            ], "candidates": [], "candidate_count": 2,
        }, 200)
        empty = {"name": "empty.jpg", "status": "empty_confirmed", "objects": [],
                 "candidates": [], "candidate_count": 0}
        for record in (pair, empty):
            append_record(self.output / "review.jsonl", record)
        self.window.records = {record["name"]: record for record in (pair, empty)}
        with patch.object(ui.QMessageBox, "information"):
            self.window.export()
        dataset = next(self.output.glob("yolo_hbb_*/dataset.yaml"))
        self.assertIn("nc: 2\nnames:\n  0: \"left\"\n  1: \"right\"",
                      dataset.read_text(encoding="utf-8"))
