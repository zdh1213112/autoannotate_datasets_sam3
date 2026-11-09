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
import yaml
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

    def test_side_camera_requires_review_of_left_right_labels(self):
        record = {"status": "auto", "objects": [{"class_id": 0}],
                  "reason": "自动候选"}
        self.assertEqual(ui.flag_side_camera_review(record, "cam_0")["status"],
                         "auto")
        side = ui.flag_side_camera_review(record, "cam_2")
        self.assertEqual(side["status"], "review")
        self.assertEqual(side["camera_id"], "cam_2")
        self.assertIn("侧视相机", side["reason"])

    def test_side_camera_uses_sam_evidence_without_screen_side_guess(self):
        glove = {"class_id": 1, "score": 0.93,
                 "polygon": [[140, 50], [180, 50], [180, 90], [140, 90]]}
        record = {"status": "auto", "reason": "自动候选", "objects": [glove],
                  "candidates": [], "candidate_count": 1}
        evidence = [{"class_id": 0, "score": 0.85, "box": [140, 10, 180, 50]},
                    {"class_id": 1, "score": 0.74, "box": [140, 10, 180, 50]}]
        scored = ui.attach_side_scores(record, evidence, (0, 40))
        left = ui.assign_glove_sides(scored, 200, "screen", "cam_3")
        left = ui.flag_side_camera_review(left, "cam_3")
        self.assertEqual(left["objects"][0]["class_id"], 0)
        self.assertFalse(left["side_pending"])
        self.assertTrue(left["side_autoclassified"])
        self.assertEqual(left["status"], "auto")
        conflicting = ui.attach_side_scores(record, [
            {"class_id": 0, "score": 0.70, "box": [140, 10, 180, 50]},
            {"class_id": 1, "score": 0.86, "box": [140, 10, 180, 50]},
        ], (0, 40))
        uncertain = ui.assign_glove_sides(conflicting, 200, "screen", "cam_3")
        self.assertTrue(uncertain["side_pending"])
        self.assertEqual(uncertain["status"], "review")

    def test_side_camera_pair_auto_when_both_prompts_agree(self):
        record = {"status": "auto", "reason": "自动候选", "objects": [
            {"class_id": 0, "polygon": [[20, 50], [50, 50], [50, 90], [20, 90]],
             "side_scores": [0.90, 0.59]},
            {"class_id": 0, "polygon": [[140, 50], [180, 50], [180, 90], [140, 90]],
             "side_scores": [0.70, 0.88]},
        ], "candidates": [], "candidate_count": 2}
        result = ui.flag_side_camera_review(
            ui.assign_glove_sides(record, 200, camera_id="cam_2"), "cam_2")
        self.assertEqual([item["class_id"] for item in result["objects"]], [0, 1])
        self.assertTrue(result["side_autoclassified"])
        self.assertEqual(result["status"], "auto")


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
        self.assertEqual(self.window.confidence_spin.value(), 0.40)
        self.assertEqual(self.window.review_score_spin.value(), 0.50)
        self.assertEqual(self.window.format_combo.currentData(), "hbb")
        self.assertEqual(self.window.output_edit.text(), str(self.output))
        self.assertIsInstance(self.window.centralWidget(), QScrollArea)
        self.assertFalse(self.window.class_table.isEnabled())
        self.assertEqual(self.window.side_combo.currentData(), "screen")
        self.assertTrue(self.window.auto_side_check.isChecked())
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

    def test_f5_retries_old_side_reviews_without_touching_confirmed_labels(self):
        self.window.records = {
            "old_side.jpg": {"camera_id": "cam_3", "status": "review"},
            "attempted.jpg": {"camera_id": "cam_2", "status": "review",
                              "side_auto_attempted": True},
            "confirmed.jpg": {"camera_id": "cam_3", "status": "confirmed"},
        }
        thread = self.window.make_detection_thread([], {}, set(self.window.records))
        self.assertEqual(thread.done_names, {"attempted.jpg", "confirmed.jpg"})
        self.assertTrue(thread.config["auto_side"])
        self.window.auto_side_check.setChecked(False)
        conservative = self.window.make_detection_thread([], {}, set(self.window.records))
        self.assertEqual(conservative.done_names, set(self.window.records))

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
        self.assertFalse(self.window.video_edit.text())
        self.assertEqual(self.window.video_list.item(0).text(), str(video))
        for button in (self.window.video_button, self.window.folder_button,
                       self.window.add_path_button, self.window.remove_video_button,
                       self.window.clear_video_button):
            self.assertFalse(button.isEnabled())
        self.assertTrue(self.window.change_video_button.isEnabled())
        self.window.change_video_button.click()
        self.assertEqual(self.window.video_list.count(), 0)
        self.assertFalse(self.window.input_edit.text())
        self.assertTrue(self.window.video_button.isEnabled())
        self.assertFalse(self.window.change_video_button.isEnabled())

    def test_detection_accepts_one_input_source(self):
        with patch.object(ui.QMessageBox, "warning") as warning, \
                patch.object(ui.GenericAnnotatorApp, "start_detection") as detect:
            self.window.start_detection()
            self.assertIn("图片目录或添加视频", warning.call_args.args[2])
            self.window.input_edit.setText(str(self.root / "images"))
            self.window.video_list.addItem(str(self.root / "recording.mcap"))
            self.window.start_detection()
            self.assertIn("请选择一种输入", warning.call_args.args[2])
            detect.assert_not_called()
            self.window.video_list.clear()
            self.window.start_detection()
            detect.assert_called_once()

    def test_video_only_detection_extracts_then_starts_detection(self):
        video = self.root / "gloves.mp4"
        make_video(video)
        self.window.add_video_paths([video])
        with patch.object(ui.GenericAnnotatorApp, "start_detection") as detect:
            self.window.start_detection()
            deadline = time.monotonic() + 10
            while self.window.video_thread is not None and time.monotonic() < deadline:
                self.qt.processEvents()
                time.sleep(0.01)
            self.qt.processEvents()
            self.assertIsNone(self.window.video_thread)
            detect.assert_called_once()
        self.assertTrue(Path(self.window.input_edit.text()).is_dir())
        self.assertEqual(self.window.video_list.count(), 1)
        self.assertEqual(self.window.video_list.item(0).text(), str(video))

    def test_imported_sources_restore_when_project_reopened(self):
        video = self.root / "gloves.mp4"
        make_video(video)
        self.window.add_video_paths([video])
        self.window.import_video()
        deadline = time.monotonic() + 10
        while self.window.video_thread is not None and time.monotonic() < deadline:
            self.qt.processEvents()
            time.sleep(0.01)
        self.qt.processEvents()
        self.assertIsNone(self.window.video_thread)
        self.window.ensure_project()
        self.window.video_list.clear()
        self.window._loaded_video_folder = None
        self.window.load_project()
        self.assertEqual(self.window.video_list.item(0).text(), str(video))
        self.assertFalse(self.window.clear_video_button.isEnabled())
        with patch.object(ui.GenericAnnotatorApp, "start_detection") as detect, \
                patch.object(ui.QMessageBox, "warning") as warning:
            self.window.start_detection()
            detect.assert_called_once()
            warning.assert_not_called()
        self.window.input_edit.setText(str(self.root / "other_images"))
        self.window.sync_video_sources()
        self.assertEqual(self.window.video_list.count(), 0)
        self.assertTrue(self.window.video_button.isEnabled())

    def test_failed_video_extraction_does_not_start_detection(self):
        video = self.root / "gloves.mp4"
        make_video(video)
        self.window.add_video_paths([video])
        with patch.object(ui, "extract_video_batch", side_effect=ValueError("抽帧失败")), \
                patch.object(ui.QMessageBox, "warning") as warning, \
                patch.object(ui.GenericAnnotatorApp, "start_detection") as detect:
            self.window.start_detection()
            deadline = time.monotonic() + 10
            while self.window.video_thread is not None and time.monotonic() < deadline:
                self.qt.processEvents()
                time.sleep(0.01)
            self.qt.processEvents()
            self.assertIsNone(self.window.video_thread)
            warning.assert_called_once()
            detect.assert_not_called()
        self.assertEqual(self.window.video_list.count(), 1)
        self.assertFalse(self.window.input_edit.text())

    def test_image_interval_controls_detection_export_and_project_reload(self):
        source = self.root / "images"
        source.mkdir()
        for index in range(7):
            cv2.imwrite(str(source / f"{index:02}.jpg"), np.zeros((10, 10, 3), np.uint8))
        self.window.input_edit.setText(str(source))
        self.window.image_interval_spin.setValue(2)
        self.assertIn("抽帧 4/7", self.window.sample_summary.text())
        with patch.object(self.window, "validate_model_ready"), \
                patch.object(ui, "GrayGloveThread") as thread_class:
            thread_class.return_value.isRunning.return_value = False
            self.window.start_detection()
            selected = thread_class.call_args.args[0]
            self.assertEqual([path.name for path in selected],
                             ["00.jpg", "02.jpg", "04.jpg", "06.jpg"])
            self.assertEqual(self.window.records["01.jpg"]["status"], "skipped")
            self.window.load_project()
            self.assertEqual(self.window.image_interval_spin.value(), 2)
            with patch("auto_obb_annotator.app_generic.export_generic",
                       return_value=(self.output, 0, 0)) as export, \
                    patch.object(ui.QMessageBox, "information"):
                self.window.export()
            self.assertEqual(export.call_args.args[-1],
                             {"00.jpg", "02.jpg", "04.jpg", "06.jpg"})

    def test_video_frames_are_not_sampled_a_second_time_as_images(self):
        video = self.root / "gloves.mp4"
        make_video(video)
        self.window.image_interval_spin.setValue(3)
        self.window.video_edit.setText(str(video))
        self.window.interval_spin.setValue(0.5)
        self.window.import_video()
        deadline = time.monotonic() + 10
        while self.window.video_thread is not None and time.monotonic() < deadline:
            self.qt.processEvents()
            time.sleep(0.01)
        self.qt.processEvents()
        self.assertIsNone(self.window.video_thread)
        files = ui.image_files(Path(self.window.input_edit.text()))
        self.assertEqual(len(files), 4)
        self.assertEqual(self.window.sampling_candidates(files, self.window.config()), files)
        self.assertIn("抽帧 4/4", self.window.sample_summary.text())

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

    def test_undistorted_import_uses_separate_project_and_keeps_raw_labels(self):
        raw_source = self.output / "video_frames" / "legacy"
        raw_source.mkdir(parents=True)
        cv2.imwrite(str(raw_source / "old.jpg"), np.zeros((60, 80, 3), np.uint8))
        self.window.input_edit.setText(str(raw_source))
        self.window.ensure_project()
        old_record = {"name": "old.jpg", "status": "confirmed", "objects": [],
                      "candidates": [], "candidate_count": 0}
        append_record(self.output / "review.jsonl", old_record)
        video = self.root / "EGO_camera_left.mp4"
        make_video(video)
        calibration = self.root / "EGO_calibration_camera.yaml"
        calibration.write_text(yaml.safe_dump({"cameras": [{
            "id": "cam_0", "distortion_model": "KB",
            "image_width": 80, "image_height": 60,
            "intrinsics": {"fx": 45, "fy": 45, "cx": 40, "cy": 30},
            "distortion": {"k1": 0.08, "k2": -0.01, "k3": 0.001, "k4": 0},
        }]}), encoding="utf-8")
        self.window.video_edit.setText(str(video))
        self.window.end_check.setChecked(True)
        self.window.end_spin.setValue(0.2)
        self.window.import_video(undistort=True)
        deadline = time.monotonic() + 10
        while self.window.video_thread is not None and time.monotonic() < deadline:
            self.qt.processEvents()
            time.sleep(0.01)
        self.qt.processEvents()
        rectified_output = self.root / "out_undistorted"
        self.assertEqual(self.window.output_edit.text(), str(rectified_output))
        self.assertEqual(self.window.input_edit.text(),
                         str(rectified_output / "video_frames" / "collection"))
        self.assertEqual(self.window.video_list.item(0).text(), str(video))
        self.assertFalse(self.window.clear_video_button.isEnabled())
        self.assertEqual(len(list((rectified_output / "video_frames" / "collection").glob("*.jpg"))), 1)
        self.assertEqual(load_records(self.output / "review.jsonl")["old.jpg"], old_record)
        self.assertEqual(self.window.video_target_dir(self.output), raw_source)

    def test_prompt_ensemble_merges_candidates_before_side_assignment(self):
        fake = [{"class_id": index, "box": [0, 0, 20, 20], "score": 0.9}
                for index in range(3)]
        with patch.object(ui.GenericSAM3Segmenter, "detect_many", return_value=fake) as run:
            segmenter = ui.GrayGloveSegmenter.__new__(ui.GrayGloveSegmenter)
            result = segmenter.detect_many(None, [])
        self.assertEqual(run.call_args.args[-1], list(ui.GRAY_GLOVE_PROMPTS))
        self.assertEqual([item["class_id"] for item in result], [0, 0, 0])

    def test_directional_prompts_are_kept_as_evidence_not_extra_gloves(self):
        fake = [{"class_id": index, "box": [0, 0, 20, 20], "score": 0.9}
                for index in range(5)]
        with patch.object(ui.GenericSAM3Segmenter, "detect_many", return_value=fake) as run:
            segmenter = ui.GrayGloveSegmenter.__new__(ui.GrayGloveSegmenter)
            segmenter.side_prompts_enabled = True
            result = segmenter.detect_many(None, [])
        self.assertEqual(run.call_args.args[-1],
                         list(ui.GRAY_GLOVE_PROMPTS) + list(ui.SIDE_GLOVE_PROMPTS))
        self.assertEqual(len(result), 3)
        self.assertEqual([item["class_id"] for item in segmenter.last_side_detections],
                         [0, 1])

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
        record = ui.flag_side_camera_review(record, "cam_3")
        self.window.records["one.jpg"] = record
        self.window.update_table(record)
        self.window.show_record("one.jpg")
        self.assertEqual(self.window.object_table.item(0, 1).text(), "左右待定")
        self.assertEqual(
            self.window.preview_pixmap.toImage().pixelColor(140, 70).getRgb()[:3],
            (210, 90, 180))
        with patch.object(ui.QMessageBox, "information"):
            self.window.confirm_current()
        self.assertEqual(self.window.records["one.jpg"]["status"], "review")
        self.window.assign_current_side(0)
        saved = load_records(self.output / "review.jsonl")["one.jpg"]
        self.assertEqual(saved["status"], "confirmed")
        self.assertEqual(saved["objects"][0]["class_id"], 0)
        self.assertFalse(saved["side_pending"])
        self.assertEqual(self.window.object_table.item(0, 1).text(), "left")
        self.assertEqual(
            self.window.preview_pixmap.toImage().pixelColor(140, 70).getRgb()[:3],
            (0, 200, 0))
        self.window.toggle_focus_view()
        self.assertFalse(self.window.settings_group.isVisible())
        self.assertGreaterEqual(self.window.preview.minimumHeight(), 620)
        self.window.toggle_focus_view()
        self.assertTrue(self.window.settings_group.isVisible())

    def test_left_right_label_colors_and_side_swap(self):
        source = self.root / "images"
        source.mkdir()
        cv2.imwrite(str(source / "both.jpg"), np.zeros((120, 220, 3), np.uint8))
        self.window.input_edit.setText(str(source))
        self.window.ensure_project()
        polygons = (
            [[20, 60], [50, 60], [50, 90], [20, 90]],
            [[120, 60], [150, 60], [150, 90], [120, 90]],
            [[170, 60], [200, 60], [200, 90], [170, 90]],
        )
        record = {
            "name": "both.jpg", "status": "auto", "reason": "自动候选",
            "width": 220, "height": 120,
            "objects": [
                {"class_id": 0, "score": 0.95, "polygon": polygons[0], "candidate_id": 0},
                {"class_id": 1, "score": 0.94, "polygon": polygons[1], "candidate_id": 1},
            ],
            "candidates": [
                {"class_id": 0, "score": 0.95, "polygon": polygons[0], "selected": True},
                {"class_id": 1, "score": 0.94, "polygon": polygons[1], "selected": True},
                {"class_id": 1, "score": 0.70, "polygon": polygons[2], "selected": False},
            ],
            "candidate_count": 3,
        }
        self.window.records["both.jpg"] = record
        self.window.show_record("both.jpg")

        def pixel(x):
            return self.window.preview_pixmap.toImage().pixelColor(x, 75).getRgb()[:3]

        self.assertEqual(pixel(20), (0, 200, 0))
        self.assertEqual(pixel(120), (0, 180, 255))
        self.assertEqual(pixel(170), (184, 134, 11))
        self.window.swap_current_sides()
        self.assertEqual(pixel(20), (0, 180, 255))
        self.assertEqual(pixel(120), (0, 200, 0))
        self.assertEqual(pixel(170), (184, 134, 11))

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
