"""Raw-frame provenance and nonlinear mask-to-label projection regressions."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from auto_obb_annotator.barcode_core import image_files
from auto_obb_annotator.generic_core import detect_generic
from auto_obb_annotator.gray_glove_projection import (
    PROJECTION_WORKFLOW, ProjectedGloveSegmenter, RawFrameReader,
)
from auto_obb_annotator.gray_glove_video import extract_video_batch, raw_frame_path
from auto_obb_annotator.orbbec_undistort import OrbbecUndistorter


class FakeSegmenter:
    def __init__(self, mask):
        self.mask = mask
        self.seen = None
        self.last_side_detections = []

    def detect_many(self, image, prompts):
        self.seen = image.copy()
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(self.mask))
        detection = {"class_id": 0, "score": 0.9, "box": [x, y, x + w, y + h],
                     "mask": self.mask}
        self.last_side_detections = [dict(detection)]
        return [detection]


class GrayGloveProjectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.video = self.root / "EGO_camera_left.mp4"
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*"mp4v"),
                                 10, (96, 64))
        self.assertTrue(writer.isOpened())
        for index in range(4):
            image = np.zeros((64, 96, 3), np.uint8)
            image[:, :, 0] = np.arange(96, dtype=np.uint8) * 2
            image[:, :, 1] = index * 40
            cv2.rectangle(image, (56, 12), (84, 47), (50, 170, 245), -1)
            writer.write(image)
        writer.release()
        self.calibration = self.root / "EGO_calibration_camera.yaml"
        self.calibration.write_text(yaml.safe_dump({"cameras": [{
            "id": "cam_0", "distortion_model": "KB", "image_width": 96, "image_height": 64,
            "intrinsics": {"fx": 55, "fy": 55, "cx": 48, "cy": 32},
            "distortion": {"k1": 0.12, "k2": -0.02, "k3": 0, "k4": 0},
        }]}))
        self.rectifier = OrbbecUndistorter([self.video])
        self.folder, _, _ = extract_video_batch(
            [self.video], self.root / "out", "frames", 1, end_seconds=0.3,
            frame_transform=self.rectifier, transform_info=self.rectifier.metadata)
        self.frames = json.loads(next(self.folder.glob("extract_*.json")).read_text())["frames"]
        self.reader = RawFrameReader(self.folder)
        self.addCleanup(self.reader.close)
        self.mask = np.zeros((64, 96), np.uint8)
        cv2.fillConvexPoly(self.mask, np.array([[60, 12], [84, 20], [76, 47], [56, 37]]), 1)

    def test_lossless_cache_can_be_used_with_source_video_offline(self):
        entry = self.frames[0]
        cap = cv2.VideoCapture(str(self.video))
        ok, expected = cap.read()
        cap.release()
        self.assertTrue(ok)
        self.video.rename(self.video.with_suffix(".offline"))
        raw, restored, key = self.reader.load(entry["name"])
        self.assertTrue(np.array_equal(raw, expected))
        self.assertTrue(np.array_equal(restored(key, raw), self.rectifier(str(self.video), raw)))
        self.assertEqual(len(image_files(self.folder)), 3)
        self.assertEqual(len(list((self.folder / "raw_frames").glob("*.png"))), 3)

    def test_old_mp4_frames_are_recovered_at_exact_indices_and_cached(self):
        entry = self.frames[2]
        cached = raw_frame_path(self.folder, entry["name"])
        cached.unlink()
        raw, _, _ = self.reader.load(entry["name"])
        cap = cv2.VideoCapture(str(self.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 2)
        ok, expected = cap.read()
        cap.release()
        self.assertTrue(ok)
        self.assertTrue(np.array_equal(raw, expected))
        self.assertTrue(np.array_equal(cv2.imread(str(cached)), expected))

    def test_missing_or_changed_sources_are_reported_without_guessing(self):
        entry = self.frames[0]
        raw_frame_path(self.folder, entry["name"]).unlink()
        self.video.touch()
        with self.assertRaisesRegex(ValueError, "原始录像已改变"):
            self.reader.load(entry["name"])
        self.video.rename(self.video.with_suffix(".offline"))
        with self.assertRaisesRegex(FileNotFoundError, "缺少原图缓存和原始录像"):
            self.reader.load(entry["name"])

    def test_nonlinear_mask_is_refit_in_output_roi_without_extra_padding(self):
        entry = self.frames[0]
        raw, rectifier, key = self.reader.load(entry["name"])
        corrected = cv2.imread(str(self.folder / entry["name"]))
        roi = [0.05, 0.10, 0.95, 0.95]
        fake = FakeSegmenter(self.mask)
        adapter = ProjectedGloveSegmenter(fake, raw, rectifier, key, roi, corrected.shape)
        result = detect_generic(corrected, roi, adapter, [{"prompt": "glove"}], "fixed", 1)
        self.assertTrue(np.array_equal(fake.seen, raw))
        self.assertFalse(np.array_equal(fake.seen, corrected))
        self.assertEqual(len(result["objects"]), 1)
        projected = self.rectifier.remap_mask(str(self.video), self.mask)
        x1, y1, x2, y2 = adapter.bounds
        cropped = projected[y1:y2, x1:x2]
        expected = cv2.boxPoints(cv2.minAreaRect(cv2.findNonZero(cropped))) + [x1, y1]
        np.testing.assert_allclose(result["objects"][0]["polygon"], expected, atol=0.011)
        x, y, w, h = cv2.boundingRect(cv2.findNonZero(cropped))
        self.assertEqual(adapter.last_side_detections[0]["box"], [x, y, x + w, y + h])

    def test_masks_outside_output_roi_are_not_replaced_with_fake_boxes(self):
        raw, rectifier, key = self.reader.load(self.frames[0]["name"])
        fake = FakeSegmenter(self.mask)
        adapter = ProjectedGloveSegmenter(fake, raw, rectifier, key, [0, 0, 0.25, 0.25], raw.shape)
        self.assertEqual(adapter.detect_many(None, []), [])

    def test_circular_roi_masks_out_pixels_inside_square_but_outside_circle(self):
        raw, rectifier, key = self.reader.load(self.frames[0]["name"])
        fake = FakeSegmenter(np.ones(raw.shape[:2], np.uint8))
        roi = {"shape": "circle", "center": [0.5, 0.5], "radius": 0.25}
        adapter = ProjectedGloveSegmenter(fake, raw, rectifier, key, roi, raw.shape)
        detections = adapter.detect_many(None, [])
        self.assertEqual(len(detections), 1)
        cropped = detections[0]["mask"]
        self.assertEqual(cropped[0, 0], 0)
        self.assertEqual(cropped[16, 16], 1)
        self.assertEqual(cropped[0, -1], 0)

    def test_missing_masks_and_mismatched_output_sizes_fail_explicitly(self):
        raw, rectifier, key = self.reader.load(self.frames[0]["name"])
        fake = FakeSegmenter(self.mask)
        adapter = ProjectedGloveSegmenter(fake, raw, rectifier, key, [0, 0, 1, 1], raw.shape)
        with patch.object(fake, "detect_many", return_value=[{
                "class_id": 0, "score": 0.9, "box": [20, 20, 50, 50]}]):
            with self.assertRaisesRegex(ValueError, "未返回原图分割掩码"):
                adapter.detect_many(None, [])
        with self.assertRaisesRegex(ValueError, "图片尺寸"):
            ProjectedGloveSegmenter(fake, raw, rectifier, key, [0, 0, 1, 1], (32, 48, 3))

    def test_worker_uses_raw_pixels_and_keeps_projected_label_provenance(self):
        from auto_obb_annotator import app_gray_gloves as ui
        entry = self.frames[0]
        raw, _, _ = self.reader.load(entry["name"])
        fake = FakeSegmenter(self.mask)
        fake.device = "cpu"
        config = {"source_dir": str(self.folder), "roi": [0, 0, 1, 1],
                  "classes": ui.GRAY_GLOVE_CLASSES, "count_mode": "fixed",
                  "target_count": 2, "confidence": 0.20,
                  "glove_selection_version": ui.GLOVE_SELECTION_VERSION}
        thread = ui.GrayGloveThread([self.folder / entry["name"]], config, set(), fake)
        records = []
        thread.result_signal.connect(records.append)
        with patch.object(ui.torch.cuda, "is_available", return_value=False):
            thread.run()
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0]["objects"]), 1, records[0])
        self.assertTrue(np.array_equal(fake.seen, raw))
        self.assertEqual(records[0]["detection_space"], PROJECTION_WORKFLOW)
        self.assertEqual(records[0]["source_frame"], entry["source_frame"])
