"""Selected-frame Orbbec KB undistortion and provenance checks."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import yaml

from auto_obb_annotator.gray_glove_video import extract_video_batch, frame_camera_ids
from auto_obb_annotator.orbbec_undistort import (
    OrbbecUndistorter, calibration_for_video, camera_id_for_video,
)


def make_video(path: Path, size=(80, 60)):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             10, size)
    if not writer.isOpened():
        raise RuntimeError("测试环境无法编码 MP4")
    for index in range(10):
        image = np.zeros((size[1], size[0], 3), np.uint8)
        cv2.rectangle(image, (5 + index, 5), (size[0] - 5, size[1] - 5),
                      (40, 160, 240), -1)
        writer.write(image)
    writer.release()


def make_calibration(path: Path, size=(80, 60)):
    cameras = []
    for index in range(4):
        cameras.append({
            "id": f"cam_{index}", "distortion_model": "KB",
            "image_width": size[0], "image_height": size[1],
            "intrinsics": {"fx": 45.0, "fy": 45.0,
                           "cx": size[0] / 2, "cy": size[1] / 2},
            "distortion": {"k1": 0.08 + index * 0.01, "k2": -0.01,
                           "k3": 0.001, "k4": 0.0},
        })
    path.write_text(yaml.safe_dump({
        "calibration_info": {"serial_number": "TEST123"},
        "cameras": cameras,
    }), encoding="utf-8")


class OrbbecUndistortTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.session = Path(self.temp.name) / "session"
        self.session.mkdir()
        self.calibration = self.session / "EGO_calibration_camera.yaml"
        make_calibration(self.calibration)
        self.video = self.session / "EGO_camera_left.mp4"
        make_video(self.video)

    def test_camera_mapping_and_automatic_calibration(self):
        mapping = {"left": "cam_0", "right": "cam_1",
                   "right_side": "cam_2", "left_side": "cam_3"}
        for suffix, expected in mapping.items():
            self.assertEqual(camera_id_for_video(
                self.session / f"EGO_camera_{suffix}.mp4"), expected)
        self.assertEqual(calibration_for_video(self.video), self.calibration)
        with self.assertRaisesRegex(ValueError, "无法判断视频"):
            camera_id_for_video(self.session / "unknown.mp4")

    def test_only_selected_frames_are_undistorted_and_resumable(self):
        rectifier = OrbbecUndistorter([self.video])
        output = self.session / "annotations"
        folder, count, reused = extract_video_batch(
            [self.video], output, "frames", 4, end_seconds=0.9,
            frame_transform=rectifier, transform_info=rectifier.metadata)
        self.assertEqual((count, reused), (3, 0))
        self.assertEqual(len(list(folder.glob("*.jpg"))), 3)
        manifest = json.loads(next(folder.glob("extract_*.json")).read_text())
        self.assertEqual([item["source_frame"] for item in manifest["frames"]],
                         [0, 4, 8])
        transform = manifest["settings"]["transform"][str(self.video)]
        self.assertEqual(transform["camera_id"], "cam_0")
        self.assertEqual(transform["image_size"], [80, 60])
        self.assertEqual(len(transform["new_camera_matrix"]), 3)
        self.assertEqual(frame_camera_ids(folder),
                         {item["name"]: "cam_0" for item in manifest["frames"]})
        cap = cv2.VideoCapture(str(self.video))
        ok, raw = cap.read()
        cap.release()
        self.assertTrue(ok)
        corrected = cv2.imread(str(folder / manifest["frames"][0]["name"]))
        self.assertEqual(corrected.shape, raw.shape)
        self.assertGreater(float(np.mean(cv2.absdiff(raw, corrected))), 2.0)
        self.assertEqual(extract_video_batch(
            [self.video], output, "frames", 4, end_seconds=0.9,
            frame_transform=rectifier, transform_info=rectifier.metadata),
            (folder, 3, 3))

    def test_wrong_calibration_resolution_is_rejected_before_writing(self):
        make_calibration(self.calibration, size=(100, 60))
        with self.assertRaisesRegex(ValueError, "实际画面"):
            OrbbecUndistorter([self.video])
