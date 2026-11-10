"""MCAP camera selection, sampling, provenance, and calibration regression tests."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import yaml

from auto_obb_annotator.gray_glove_video import (
    collect_videos, extract_video_batch, frame_camera_ids,
)
from auto_obb_annotator.orbbec_undistort import OrbbecUndistorter
from auto_obb_annotator.gray_glove_projection import RawFrameReader


HAS_MCAP = all(importlib.util.find_spec(name) is not None
               for name in ("mcap", "mcap_protobuf", "av"))


@unittest.skipUnless(HAS_MCAP, "MCAP optional dependencies are not installed")
class GrayGloveMcapTests(unittest.TestCase):
    def setUp(self):
        from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
        from mcap_protobuf.writer import Writer

        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        session = self.root / "EGO-4B_TEST123_20261109_150103"
        part = session / "part0001"
        part.mkdir(parents=True)
        self.mcap = part / "EGO-4B_TEST123_20261109_150103_part0001.mcap"
        descriptor = descriptor_pb2.FileDescriptorProto()
        descriptor.name = "compressed_video_test.proto"
        descriptor.package = "foxglove"
        descriptor.syntax = "proto3"
        message = descriptor.message_type.add()
        message.name = "CompressedVideo"
        for number, (name, field_type) in enumerate(
                (("data", 12), ("format", 9)), start=1):
            field = message.field.add()
            field.name = name
            field.number = number
            field.label = 1
            field.type = field_type
        pool = descriptor_pool.DescriptorPool()
        pool.Add(descriptor)
        Video = message_factory.GetMessageClass(
            pool.FindMessageTypeByName("foxglove.CompressedVideo"))
        with self.mcap.open("wb") as stream:
            writer = Writer(stream)
            for index in range(5):
                for topic, value in (("/camera/left", 40), ("/camera/right", 130)):
                    image = np.full((48, 64, 3), value + index, np.uint8)
                    ok, encoded = cv2.imencode(".jpg", image)
                    self.assertTrue(ok)
                    writer.write_message(topic, Video(data=encoded.tobytes(), format="jpeg"),
                                         log_time=1_000_000_000 + index * 100_000_000)
            writer.finish()
        cameras = []
        for index in range(4):
            cameras.append({
                "id": f"cam_{index}", "distortion_model": "KB",
                "image_width": 64, "image_height": 48,
                "intrinsics": {"fx": 40, "fy": 40, "cx": 32, "cy": 24},
                "distortion": {"k1": 0.05, "k2": 0, "k3": 0, "k4": 0},
            })
        self.calibration = session / "EGO-4B_TEST123_20261109_150103_params_cam_64x48.yaml"
        self.calibration.write_text(yaml.safe_dump({"cameras": cameras}), encoding="utf-8")

    def test_sample_selected_camera_and_reuse(self):
        self.assertEqual(collect_videos([self.mcap.parent.parent]), [self.mcap])
        output = self.root / "out"
        folder, count, reused = extract_video_batch(
            [self.mcap], output, "seconds", 0.2,
            mcap_camera_ids=("cam_0",))
        self.assertEqual((count, reused), (3, 0))
        manifest = json.loads(next(folder.glob("extract_*.json")).read_text())
        self.assertTrue(manifest["complete"])
        self.assertEqual([item["source_frame"] for item in manifest["frames"]], [0, 2, 4])
        self.assertEqual(set(frame_camera_ids(folder).values()), {"cam_0"})
        self.assertEqual(extract_video_batch(
            [self.mcap], output, "seconds", 0.2,
            mcap_camera_ids=("cam_0",))[1:], (3, 3))

    def test_incomplete_files_are_reported_before_batch_writes(self):
        empty = self.root / "empty.mcap"
        empty.touch()
        truncated = self.root / "truncated.mcap"
        truncated.write_bytes(self.mcap.read_bytes()[:-8])
        output = self.root / "out"
        with self.assertRaises(ValueError) as error:
            extract_video_batch([self.mcap, empty, truncated], output,
                                "seconds", 1.0, mcap_camera_ids=("cam_0",))
        self.assertIn("empty.mcap: 文件为空或过短", str(error.exception))
        self.assertIn("truncated.mcap: 缺少 MCAP 结束标记", str(error.exception))
        self.assertFalse(output.exists())

    def test_mcap_undistortion_uses_parent_calibration(self):
        rectifier = OrbbecUndistorter([self.mcap], mcap_camera_ids=("cam_1",))
        self.assertEqual(set(rectifier.metadata), {f"{self.mcap}::/camera/right"})
        folder, count, reused = extract_video_batch(
            [self.mcap], self.root / "rectified", "frames", 2,
            frame_transform=rectifier, transform_info=rectifier.metadata,
            mcap_camera_ids=("cam_1",))
        self.assertEqual((count, reused), (3, 0))
        self.assertEqual(set(frame_camera_ids(folder).values()), {"cam_1"})
        self.assertEqual(cv2.imread(str(next(folder.glob("*.jpg")))).shape, (48, 64, 3))

    def test_old_mcap_frames_recover_correct_camera_and_message_index(self):
        rectifier = OrbbecUndistorter([self.mcap], mcap_camera_ids=("cam_0", "cam_1"))
        folder, _, _ = extract_video_batch(
            [self.mcap], self.root / "rectified", "frames", 2,
            frame_transform=rectifier, transform_info=rectifier.metadata,
            mcap_camera_ids=("cam_0", "cam_1"))
        # Simulate an existing extraction made before raw caching was added.
        for path in (folder / "raw_frames").glob("*.png"):
            path.unlink()
        reader = RawFrameReader(folder)
        self.addCleanup(reader.close)
        for image in sorted(folder.glob("*.jpg"), key=reader.sort_key):
            entry = reader.entries[image.name]
            raw, _, source_key = reader.load(image.name)
            expected = (40 if entry["camera_id"] == "cam_0" else 130) + entry["source_frame"]
            self.assertAlmostEqual(float(raw.mean()), expected, delta=1)
            self.assertTrue(source_key.endswith("::" + entry["topic"]))
        self.assertEqual(len(reader._mcap_readers), 2)
        self.assertEqual(len(list((folder / "raw_frames").glob("*.png"))), 6)
