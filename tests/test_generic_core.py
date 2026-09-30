import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from auto_obb_annotator.generic_core import (
    detect_generic, export_generic, select_detections, validate_classes,
)


CLASSES = [{"name": "cup", "prompt": "cup"},
           {"name": "瓶子", "prompt": "water bottle"}]


class FakeSegmenter:
    def __init__(self, detections):
        self.detections = detections
        self.prompts = None
        self.crop_shape = None

    def detect_many(self, crop, prompts):
        self.prompts = prompts
        self.crop_shape = crop.shape
        return self.detections


class GenericCoreTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((200, 300, 3), dtype=np.uint8)
        self.roi = [0.1, 0.2, 0.9, 0.8]
        self.detections = [
            {"class_id": 0, "box": [20, 20, 60, 70], "score": 0.95},
            {"class_id": 1, "box": [90, 20, 130, 70], "score": 0.88},
            {"class_id": 0, "box": [160, 20, 200, 70], "score": 0.80},
        ]

    def test_custom_classes_need_distinct_names_and_prompts(self):
        self.assertEqual(validate_classes(CLASSES), CLASSES)
        with self.assertRaises(ValueError):
            validate_classes([CLASSES[0], {"name": "cup", "prompt": "glass"}])
        with self.assertRaises(ValueError):
            validate_classes([CLASSES[0], {"name": "glass", "prompt": "cup"}])

    def test_fixed_count_selects_top_n_and_flags_extra_high_candidate(self):
        segmenter = FakeSegmenter(self.detections)
        result = detect_generic(self.image, self.roi, segmenter, CLASSES, "fixed", 2)
        self.assertEqual(segmenter.prompts, ["cup", "water bottle"])
        self.assertEqual(segmenter.crop_shape, (120, 240, 3))
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual([obj["class_id"] for obj in result["objects"]], [0, 1])
        self.assertEqual(result["status"], "review")

    def test_all_mode_keeps_every_distinct_candidate(self):
        result = select_detections(self.image, self.roi, self.detections, 2, "all", 1)
        self.assertEqual(len(result["objects"]), 3)
        self.assertEqual(result["status"], "auto")

    def test_duplicate_across_classes_requires_review(self):
        detections = [
            {"class_id": 0, "box": [20, 20, 80, 80], "score": 0.91},
            {"class_id": 1, "box": [21, 21, 81, 81], "score": 0.87},
        ]
        result = select_detections(self.image, self.roi, detections, 2, "all", 1)
        self.assertEqual(len(result["objects"]), 1)
        self.assertEqual(result["status"], "review")

    def test_export_writes_multiple_classes_and_excludes_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "images"
            source.mkdir()
            for name in ("one.png", "two.png", "three.png"):
                self.assertTrue(cv2.imwrite(str(source / name), self.image))
            polygon_a = [[50, 60], [90, 60], [90, 110], [50, 110]]
            polygon_b = [[160, 60], [200, 60], [200, 110], [160, 110]]
            records = {
                "one.png": {"name": "one.png", "status": "confirmed", "objects": [
                    {"class_id": 0, "polygon": polygon_a},
                    {"class_id": 1, "polygon": polygon_b},
                ]},
                "two.png": {"name": "two.png", "status": "empty_confirmed", "objects": []},
                "three.png": {"name": "three.png", "status": "review", "objects": []},
            }
            target, images, objects = export_generic(source, root, records, CLASSES, "obb")
            self.assertEqual((images, objects), (1, 2))
            self.assertEqual(len(list((target / "images" / "train").iterdir())), 1)
            self.assertEqual(len(list((target / "images" / "val").iterdir())), 1)
            labels = list((target / "labels").rglob("*.txt"))
            self.assertEqual({path.name for path in labels}, {"one.txt", "two.txt"})
            lines = next(path for path in labels if path.name == "one.txt").read_text().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual([line.split()[0] for line in lines], ["0", "1"])
            self.assertIn('1: "瓶子"', (target / "dataset.yaml").read_text())


if __name__ == "__main__":
    unittest.main()
