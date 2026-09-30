import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from auto_obb_annotator.barcode_core import (
    append_record, detect_barcode, export_dataset, load_records,
    polygon_in_roi, select_barcode, yolo_label,
)


class FakeSegmenter:
    def __init__(self, result):
        self.result = result
        self.crop_shape = None

    def detect(self, crop, prompt):
        self.crop_shape = crop.shape
        assert prompt == "barcode"
        return self.result


class BarcodeCoreTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((200, 300, 3), dtype=np.uint8)
        self.roi = [0.2, 0.25, 0.8, 0.75]

    def test_roi_crop_and_rotated_polygon_are_in_full_image_coordinates(self):
        mask = np.zeros((100, 180), dtype=np.uint8)
        points = cv2.boxPoints(((90, 50), (70, 26), 23)).astype(np.int32)
        cv2.fillConvexPoly(mask, points, 1)
        detector = FakeSegmenter(([mask], [[40, 10, 140, 90]], [0.95]))

        result = detect_barcode(self.image, self.roi, detector)

        self.assertEqual(detector.crop_shape, (100, 180, 3))
        self.assertEqual(result["status"], "auto")
        polygon = np.asarray(result["polygon"])
        np.testing.assert_allclose(polygon.mean(axis=0), [150, 100], atol=3)
        self.assertTrue(polygon_in_roi(polygon, self.roi, 300, 200))
        self.assertEqual(len(yolo_label(polygon, 300, 200, "obb").split()), 9)
        self.assertEqual(len(yolo_label(polygon, 300, 200, "hbb").split()), 5)

    def test_separate_competing_candidate_requires_review(self):
        result = select_barcode(
            self.image, self.roi, [],
            [[20, 20, 70, 70], [110, 20, 160, 70]], [0.93, 0.81],
        )
        self.assertEqual(result["status"], "review")
        self.assertEqual(result["candidates"], 2)
        self.assertEqual(len(result["polygon"]), 4)

    def test_missing_candidate_requires_review(self):
        result = select_barcode(self.image, self.roi, [], [], [])
        self.assertEqual(result["status"], "review")
        self.assertIsNone(result["polygon"])

    def test_latest_review_wins_and_export_omits_unresolved(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "color"
            source.mkdir()
            for name in ("one.png", "two.png", "three.png"):
                self.assertTrue(cv2.imwrite(str(source / name), self.image))
            log = root / "review.jsonl"
            candidate = [[90, 70], [150, 70], [150, 110], [90, 110]]
            append_record(log, {"name": "one.png", "status": "review",
                                "polygon": candidate})
            append_record(log, {"name": "one.png", "status": "confirmed",
                                "polygon": candidate})
            append_record(log, {"name": "two.png", "status": "empty_confirmed",
                                "polygon": None})
            append_record(log, {"name": "three.png", "status": "review",
                                "polygon": None})
            records = load_records(log)
            self.assertEqual(records["one.png"]["status"], "confirmed")

            target, positive, negative = export_dataset(source, root, records, "obb")

            self.assertEqual((positive, negative), (1, 1))
            self.assertTrue((target / "dataset.yaml").is_file())
            self.assertEqual(len(list((target / "images" / "train").iterdir())), 1)
            self.assertEqual(len(list((target / "images" / "val").iterdir())), 1)
            labels = list((target / "labels").rglob("*.txt"))
            self.assertEqual({path.name for path in labels}, {"one.txt", "two.txt"})
            self.assertEqual(next(path for path in labels if path.name == "two.txt").read_text(), "")


if __name__ == "__main__":
    unittest.main()
