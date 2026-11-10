"""Representative export samples must match final labels and preserve originals."""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from auto_obb_annotator.dataset_preview import choose_preview_records
from auto_obb_annotator.generic_core import export_generic


CLASSES = [{"name": "left", "prompt": "left glove"},
           {"name": "right", "prompt": "right glove"}]


class DatasetPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.image = np.zeros((200, 200, 3), np.uint8)
        self.diamond = [[60, 80], [100, 40], [140, 80], [100, 120]]

    def image_record(self, name, objects, status="confirmed"):
        self.assertTrue(cv2.imwrite(str(self.source / name), self.image))
        return {"name": name, "status": status, "objects": objects}

    def test_samples_are_reproducible_and_cover_splits_classes_cameras_and_negatives(self):
        records = [{"name": f"frame_{index:04d}.png", "camera_id": f"cam_{index % 4}",
                    "objects": [{"class_id": index % 2}]} for index in range(60)]
        records[-1]["objects"] = []
        val_names = {records[1]["name"], records[8]["name"]}
        samples = choose_preview_records(records, val_names)
        self.assertEqual(len(samples), 10)
        self.assertEqual(len({record["name"] for record in samples}), 10)
        self.assertEqual(samples, choose_preview_records(list(reversed(records)), val_names))
        self.assertEqual({record["camera_id"] for record in samples},
                         {"cam_0", "cam_1", "cam_2", "cam_3"})
        self.assertEqual({obj["class_id"] for record in samples for obj in record["objects"]}, {0, 1})
        self.assertTrue(any(record["name"] in val_names for record in samples))
        self.assertTrue(any(record["name"] not in val_names for record in samples))
        self.assertTrue(any(not record["objects"] for record in samples))

    def test_small_dataset_previews_final_hbb_and_obb_without_touching_training_images(self):
        selected = self.image_record("hand.png", [{"class_id": 0, "polygon": self.diamond}])
        selected["candidates"] = [{"class_id": 1, "selected": False,
                                   "polygon": [[160, 160], [190, 160], [190, 190], [160, 190]]}]
        empty = self.image_record("empty.png", [], "empty_confirmed")
        review = self.image_record("review.png", [], "review")
        excluded = self.image_record("excluded.png", [])
        records = {record["name"]: record for record in (selected, empty, review, excluded)}
        original = (self.source / "hand.png").read_bytes()
        for box_format in ("hbb", "obb"):
            with self.subTest(box_format=box_format):
                target, _, _ = export_generic(self.source, self.root, records, CLASSES,
                                              box_format, {"hand.png", "empty.png", "review.png"})
                folder = target / "visualizations"
                self.assertEqual(len(list(folder.glob("*.jpg"))), 3)  # two samples + overview
                self.assertTrue((folder / "overview.jpg").is_file())
                self.assertFalse(any("review" in path.name or "excluded" in path.name
                                     for path in folder.glob("*.jpg")))
                readme = (folder / "README.md").read_text()
                self.assertIn("2 张图片中选择 2 张", readme)
                self.assertIn("left", readme)
                self.assertIn(box_format.upper(), readme)
                preview = cv2.imread(str(next(folder.glob("*__hand.jpg"))))
                self.assertIsNotNone(preview)
                # HBB is the diamond's axis-aligned enclosure; OBB follows its
                # diagonal edge. The 54px header leaves label coordinates intact.
                x, y = (60, 60) if box_format == "hbb" else (80, 60)
                b, g, r = preview[y + 54, x]
                self.assertGreater(int(g), 120)
                self.assertLess(int(b), 100)
                self.assertLess(int(r), 100)
                self.assertLess(int(preview[170 + 54, 170].max()), 30)
                training = next((target / "images").rglob("hand.png"))
                self.assertEqual(training.read_bytes(), original)
                self.assertEqual((self.source / "hand.png").read_bytes(), original)

    def test_large_dataset_writes_ten_samples_and_one_overview(self):
        records = {}
        for index in range(25):
            record = self.image_record(f"image_{index:03d}.png", [
                {"class_id": index % 2, "polygon": self.diamond}])
            record["camera_id"] = f"cam_{index % 4}"
            records[record["name"]] = record
        target, _, _ = export_generic(self.source, self.root, records, CLASSES, "hbb")
        folder = target / "visualizations"
        self.assertEqual(len(list(folder.glob("*.jpg"))), 11)
        self.assertTrue(list(folder.glob("train__*.jpg")))
        self.assertTrue(list(folder.glob("val__*.jpg")))
        self.assertIn("25 张图片中选择 10 张", (folder / "README.md").read_text())
        self.assertEqual(len(list((target / "images").rglob("*.png"))), 25)
        self.assertEqual(len(list((target / "labels").rglob("*.txt"))), 25)

    def test_all_empty_dataset_still_has_visual_checks(self):
        records = {name: self.image_record(name, [], "empty_confirmed")
                   for name in ("a.png", "b.png", "c.png")}
        target, images, objects = export_generic(self.source, self.root, records, CLASSES, "hbb")
        self.assertEqual((images, objects), (0, 0))
        self.assertEqual(len(list((target / "visualizations").glob("*.jpg"))), 4)
        for label in (target / "labels").rglob("*.txt"):
            self.assertEqual(label.read_text(), "")
