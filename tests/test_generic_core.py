import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from auto_obb_annotator.generic_core import (
    detect_generic, export_generic, normalize_sample_percent,
    normalize_sample_seed, normalize_selection_rule, sample_image_files,
    select_detections,
    validate_classes,
)
from auto_obb_annotator.generic_region import (
    crop_region, normalize_region, polygon_in_region, region_bounds,
)


CLASSES = [{"name": "cup", "prompt": "cup"},
           {"name": "瓶子", "prompt": "water bottle"}]


class FakeSegmenter:
    def __init__(self, detections):
        self.detections = detections
        self.prompts = None
        self.crop_shape = None
        self.crop = None

    def detect_many(self, crop, prompts):
        self.prompts = prompts
        self.crop_shape = crop.shape
        self.crop = crop.copy()
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

    def test_sample_images_randomly_and_reproducibly_with_seed(self):
        files = [Path(f"{index:02d}.png") for index in range(10)]
        self.assertEqual(sample_image_files(files, 100, 42), files)
        first = sample_image_files(files, 30, 42)
        self.assertEqual(first, sample_image_files(list(reversed(files)), 30, 42))
        self.assertEqual(first, [files[1], files[0], files[4]])
        self.assertNotEqual(first, sample_image_files(files, 30, 43))
        self.assertEqual(len(sample_image_files(files, 1, 42)), 1)
        self.assertEqual(sample_image_files([], 50, 42), [])
        self.assertEqual(normalize_sample_percent(), 100)
        self.assertEqual(normalize_sample_seed(42), 42)
        for invalid in (0, 101, 2.5, True, "50"):
            with self.assertRaises(ValueError):
                sample_image_files(files, invalid, 42)
        for invalid_seed in (-1, True, 2.5, "42"):
            with self.assertRaises(ValueError):
                sample_image_files(files, 30, invalid_seed)

    def test_fixed_count_selects_top_n_without_review_for_extra_candidates(self):
        segmenter = FakeSegmenter(self.detections)
        result = detect_generic(self.image, self.roi, segmenter, CLASSES, "fixed", 2)
        self.assertEqual(segmenter.prompts, ["cup", "water bottle"])
        self.assertEqual(segmenter.crop_shape, (120, 240, 3))
        self.assertEqual(result["candidate_count"], 3)
        self.assertEqual([obj["class_id"] for obj in result["objects"]], [0, 1])
        self.assertEqual([obj["score"] for obj in result["objects"]], [0.95, 0.88])
        self.assertEqual([obj["candidate_id"] for obj in result["objects"]], [0, 1])
        self.assertEqual([item["selected"] for item in result["candidates"]],
                         [True, True, False])
        self.assertEqual([item["score"] for item in result["candidates"]],
                         [0.95, 0.88, 0.80])
        self.assertEqual(result["status"], "auto")

    def test_fixed_count_still_reviews_when_fewer_targets_found(self):
        result = select_detections(self.image, self.roi, self.detections[:1],
                                   2, "fixed", 2)
        self.assertEqual(len(result["objects"]), 1)
        self.assertEqual(result["status"], "review")

    def test_fixed_count_can_select_largest_area_instead_of_highest_score(self):
        detections = [
            {"class_id": 0, "box": [10, 10, 40, 40], "score": 0.96},
            {"class_id": 0, "box": [50, 10, 150, 90], "score": 0.75},
            {"class_id": 1, "box": [170, 10, 210, 60], "score": 0.85},
        ]
        by_score = select_detections(self.image, self.roi, detections,
                                     2, "fixed", 1)
        by_area = select_detections(self.image, self.roi, detections,
                                    2, "fixed", 1, selection_rule="area")
        self.assertEqual(by_score["objects"][0]["score"], 0.96)
        self.assertEqual(by_area["objects"][0]["score"], 0.75)
        self.assertEqual(by_area["objects"][0]["candidate_id"], 2)
        self.assertEqual([item["selected"] for item in by_area["candidates"]],
                         [False, False, True])
        self.assertEqual(normalize_selection_rule(), "confidence")
        with self.assertRaises(ValueError):
            normalize_selection_rule("unknown")

    def test_area_rule_keeps_larger_box_when_predictions_overlap(self):
        detections = [
            {"class_id": 0, "box": [10, 10, 80, 80], "score": 0.95},
            {"class_id": 0, "box": [12, 12, 90, 90], "score": 0.80},
        ]
        by_area = select_detections(self.image, self.roi, detections,
                                    2, "fixed", 1, selection_rule="area")
        self.assertEqual(by_area["objects"][0]["score"], 0.80)
        self.assertEqual(len(by_area["candidates"]), 2)
        self.assertEqual([item["selected"] for item in by_area["candidates"]],
                         [False, True])

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
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual([item["selected"] for item in result["candidates"]],
                         [True, False])
        self.assertEqual(result["status"], "review")

    def test_review_reasons_can_be_selected_independently(self):
        near_edge = [{"class_id": 0, "box": [0, 30, 30, 70], "score": 0.93}]
        default = select_detections(self.image, self.roi, near_edge, 2, "all", 1)
        ignored = select_detections(self.image, self.roi, near_edge, 2,
                                    "all", 1, ["low_score"])
        self.assertEqual(default["status"], "review")
        self.assertIn("边缘", default["reason"])
        self.assertEqual(ignored["status"], "auto")
        self.assertIn("已关闭", ignored["reason"])
        low = [{**near_edge[0], "score": 0.65}]
        low_only = select_detections(self.image, self.roi, low, 2,
                                     "all", 1, ["low_score"])
        self.assertEqual(low_only["status"], "review")
        self.assertIn("低置信度", low_only["reason"])
        missing = select_detections(self.image, self.roi, [], 2, "all", 1, [])
        self.assertEqual(missing["status"], "auto")
        short = select_detections(self.image, self.roi, [], 2,
                                  "fixed", 2, ["short_count"])
        self.assertEqual(short["status"], "review")
        self.assertIn("少于", short["reason"])

    def test_circle_region_masks_crop_and_excludes_outside_candidates(self):
        circle = {"shape": "circle", "center": [0.5, 0.5], "radius": 0.3}
        self.assertEqual(normalize_region(circle), circle)
        self.assertEqual(region_bounds(circle, 300, 200), (90, 40, 210, 160))
        crop = crop_region(np.full_like(self.image, 255), circle)
        self.assertEqual(crop.shape, (120, 120, 3))
        self.assertEqual(crop[0, 0, 0], 0)
        self.assertEqual(crop[60, 60, 0], 255)
        detections = [
            {"class_id": 0, "box": [0, 0, 25, 25], "score": 0.99},
            {"class_id": 1, "box": [40, 35, 80, 75], "score": 0.88},
        ]
        segmenter = FakeSegmenter(detections)
        result = detect_generic(np.full_like(self.image, 255), circle,
                                segmenter, CLASSES, "fixed", 1)
        self.assertEqual(segmenter.crop.shape, (120, 120, 3))
        self.assertEqual(segmenter.crop[0, 0, 0], 0)
        self.assertEqual(segmenter.crop[60, 60, 0], 255)
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual([obj["class_id"] for obj in result["objects"]], [1])
        self.assertEqual(result["status"], "auto")
        self.assertTrue(polygon_in_region(result["objects"][0]["polygon"],
                                          circle, 300, 200))

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
                ], "candidates": [
                    {"class_id": 0, "polygon": polygon_a, "selected": True},
                    {"class_id": 1, "polygon": polygon_b, "selected": True},
                    {"class_id": 0, "polygon": [[210, 60], [240, 60],
                                                  [240, 110], [210, 110]],
                     "selected": False},
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
            with self.assertRaisesRegex(ValueError, "至少需要 2 张"):
                export_generic(source, root, records, CLASSES, "obb",
                               {"one.png", "three.png"})
            self.assertEqual([line.split()[0] for line in lines], ["0", "1"])
            self.assertIn('1: "瓶子"', (target / "dataset.yaml").read_text())


if __name__ == "__main__":
    unittest.main()
