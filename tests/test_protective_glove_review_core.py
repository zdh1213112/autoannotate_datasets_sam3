import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from auto_obb_annotator.protective_glove_review_core import (
    apply_temporal_side,
    assign_left_right_classes,
    export_dataset,
    make_glove_record,
    normalize_record_sides,
    select_prompted_glove_objects,
)


class ProtectiveGloveReviewCoreTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((120, 240, 3), dtype=np.uint8)

    @staticmethod
    def box(x1, y1=25, x2=None, y2=95):
        x2 = x2 if x2 is not None else x1 + 40
        return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    def test_position_assignment_is_independent_of_detection_order(self):
        objects = [
            {"class_id": 0, "score": 0.8, "polygon": self.box(150)},
            {"class_id": 0, "score": 0.9, "polygon": self.box(20)},
        ]
        assign_left_right_classes(objects, 240)
        self.assertEqual([obj["class_id"] for obj in objects], [1, 0])

    def test_explicit_semantic_prompts_can_auto_confirm_both_sides(self):
        left = np.zeros((120, 240), dtype=np.uint8)
        left[25:95, 20:60] = 1
        right = np.zeros((120, 240), dtype=np.uint8)
        right[25:95, 150:190] = 1
        result = select_prompted_glove_objects(
            self.image,
            {
                "left hand wearing a protective work glove":
                    ([left], [[20, 25, 60, 95]], [0.90]),
                "right hand wearing a protective work glove":
                    ([right], [[150, 25, 190, 95]], [0.92]),
            },
        )
        self.assertEqual(result["status"], "auto")
        self.assertEqual({obj["class_id"] for obj in result["objects"]}, {0, 1})

    def test_two_generic_candidates_can_be_auto_in_screen_mode(self):
        result = select_prompted_glove_objects(
            self.image,
            {"hand wearing a protective work glove":
             ([], [[20, 25, 60, 95], [150, 25, 190, 95]], [0.9, 0.9])},
        )
        self.assertEqual(result["status"], "auto")
        self.assertEqual(len(result["objects"]), 2)
        self.assertEqual([obj["class_id"] for obj in result["objects"]], [0, 1])

    def test_border_contact_alone_does_not_force_review(self):
        result = select_prompted_glove_objects(self.image, {
            "hand": ([], [[0, 25, 60, 120], [150, 25, 239, 119]], [.95, .92]),
        })
        self.assertEqual(result["status"], "auto")

    def test_genuine_ambiguities_still_require_review(self):
        for boxes, scores in (
            ([], []),
            ([[20, 25, 60, 95]], [.95]),
            ([[20, 25, 60, 95], [150, 25, 190, 95]], [.95, .40]),
            ([[20, 25, 100, 95], [60, 25, 140, 95]], [.95, .95]),
            ([[20, 10, 60, 50], [21, 70, 61, 110]], [.95, .95]),
            ([[20, 25, 60, 95], [90, 25, 130, 95], [150, 25, 190, 95]], [.95] * 3),
        ):
            with self.subTest(boxes=boxes, scores=scores):
                result = select_prompted_glove_objects(
                    self.image, {"hand": ([], boxes, scores)})
                self.assertEqual(result["status"], "review")

    def test_generic_score_is_used_when_side_prompts_are_weak(self):
        boxes = [[20, 25, 60, 95], [150, 25, 190, 95]]
        result = select_prompted_glove_objects(self.image, {
            "left hand": ([], boxes, [.4, .45]),
            "right hand": ([], boxes, [.45, .4]),
            "hand": ([], boxes, [.9, .95]),
        }, side_mapping="mirror")
        self.assertEqual(result["status"], "auto")
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual([obj["class_id"] for obj in result["objects"]], [1, 0])

    def test_missing_scores_do_not_become_high_confidence(self):
        result = select_prompted_glove_objects(self.image, {
            "hand": ([], [[20, 25, 60, 95], [150, 25, 190, 95]], []),
        })
        self.assertEqual(result["status"], "review")

    def test_duplicate_hits_from_both_side_prompts_are_merged(self):
        masks = []
        boxes = []
        for x1, x2 in ((20, 60), (150, 190)):
            mask = np.zeros((120, 240), dtype=np.uint8)
            mask[25:95, x1:x2] = 1
            masks.append(mask)
            boxes.append([x1, 25, x2, 95])
        result = select_prompted_glove_objects(
            self.image,
            {
                "left hand wearing a protective work glove":
                    (masks, boxes, [0.90, 0.89]),
                "right hand wearing a protective work glove":
                    (masks, boxes, [0.91, 0.88]),
            },
        )
        self.assertEqual(result["status"], "auto")
        self.assertEqual(result["candidate_count"], 2)

    def test_single_hand_uses_semantic_prompt_even_when_on_screen_left(self):
        result = select_prompted_glove_objects(
            self.image,
            {
                "left hand wearing a protective work glove":
                    ([], [[20, 25, 60, 95]], [0.51]),
                "right hand wearing a protective work glove":
                    ([], [[20, 25, 60, 95]], [0.93]),
                "hand wearing a protective work glove":
                    ([], [[20, 25, 60, 95]], [0.91]),
            },
        )
        self.assertEqual(result["status"], "auto")
        self.assertEqual(result["objects"][0]["class_id"], 1)
        self.assertEqual(result["objects"][0]["side_source"], "semantic")

    def test_single_hand_semantic_tie_stays_review(self):
        result = select_prompted_glove_objects(
            self.image,
            {
                "left hand wearing a protective work glove":
                    ([], [[20, 25, 60, 95]], [0.82]),
                "right hand wearing a protective work glove":
                    ([], [[20, 25, 60, 95]], [0.80]),
            },
        )
        self.assertEqual(result["status"], "review")
        self.assertIsNone(result["objects"][0]["class_id"])

    def test_temporal_side_follows_a_right_hand_across_screen_position(self):
        payload = {
            "objects": [{"class_id": None, "side": "unknown",
                         "side_source": "unresolved", "score": 0.9,
                         "polygon": self.box(20)}],
            "status": "review", "candidate_count": 1,
        }
        previous = [{"class_id": 1, "side": "right", "side_source": "manual",
                     "polygon": self.box(24), "score": 1.0}]
        result = apply_temporal_side(payload, previous, 240, 120)
        self.assertEqual(result["status"], "auto")
        self.assertEqual(result["objects"][0]["class_id"], 1)
        self.assertEqual(result["objects"][0]["side_source"], "temporal")

    def test_single_hand_semantic_and_temporal_results_are_exportable(self):
        for source in ("semantic", "temporal"):
            record = make_glove_record(
                f"{source}.png", 240, 120,
                [{"class_id": 1, "score": .9, "side_source": source,
                  "polygon": self.box(20)}],
                "auto", "automatic", 1)
            self.assertEqual(record["status"], "auto")
            self.assertEqual(record["objects"][0]["side_source"], source)

    def test_legacy_single_hand_position_guess_is_not_reused(self):
        record = normalize_record_sides({
            "name": "legacy.png", "status": "auto",
            "objects": [{"class_id": 0, "side": "left",
                          "side_source": "position_pair",
                          "polygon": self.box(20), "score": .9}],
        })
        self.assertEqual(record["status"], "review")
        self.assertIsNone(record["objects"][0]["class_id"])

    def test_export_writes_left_right_classes_and_omits_review(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "color"
            source.mkdir()
            for name in ("one.png", "two.png", "three.png"):
                self.assertTrue(cv2.imwrite(str(source / name), self.image))
            left = make_glove_record(
                "one.png", 240, 120,
                [{"class_id": 0, "score": 1.0, "polygon": self.box(20)}],
                "confirmed", "manual")
            empty = make_glove_record("two.png", 240, 120, [],
                                      "empty_confirmed", "manual")
            review = make_glove_record(
                "three.png", 240, 120,
                [{"class_id": 1, "score": 0.4, "polygon": self.box(150)}],
                "review", "check")
            target, images, objects = export_dataset(
                source, root, {item["name"]: item for item in (left, empty, review)}, "hbb")
            self.assertEqual((images, objects), (1, 1))
            self.assertIn("0: \"left\"", (target / "dataset.yaml").read_text())
            self.assertIn("1: \"right\"", (target / "dataset.yaml").read_text())
            labels = list((target / "labels").rglob("*.txt"))
            self.assertEqual({path.name for path in labels}, {"one.txt", "two.txt"})


if __name__ == "__main__":
    unittest.main()
