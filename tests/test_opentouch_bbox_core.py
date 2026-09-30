import json
import unittest

from auto_obb_annotator.opentouch_bbox_core import (
    Detection,
    expand_bbox_xyxy,
    make_bbox_record,
    prepare_detections,
    select_verified_right_hand,
)


class OpenTouchBBoxCoreTests(unittest.TestCase):
    def setUp(self):
        self.generic = [Detection((100.0, 80.0, 300.0, 280.0), 0.82)]
        self.right = [Detection((105.0, 84.0, 298.0, 276.0), 0.80)]

    def test_selects_one_verified_right_hand(self):
        result = select_verified_right_hand(
            self.generic,
            self.right,
            [],
            image_width=640,
            image_height=480,
        )
        self.assertEqual(result.reason, "useful")
        self.assertEqual(result.bbox_xyxy, self.generic[0].bbox_xyxy)

    def test_left_right_conflict_returns_null(self):
        left = [Detection((102.0, 82.0, 301.0, 279.0), 0.78)]
        result = select_verified_right_hand(
            self.generic,
            self.right,
            left,
            image_width=640,
            image_height=480,
            side_score_margin=0.05,
        )
        self.assertEqual(result.reason, "left_right_ambiguous")
        self.assertIsNone(result.bbox_xyxy)

    def test_left_hand_elsewhere_is_ignored(self):
        left = [Detection((350.0, 100.0, 500.0, 270.0), 0.95)]
        result = select_verified_right_hand(
            self.generic,
            self.right,
            left,
            image_width=640,
            image_height=480,
        )
        self.assertEqual(result.reason, "useful")

    def test_border_candidate_returns_null(self):
        generic = [Detection((0.0, 80.0, 200.0, 280.0), 0.90)]
        right = [Detection((0.0, 82.0, 198.0, 278.0), 0.88)]
        result = select_verified_right_hand(
            generic,
            right,
            [],
            image_width=640,
            image_height=480,
        )
        self.assertEqual(result.reason, "right_hand_not_fully_visible")
        self.assertIsNone(result.bbox_xyxy)

    def test_multiple_equally_plausible_right_hands_returns_null(self):
        generic = self.generic + [
            Detection((350.0, 90.0, 520.0, 270.0), 0.80)
        ]
        right = self.right + [
            Detection((352.0, 92.0, 518.0, 268.0), 0.78)
        ]
        result = select_verified_right_hand(
            generic,
            right,
            [],
            image_width=640,
            image_height=480,
            side_score_margin=0.05,
        )
        self.assertEqual(result.reason, "multiple_right_candidates")
        self.assertIsNone(result.bbox_xyxy)

    def test_prepare_detections_clips_and_deduplicates(self):
        detections = prepare_detections(
            [[-5, 10, 100, 120], [0, 12, 99, 118], [30, 40, 30, 50]],
            [0.9, 0.8, 0.99],
            image_width=640,
            image_height=480,
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].bbox_xyxy, (0.0, 10.0, 100.0, 120.0))

    def test_record_matches_delivery_contract(self):
        record = make_bbox_record(
            "eat_mcdonalds.hdf5",
            "demo_00",
            0,
            [312.501, 184, 521, 431.499],
        )
        self.assertEqual(
            record,
            {
                "sample_id": "eat_mcdonalds::demo_00::000000",
                "source_file": "eat_mcdonalds.hdf5",
                "clip_id": "demo_00",
                "source_frame_index": 0,
                "bbox_xyxy": [312.5, 184.0, 521.0, 431.5],
            },
        )
        # Ensure None is emitted as JSON null, not as a missing field.
        null_record = make_bbox_record(
            "eat_mcdonalds.hdf5", "demo_00", 1, None
        )
        self.assertIn('"bbox_xyxy": null', json.dumps(null_record))

    def test_expands_bbox_on_every_side(self):
        expanded = expand_bbox_xyxy(
            [100.0, 80.0, 300.0, 280.0],
            image_width=640,
            image_height=480,
            padding_ratio=0.10,
        )
        self.assertEqual(expanded, (80.0, 60.0, 320.0, 300.0))

    def test_expanded_bbox_is_clipped_to_image(self):
        expanded = expand_bbox_xyxy(
            [5.0, 10.0, 105.0, 110.0],
            image_width=120,
            image_height=100,
            padding_ratio=0.20,
        )
        self.assertEqual(expanded, (0.0, 0.0, 120.0, 100.0))


if __name__ == "__main__":
    unittest.main()
