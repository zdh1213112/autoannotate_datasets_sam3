import unittest

import numpy as np

from auto_obb_annotator.protective_glove_core import (
    ProtectiveGloveFilterConfig,
    complete_glove_mask,
    select_protective_gloved_hands,
)


class ProtectiveGloveCoreTests(unittest.TestCase):
    def test_complete_mask_covers_fragments_semantic_box_and_padding(self):
        mask = np.zeros((200, 200), dtype=np.uint8)
        mask[80:135, 70:125] = 1       # palm
        mask[48:76, 72:81] = 1         # separated fingers
        mask[45:77, 92:101] = 1
        mask[140:158, 82:115] = 1      # separated wrist guard

        completed, box = complete_glove_mask(
            mask,
            [50, 40, 130, 160],
            image_width=200,
            image_height=200,
            padding_ratio=0.10,
            fragment_gap_ratio=0.04,
        )

        self.assertTrue(completed[50, 75])
        self.assertTrue(completed[150, 90])
        self.assertLessEqual(box[0], 42.0)
        self.assertLessEqual(box[1], 28.0)
        self.assertGreaterEqual(box[2], 138.0)
        self.assertGreaterEqual(box[3], 172.0)

    def test_mixed_white_gray_and_black_glove_is_accepted_by_default(self):
        image = np.full((160, 200, 3), 220, dtype=np.uint8)
        mask = np.zeros((160, 200), dtype=np.uint8)
        mask[60:145, 55:145] = 1
        image[60:145, 55:145] = (190, 190, 190)  # gray armor
        image[60:95, 60:85] = (30, 30, 30)       # black fingers

        masks, boxes, scores = select_protective_gloved_hands(
            image,
            [mask],
            [[50, 55, 150, 148]],
            [0.86],
            ProtectiveGloveFilterConfig(complete_box_enabled=True),
        )

        self.assertEqual(len(masks), 1)
        self.assertEqual(len(boxes), 1)
        self.assertEqual(scores, [0.86])
        self.assertLess(boxes[0][0], 50)
        self.assertGreater(boxes[0][2], 150)

    def test_optional_dark_ratio_filter_can_still_be_enabled(self):
        image = np.full((160, 200, 3), 220, dtype=np.uint8)
        mask = np.zeros((160, 200), dtype=np.uint8)
        mask[60:145, 55:145] = 1

        selected = select_protective_gloved_hands(
            image,
            [mask],
            [[50, 55, 150, 148]],
            [0.86],
            ProtectiveGloveFilterConfig(min_dark_ratio=0.50),
        )

        self.assertEqual(selected, ([], [], []))

    def test_completion_is_opt_in_for_other_existing_launchers(self):
        image = np.full((160, 200, 3), 80, dtype=np.uint8)
        mask = np.zeros((160, 200), dtype=np.uint8)
        mask[60:145, 55:145] = 1
        original_box = [50, 55, 150, 148]

        masks, boxes, _ = select_protective_gloved_hands(
            image,
            [mask],
            [original_box],
            [0.86],
            ProtectiveGloveFilterConfig(min_dark_ratio=0.5),
        )

        self.assertEqual(len(masks), 1)
        self.assertEqual(boxes, [original_box])

    def test_padded_complete_box_is_clipped_at_frame_edge(self):
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[5:60, 2:50] = 1
        _, box = complete_glove_mask(
            mask,
            [0, 0, 55, 65],
            image_width=120,
            image_height=100,
            padding_ratio=0.20,
        )
        self.assertEqual(box[0], 0.0)
        self.assertEqual(box[1], 0.0)
        self.assertLessEqual(box[2], 120.0)
        self.assertLessEqual(box[3], 100.0)


if __name__ == "__main__":
    unittest.main()
