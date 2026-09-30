import unittest

from auto_obb_annotator.app import make_dataset_record


class DatasetRecordTests(unittest.TestCase):
    def test_record_keeps_metadata_without_decoded_image(self):
        labels = [(0, [0.5, 0.5, 0.25, 0.25])]

        record = make_dataset_record(
            "/input/frame.jpg", "frame.jpg", labels, 1600, 1300
        )

        self.assertEqual(record["path"], "/input/frame.jpg")
        self.assertEqual(record["labels"], labels)
        self.assertEqual(record["img_w"], 1600)
        self.assertEqual(record["img_h"], 1300)
        self.assertNotIn("img", record)


if __name__ == "__main__":
    unittest.main()
