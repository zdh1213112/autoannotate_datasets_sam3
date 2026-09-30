import os
import tempfile
import unittest
import weakref

import cv2
import numpy as np

from auto_obb_annotator import app_protective_gloves as protective_app


class _SignalRecorder:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _PreviewReceiver:
    def __init__(self):
        self.live_preview_signal = _SignalRecorder()
        self.preview_error_signal = _SignalRecorder()


class ProtectivePreviewTests(unittest.TestCase):
    def tearDown(self):
        protective_app._ACTIVE_PREVIEW_WINDOW = None
        protective_app.base_app.make_dataset_record = (
            protective_app._ORIGINAL_MAKE_DATASET_RECORD)

    def test_record_hook_emits_completed_frame_immediately(self):
        receiver = _PreviewReceiver()
        protective_app._ACTIVE_PREVIEW_WINDOW = weakref.ref(receiver)

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "frame.jpg")
            image = np.zeros((120, 180, 3), dtype=np.uint8)
            self.assertTrue(cv2.imwrite(image_path, image))
            labels = [(0, [0.50, 0.50, 0.40, 0.50])]

            record = protective_app._make_dataset_record_with_live_preview(
                image_path, "frame.jpg", labels, 180, 120)

        self.assertEqual(record["labels"], labels)
        self.assertNotIn("img", record)
        self.assertEqual(len(receiver.live_preview_signal.calls), 1)
        preview, name = receiver.live_preview_signal.calls[0]
        self.assertEqual(name, "frame.jpg")
        self.assertGreater(np.count_nonzero(preview), 0)
        self.assertEqual(receiver.preview_error_signal.calls, [])


if __name__ == "__main__":
    unittest.main()
