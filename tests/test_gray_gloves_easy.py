"""Exercise the independent editor using real Qt keyboard/mouse events."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import cv2
import numpy as np
from PyQt5.QtCore import QCoreApplication, QLibraryInfo, QPointF, Qt, QEvent
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication
import auto_obb_annotator.app_gray_gloves as gray
from auto_obb_annotator.app_gray_gloves_easy import EasyGrayGloveApp


class EasyEditorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plugins = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugins
        QCoreApplication.setLibraryPaths([plugins])
        cls.app = QApplication.instance() or QApplication([])

    def test_draw_classify_cancel_and_undo(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            src, out = root / 'images', root / 'result'
            src.mkdir()
            out.mkdir()
            cv2.imwrite(str(src / 'test.png'), np.zeros((480, 640, 3), np.uint8))
            with patch.object(gray, 'DEFAULT_OUTPUT', out), \
                    patch.object(EasyGrayGloveApp, 'previous_review_project', return_value=(None, None)), \
                    patch.object(EasyGrayGloveApp, 'remember_review_project'):
                w = EasyGrayGloveApp()
                try:
                    w.show()
                    w.activateWindow()
                    self.app.processEvents()
                    w.active_source, w.active_output = src, out
                    w.active_config = {'classes': gray.GRAY_GLOVE_CLASSES}
                    w.records = {'test.png': {'name': 'test.png', 'status': 'review',
                                             'objects': [], 'candidates': [], 'width': 640, 'height': 480}}
                    w.update_table(w.records['test.png'])
                    w.show_record('test.png')
                    self.app.processEvents()
                    keys = [key for _, key in w.button_shortcuts.bindings]
                    self.assertEqual(len(keys), len(set(keys)))
                    c = w.canvas
                    c.setFocus()
                    QTest.keyClick(c, Qt.Key_A)
                    self.assertTrue(c.drawing)
                    start = c.mapFromScene(QPointF(50, 50))
                    end = c.mapFromScene(QPointF(200, 200))
                    QTest.mousePress(c.viewport(), Qt.LeftButton, Qt.NoModifier, start)
                    QApplication.sendEvent(c.viewport(), QMouseEvent(
                        QEvent.MouseMove, QPointF(end), Qt.NoButton, Qt.LeftButton, Qt.NoModifier))
                    QTest.mouseRelease(c.viewport(), Qt.LeftButton, Qt.NoModifier, end)
                    self.assertEqual(len(w.records['test.png']['objects']), 1)
                    QTest.keyClick(c, Qt.Key_1)
                    self.assertFalse(w.records['test.png']['side_pending'])
                    c.set_draw()
                    QTest.keyClick(c, Qt.Key_Escape)
                    self.assertFalse(c.drawing)
                    QTest.keyClick(c, Qt.Key_Z, Qt.ControlModifier)
                    self.assertTrue(w.records['test.png']['side_pending'])
                    self.assertTrue((out / 'review.jsonl').is_file())
                finally:
                    w.close()
                    w.deleteLater()
                    self.app.sendPostedEvents(None, QEvent.DeferredDelete)

    def test_open_yolo_edit_undo_reopen_and_return_to_native_project(self):
        import json
        from auto_obb_annotator.barcode_core import load_records
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'dataset/images'
            labels = root / 'dataset/labels'
            source.mkdir(parents=True)
            labels.mkdir()
            (source.parent / 'classes.txt').write_text('left\nright\n')
            cv2.imwrite(str(source / 'first.png'), np.zeros((100, 200, 3), np.uint8))
            cv2.imwrite(str(source / 'second.png'), np.zeros((100, 200, 3), np.uint8))
            original = b'1 0.500000000 0.5 0.4 0.2\r\n'
            label = labels / 'first.txt'
            label.write_bytes(original)
            (labels / 'second.txt').write_text('0 0.5 0.5 0.4 0.2\n')
            native = root / 'native'
            native.mkdir()
            with patch.object(gray, 'DEFAULT_OUTPUT', native), \
                    patch.object(EasyGrayGloveApp, 'previous_review_project', return_value=(None, None)), \
                    patch.object(EasyGrayGloveApp, 'remember_review_project'):
                w = EasyGrayGloveApp()
                try:
                    w.show()
                    w.activateWindow()
                    self.app.processEvents()
                    w.input_edit.setText(str(source))
                    (native / 'project.json').write_text(json.dumps(w.config()))
                    native_record = {'name': 'second.png', 'status': 'review', 'objects': [],
                                     'candidates': [], 'width': 200, 'height': 100}
                    (native / 'review.jsonl').write_text(json.dumps(native_record) + '\n')
                    self.assertTrue(w.open_path(source))
                    self.assertEqual(w.table.rowCount(), 2)
                    self.assertFalse(w.start_button.isEnabled())
                    self.assertEqual(w.current_name, 'images/first.png')
                    self.assertEqual(w.object_table.item(0, 2).text(), '—')
                    self.assertEqual(label.read_bytes(), original)
                    self.assertFalse((source.parent / '.gray_glove_editor').exists())
                    self.app.processEvents()
                    w.activateWindow()
                    w.object_table.selectRow(0)
                    w.canvas.setFocus()
                    self.app.processEvents()
                    QTest.keyClick(w.canvas, Qt.Key_1)
                    self.assertTrue(label.read_text().startswith('0 '))
                    self.assertEqual((source.parent / '.gray_glove_editor/backups/labels/first.txt').read_bytes(), original)
                    QTest.keyClick(w.canvas, Qt.Key_Z, Qt.ControlModifier)
                    self.assertEqual(label.read_bytes(), original)
                    self.assertEqual(w.records[w.current_name]['objects'][0]['class_id'], 1)
                    self.assertTrue(w.commit_polygon(0, [[65,40],[145,40],[145,60],[65,60]]))
                    saved = label.read_bytes()
                    self.assertNotEqual(saved, original)
                    self.assertTrue(w.open_path(labels, 'images/first.png'))
                    self.assertEqual(w.records[w.current_name]['objects'][0]['class_id'], 1)
                    self.assertAlmostEqual(w.records[w.current_name]['objects'][0]['polygon'][0][0],65,places=4)
                    self.assertEqual(w.records[w.current_name]['status'], 'confirmed')
                    self.assertFalse(w._undo_stack)
                    self.assertTrue(w.open_path(native))
                    self.assertIsNone(w._yolo_dataset)
                    self.assertTrue(w.start_button.isEnabled())
                    self.assertTrue(w.settings_group.isEnabled())
                    self.assertEqual(w.current_name, 'second.png')
                    w.mark_empty()
                    self.assertEqual(load_records(native / 'review.jsonl')['second.png']['status'], 'empty_confirmed')
                    self.assertEqual(label.read_bytes(), saved)
                finally:
                    w.close()
                    w.deleteLater()
                    self.app.sendPostedEvents(None, QEvent.DeferredDelete)


class EasyVideoModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plugins = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugins
        QCoreApplication.setLibraryPaths([plugins])
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'out'
        self.patches = [patch.object(gray, 'DEFAULT_OUTPUT', self.output),
                        patch.object(EasyGrayGloveApp, 'previous_review_project', return_value=(None, None)),
                        patch.object(EasyGrayGloveApp, 'remember_review_project')]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)
        self.w = EasyGrayGloveApp()
        self.w.show()
        self.app.processEvents()

    def tearDown(self):
        self.w.close()
        self.w.deleteLater()
        self.app.sendPostedEvents(None, QEvent.DeferredDelete)
        self.app.processEvents()

    def video(self, calibrated=False):
        import yaml
        path = self.root / 'EGO_camera_left.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (80, 60))
        self.assertTrue(writer.isOpened())
        image = np.zeros((60, 80, 3), np.uint8)
        image[:, :, 0] = np.arange(80, dtype=np.uint8) * 3
        image[:, :, 1] = np.arange(60, dtype=np.uint8)[:, None] * 4
        for _ in range(10):
            writer.write(image)
        writer.release()
        if calibrated:
            (self.root / 'EGO_calibration_camera.yaml').write_text(yaml.safe_dump({'cameras': [{
                'id': 'cam_0', 'distortion_model': 'KB', 'image_width': 80, 'image_height': 60,
                'intrinsics': {'fx': 45, 'fy': 45, 'cx': 40, 'cy': 30},
                'distortion': {'k1': .08, 'k2': -.01, 'k3': .001, 'k4': 0},
            }]}))
        return path

    def finish_import(self):
        import time
        deadline = time.monotonic() + 10
        while self.w.video_thread is not None and time.monotonic() < deadline:
            self.app.processEvents()
            QTest.qWait(10)
        self.assertIsNone(self.w.video_thread)

    def mode(self, mode):
        self.w.video_geometry_combo.setCurrentIndex(self.w.video_geometry_combo.findData(mode))

    def test_modes_mcap_and_calibration_controls(self):
        self.assertTrue(self.w.should_undistort_videos([Path('recording.mcap')]))
        self.mode('raw')
        self.assertFalse(self.w.should_undistort_videos([Path('recording.mcap')]))
        self.assertFalse(self.w.should_undistort_videos([Path('EGO_camera_right.mp4')]))
        self.assertFalse(self.w.balance_spin.isEnabled())
        self.assertFalse(self.w.calibration_edit.isEnabled())
        self.mode('rectified')
        self.assertTrue(self.w.balance_spin.isEnabled())
        with self.assertRaises(ValueError):
            self.w.should_undistort_videos([Path('ordinary.mp4')])

    def test_f5_raw_import_needs_no_calibration_and_does_not_upgrade_again(self):
        import json
        video = self.video()
        self.mode('raw')
        self.w.add_video_paths([video])
        with patch.object(gray.GenericAnnotatorApp, 'start_detection') as detect, \
                patch.object(gray, 'OrbbecUndistorter', side_effect=AssertionError('must not undistort')), \
                patch.object(gray.QMessageBox, 'warning') as warning:
            self.w.start_detection()
            self.assertFalse(self.w.video_geometry_combo.isEnabled())
            self.finish_import()
            self.assertEqual(detect.call_count, 1)
            self.w.start_detection()
            self.assertEqual(detect.call_count, 2)
            warning.assert_not_called()
        folder = self.w.source_dir()
        self.assertEqual(self.w.output_dir(), self.output)
        self.assertFalse(gray.frame_projection_sources(folder))
        self.assertNotIn('undistorted_detection', self.w.config())
        manifest = json.loads(next(folder.glob('extract_*.json')).read_text())
        self.assertNotIn('transform', manifest['settings'])
        self.assertTrue(self.w.video_geometry_combo.isEnabled())
        self.assertFalse((self.root / 'out_undistorted').exists())

    def test_switch_rectified_project_to_raw_preserves_old_frames_and_labels(self):
        from auto_obb_annotator.barcode_core import append_record
        video = self.video(calibrated=True)
        self.mode('rectified')
        self.w.add_video_paths([video])
        self.w.import_button.click()
        self.finish_import()
        rectified = self.w.output_dir()
        self.assertEqual(rectified, self.root / 'out_undistorted')
        self.assertTrue(gray.frame_projection_sources(self.w.source_dir()))
        self.w.ensure_project()
        image = next(self.w.source_dir().glob('*.jpg'))
        append_record(rectified / 'review.jsonl', {'name': image.name, 'status': 'empty_confirmed', 'objects': []})
        before_labels = (rectified / 'review.jsonl').read_bytes()
        before_config = (rectified / 'project.json').read_bytes()
        before_image = image.read_bytes()
        self.mode('raw')
        with patch.object(gray.GenericAnnotatorApp, 'start_detection') as detect, \
                patch.object(gray, 'OrbbecUndistorter', side_effect=AssertionError('must not undistort')), \
                patch.object(gray.QMessageBox, 'warning') as warning:
            self.w.start_detection()
            self.finish_import()
            detect.assert_called_once()
            warning.assert_not_called()
        self.assertEqual(self.w.output_dir(), self.output)
        self.assertFalse(gray.frame_projection_sources(self.w.source_dir()))
        self.assertEqual((rectified / 'review.jsonl').read_bytes(), before_labels)
        self.assertEqual((rectified / 'project.json').read_bytes(), before_config)
        self.assertEqual(image.read_bytes(), before_image)
        self.assertNotIn('undistorted_detection', self.w.config())

    def test_projected_images_without_original_video_do_not_run_in_raw_mode(self):
        self.w.input_edit.setText(str(self.root / 'images'))
        self.mode('raw')
        with patch('auto_obb_annotator.app_gray_gloves_easy.frame_projection_sources', return_value={'frame.jpg': {}}), \
                patch.object(gray.GenericAnnotatorApp, 'start_detection') as detect, \
                patch.object(gray.QMessageBox, 'warning') as warning:
            self.w.start_detection()
            detect.assert_not_called()
            warning.assert_called_once()
            self.assertIn('原始录像', warning.call_args.args[1])

    def test_detection_focus_returns_to_canvas(self):
        self.w.open_settings()
        self.w.toggle_focus_view()
        self.assertFalse(self.w.settings_dialog.isVisible())
        self.assertTrue(self.w._focus_view)

    def test_recheck_cannot_silently_ignore_selected_raw_mode(self):
        self.w.input_edit.setText(str(self.root / 'images'))
        self.mode('raw')
        with patch('auto_obb_annotator.app_gray_gloves_easy.frame_projection_sources', return_value={'frame.jpg': {}}), \
                patch.object(gray.GrayGloveAnnotatorApp, 'recheck_review_images') as recheck, \
                patch.object(gray.QMessageBox, 'information') as message:
            self.w.recheck_review_images()
            recheck.assert_not_called()
            message.assert_called_once()

    def test_new_project_unlocks_dataset_settings_without_touching_labels(self):
        from auto_obb_annotator.gray_glove_dataset import YoloGloveDataset
        source = self.root / 'dataset/images'
        labels = self.root / 'dataset/labels'
        source.mkdir(parents=True)
        labels.mkdir()
        cv2.imwrite(str(source / 'frame.png'), np.zeros((60, 80, 3), np.uint8))
        original = b'1 0.5 0.5 0.2 0.2\n'
        label = labels / 'frame.txt'
        label.write_bytes(original)
        self.w.video_list.addItem('/old/video.mcap')
        self.w.activate_dataset(YoloGloveDataset.load(source.parent))
        self.w.open_settings()
        self.assertFalse(self.w.video_group.isEnabled())
        self.assertTrue(self.w.dataset_settings_banner.isVisible())
        self.assertTrue(self.w.dataset_new_project_button.isEnabled())
        self.assertEqual(self.w.video_list.count(), 0)
        with patch('auto_obb_annotator.app_gray_gloves_easy.ROOT_DIR', self.root):
            self.w.dataset_new_project_button.click()
        self.assertIsNone(self.w._yolo_dataset)
        self.assertTrue(self.w.settings_group.isEnabled())
        self.assertTrue(self.w.video_group.isEnabled())
        self.assertTrue(self.w.video_geometry_combo.isEnabled())
        self.assertTrue(self.w.video_button.isEnabled())
        self.assertTrue(self.w.start_button.isEnabled())
        self.assertFalse(self.w.dataset_settings_banner.isVisible())
        self.assertFalse(self.w.records)
        self.assertFalse(self.w.input_edit.text())
        self.assertFalse(self.w.video_edit.text())
        self.assertFalse(self.w._undo_stack)
        self.assertIsNone(self.w._loaded_video_folder)
        self.assertTrue(self.w.output_dir().is_relative_to(self.root / 'data'))
        self.assertFalse(self.w.output_dir().exists())
        self.assertEqual(label.read_bytes(), original)
        self.mode('raw')
        self.assertFalse(self.w.should_undistort_videos([Path('recording.mcap')]))
        # The newly unlocked workflow can actually extract, then start inference.
        self.w.add_video_paths([self.video()])
        with patch.object(gray.GenericAnnotatorApp, 'start_detection') as detect:
            self.w.start_detection()
            self.finish_import()
            detect.assert_called_once()
        self.assertEqual(label.read_bytes(), original)

    def test_new_project_is_blocked_while_worker_is_active(self):
        from unittest.mock import Mock
        worker = Mock()
        worker.isRunning.return_value = True
        self.w.thread = worker
        before = self.w.output_edit.text()
        try:
            self.assertFalse(self.w.new_detection_project())
            self.assertEqual(self.w.output_edit.text(), before)
        finally:
            self.w.thread = None
