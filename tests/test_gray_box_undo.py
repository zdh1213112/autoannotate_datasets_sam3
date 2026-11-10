"""Exercise undo inside the gray-glove box editor with Qt input events."""

import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PyQt5.QtCore import QCoreApplication, QEvent, QLibraryInfo, QPoint, QPointF, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtTest import QTest
from PyQt5.QtWidgets import QApplication, QPushButton

from auto_obb_annotator.app_generic import GenericBoxDialog


class GrayBoxUndoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plugin_dir = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_dir
        QCoreApplication.setLibraryPaths([plugin_dir])
        cls.qt = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        image = Path(self.temp.name) / "frame.png"
        cv2.imwrite(str(image), np.zeros((180, 240, 3), np.uint8))
        self.image = image
        self.dialog = GenericBoxDialog(
            image, "修正目标", [[60, 60], [140, 60], [140, 120], [60, 120]])
        self.dialog.show()
        self.dialog.activateWindow()
        self.dialog.canvas.setFocus()
        self.qt.processEvents()

    def tearDown(self):
        self.dialog.close()
        self.dialog.deleteLater()
        self.qt.sendPostedEvents(None, QEvent.DeferredDelete)
        self.qt.processEvents()

    def drag(self, start, end):
        canvas = self.dialog.canvas
        QTest.mousePress(canvas, Qt.LeftButton, Qt.NoModifier, start)
        move = QMouseEvent(QEvent.MouseMove, QPointF(end), Qt.NoButton,
                           Qt.LeftButton, Qt.NoModifier)
        QApplication.sendEvent(canvas, move)
        QTest.mouseRelease(canvas, Qt.LeftButton, Qt.NoModifier, end)
        self.qt.processEvents()

    def undo(self, widget=None):
        QTest.keyClick(widget or self.dialog.canvas, Qt.Key_Z, Qt.ControlModifier)
        self.qt.processEvents()

    def test_move_resize_and_rotate_undo_one_gesture_at_a_time(self):
        canvas = self.dialog.canvas
        initial = canvas._snapshot()
        center = QPoint(round(canvas._cx), round(canvas._cy))
        self.drag(center, center + QPoint(20, 10))
        moved = canvas._snapshot()
        self.assertNotEqual(moved, initial)
        handle = canvas._world_handles()[4]
        center = QPoint(round(canvas._cx), round(canvas._cy))
        vector = handle - center
        self.drag(handle, center + QPoint(round(vector.x() * 1.5), round(vector.y() * 1.5)))
        resized = canvas._snapshot()
        self.assertNotEqual(resized, moved)
        rotate = canvas._world_handles()[8]
        target = QPoint(round(canvas._cx + 100), round(canvas._cy))
        self.drag(rotate, target)
        self.assertNotEqual(canvas._snapshot(), resized)
        self.undo()
        self.assertEqual(canvas._snapshot(), resized)
        self.undo()
        self.assertEqual(canvas._snapshot(), moved)
        self.undo()
        self.assertEqual(canvas._snapshot(), initial)

    def test_draw_and_clear_can_be_undone(self):
        canvas = self.dialog.canvas
        initial = canvas._snapshot()
        self.drag(QPoint(5, 5), QPoint(45, 50))
        drawn = canvas._snapshot()
        self.assertNotEqual(drawn, initial)
        clear = next(b for b in self.dialog.findChildren(QPushButton)
                     if b.text().startswith("清除框"))
        clear.click()
        self.assertFalse(canvas._has_pending)
        self.undo(clear)
        self.assertEqual(canvas._snapshot(), drawn)
        self.undo()
        self.assertEqual(canvas._snapshot(), initial)

    def test_ctrl_z_does_not_rotate_and_plain_z_remains_rotation(self):
        canvas = self.dialog.canvas
        before = canvas._snapshot()
        self.undo()
        self.assertEqual(canvas._snapshot(), before)
        QTest.keyClick(canvas, Qt.Key_Z)
        self.assertNotEqual(canvas._angle, before[-1])
        self.undo()
        self.assertEqual(canvas._snapshot(), before)


if __name__ == "__main__":
    unittest.main()
