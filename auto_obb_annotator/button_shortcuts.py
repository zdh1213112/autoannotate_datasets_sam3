"""Window-local shortcuts for buttons with a trailing ``[key]`` label."""

import re

from PyQt5.QtCore import QObject, Qt
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAbstractButton, QAbstractSpinBox, QApplication, QComboBox, QLineEdit,
    QPushButton, QShortcut,
)


class ButtonShortcuts(QObject):
    """Bind the key shown on each button to its normal enabled click action."""

    def __init__(self, window, text_entry_widgets=()):
        super().__init__(window)
        self.window = window
        self.text_entry_widgets = text_entry_widgets
        self.bindings = []
        for button in window.findChildren(QAbstractButton):
            if button.window() is not window:
                continue
            match = re.search(r"\[([^\]]+)\]$", button.text())
            if match is None:
                continue
            key = match.group(1)
            shortcut = QShortcut(QKeySequence(key), window)
            shortcut.setContext(Qt.WindowShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.activated.connect(button.click)
            button.setToolTip(button.toolTip() or f"快捷键：{key}")
            button.setAutoRepeat(False)
            if isinstance(button, QPushButton):
                button.setAutoDefault(False)
            self.bindings.append((shortcut, key))
        QApplication.instance().focusChanged.connect(self.update)
        self.update(None, QApplication.focusWidget())

    def update(self, _old, current):
        modal = QApplication.activeModalWidget()
        in_this_window = current is not None and current.window() is self.window
        typing = in_this_window and isinstance(
            current, (QLineEdit, QAbstractSpinBox, QComboBox))
        if in_this_window:
            typing = typing or any(
                current is widget or widget.isAncestorOf(current)
                for widget in self.text_entry_widgets)
        for shortcut, key in self.bindings:
            text_key = (
                len(key) == 1
                or (key.startswith("Shift+") and len(key) == len("Shift+") + 1)
                or key in {"Space", "Delete", "Enter", "Ctrl+Delete", "Ctrl+Backspace"}
            )
            shortcut.setEnabled((modal is None or modal is self.window)
                                and not (typing and text_key))
