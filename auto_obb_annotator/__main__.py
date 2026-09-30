from .app import AutoAnnotatorApp


def main() -> int:
    import os
    import sys
    from PyQt5.QtCore import QCoreApplication, QLibraryInfo
    from PyQt5.QtWidgets import QApplication

    # The non-headless OpenCV wheel redirects Qt to its own plugin directory
    # when cv2 is imported by app.py. Use the plugins bundled with PyQt5.
    pyqt_plugin_root = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = pyqt_plugin_root
    QCoreApplication.setLibraryPaths([pyqt_plugin_root])

    app = QApplication(sys.argv)
    window = AutoAnnotatorApp()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
