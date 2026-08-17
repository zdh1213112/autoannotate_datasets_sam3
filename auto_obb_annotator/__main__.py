from .app import AutoAnnotatorApp


def main() -> int:
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = AutoAnnotatorApp()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
