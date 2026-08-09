"""Entry point: python -m s2p_trace_curation"""

from __future__ import annotations

import sys


def main() -> int:
    from pyqtgraph.Qt import QtWidgets

    from s2p_trace_curation.gui.main_window import MainWindow

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("s2p Trace Curation")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())