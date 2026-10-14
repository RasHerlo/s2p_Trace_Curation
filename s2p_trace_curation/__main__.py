"""Entry point: python -m s2p_trace_curation"""

from __future__ import annotations

import os
import sys
from pathlib import Path

FROZEN = bool(getattr(sys, "frozen", False))


def log_path() -> Path:
    return Path.home() / ".s2p_trace_curation" / "error.log"


def _write_log(text: str) -> Path | None:
    from datetime import datetime

    path = log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} ===\n{text}")
    except OSError:
        return None
    return path


def _install_frozen_excepthook() -> None:
    """The built app has no console: show unhandled errors, log them, then quit.

    Quitting matches what PyQt5 does without a hook, so curation state is never
    left half-updated and saved afterwards.
    """
    import traceback

    def hook(exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        path = _write_log(text)
        try:
            from pyqtgraph.Qt import QtWidgets

            if QtWidgets.QApplication.instance() is not None:
                where = f"\n\nDetails were written to:\n{path}" if path else ""
                QtWidgets.QMessageBox.critical(
                    None,
                    "s2p Trace Curation — unexpected error",
                    f"{exc_type.__name__}: {exc}{where}\n\nThe app will now close.",
                )
        finally:
            os._exit(1)

    sys.excepthook = hook


def selftest() -> int:
    """Import every module and build the main window without showing it."""
    import importlib
    import pkgutil
    import traceback

    try:
        import s2p_trace_curation as pkg

        names = [m.name for m in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + ".")]
        if not names:
            raise RuntimeError("no s2p_trace_curation submodules found")
        for name in names:
            importlib.import_module(name)

        from pyqtgraph.Qt import QtWidgets

        from s2p_trace_curation.gui.main_window import MainWindow

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
        window = MainWindow()
        app.processEvents()
        window.deleteLater()
        app.processEvents()
    except Exception:
        text = traceback.format_exc()
        _write_log("selftest failed\n" + text)
        if sys.stderr is not None:
            sys.stderr.write(text)
        return 1
    if sys.stdout is not None:
        print(f"selftest ok ({len(names)} modules)")
    return 0


def main() -> int:
    if "--selftest" in sys.argv[1:]:
        return selftest()

    if FROZEN:
        _install_frozen_excepthook()

    from pyqtgraph.Qt import QtCore, QtWidgets

    from s2p_trace_curation.gui.main_window import MainWindow

    if FROZEN:
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
        QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("s2p Trace Curation")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
