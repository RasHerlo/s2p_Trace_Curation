"""PyInstaller entry script for the Windows app."""

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt5")

from s2p_trace_curation.__main__ import main  # noqa: E402

raise SystemExit(main())
