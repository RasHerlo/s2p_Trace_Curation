"""Find trc_curation.pkl files under a root folder and pick one.

Each file is listed with the number of edited ROIs (``edited/all``), computed in a
background thread so the list opens at once and the numbers fill in as they are read.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from s2p_trace_curation import PICKLE_NAME
from s2p_trace_curation.edits import EditSummary, summarize_pickle
from s2p_trace_curation.gui.edits_dialog import DEFAULT_COLOR, EDITED_COLOR

QDialog = QtWidgets.QDialog
QDialogButtonBox = QtWidgets.QDialogButtonBox
QHeaderView = QtWidgets.QHeaderView
QLabel = QtWidgets.QLabel
QTreeWidget = QtWidgets.QTreeWidget
QTreeWidgetItem = QtWidgets.QTreeWidgetItem
QVBoxLayout = QtWidgets.QVBoxLayout
Qt = QtCore.Qt

COL_NAME = 0
COL_EDITS = 1


def find_curation_pickles(root: Path) -> list[Path]:
    """Return ``trc_curation.pkl`` paths under ``root``, sorted by relative path."""
    root = Path(root)
    found: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if PICKLE_NAME in filenames:
            found.append(Path(dirpath) / PICKLE_NAME)
    found.sort(key=lambda p: tuple(part.casefold() for part in p.relative_to(root).parts))
    return found


class _EditCountWorker(QtCore.QThread):
    """Reads the pickles one by one and reports each file's edit summary."""

    summarized = QtCore.Signal(str, object)

    def __init__(self, paths: list[Path], parent: Any = None) -> None:
        super().__init__(parent)
        self._paths = list(paths)

    def run(self) -> None:
        master_cache: dict[str, Any] = {}
        for pkl in self._paths:
            if self.isInterruptionRequested():
                return
            self.summarized.emit(str(pkl), summarize_pickle(pkl, master_cache))


class FindPickleDialog(QDialog):
    """Folder tree of curation pickles. ``selected`` is set when the user opens one."""

    def __init__(self, root: Path, paths: list[Path], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Find pickle file")
        self.setMinimumSize(620, 420)
        self.selected: Path | None = None
        self._root = Path(root)
        self._paths = list(paths)
        self._leaf: dict[str, QTreeWidgetItem] = {}
        self._summaries: dict[str, EditSummary] = {}
        self._worker: _EditCountWorker | None = None

        layout = QVBoxLayout(self)
        n = len(paths)
        noun = "file" if n == 1 else "files"
        self._head_text = f"{n} {PICKLE_NAME} {noun} under {self._root}"
        self.label = QLabel(self._head_text)
        self.label.setWordWrap(True)
        layout.addWidget(self.label)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(2)
        self.tree.setHeaderLabels(["Experiment", "Edited ROIs"])
        self.tree.headerItem().setToolTip(
            COL_EDITS,
            "Number of ROIs that differ from their baseline (suite2p files or shared "
            "ROI set) / all ROIs in the file. Same count as View \u2192 Show Edits.",
        )
        header = self.tree.header()
        header.setSectionResizeMode(COL_NAME, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(COL_EDITS, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(False)
        self.tree.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._fill_tree(paths)
        self.tree.expandToDepth(1 if n > 40 else 8)
        self.tree.itemSelectionChanged.connect(self._sync_open_button)
        self.tree.itemDoubleClicked.connect(self._on_double_click)
        layout.addWidget(self.tree, stretch=1)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Open")
        self.buttons.accepted.connect(self._accept_current)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self._sync_open_button()
        self._start_counting()

    # ------------------------------------------------------------ tree
    def _fill_tree(self, paths: list[Path]) -> None:
        nodes: dict[tuple[str, ...], QTreeWidgetItem] = {}
        for pkl in paths:
            rel = pkl.relative_to(self._root)
            parent: QTreeWidgetItem | None = None
            acc: list[str] = []
            parts = rel.parts
            for i, part in enumerate(parts):
                acc.append(part)
                key = tuple(acc)
                item = nodes.get(key)
                if item is None:
                    item = QTreeWidgetItem([part])
                    item.setToolTip(COL_NAME, str(self._root.joinpath(*acc)))
                    if i == len(parts) - 1:
                        item.setData(COL_NAME, Qt.ItemDataRole.UserRole, str(pkl))
                        item.setText(COL_EDITS, "\u2026")
                        item.setToolTip(COL_EDITS, "Counting edited ROIs\u2026")
                        item.setTextAlignment(COL_EDITS, int(Qt.AlignmentFlag.AlignCenter))
                        item.setForeground(COL_EDITS, QtGui.QBrush(QtGui.QColor(DEFAULT_COLOR)))
                        self._leaf[str(pkl)] = item
                    if parent is None:
                        self.tree.addTopLevelItem(item)
                    else:
                        parent.addChild(item)
                    nodes[key] = item
                parent = item

    # ------------------------------------------------------ edit counts
    def _start_counting(self) -> None:
        if not self._paths:
            return
        self._worker = _EditCountWorker(self._paths, self)
        self._worker.summarized.connect(self._on_summary)
        self._worker.finished.connect(self._on_counting_finished)
        self._worker.start()

    def _on_summary(self, pkl: str, summary: EditSummary) -> None:
        item = self._leaf.get(pkl)
        if item is None:
            return
        self._summaries[pkl] = summary
        item.setText(COL_EDITS, summary.text)
        font = QtGui.QFont(item.font(COL_EDITS))
        edited = (not summary.failed) and summary.edited > 0
        font.setBold(edited)
        item.setFont(COL_EDITS, font)
        item.setForeground(
            COL_EDITS, QtGui.QBrush(QtGui.QColor(EDITED_COLOR if edited else DEFAULT_COLOR))
        )
        item.setToolTip(COL_EDITS, self._tooltip(summary))
        self.label.setText(self._progress_text())

    @staticmethod
    def _tooltip(s: EditSummary) -> str:
        if s.failed:
            return f"Edits could not be counted: {s.note}"
        lines = [f"{s.edited} of {s.total} ROIs edited"]
        if s.new:
            lines.append(f"{s.new} new (drawn or imported, nothing to compare with)")
        if s.unknown:
            lines.append(f"{s.unknown} cannot be compared")
        if s.note:
            lines.append(s.note)
        return "\n".join(lines)

    def _progress_text(self) -> str:
        done = len(self._summaries)
        total = len(self._paths)
        with_edits = sum(1 for s in self._summaries.values() if not s.failed and s.edited > 0)
        if done < total:
            return f"{self._head_text}\nCounting edits\u2026 {done}/{total}"
        return f"{self._head_text}\n{with_edits} of {total} with edits"

    def _on_counting_finished(self) -> None:
        self.label.setText(self._progress_text())

    def _stop_worker(self) -> None:
        worker = self._worker
        if worker is not None and worker.isRunning():
            worker.requestInterruption()
            worker.wait(5000)

    def done(self, result: int) -> None:  # noqa: D401 - Qt override
        self._stop_worker()
        super().done(result)

    # --------------------------------------------------------- selection
    def _current_pickle(self) -> Path | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        raw = item.data(COL_NAME, Qt.ItemDataRole.UserRole)
        if not raw:
            return None
        return Path(str(raw))

    def _sync_open_button(self) -> None:
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            self._current_pickle() is not None
        )

    def _on_double_click(self, item: QTreeWidgetItem, _column: int) -> None:
        if item.data(COL_NAME, Qt.ItemDataRole.UserRole):
            self._accept_current()

    def _accept_current(self) -> None:
        picked = self._current_pickle()
        if picked is None:
            return
        self.selected = picked
        self.accept()
