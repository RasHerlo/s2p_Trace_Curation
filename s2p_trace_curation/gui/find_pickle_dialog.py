"""Find trc_curation.pkl files under a root folder and pick one."""

from __future__ import annotations

import os
from pathlib import Path

from pyqtgraph.Qt import QtCore, QtWidgets

from s2p_trace_curation import PICKLE_NAME

QDialog = QtWidgets.QDialog
QDialogButtonBox = QtWidgets.QDialogButtonBox
QLabel = QtWidgets.QLabel
QTreeWidget = QtWidgets.QTreeWidget
QTreeWidgetItem = QtWidgets.QTreeWidgetItem
QVBoxLayout = QtWidgets.QVBoxLayout
Qt = QtCore.Qt


def find_curation_pickles(root: Path) -> list[Path]:
    """Return ``trc_curation.pkl`` paths under ``root``, sorted by relative path."""
    root = Path(root)
    found: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        if PICKLE_NAME in filenames:
            found.append(Path(dirpath) / PICKLE_NAME)
    found.sort(key=lambda p: tuple(part.casefold() for part in p.relative_to(root).parts))
    return found


class FindPickleDialog(QDialog):
    """Folder tree of curation pickles. ``selected`` is set when the user opens one."""

    def __init__(self, root: Path, paths: list[Path], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Find pickle file")
        self.setMinimumSize(520, 420)
        self.selected: Path | None = None
        self._root = Path(root)

        layout = QVBoxLayout(self)
        n = len(paths)
        noun = "file" if n == 1 else "files"
        label = QLabel(f"{n} {PICKLE_NAME} {noun} under {self._root}")
        label.setWordWrap(True)
        layout.addWidget(label)

        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
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
                    item.setToolTip(0, str(self._root.joinpath(*acc)))
                    if i == len(parts) - 1:
                        item.setData(0, Qt.ItemDataRole.UserRole, str(pkl))
                    if parent is None:
                        self.tree.addTopLevelItem(item)
                    else:
                        parent.addChild(item)
                    nodes[key] = item
                parent = item

    def _current_pickle(self) -> Path | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        raw = item.data(0, Qt.ItemDataRole.UserRole)
        if not raw:
            return None
        return Path(str(raw))

    def _sync_open_button(self) -> None:
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(
            self._current_pickle() is not None
        )

    def _on_double_click(self, item: QTreeWidgetItem, _column: int) -> None:
        if item.data(0, Qt.ItemDataRole.UserRole):
            self._accept_current()

    def _accept_current(self) -> None:
        picked = self._current_pickle()
        if picked is None:
            return
        self.selected = picked
        self.accept()
