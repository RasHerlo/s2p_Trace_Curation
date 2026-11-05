"""Edit windows: the per-ROI component list (with reset) and the all-ROI overview."""

from __future__ import annotations

from typing import Any

from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from s2p_trace_curation.edits import (
    COMPONENT_LABELS,
    COMPONENTS,
    EXPERIMENT_WIDE_NOTE,
    RESETTABLE,
    STATUS_EDITED,
    STATUS_NEW,
    STATUS_UNCHANGED,
    STATUS_UNKNOWN,
    RoiEdits,
)

Qt = QtCore.Qt
QAbstractItemView = QtWidgets.QAbstractItemView
QDialog = QtWidgets.QDialog
QHBoxLayout = QtWidgets.QHBoxLayout
QHeaderView = QtWidgets.QHeaderView
QLabel = QtWidgets.QLabel
QMessageBox = QtWidgets.QMessageBox
QPushButton = QtWidgets.QPushButton
QTableWidget = QtWidgets.QTableWidget
QTableWidgetItem = QtWidgets.QTableWidgetItem
QVBoxLayout = QtWidgets.QVBoxLayout

EDITED_COLOR = QtGui.QColor("#1b5e20")  # dark green
DEFAULT_COLOR = QtGui.QColor("#808080")  # grey
TICK = "\u2713"


def _when_text(iso: str | None) -> str:
    if not iso:
        return ""
    return iso.replace("T", " ")[:16] + " UTC"


class _KeepForegroundDelegate(QtWidgets.QStyledItemDelegate):
    """Selected rows keep their own text colour (edited stays green, default stays grey)."""

    def initStyleOption(self, option: Any, index: Any) -> None:
        super().initStyleOption(option, index)
        brush = index.data(Qt.ItemDataRole.ForegroundRole)
        color = brush.color() if isinstance(brush, QtGui.QBrush) else brush
        if isinstance(color, QtGui.QColor):
            option.palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, color)


def _style_item(item: QTableWidgetItem, *, edited: bool, base_font: QtGui.QFont, kind: str = "value") -> None:
    """Edited: bold dark green. Same as default: normal grey. ``kind="note"``: grey italic."""
    font = QtGui.QFont(base_font)
    if edited:
        font.setBold(True)
        item.setForeground(QtGui.QBrush(EDITED_COLOR))
    else:
        item.setForeground(QtGui.QBrush(DEFAULT_COLOR))
        if kind == "note":
            font.setItalic(True)
    item.setFont(font)


class RoiEditsWindow(QDialog):
    """Small non-modal window: which components of the active ROI were edited, and reset them."""

    COL_EDITED, COL_RESET, COL_COMPONENT, COL_CURRENT, COL_DEFAULT = range(5)

    def __init__(self, main: Any) -> None:
        super().__init__(main)
        self.main = main
        self.setWindowTitle("ROI edits")
        self.setModal(False)
        self.setMinimumWidth(680)
        self._filling = False
        self._roi_id: int | None = None
        self._checked: set[str] = set()
        self._edits: RoiEdits | None = None

        layout = QVBoxLayout(self)
        self.lbl_head = QLabel("")
        self.lbl_head.setWordWrap(True)
        layout.addWidget(self.lbl_head)

        self.table = QTableWidget(len(COMPONENTS), 5)
        self.table.setHorizontalHeaderLabels(["Edited", "Reset", "Component", "Current", "Default"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(self.COL_COMPONENT, QHeaderView.ResizeMode.Stretch)
        for col in (self.COL_EDITED, self.COL_RESET, self.COL_CURRENT, self.COL_DEFAULT):
            header.setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.table.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self.table)
        self._fit_table_height()

        self.lbl_note = QLabel(
            "Tick = differs from the default. Tick \u201cReset\u201d for the edited components "
            "you want back at the default, then press the button. "
            "Smoothing and bleach are experiment-wide (Trace Processing) and not reset here."
        )
        self.lbl_note.setWordWrap(True)
        self.lbl_note.setStyleSheet("color: #808080;")
        layout.addWidget(self.lbl_note)

        btns = QHBoxLayout()
        self.btn_reset = QPushButton("Reset checked components")
        self.btn_reset.setEnabled(False)
        self.btn_reset.clicked.connect(self._on_reset)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        btns.addWidget(self.btn_reset)
        btns.addStretch(1)
        btns.addWidget(btn_close)
        layout.addLayout(btns)

        self.refresh()

    def _fit_table_height(self) -> None:
        h = self.table.horizontalHeader().height() + 2 * self.table.frameWidth()
        for r in range(self.table.rowCount()):
            h += self.table.rowHeight(r)
        self.table.setFixedHeight(h + 4)

    # ------------------------------------------------------------ content
    def refresh(self) -> None:
        main = self.main
        if main.doc is None:
            self._edits = None
            self._roi_id = None
            self._checked.clear()
            self.lbl_head.setText("No experiment open.")
            self.setWindowTitle("ROI edits")
            self._fill(None, can_reset=False)
            return
        roi_id = int(main.active_roi_id)
        if roi_id != self._roi_id:
            self._checked.clear()
            self._roi_id = roi_id
        edits = main.edits_for_active()
        self._edits = edits
        self.setWindowTitle(f"ROI edits \u2014 ROI {roi_id}")
        if edits is None:
            self.lbl_head.setText(f"ROI {roi_id}: not found.")
        elif edits.status == STATUS_NEW:
            self.lbl_head.setText(
                f"<b>ROI {roi_id}</b> was drawn or imported in this experiment, so there is "
                "nothing to compare it with."
            )
        elif edits.status == STATUS_UNKNOWN:
            self.lbl_head.setText(
                f"<b>ROI {roi_id}</b>: cannot compare \u2014 {main.edit_baseline_error() or 'no baseline'}."
            )
        else:
            n = len(edits.edited_keys)
            self.lbl_head.setText(
                f"<b>ROI {roi_id}</b> compared with {edits.baseline_label}: "
                + (f"{n} of {len(RESETTABLE)} components edited." if n else "nothing edited.")
            )
        can_reset = bool(edits is not None and edits.has_baseline and main.can_reset_roi_components())
        self._fill(edits, can_reset=can_reset)

    def _fill(self, edits: RoiEdits | None, *, can_reset: bool) -> None:
        self._filling = True
        try:
            base_font = self.table.font()
            edited_now = set(edits.edited_keys) if edits is not None else set()
            self._checked &= edited_now
            for r, key in enumerate(COMPONENTS):
                comp = edits.components.get(key) if edits is not None else None
                edited = bool(comp and comp.edited)
                tick = QTableWidgetItem(TICK if edited else "")
                tick.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                tick.setFlags(Qt.ItemFlag.ItemIsEnabled)
                _style_item(tick, edited=edited, base_font=base_font)
                if comp is not None and comp.when:
                    tick.setToolTip(f"Last edited {_when_text(comp.when)}")
                self.table.setItem(r, self.COL_EDITED, tick)

                chk = QTableWidgetItem("")
                pickable = bool(can_reset and edited and key in RESETTABLE)
                flags = Qt.ItemFlag.ItemIsEnabled
                if pickable:
                    flags |= Qt.ItemFlag.ItemIsUserCheckable
                chk.setFlags(flags)
                if pickable:
                    chk.setCheckState(
                        Qt.CheckState.Checked if key in self._checked else Qt.CheckState.Unchecked
                    )
                if key not in RESETTABLE:
                    chk.setToolTip(EXPERIMENT_WIDE_NOTE)
                elif not edited:
                    chk.setToolTip("Not edited")
                self.table.setItem(r, self.COL_RESET, chk)

                label = QTableWidgetItem(COMPONENT_LABELS[key])
                label.setFlags(Qt.ItemFlag.ItemIsEnabled)
                if key not in RESETTABLE:
                    label.setToolTip(EXPERIMENT_WIDE_NOTE)
                self.table.setItem(r, self.COL_COMPONENT, label)

                cur = QTableWidgetItem(comp.current if comp else "")
                cur.setFlags(Qt.ItemFlag.ItemIsEnabled)
                _style_item(cur, edited=edited, base_font=base_font)
                self.table.setItem(r, self.COL_CURRENT, cur)

                dflt = QTableWidgetItem(comp.default if comp else "")
                dflt.setFlags(Qt.ItemFlag.ItemIsEnabled)
                _style_item(dflt, edited=False, base_font=base_font)
                self.table.setItem(r, self.COL_DEFAULT, dflt)
        finally:
            self._filling = False
        self.btn_reset.setEnabled(bool(self._checked) and can_reset)

    # ------------------------------------------------------------ actions
    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        if self._filling or item.column() != self.COL_RESET:
            return
        key = COMPONENTS[item.row()]
        if item.checkState() == Qt.CheckState.Checked:
            self._checked.add(key)
        else:
            self._checked.discard(key)
        self.btn_reset.setEnabled(bool(self._checked))

    def _on_reset(self) -> None:
        edits = self._edits
        if edits is None or not self._checked:
            return
        keys = [k for k in RESETTABLE if k in self._checked]
        names = ", ".join(COMPONENT_LABELS[k] for k in keys)
        text = f"Reset ROI {edits.roi_id}: {names}\nto {edits.baseline_label}?"
        if {"roi_mask", "neuropil_mask"} & set(keys) and self.main.doc and self.main.doc.get("master_link"):
            text += "\n\nThe traces are re-extracted from data.bin (a few seconds)."
        text += "\n\nThe change is not saved until you save the session."
        if QMessageBox.question(self, "Reset ROI components", text) != QMessageBox.StandardButton.Yes:
            return
        self._checked.clear()
        self.main.reset_roi_components(keys)
        self.refresh()


class EditsOverviewWindow(QDialog):
    """All ROIs and their edit status, with the current value of every component."""

    HEADERS = [
        "ROI #",
        "Edited",
        "Selected",
        "ROI-mask",
        "Neuropil-mask",
        "Fneu-compensation (x, offset)",
        "Trace smoothing (SG)",
        "Bleach-correction",
    ]

    def __init__(self, main: Any) -> None:
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Edits overview")
        self.setModal(False)
        self.resize(1180, 640)
        self._filling = False
        self._fitting = False
        self._user_sized = False
        self._ids: list[int] = []
        self._cache: dict[tuple[int, int], tuple[str, str]] = {}
        ncols = len(self.HEADERS)

        layout = QVBoxLayout(self)
        self.lbl_head = QLabel("")
        self.lbl_head.setWordWrap(True)
        layout.addWidget(self.lbl_head)

        # Column names + the pinned "Total:" row live in their own one-row table so
        # they stay put while the ROI table below scrolls. Column widths are kept equal.
        self.total_table = QTableWidget(1, ncols)
        self.total_table.setHorizontalHeaderLabels(self.HEADERS)
        self.total_table.verticalHeader().setVisible(False)
        self.total_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.total_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.total_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.total_table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.total_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.total_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.total_table.horizontalHeader().sectionResized.connect(self._on_total_section_resized)
        self.total_table.setFixedHeight(
            self.total_table.horizontalHeader().sizeHint().height()
            + self.total_table.verticalHeader().defaultSectionSize()
            + 2 * self.total_table.frameWidth()
        )

        self.table = QTableWidget(0, ncols)
        self.table.horizontalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setItemDelegate(_KeepForegroundDelegate(self.table))
        self.table.setStyleSheet("QTableWidget { selection-background-color: #dce8f6; }")
        # A permanent scrollbar keeps both tables the same width; the spacer below
        # leaves the same room next to the (scrollbar-less) total table.
        self.table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.table.itemSelectionChanged.connect(self._on_select)
        self.table.horizontalScrollBar().valueChanged.connect(
            self.total_table.horizontalScrollBar().setValue
        )

        total_row = QHBoxLayout()
        total_row.setSpacing(0)
        total_row.setContentsMargins(0, 0, 0, 0)
        total_row.addWidget(self.total_table, stretch=1)
        spacer = QtWidgets.QWidget()
        spacer.setFixedWidth(self.table.verticalScrollBar().sizeHint().width())
        total_row.addWidget(spacer)
        stack = QVBoxLayout()
        stack.setSpacing(0)
        stack.setContentsMargins(0, 0, 0, 0)
        stack.addLayout(total_row)
        stack.addWidget(self.table, stretch=1)
        layout.addLayout(stack, stretch=1)

        legend = QLabel(
            "<span style='color:#1b5e20'><b>Bold green</b></span> = edited (differs from the default); "
            "<span style='color:#808080'>grey</span> = same as the default. "
            "<i>new</i> = drawn or imported here, nothing to compare with. "
            "Smoothing and bleach are experiment-wide and not counted in \u201cEdited\u201d. "
            "<b>Total:</b> = ROIs edited in that category / all ROIs. "
            "Click a row to open that ROI."
        )
        legend.setWordWrap(True)
        layout.addWidget(legend)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        close_row.addWidget(btn_close)
        layout.addLayout(close_row)

        self.refresh()

    # ------------------------------------------------------------ content
    def refresh(self) -> None:
        main = self.main
        all_edits = main.edits_for_all() if main.doc is not None else []
        ids = [e.roi_id for e in all_edits]
        base_font = self.table.font()

        self._filling = True
        try:
            if ids != self._ids:
                scroll = self.table.verticalScrollBar().value()
                self.table.setRowCount(len(ids))
                self._ids = ids
                self._cache.clear()
                self.table.verticalScrollBar().setValue(scroll)
            for r, e in enumerate(all_edits):
                self._set_cell(r, 0, str(e.roi_id), "plain", base_font)
                if e.status == STATUS_EDITED:
                    self._set_cell(r, 1, "yes", "edited", base_font)
                elif e.status == STATUS_UNCHANGED:
                    self._set_cell(r, 1, "no", "default", base_font)
                elif e.status == STATUS_NEW:
                    self._set_cell(r, 1, "new", "note", base_font)
                else:
                    self._set_cell(r, 1, "?", "note", base_font)
                for c, key in enumerate(COMPONENTS, start=2):
                    comp = e.components[key]
                    self._set_cell(
                        r, c, comp.current, "edited" if comp.edited else "default", base_font
                    )
        finally:
            self._filling = False

        n_edit = sum(1 for e in all_edits if e.status == STATUS_EDITED)
        n_new = sum(1 for e in all_edits if e.status == STATUS_NEW)
        n_unknown = sum(1 for e in all_edits if e.status == STATUS_UNKNOWN)
        if not all_edits:
            self.lbl_head.setText("No experiment open.")
        else:
            parts = [f"<b>{n_edit}</b> of {len(all_edits)} ROIs edited"]
            if n_new:
                parts.append(f"{n_new} new")
            if n_unknown:
                parts.append(f"{n_unknown} cannot be compared ({main.edit_baseline_error() or 'no baseline'})")
            self.lbl_head.setText(
                " \u00b7 ".join(parts) + f" \u2014 compared with {main.edit_baseline_label()}."
            )
        self._fill_totals(all_edits, n_edit, n_new, n_unknown)
        self._fit_columns()
        self.sync_active()

    # --------------------------------------------------------- total row
    def _fill_totals(self, all_edits: list[RoiEdits], n_edit: int, n_new: int, n_unknown: int) -> None:
        """``edited / all ROIs`` for the Edited column and every component."""
        total = len(all_edits)
        base_font = self.total_table.font()
        counts = [n_edit] + [
            sum(1 for e in all_edits if e.components[key].edited) for key in COMPONENTS
        ]
        label = self._total_item(0)
        label.setText("Total:")
        font = QtGui.QFont(base_font)
        font.setBold(True)
        label.setFont(font)
        label.setForeground(self.total_table.palette().brush(QtGui.QPalette.ColorRole.Text))
        label.setToolTip("Number of ROIs edited in each category / all ROIs in this experiment")
        for c, n in enumerate(counts, start=1):
            item = self._total_item(c)
            item.setText(f"{n}/{total}")
            _style_item(item, edited=n > 0, base_font=base_font)
            if c == 1:
                extra = []
                if n_new:
                    extra.append(f"{n_new} new (nothing to compare with)")
                if n_unknown:
                    extra.append(f"{n_unknown} cannot be compared")
                item.setToolTip("; ".join(extra) if extra else "")
            elif COMPONENTS[c - 2] not in RESETTABLE:
                item.setToolTip(EXPERIMENT_WIDE_NOTE)
            else:
                item.setToolTip("")

    def _total_item(self, col: int) -> QTableWidgetItem:
        item = self.total_table.item(0, col)
        if item is None:
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            self.total_table.setItem(0, col, item)
        return item

    # ------------------------------------------------------ column widths
    def _fit_columns(self) -> None:
        """Same width for a column in both tables; the last one takes the slack."""
        if not hasattr(self, "total_table") or self._user_sized or self._fitting:
            return  # not built yet, or the user sized the columns
        self._fitting = True
        try:
            n = self.table.columnCount()
            head = self.total_table.horizontalHeader()
            widths = [
                max(
                    head.sectionSizeHint(c),
                    self.total_table.sizeHintForColumn(c),
                    self.table.sizeHintForColumn(c),
                )
                + 14
                for c in range(n)
            ]
            extra = self.table.viewport().width() - sum(widths)
            if extra > 0:
                widths[-1] += extra
            for c, w in enumerate(widths):
                self.table.setColumnWidth(c, w)
                self.total_table.setColumnWidth(c, w)
        finally:
            self._fitting = False

    def _on_total_section_resized(self, col: int, _old: int, new: int) -> None:
        if self._fitting:
            return
        self._user_sized = True  # the user dragged a column edge: stop auto-fitting
        self.table.setColumnWidth(col, new)

    def showEvent(self, event: Any) -> None:  # noqa: N802
        super().showEvent(event)
        self._fit_columns()

    def resizeEvent(self, event: Any) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit_columns()

    def _set_cell(self, row: int, col: int, text: str, style: str, base_font: QtGui.QFont) -> None:
        cache_key = (row, col)
        if self._cache.get(cache_key) == (text, style):
            return
        item = self.table.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            self.table.setItem(row, col, item)
        item.setText(text)
        if style == "plain":
            item.setFont(QtGui.QFont(base_font))
            item.setForeground(self.table.palette().brush(QtGui.QPalette.ColorRole.Text))
        elif style == "edited":
            _style_item(item, edited=True, base_font=base_font)
        elif style == "note":
            _style_item(item, edited=False, base_font=base_font, kind="note")
        else:
            _style_item(item, edited=False, base_font=base_font)
        self._cache[cache_key] = (text, style)

    # ----------------------------------------------------------- selection
    def sync_active(self) -> None:
        """Highlight the main window's active ROI (no jump back)."""
        main = self.main
        if main.doc is None or not self._ids:
            return
        try:
            row = self._ids.index(int(main.active_roi_id))
        except ValueError:
            return
        current = self.table.currentRow()
        if current == row and self.table.item(row, 0) is not None and self.table.item(row, 0).isSelected():
            return
        self._filling = True
        try:
            self.table.selectRow(row)
            item = self.table.item(row, 0)
            if item is not None:
                self.table.scrollToItem(item, QAbstractItemView.ScrollHint.EnsureVisible)
        finally:
            self._filling = False

    def _on_select(self) -> None:
        if self._filling:
            return
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return
        r = rows[0].row()
        if 0 <= r < len(self._ids):
            self.main.jump_to_roi(self._ids[r])
