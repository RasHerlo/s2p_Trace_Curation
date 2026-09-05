"""Non-modal Annotation Tools window: heatmap-style ranges on a mirrored raster."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from s2p_trace_curation.analyses import active_sort_run, apply_raster_sort
from s2p_trace_curation.annotations import (
    ANNOTATION_PROPERTIES,
    PROPERTY_BG_MOTION,
    PROPERTY_PMT_NOISE,
    SPAN_FILL_ALPHA,
    SPAN_PEN_ALPHA,
    annotation_kind,
    annotation_list_text,
    annotation_ranges,
    ensure_annotations,
    get_annotation,
    is_bg_motion,
    is_bundle_kind,
    is_led_shutter,
    is_pmt_noise,
    make_annotation,
    next_ann_id,
    normalize_property_name,
    property_spec,
    set_annotation_ranges,
    validate_annotation_frames,
)
from s2p_trace_curation.bg_rois import ensure_bg_rois
from s2p_trace_curation.pmt_noise import RMS_COLUMN, read_per_frame_rms
from s2p_trace_curation.gui.colormaps import (
    colorize_raster,
    lut_with_revert,
    selected_row_lut,
)
from s2p_trace_curation.heatmaps import normalize_ranges
from s2p_trace_curation.raster import rois_for_raster
from s2p_trace_curation.trace_processing import (
    TRACE_FIELD_LABELS,
    TRACE_FIELD_NORM,
    TRACE_FIELDS,
    raster_trace_field,
    stack_trace_field,
)
from s2p_trace_curation.gui.x_units import (
    UNITS_FRAMES,
    UNITS_SECONDS,
    combo_x_units,
    fill_x_units_combo,
)
from s2p_trace_curation.user_settings import load_settings, save_settings

Qt = QtCore.Qt
QComboBox = QtWidgets.QComboBox
QDialog = QtWidgets.QDialog
QFileDialog = QtWidgets.QFileDialog
QFormLayout = QtWidgets.QFormLayout
QGroupBox = QtWidgets.QGroupBox
QHBoxLayout = QtWidgets.QHBoxLayout
QLabel = QtWidgets.QLabel
QListWidget = QtWidgets.QListWidget
QListWidgetItem = QtWidgets.QListWidgetItem
QMessageBox = QtWidgets.QMessageBox
QPushButton = QtWidgets.QPushButton
QSpinBox = QtWidgets.QSpinBox
QSplitter = QtWidgets.QSplitter
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget

RANGE_BRUSH = (241, 196, 15, 60)
RANGE_RASTER_BRUSH = (241, 196, 15, 45)
RANGE_PEN = "#f1c40f"

# Longest preset is "LED+Shutter"; keep the combo wide enough for the text
# plus the drop-down arrow (File lives on the right action row).
_KIND_COMBO_MIN_WIDTH = 220
_LEFT_PANEL_MIN_WIDTH = 280


class AnnotationEditorWindow(QDialog):
    def __init__(self, main: Any) -> None:
        super().__init__(main)
        from s2p_trace_curation.gui.main_window import ClickableImageView

        self.main = main
        self.setWindowTitle("Annotation Tools")
        self.setModal(False)
        self.setMinimumSize(1000, 680)
        self._editing_id: int | None = None
        self._loading = False
        self._updating = False
        self._trace_field = TRACE_FIELD_NORM
        self._ranges: list[list[int]] = []
        self._draft_label = ""
        self._range_items: list[pg.LinearRegionItem] = []
        self._raster_range_items: list[pg.LinearRegionItem] = []
        self._guide_spans: list[pg.LinearRegionItem] = []
        self._raster_row_ids: list[int] = []
        self._raster_shape: tuple[int, int] | None = None

        layout = QVBoxLayout(self)
        outer = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left.setMinimumWidth(_LEFT_PANEL_MIN_WIDTH)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Saved annotations"))
        self.list_ann = QListWidget()
        self.list_ann.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list_ann.itemSelectionChanged.connect(self._on_selected)
        self.list_ann.itemClicked.connect(self._on_item_clicked)
        self.list_ann.itemDoubleClicked.connect(self._on_item_clicked)
        left_layout.addWidget(self.list_ann, stretch=1)
        list_btns = QHBoxLayout()
        self.btn_new = QPushButton("New")
        self.btn_new.clicked.connect(self._on_new)
        self.btn_delete = QPushButton("Delete selected")
        self.btn_delete.clicked.connect(self._on_delete)
        list_btns.addWidget(self.btn_new)
        list_btns.addWidget(self.btn_delete)
        left_layout.addLayout(list_btns)

        form = QFormLayout()
        self.cmb_prop = QComboBox()
        self.cmb_prop.setEditable(True)
        self.cmb_prop.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self.cmb_prop.setToolTip(
            "Pick LED+Shutter, AirPuff, PMT-noise, or BG-motion, or type a "
            "custom name. Selecting an LED+Shutter row NaNs that span on the traces."
        )
        self.cmb_prop.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.cmb_prop.setMinimumContentsLength(12)
        self.cmb_prop.setMinimumWidth(_KIND_COMBO_MIN_WIDTH)
        self.cmb_prop.currentTextChanged.connect(self._on_kind_text_changed)
        form.addRow("Kind", self.cmb_prop)
        left_layout.addLayout(form)
        self.lbl_status = QLabel("New draft — add ranges, then Save")
        self.lbl_status.setWordWrap(True)
        left_layout.addWidget(self.lbl_status)
        outer.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        raster_box = QGroupBox("Raster")
        raster_layout = QVBoxLayout(raster_box)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Trace"))
        self.cmb_trace = QComboBox()
        self.cmb_trace.setToolTip(
            "Which stored trace to show in the raster and in the plot below. "
            "Click a raster row to select that ROI."
        )
        for field in TRACE_FIELDS:
            self.cmb_trace.addItem(TRACE_FIELD_LABELS[field], field)
        self.cmb_trace.currentIndexChanged.connect(self._on_trace_field_changed)
        mode_row.addWidget(self.cmb_trace)
        mode_row.addSpacing(14)
        mode_row.addWidget(QLabel("X units"))
        self.cmb_units = QComboBox()
        fill_x_units_combo(
            self.cmb_units,
            self._fs(),
            load_settings().get("heatmap_x_units"),
        )
        self.cmb_units.setToolTip(
            "Tick labels on the raster and trace x-axes. Seconds needs the "
            "frame rate (meta 'fs' from ops.npy); ranges stay in frames."
        )
        self.cmb_units.currentIndexChanged.connect(self._on_units_changed)
        mode_row.addWidget(self.cmb_units)
        mode_row.addStretch(1)
        self.lbl_raster_info = QLabel("")
        mode_row.addWidget(self.lbl_raster_info)
        raster_layout.addLayout(mode_row)
        self.raster_plot = pg.PlotItem()
        self.raster_view = ClickableImageView(
            view=self.raster_plot, on_click=self._on_raster_click
        )
        self.raster_view.ui.histogram.hide()
        self.raster_plot.showAxis("bottom", True)
        self.raster_plot.showAxis("left", True)
        self.raster_plot.setLabel("left", "row")
        self.raster_plot.setLabel("bottom", "frame")
        self.raster_plot.showGrid(x=True, y=False, alpha=0.25)
        self.raster_plot.getAxis("bottom").enableAutoSIPrefix(False)
        raster_layout.addWidget(self.raster_view)
        right_layout.addWidget(raster_box, stretch=3)

        self.plot_trace = pg.PlotWidget(title="Trace")
        self.plot_trace.setLabel("bottom", "frame")
        self.plot_trace.showGrid(x=True, y=False, alpha=0.2)
        self.plot_trace.getAxis("bottom").enableAutoSIPrefix(False)
        self.curve_trace = self.plot_trace.plot(pen=pg.mkPen("#2ca02c", width=1.4))
        right_layout.addWidget(self.plot_trace, stretch=2)

        ranges_box = QGroupBox("Draft ranges")
        ranges_layout = QHBoxLayout(ranges_box)
        self.list_ranges = QListWidget()
        self.list_ranges.setMaximumHeight(110)
        ranges_layout.addWidget(self.list_ranges, stretch=1)
        edit_col = QFormLayout()
        self.spin_start = QSpinBox()
        self.spin_end = QSpinBox()
        for s in (self.spin_start, self.spin_end):
            s.setRange(0, 10**9)
        self.btn_add_range = QPushButton("Add range")
        self.btn_add_range.clicked.connect(self._on_add_range)
        self.btn_remove_range = QPushButton("Remove selected")
        self.btn_remove_range.clicked.connect(self._on_remove_range)
        edit_col.addRow("Start", self.spin_start)
        edit_col.addRow("End", self.spin_end)
        edit_col.addRow(self.btn_add_range)
        edit_col.addRow(self.btn_remove_range)
        ranges_layout.addLayout(edit_col)
        right_layout.addWidget(ranges_box)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self.btn_pmt_file = QPushButton("File\u2026")
        self.btn_pmt_file.setEnabled(False)
        self.btn_pmt_file.setToolTip(
            f"PMT-noise only: load a per_frame.csv and threshold its "
            f"'{RMS_COLUMN}' column into ranges."
        )
        self.btn_pmt_file.clicked.connect(self._on_pmt_file)
        self.btn_bg_rois = QPushButton("BG ROIs")
        self.btn_bg_rois.setEnabled(False)
        self.btn_bg_rois.setToolTip(
            "BG-motion only: plot saved BG-ROI traces and threshold their raw sum."
        )
        self.btn_bg_rois.clicked.connect(self._on_bg_rois)
        self.btn_save = QPushButton("Save")
        self.btn_save.setToolTip(
            "Write every draft range as an annotation of the selected kind. "
            "Click a saved row to load it for editing."
        )
        self.btn_save.clicked.connect(self._on_save)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        action_row.addWidget(self.btn_pmt_file)
        action_row.addWidget(self.btn_bg_rois)
        action_row.addWidget(self.btn_save)
        action_row.addWidget(btn_close)
        right_layout.addLayout(action_row)

        hint = QLabel(
            "Kind is a preset (LED+Shutter, AirPuff, PMT-noise, BG-motion) or a "
            "name you type. Click a saved row to load it, or New for a draft. "
            "Drag a range edge on the trace, or type Start/End and Add. "
            "Save writes each draft range as its own annotation, except PMT-noise "
            "and BG-motion, which become one annotation holding every interval. "
            "Select an LED+Shutter row to NaN those spans on the main traces."
        )
        hint.setWordWrap(True)
        right_layout.addWidget(hint)

        outer.addWidget(right)
        outer.setStretchFactor(0, 1)
        outer.setStretchFactor(1, 4)
        layout.addWidget(outer)

        if self.main.doc is not None:
            self._trace_field = raster_trace_field(self.main.doc)
        self._sync_trace_combo()
        self.refresh_from_doc()

    def _doc(self) -> dict[str, Any] | None:
        return self.main.doc

    def _nframes(self) -> int:
        doc = self._doc()
        return int(doc["meta"]["nframes"]) if doc is not None else 0

    def refresh_from_doc(self) -> None:
        n = self._nframes()
        for s in (self.spin_start, self.spin_end):
            s.setRange(0, max(n - 1, 0))
        if n:
            self.spin_start.setValue(int(0.2 * (n - 1)))
            self.spin_end.setValue(int(0.3 * (n - 1)))
        selected = combo_x_units(self.cmb_units)
        if selected == UNITS_SECONDS and self._fs() is None:
            selected = UNITS_FRAMES
        fill_x_units_combo(self.cmb_units, self._fs(), selected)
        self._apply_x_units()
        self.reload_list(load_form=True)
        self.refresh_raster()

    def _fs(self) -> float | None:
        """Frame rate from the pickle meta, or None when it is unusable."""
        doc = self._doc()
        if doc is None:
            return None
        try:
            fs = float((doc.get("meta") or {}).get("fs"))
        except (TypeError, ValueError):
            return None
        return fs if np.isfinite(fs) and fs > 0 else None

    def _seconds_mode(self) -> bool:
        return combo_x_units(self.cmb_units) == UNITS_SECONDS and self._fs() is not None

    def _apply_x_units(self) -> None:
        """Rescale only the tick labels; plot data stays in frame coordinates."""
        fs = self._fs()
        seconds = self._seconds_mode()
        scale = 1.0 / fs if (seconds and fs) else 1.0
        label = "time (s)" if seconds else "frame"
        for plot in (self.plot_trace, self.raster_plot):
            plot.getAxis("bottom").setScale(scale)
            plot.setLabel("bottom", label)

    def _set_units(self, units: str) -> None:
        fill_x_units_combo(self.cmb_units, self._fs(), units)
        self._apply_x_units()

    def _on_units_changed(self) -> None:
        if combo_x_units(self.cmb_units) == UNITS_SECONDS and self._fs() is None:
            QMessageBox.information(
                self,
                "Seconds unavailable",
                "This pickle has no frame rate (meta 'fs' from ops.npy), so the "
                "x-axes stay in frames.",
            )
            self._set_units(UNITS_FRAMES)
            return
        save_settings({"heatmap_x_units": combo_x_units(self.cmb_units)})
        self._apply_x_units()
        self._rebuild_ranges_ui()

    def reload_list(self, *, load_form: bool = False) -> None:
        doc = self._doc()
        self._loading = True
        self.list_ann.clear()
        if doc is None:
            self._loading = False
            if load_form:
                self._set_draft_form()
            return
        keep = self._editing_id
        selected = set(self.main._ann_selected_ids)
        self._fill_kind_combo()
        for ann in ensure_annotations(doc):
            item = QListWidgetItem(annotation_list_text(ann))
            item.setData(Qt.ItemDataRole.UserRole, int(ann["ann_id"]))
            color = property_spec(str(ann["property"])).get("color", "#888888")
            item.setForeground(QtGui.QColor(color))
            self.list_ann.addItem(item)
        if keep is not None:
            self._select_ids({int(keep)})
        elif selected:
            self._select_ids(selected)
        elif load_form:
            self._set_draft_form()
        self._loading = False
        self._sync_selection_to_main()
        if load_form:
            ids = self._selected_saved_ids()
            if len(ids) == 1:
                self._load_saved(next(iter(ids)))
        self._update_buttons()

    def _select_ids(self, ids: set[int]) -> None:
        self.list_ann.blockSignals(True)
        self.list_ann.clearSelection()
        for i in range(self.list_ann.count()):
            item = self.list_ann.item(i)
            if item is not None and int(item.data(Qt.ItemDataRole.UserRole)) in ids:
                item.setSelected(True)
        self.list_ann.blockSignals(False)

    def _fill_kind_combo(self, current: str | None = None) -> None:
        current = (
            current if current is not None else self.cmb_prop.currentText()
        ).strip()
        self.cmb_prop.blockSignals(True)
        self.cmb_prop.clear()
        seen: set[str] = set()
        for name in ANNOTATION_PROPERTIES:
            self.cmb_prop.addItem(name)
            seen.add(name)
        doc = self._doc()
        if doc is not None:
            extras = sorted(
                {
                    str(ann.get("property") or "").strip()
                    for ann in ensure_annotations(doc)
                }
                - seen
                - {""}
            )
            for name in extras:
                self.cmb_prop.addItem(name)
        if current:
            idx = self.cmb_prop.findText(current)
            if idx >= 0:
                self.cmb_prop.setCurrentIndex(idx)
            else:
                self.cmb_prop.setEditText(current)
        elif self.cmb_prop.count():
            self.cmb_prop.setCurrentIndex(0)
        self.cmb_prop.blockSignals(False)
        self._update_kind_buttons()

    def _form_kind(self) -> str:
        return normalize_property_name(self.cmb_prop.currentText())

    def _set_kind(self, name: str) -> None:
        text = str(name or "").strip()
        idx = self.cmb_prop.findText(text)
        if idx >= 0:
            self.cmb_prop.setCurrentIndex(idx)
        elif text:
            self.cmb_prop.setEditText(text)
        elif self.cmb_prop.count():
            self.cmb_prop.setCurrentIndex(0)
        # currentTextChanged stays silent when the text is already correct.
        self._update_kind_buttons()

    # ----------------------------------------------------------- PMT-noise csv
    def _on_kind_text_changed(self, _text: str = "") -> None:
        self._update_kind_buttons()

    def _update_kind_buttons(self) -> None:
        """File… is PMT-noise; BG ROIs is BG-motion (and needs saved BG ROIs)."""
        if not hasattr(self, "btn_pmt_file") or not hasattr(self, "btn_bg_rois"):
            return
        doc = self._doc()
        has = doc is not None
        kind = self.cmb_prop.currentText()
        self.btn_pmt_file.setEnabled(has and is_pmt_noise(kind))
        n_bg = len(ensure_bg_rois(doc)) if has else 0
        self.btn_bg_rois.setEnabled(has and is_bg_motion(kind) and n_bg > 0)
        if has and is_bg_motion(kind) and n_bg == 0:
            self.btn_bg_rois.setToolTip(
                "Draw at least one BG ROI in Mask Tools first."
            )
        else:
            self.btn_bg_rois.setToolTip(
                "BG-motion only: plot saved BG-ROI traces and threshold their raw sum."
            )

    def _pmt_start_dir(self) -> str:
        raw = load_settings().get("last_pmt_csv_dir")
        if raw and Path(raw).is_dir():
            return str(raw)
        suite2p_dir = getattr(self.main, "suite2p_dir", None)
        if suite2p_dir is not None and Path(suite2p_dir).is_dir():
            return str(suite2p_dir)
        return ""

    def _on_pmt_file(self) -> None:
        doc = self._doc()
        if doc is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select per_frame.csv",
            self._pmt_start_dir(),
            "CSV files (*.csv);;All files (*)",
        )
        if not path:
            return
        nframes = self._nframes()
        try:
            rms = read_per_frame_rms(path, nframes=nframes)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not read per_frame.csv", str(exc))
            return
        if rms.n_mapped == 0:
            QMessageBox.warning(
                self,
                "No usable values",
                f"{rms.path.name} has a '{RMS_COLUMN}' column, but none of its "
                "rows produced a numeric value for a frame in this movie.",
            )
            return
        save_settings({"last_pmt_csv_dir": str(rms.path.parent)})
        if rms.n_rows != nframes:
            QMessageBox.information(
                self,
                "Frame count differs",
                f"{rms.path.name} has {rms.n_rows} row(s) but this movie has "
                f"{nframes} frame(s). Values are mapped by "
                + (
                    f"the '{rms.frame_column}' column"
                    if rms.frame_column
                    else "row order"
                )
                + "; frames without a value are ignored.",
            )

        from s2p_trace_curation.gui.pmt_noise_dialog import PmtNoiseThresholdDialog

        dlg = PmtNoiseThresholdDialog(
            rms,
            parent=self,
            fs=self._fs(),
            seconds=self._seconds_mode(),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        ranges = dlg.ranges()
        if not ranges:
            return
        self._editing_id = None
        self._loading = True
        self.list_ann.clearSelection()
        self._loading = False
        self.main._set_ann_selection(set())
        self._set_kind(PROPERTY_PMT_NOISE)
        self._ranges = ranges
        self._draft_label = (
            f"{rms.path.name} {RMS_COLUMN}>{dlg.threshold():g}"
        )
        self._rebuild_ranges_ui()
        self._refresh_guide_spans()
        n_frames = sum(b - a + 1 for a, b in self._ranges)
        self.lbl_status.setText(
            f"{len(self._ranges)} interval(s) from {rms.path.name} "
            f"({n_frames} frame(s), {RMS_COLUMN} > {dlg.threshold():g}) — "
            "Save writes them as one PMT-noise annotation"
        )
        self.main.statusBar().showMessage(
            f"PMT-noise: {len(self._ranges)} interval(s) above "
            f"{RMS_COLUMN} {dlg.threshold():g}"
        )

    def _on_bg_rois(self) -> None:
        doc = self._doc()
        if doc is None:
            return
        rows = ensure_bg_rois(doc)
        if not rows:
            QMessageBox.information(
                self,
                "No BG ROIs",
                "Draw at least one BG ROI with Mask Tools → Add BG ROI first.",
            )
            return
        from s2p_trace_curation.gui.bg_roi_dialog import BgRoiThresholdDialog

        dlg = BgRoiThresholdDialog(
            doc,
            parent=self,
            fs=self._fs(),
            seconds=self._seconds_mode(),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        ranges = dlg.ranges()
        if not ranges:
            return
        self._editing_id = None
        self._loading = True
        self.list_ann.clearSelection()
        self._loading = False
        self.main._set_ann_selection(set())
        self._set_kind(PROPERTY_BG_MOTION)
        self._ranges = ranges
        ids = dlg.selected_ids()
        id_txt = ",".join(str(i) for i in ids)
        self._draft_label = f"BG {id_txt} sumF>{dlg.threshold():g}"
        self._rebuild_ranges_ui()
        self._refresh_guide_spans()
        n_frames = sum(b - a + 1 for a, b in self._ranges)
        self.lbl_status.setText(
            f"{len(self._ranges)} interval(s) from {len(ids)} BG ROI(s) "
            f"({n_frames} frame(s), sum F > {dlg.threshold():g}) — "
            "Save writes them as one BG-motion annotation"
        )
        self.main.statusBar().showMessage(
            f"BG-motion: {len(self._ranges)} interval(s) above "
            f"sum F {dlg.threshold():g}"
        )

    def _set_draft_form(self) -> None:
        self._editing_id = None
        self._set_kind(ANNOTATION_PROPERTIES[0])
        self._ranges = []
        self._draft_label = ""
        self._rebuild_ranges_ui()
        self.lbl_status.setText("New draft — pick or type a kind, add ranges, then Save")
        self._update_buttons()

    def _on_new(self) -> None:
        self._loading = True
        self.list_ann.clearSelection()
        self._loading = False
        self.main._set_ann_selection(set())
        self._set_draft_form()
        self._refresh_guide_spans()

    def _on_item_clicked(self, item: QListWidgetItem) -> None:
        if self._loading:
            return
        data = item.data(Qt.ItemDataRole.UserRole)
        if data is None:
            return
        self._load_saved(int(data))

    def _on_selected(self) -> None:
        if self._loading:
            return
        ids = self._selected_saved_ids()
        self.main._set_ann_selection(ids)
        if len(ids) == 1:
            self._load_saved(next(iter(ids)))
            return
        self._update_buttons()

    def _load_saved(self, ann_id: int) -> None:
        doc = self._doc()
        if doc is None:
            return
        ann = get_annotation(doc, ann_id)
        if ann is None:
            return
        self._editing_id = int(ann["ann_id"])
        self._set_kind(annotation_kind(ann))
        self._draft_label = str(ann.get("label") or "")
        self._ranges = annotation_ranges(ann)
        self._rebuild_ranges_ui()
        self._refresh_guide_spans()
        name = annotation_kind(ann)
        n = len(self._ranges)
        detail = (
            "drag the range to adjust, or Add more before Save"
            if n <= 1
            else f"{n} intervals loaded — adjust or Add more before Save"
        )
        self.lbl_status.setText(f"Editing {name} — {detail}")
        self._update_buttons()

    def _selected_saved_ids(self) -> set[int]:
        ids: set[int] = set()
        for item in self.list_ann.selectedItems():
            data = item.data(Qt.ItemDataRole.UserRole)
            if data is not None:
                ids.add(int(data))
        return ids

    def _sync_selection_to_main(self) -> None:
        self.main._set_ann_selection(self._selected_saved_ids())

    def _update_buttons(self) -> None:
        has = self._doc() is not None
        self.btn_new.setEnabled(has)
        self.btn_delete.setEnabled(has and bool(self._selected_saved_ids()))
        self.btn_save.setEnabled(has and bool(self._ranges))
        self.btn_add_range.setEnabled(has)
        self.btn_remove_range.setEnabled(has and bool(self._ranges))
        self._update_kind_buttons()

    def _clear_range_items(self) -> None:
        for item in self._range_items:
            try:
                self.plot_trace.removeItem(item)
            except Exception:
                pass
        self._range_items.clear()

    def _clear_raster_range_items(self) -> None:
        view = self.raster_view.getView()
        for item in self._raster_range_items:
            try:
                view.removeItem(item)
            except Exception:
                pass
        self._raster_range_items.clear()

    def _rebuild_raster_range_items(self) -> None:
        self._clear_raster_range_items()
        view = self.raster_view.getView()
        for a, b in self._ranges:
            region = pg.LinearRegionItem(
                values=(float(a) - 0.5, float(b) + 0.5),
                movable=False,
                brush=QtGui.QColor(*RANGE_RASTER_BRUSH),
                pen=pg.mkPen(RANGE_PEN, width=1),
            )
            region.setZValue(20)
            view.addItem(region)
            self._raster_range_items.append(region)

    def _rebuild_ranges_ui(self) -> None:
        self._ranges = normalize_ranges(self._ranges, self._nframes())
        self._clear_range_items()
        for a, b in self._ranges:
            region = pg.LinearRegionItem(
                values=(float(a), float(b) + 1.0),
                movable=True,
                brush=QtGui.QColor(*RANGE_BRUSH),
                pen=pg.mkPen(RANGE_PEN, width=1),
            )
            region.setZValue(-5)
            region.sigRegionChangeFinished.connect(self._on_region_changed)
            self.plot_trace.addItem(region)
            self._range_items.append(region)

        fs = self._fs() if self._seconds_mode() else None
        self.list_ranges.blockSignals(True)
        self.list_ranges.clear()
        for i, (a, b) in enumerate(self._ranges):
            n = int(b) - int(a) + 1
            text = f"{i}:  {int(a)}–{int(b)}   ({n} frames)"
            if fs:
                text += f"   [{int(a) / fs:.2f}–{(int(b) + 1) / fs:.2f} s]"
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.list_ranges.addItem(item)
        self.list_ranges.blockSignals(False)
        self._rebuild_raster_range_items()
        self._update_trace_title()
        self._update_buttons()

    def _on_region_changed(self) -> None:
        if self._loading:
            return
        last = max(self._nframes() - 1, 0)
        collected: list[list[int]] = []
        for region in self._range_items:
            v0, v1 = region.getRegion()
            a = int(round(float(v0)))
            b = int(round(float(v1))) - 1
            if b < a:
                b = a
            collected.append([max(0, min(a, last)), max(0, min(b, last))])
        self._ranges = collected
        self._rebuild_ranges_ui()

    def _on_add_range(self) -> None:
        a = int(self.spin_start.value())
        b = int(self.spin_end.value())
        if b < a:
            a, b = b, a
        self._ranges.append([a, b])
        self._rebuild_ranges_ui()

    def _on_remove_range(self) -> None:
        items = self.list_ranges.selectedItems()
        if not items:
            if self._ranges:
                self._ranges.pop()
        else:
            drop = {int(i.data(Qt.ItemDataRole.UserRole)) for i in items}
            self._ranges = [r for i, r in enumerate(self._ranges) if i not in drop]
        self._rebuild_ranges_ui()

    def _raster_rows(self) -> list[dict[str, Any]]:
        doc = self._doc()
        if doc is None:
            return []
        rows = rois_for_raster(
            doc["rois"],
            self.main._overlay_filter(),
            active_roi_id=self.main.active_roi_id,
        )
        return apply_raster_sort(rows, active_sort_run(doc))

    def refresh_raster(self) -> None:
        doc = self._doc()
        if doc is None:
            self._raster_row_ids = []
            self._raster_shape = None
            self.curve_trace.setData([], [])
            return
        n = self._nframes()
        rows = self._raster_rows()
        self._raster_row_ids = [int(r["roi_id"]) for r in rows]
        field = self._trace_field
        if rows:
            matrix = stack_trace_field(rows, n, field)
        else:
            matrix = np.full((1, n), np.nan, dtype=np.float64)
        lut = lut_with_revert(
            self.main.cmb_raster_lut.currentText(),
            self.main.chk_raster_revert.isChecked(),
        )
        highlight_row: int | None = None
        highlight_lut = None
        if rows:
            try:
                highlight_row = self._raster_row_ids.index(int(self.main.active_roi_id))
            except ValueError:
                highlight_row = None
            if highlight_row is not None:
                highlight_lut = selected_row_lut(self.main.chk_raster_revert.isChecked())
        rgb = colorize_raster(
            matrix, lut, highlight_row=highlight_row, highlight_lut=highlight_lut
        )
        from s2p_trace_curation.gui.main_window import keep_image_zoom

        shape = keep_image_zoom(
            self.raster_view, rgb, prev_shape=self._raster_shape
        )
        self._raster_shape = shape
        self.lbl_raster_info.setText(f"{len(rows)} row(s) — {field}")
        self._refresh_trace(rows, matrix, field)
        self._refresh_guide_spans()
        self._rebuild_raster_range_items()
        self._apply_x_units()

    def _refresh_trace(
        self, rows: list[dict[str, Any]], matrix: np.ndarray, field: str
    ) -> None:
        n = matrix.shape[1]
        xs = np.arange(n)
        y = np.full(n, np.nan, dtype=np.float64)
        roi_id = int(self.main.active_roi_id)
        by_id = {int(r["roi_id"]): r for r in rows}
        row = by_id.get(roi_id)
        if row is not None and row.get(field) is not None:
            arr = np.asarray(row[field], dtype=np.float64)
            n_copy = min(arr.shape[0], n)
            y[:n_copy] = arr[:n_copy]
        self.curve_trace.setData(xs, y)
        self._update_trace_title()

    def _update_trace_title(self) -> None:
        field = TRACE_FIELD_LABELS.get(self._trace_field, self._trace_field)
        roi_id = int(self.main.active_roi_id)
        n = len(self._ranges)
        extra = "no draft ranges" if n == 0 else f"{n} draft range(s)"
        self.plot_trace.setTitle(f"{field} — ROI {roi_id} — {extra}")

    def _clear_guide_spans(self) -> None:
        for item in self._guide_spans:
            try:
                self.plot_trace.removeItem(item)
            except Exception:
                pass
        self._guide_spans.clear()

    def _refresh_guide_spans(self) -> None:
        self._clear_guide_spans()
        doc = self._doc()
        if doc is None:
            return
        skip = {self._editing_id} if self._editing_id is not None else set()
        for ann in ensure_annotations(doc):
            if int(ann["ann_id"]) in skip:
                continue
            prop = str(ann["property"])
            if not self.main._ann_kind_visible(prop):
                continue
            color = property_spec(prop).get("color", "#888888")
            fill = QtGui.QColor(color)
            fill.setAlpha(SPAN_FILL_ALPHA)
            edge = QtGui.QColor(color)
            edge.setAlpha(SPAN_PEN_ALPHA)
            for a, b in annotation_ranges(ann):
                region = pg.LinearRegionItem(
                    values=(float(a), float(b) + 1.0),
                    movable=False,
                    brush=fill,
                    pen=pg.mkPen(edge, width=1),
                )
                region.setHoverBrush(fill)
                region.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
                region.setZValue(-10)
                self.plot_trace.addItem(region)
                self._guide_spans.append(region)

    def _sync_trace_combo(self) -> None:
        idx = self.cmb_trace.findData(self._trace_field)
        was = self._updating
        self._updating = True
        self.cmb_trace.setCurrentIndex(idx if idx >= 0 else 0)
        self._updating = was

    def _on_trace_field_changed(self) -> None:
        if self._updating:
            return
        data = self.cmb_trace.currentData()
        field = str(data) if data else TRACE_FIELD_NORM
        if field == self._trace_field:
            return
        if not self.main._ensure_analysis_inputs(field):
            self._sync_trace_combo()
            return
        self._trace_field = field
        self.refresh_raster()

    def _on_raster_click(self, y: int, x: int) -> None:
        if not self._raster_row_ids:
            return
        n = self._nframes()
        if n and not 0 <= int(x) < n:
            return
        row = int(y)
        if 0 <= row < len(self._raster_row_ids):
            self.main._select_roi(self._raster_row_ids[row])

    def on_active_roi_changed(self) -> None:
        self.refresh_raster()

    def _on_save(self) -> None:
        doc = self._doc()
        if doc is None:
            return
        nframes = self._nframes()
        ranges = normalize_ranges(self._ranges, nframes)
        if not ranges:
            QMessageBox.information(self, "Save", "Add at least one range first.")
            return
        try:
            prop = self._form_kind()
        except ValueError as exc:
            QMessageBox.warning(self, "Kind required", str(exc))
            return
        anns = ensure_annotations(doc)
        led = is_led_shutter(prop)
        label = self._draft_label
        # PMT-noise and BG-motion are one feature made of many intervals.
        if is_bundle_kind(prop):
            self._save_as_one(doc, anns, prop, ranges, label, led)
            return
        if len(ranges) > 50:
            reply = QMessageBox.question(
                self,
                "Save many annotations",
                f"This writes {len(ranges)} separate {prop} annotations to the "
                "pickle. Continue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        created: list[int] = []
        first = ranges[0]
        rest = ranges[1:]
        try:
            s, e = validate_annotation_frames(first[0], first[1], nframes)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid frames", str(exc))
            return
        if self._editing_id is not None:
            ann = get_annotation(doc, self._editing_id)
            if ann is None:
                QMessageBox.warning(
                    self, "Save", "That annotation is no longer in the pickle."
                )
                return
            if is_led_shutter(str(ann.get("property"))):
                led = True
            ann["property"] = prop
            ann["label"] = label
            set_annotation_ranges(ann, [[s, e]], nframes)
            created.append(int(ann["ann_id"]))
        else:
            ann = make_annotation(next_ann_id(doc), prop, s, e, label=label)
            anns.append(ann)
            created.append(int(ann["ann_id"]))
        for a, b in rest:
            try:
                s2, e2 = validate_annotation_frames(a, b, nframes)
            except ValueError as exc:
                QMessageBox.warning(self, "Invalid frames", str(exc))
                break
            extra = make_annotation(next_ann_id(doc), prop, s2, e2, label=label)
            anns.append(extra)
            created.append(int(extra["ann_id"]))
        self._editing_id = created[0] if created else None
        self._after_save(led)
        self.lbl_status.setText(f"Saved {len(created)} annotation(s) as {prop}")
        self.main.statusBar().showMessage(f"Saved {len(created)} {prop} annotation(s)")

    def _save_as_one(
        self,
        doc: dict[str, Any],
        anns: list[dict[str, Any]],
        prop: str,
        ranges: list[list[int]],
        label: str,
        led: bool,
    ) -> None:
        """Write every draft interval into a single annotation."""
        nframes = self._nframes()
        if self._editing_id is not None:
            ann = get_annotation(doc, self._editing_id)
            if ann is None:
                QMessageBox.warning(
                    self, "Save", "That annotation is no longer in the pickle."
                )
                return
            if is_led_shutter(str(ann.get("property"))):
                led = True
            ann["property"] = prop
            ann["label"] = label
        else:
            ann = make_annotation(
                next_ann_id(doc), prop, ranges[0][0], ranges[0][1], label=label
            )
            anns.append(ann)
        try:
            saved = set_annotation_ranges(ann, ranges, nframes)
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid frames", str(exc))
            return
        self._editing_id = int(ann["ann_id"])
        self._ranges = [list(r) for r in saved]
        self._after_save(led)
        frames = sum(b - a + 1 for a, b in saved)
        msg = (
            f"Saved 1 {prop} annotation — {len(saved)} interval(s), "
            f"{frames} frame(s)"
        )
        self.lbl_status.setText(msg)
        self.main.statusBar().showMessage(msg)

    def _after_save(self, led: bool) -> None:
        self.main._annotations_changed(led_changed=led)
        self.reload_list(load_form=True)
        self._refresh_guide_spans()

    def _on_delete(self) -> None:
        doc = self._doc()
        if doc is None:
            return
        ids = self._selected_saved_ids()
        if not ids:
            return
        reply = QMessageBox.question(
            self,
            "Delete annotations",
            f"Delete {len(ids)} annotation(s) from the pickle?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        anns = ensure_annotations(doc)
        dropped = [a for a in anns if int(a["ann_id"]) in ids]
        doc["annotations"] = [a for a in anns if int(a["ann_id"]) not in ids]
        led = any(is_led_shutter(str(a["property"])) for a in dropped)
        if self._editing_id in ids:
            self._editing_id = None
            self._ranges = []
        self.main._set_ann_selection(set())
        self.main._annotations_changed(led_changed=led)
        self.reload_list(load_form=True)
        self._rebuild_ranges_ui()
        self._refresh_guide_spans()
        self.main.statusBar().showMessage(f"Deleted {len(ids)} annotation(s)")
