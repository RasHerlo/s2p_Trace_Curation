"""Non-modal Edit HeatMaps window: raster + trace ranges → named maps."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from s2p_trace_curation.analyses import active_sort_run, apply_raster_sort
from s2p_trace_curation.annotations import PROPERTY_SPEC, ensure_annotations
from s2p_trace_curation.gui.colormaps import (
    LUT_NAMES,
    colorize_raster,
    lut_with_revert,
    make_lut,
    selected_row_lut,
)
from s2p_trace_curation.heatmaps import (
    HEATMAP_KIND_LABELS,
    KIND_AUC_RATIO,
    HeatmapCancelled,
    apply_heatmap_result,
    compute_heatmap_map,
    default_heatmap_params,
    ensure_heatmaps,
    format_ranges,
    get_heatmap,
    heatmap_combo_label,
    make_heatmap,
    next_heatmap_id,
    normalize_heatmap_params,
    normalize_ranges,
    split_frame_weights,
)
from s2p_trace_curation.gui.overlays import iter_visible_rois, thick_outline_mask
from s2p_trace_curation.raster import led_shutter_nan_mask, rois_for_raster
from s2p_trace_curation.trace_processing import raster_trace_field, stack_trace_field

Qt = QtCore.Qt
QCheckBox = QtWidgets.QCheckBox
QComboBox = QtWidgets.QComboBox
QDialog = QtWidgets.QDialog
QFormLayout = QtWidgets.QFormLayout
QGroupBox = QtWidgets.QGroupBox
QHBoxLayout = QtWidgets.QHBoxLayout
QInputDialog = QtWidgets.QInputDialog
QLabel = QtWidgets.QLabel
QLineEdit = QtWidgets.QLineEdit
QListWidget = QtWidgets.QListWidget
QListWidgetItem = QtWidgets.QListWidgetItem
QMessageBox = QtWidgets.QMessageBox
QProgressBar = QtWidgets.QProgressBar
QPushButton = QtWidgets.QPushButton
QSlider = QtWidgets.QSlider
QSpinBox = QtWidgets.QSpinBox
QSplitter = QtWidgets.QSplitter
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget

RANGE_BRUSH = (241, 196, 15, 60)
RANGE_RASTER_BRUSH = (241, 196, 15, 45)
RANGE_PEN = "#f1c40f"
OUTLINE_RGB = (255, 0, 0)
OUTLINE_ACTIVE_RGB = (0, 255, 255)


class HeatmapEditorWindow(QDialog):
    def __init__(self, main: Any) -> None:
        super().__init__(main)
        from s2p_trace_curation.gui.main_window import ClickableImageView

        self.main = main
        self.setWindowTitle("Edit HeatMaps")
        self.setModal(False)
        self.setMinimumSize(1100, 720)
        self._editing_id: str | None = None
        self._preview_map: np.ndarray | None = None
        self._loading = False
        self._cancel = False
        self._computing = False
        self._batch = False
        self._ranges: list[list[int]] = []
        self._range_items: list[pg.LinearRegionItem] = []
        self._raster_range_items: list[pg.LinearRegionItem] = []
        self._ann_spans: list[pg.LinearRegionItem] = []
        self._raster_row_ids: list[int] = []

        layout = QVBoxLayout(self)
        outer = QSplitter(Qt.Orientation.Horizontal)

        # ---------------------------------------------------------- left list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Saved heatmaps"))
        self.list_maps = QListWidget()
        self.list_maps.itemSelectionChanged.connect(self._on_selected)
        left_layout.addWidget(self.list_maps, stretch=1)
        list_btns = QHBoxLayout()
        self.btn_new = QPushButton("New")
        self.btn_new.clicked.connect(self._on_new)
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self._on_delete)
        list_btns.addWidget(self.btn_new)
        list_btns.addWidget(self.btn_delete)
        left_layout.addLayout(list_btns)

        form = QFormLayout()
        self.edit_label = QLineEdit()
        self.edit_label.setPlaceholderText("Untitled")
        self.cmb_kind = QComboBox()
        for kind, text in HEATMAP_KIND_LABELS.items():
            self.cmb_kind.addItem(text, kind)
        self.cmb_kind.setToolTip(
            "AUC ratio: span-normalized AUC inside the ranges divided by the "
            "span-normalized AUC outside them (shutter frames excluded)"
        )
        form.addRow("Label", self.edit_label)
        form.addRow("Metric", self.cmb_kind)
        left_layout.addLayout(form)
        self.lbl_status = QLabel("New draft — set ranges, Compute, then Save")
        self.lbl_status.setWordWrap(True)
        left_layout.addWidget(self.lbl_status)
        outer.addWidget(left)

        # ----------------------------------------------------------- right side
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        top = QSplitter(Qt.Orientation.Horizontal)

        map_box = QGroupBox("HeatMap preview")
        map_layout = QVBoxLayout(map_box)
        map_ctl = QHBoxLayout()
        map_ctl.addWidget(QLabel("LUT"))
        self.cmb_map_lut = QComboBox()
        self.cmb_map_lut.addItems(list(LUT_NAMES))
        idx_turbo = self.cmb_map_lut.findText("turbo")
        if idx_turbo >= 0:
            self.cmb_map_lut.setCurrentIndex(idx_turbo)
        self.cmb_map_lut.currentIndexChanged.connect(self._redraw_map)
        map_ctl.addWidget(self.cmb_map_lut)
        self.chk_outlines = QCheckBox("ROI outlines")
        self.chk_outlines.setToolTip(
            "Draw outlines of the ROIs passing Show ROIs on top of the map "
            "(active ROI in cyan)"
        )
        self.chk_outlines.toggled.connect(self._refresh_map_overlay)
        map_ctl.addWidget(self.chk_outlines)
        map_ctl.addStretch(1)
        map_layout.addLayout(map_ctl)
        self.plot_map = pg.PlotWidget()
        self.plot_map.invertY(True)
        self.plot_map.setAspectLocked(True)
        self.map_img = pg.ImageItem()
        self.plot_map.addItem(self.map_img)
        self.map_outlines = pg.ImageItem()
        self.map_outlines.setZValue(10)
        self.plot_map.addItem(self.map_outlines)
        map_layout.addWidget(self.plot_map)
        top.addWidget(map_box)

        raster_box = QGroupBox("Raster (mirrors Raster Tools)")
        raster_layout = QVBoxLayout(raster_box)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Single"))
        self.slider_batch = QSlider(Qt.Orientation.Horizontal)
        self.slider_batch.setMinimum(0)
        self.slider_batch.setMaximum(1)
        self.slider_batch.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_batch.setTickInterval(1)
        self.slider_batch.setPageStep(1)
        self.slider_batch.setFixedWidth(80)
        self.slider_batch.setToolTip(
            "Single: selected ROI's trace. Batch: Co-Activity mean of raster rows."
        )
        self.slider_batch.valueChanged.connect(self._on_batch_slider)
        mode_row.addWidget(self.slider_batch)
        mode_row.addWidget(QLabel("Batch"))
        mode_row.addStretch(1)
        self.lbl_raster_info = QLabel("")
        mode_row.addWidget(self.lbl_raster_info)
        raster_layout.addLayout(mode_row)
        self.raster_view = ClickableImageView(on_click=self._on_raster_click)
        self.raster_view.ui.histogram.hide()
        raster_layout.addWidget(self.raster_view)
        top.addWidget(raster_box)
        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 1)
        right_layout.addWidget(top, stretch=3)

        self.plot_trace = pg.PlotWidget(title="Trace")
        self.plot_trace.setLabel("bottom", "frame")
        self.plot_trace.showGrid(x=True, y=False, alpha=0.2)
        self.curve_trace = self.plot_trace.plot(pen=pg.mkPen("#2ca02c", width=1.4))
        right_layout.addWidget(self.plot_trace, stretch=2)

        ranges_box = QGroupBox("Ranges (inside)")
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

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        right_layout.addWidget(self.progress)

        action_row = QHBoxLayout()
        self.btn_compute = QPushButton("Compute / Update")
        self.btn_compute.setToolTip("Stream data.bin once and build the map")
        self.btn_compute.clicked.connect(self._on_compute)
        self.btn_cancel = QPushButton("Cancel compute")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel_compute)
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save_as = QPushButton("Save as")
        self.btn_save_as.clicked.connect(self._on_save_as)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        action_row.addWidget(self.btn_compute)
        action_row.addWidget(self.btn_cancel)
        action_row.addWidget(self.btn_save)
        action_row.addWidget(self.btn_save_as)
        action_row.addStretch(1)
        action_row.addWidget(btn_close)
        right_layout.addLayout(action_row)

        hint = QLabel(
            "Drag a range edge on the trace, or type Start/End and Add; ranges "
            "are mirrored onto the raster. Annotation spans are shown for "
            "guidance; LED+Shutter frames are dropped from both sides of the ratio."
        )
        hint.setWordWrap(True)
        right_layout.addWidget(hint)

        outer.addWidget(right)
        outer.setStretchFactor(0, 1)
        outer.setStretchFactor(1, 4)
        layout.addWidget(outer)

        self.refresh_from_doc()

    # ------------------------------------------------------------------ state
    def _doc(self) -> dict[str, Any] | None:
        return self.main.doc

    def _nframes(self) -> int:
        doc = self._doc()
        return int(doc["meta"]["nframes"]) if doc is not None else 0

    def refresh_from_doc(self) -> None:
        self._preview_map = None
        doc = self._doc()
        n = self._nframes()
        for s in (self.spin_start, self.spin_end):
            s.setRange(0, max(n - 1, 0))
        if doc is not None and n:
            self.spin_start.setValue(int(0.2 * (n - 1)))
            self.spin_end.setValue(int(0.3 * (n - 1)))
        self.reload_list(load_form=True)
        self.refresh_raster()

    def reload_list(self, *, load_form: bool = False) -> None:
        doc = self._doc()
        self._loading = True
        self.list_maps.clear()
        if doc is None:
            self._loading = False
            if load_form:
                self._set_draft_form()
            return
        keep = self._editing_id
        for hm in ensure_heatmaps(doc):
            item = QListWidgetItem(heatmap_combo_label(hm))
            item.setData(Qt.ItemDataRole.UserRole, str(hm["id"]))
            self.list_maps.addItem(item)
        if keep:
            self._select_id(keep)
        elif load_form or self.list_maps.count() == 0:
            self._set_draft_form()
        self._loading = False
        if load_form and keep:
            self._on_selected()
        self._update_buttons()

    def _select_id(self, heatmap_id: str) -> None:
        for i in range(self.list_maps.count()):
            item = self.list_maps.item(i)
            if item is not None and str(item.data(Qt.ItemDataRole.UserRole)) == heatmap_id:
                self.list_maps.setCurrentRow(i)
                return
        self.list_maps.clearSelection()
        self._set_draft_form()

    def _set_draft_form(self) -> None:
        self._editing_id = None
        self._preview_map = None
        self.edit_label.setText("")
        params = default_heatmap_params()
        idx = self.cmb_kind.findData(params["kind"])
        if idx >= 0:
            self.cmb_kind.setCurrentIndex(idx)
        self._ranges = []
        self._rebuild_ranges_ui()
        self._clear_map()
        self.lbl_status.setText("New draft — set ranges, Compute, then Save")
        self._update_buttons()

    def _on_new(self) -> None:
        self._loading = True
        self.list_maps.clearSelection()
        self._loading = False
        self._set_draft_form()

    def _on_selected(self) -> None:
        if self._loading:
            return
        items = self.list_maps.selectedItems()
        if not items:
            return
        doc = self._doc()
        if doc is None:
            return
        hid = str(items[0].data(Qt.ItemDataRole.UserRole))
        hm = get_heatmap(doc, hid)
        if hm is None:
            return
        self._editing_id = hid
        self._preview_map = None
        self.edit_label.setText(str(hm.get("label") or ""))
        params = normalize_heatmap_params(hm.get("params") or {}, self._nframes())
        idx = self.cmb_kind.findData(params["kind"])
        if idx >= 0:
            self.cmb_kind.setCurrentIndex(idx)
        self._ranges = [list(p) for p in params["ranges"]]
        self._rebuild_ranges_ui()
        stored = hm.get("map")
        if stored is not None:
            self._show_map(np.asarray(stored, dtype=np.float64))
        else:
            self._clear_map()
        if not self._ranges:
            self.lbl_status.setText(
                f"{hm['id']} — no ranges stored (legacy). Set ranges and Compute."
            )
        else:
            self.lbl_status.setText(f"{hm['id']} — saved. Compute to replace the map.")
        self._update_buttons()

    def _update_buttons(self) -> None:
        has = self._doc() is not None and not self._computing
        self.btn_new.setEnabled(has)
        self.btn_delete.setEnabled(has and self._editing_id is not None)
        self.btn_compute.setEnabled(has and bool(self._ranges))
        self.btn_save.setEnabled(has)
        self.btn_save_as.setEnabled(has)
        self.btn_add_range.setEnabled(has)
        self.btn_remove_range.setEnabled(has and bool(self._ranges))

    def _form_params(self) -> dict[str, Any]:
        return normalize_heatmap_params(
            {
                "kind": str(self.cmb_kind.currentData() or KIND_AUC_RATIO),
                "ranges": self._ranges,
            },
            self._nframes(),
        )

    def _form_label(self) -> str:
        return self.edit_label.text().strip() or "Untitled"

    # ----------------------------------------------------------------- ranges
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
        """Mirror the ranges onto the raster (display only, so clicks still select)."""
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

        self.list_ranges.blockSignals(True)
        self.list_ranges.clear()
        for i, (a, b) in enumerate(self._ranges):
            n = int(b) - int(a) + 1
            item = QListWidgetItem(f"{i}:  {int(a)}–{int(b)}   ({n} frames)")
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.list_ranges.addItem(item)
        self.list_ranges.blockSignals(False)
        self._rebuild_raster_range_items()
        self._update_status_counts()
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
        self._preview_map = None
        self._rebuild_ranges_ui()

    def _on_add_range(self) -> None:
        a = int(self.spin_start.value())
        b = int(self.spin_end.value())
        if b < a:
            a, b = b, a
        self._ranges.append([a, b])
        self._preview_map = None
        self._rebuild_ranges_ui()

    def _on_remove_range(self) -> None:
        items = self.list_ranges.selectedItems()
        if not items:
            if self._ranges:
                self._ranges.pop()
        else:
            drop = {int(i.data(Qt.ItemDataRole.UserRole)) for i in items}
            self._ranges = [r for i, r in enumerate(self._ranges) if i not in drop]
        self._preview_map = None
        self._rebuild_ranges_ui()

    def _update_status_counts(self) -> None:
        doc = self._doc()
        if doc is None:
            return
        n = self._nframes()
        if not self._ranges:
            self.plot_trace.setTitle(f"Trace — {format_ranges(self._ranges)}")
            return
        mask = led_shutter_nan_mask(doc, n)
        w_in, w_out = split_frame_weights(self._ranges, mask)
        self.plot_trace.setTitle(
            f"Trace — inside {format_ranges(self._ranges)} "
            f"(span {w_in.sum():.0f} frames, outside {w_out.sum():.0f})"
        )

    # ----------------------------------------------------------------- raster
    def _raster_rows(self) -> list[dict[str, Any]]:
        doc = self._doc()
        if doc is None:
            return []
        rows = rois_for_raster(doc["rois"], self.main._overlay_filter())
        return apply_raster_sort(rows, active_sort_run(doc))

    def refresh_raster(self) -> None:
        doc = self._doc()
        if doc is None:
            self._raster_row_ids = []
            self.curve_trace.setData([], [])
            self._refresh_map_overlay()
            return
        n = self._nframes()
        rows = self._raster_rows()
        self._raster_row_ids = [int(r["roi_id"]) for r in rows]
        field = raster_trace_field(doc)
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
        if not self._batch and rows:
            try:
                highlight_row = self._raster_row_ids.index(int(self.main.active_roi_id))
            except ValueError:
                highlight_row = None
            if highlight_row is not None:
                highlight_lut = selected_row_lut(self.main.chk_raster_revert.isChecked())
        rgb = colorize_raster(
            matrix, lut, highlight_row=highlight_row, highlight_lut=highlight_lut
        )
        self.raster_view.setImage(rgb, autoLevels=False, levels=(0, 255))
        self.lbl_raster_info.setText(f"{len(rows)} row(s) — {field}")
        self._refresh_trace(rows, matrix, field)
        self._refresh_ann_spans()
        self._rebuild_raster_range_items()
        self._refresh_map_overlay()

    def _refresh_trace(
        self, rows: list[dict[str, Any]], matrix: np.ndarray, field: str
    ) -> None:
        n = matrix.shape[1]
        xs = np.arange(n)
        if self._batch:
            if rows:
                valid = np.isfinite(matrix)
                counts = valid.sum(axis=0)
                sums = np.nansum(np.where(valid, matrix, 0.0), axis=0)
                y = np.divide(
                    sums,
                    counts,
                    out=np.full(counts.shape, np.nan, dtype=np.float64),
                    where=counts > 0,
                )
            else:
                y = np.full(n, np.nan, dtype=np.float64)
        else:
            y = np.full(n, np.nan, dtype=np.float64)
            by_id = {int(r["roi_id"]): r for r in rows}
            row = by_id.get(int(self.main.active_roi_id))
            if row is not None and row.get(field) is not None:
                arr = np.asarray(row[field], dtype=np.float64)
                n_copy = min(arr.shape[0], n)
                y[:n_copy] = arr[:n_copy]
        self.curve_trace.setData(xs, y)
        self._update_status_counts()

    def _clear_ann_spans(self) -> None:
        for item in self._ann_spans:
            try:
                self.plot_trace.removeItem(item)
            except Exception:
                pass
        self._ann_spans.clear()

    def _refresh_ann_spans(self) -> None:
        self._clear_ann_spans()
        doc = self._doc()
        if doc is None:
            return
        for ann in ensure_annotations(doc):
            prop = str(ann["property"])
            color = PROPERTY_SPEC.get(prop, {}).get("color", "#888888")
            c = QtGui.QColor(color)
            c.setAlpha(50)
            region = pg.LinearRegionItem(
                values=(float(ann["start_frame"]), float(ann["end_frame"]) + 1.0),
                movable=False,
                brush=c,
                pen=pg.mkPen(color, width=1),
            )
            region.setZValue(-10)
            self.plot_trace.addItem(region)
            self._ann_spans.append(region)

    def _on_batch_slider(self, value: int) -> None:
        self._batch = int(value) == 1
        self.refresh_raster()

    def _on_raster_click(self, y: int, x: int) -> None:
        if not self._raster_row_ids:
            return
        row = int(y)
        if 0 <= row < len(self._raster_row_ids):
            self.main._select_roi(self._raster_row_ids[row])

    def on_active_roi_changed(self) -> None:
        if not self._batch:
            self.refresh_raster()

    # -------------------------------------------------------------------- map
    def _clear_map(self) -> None:
        self.map_img.clear()
        self.plot_map.setTitle("")
        self._refresh_map_overlay()

    def _refresh_map_overlay(self) -> None:
        doc = self._doc()
        if doc is None or not self.chk_outlines.isChecked():
            self.map_outlines.clear()
            return
        Ly = int(doc["meta"]["Ly"])
        Lx = int(doc["meta"]["Lx"])
        rgba = np.zeros((Ly, Lx, 4), dtype=np.uint8)
        active = int(self.main.active_roi_id)
        for row in iter_visible_rois(doc["rois"], self.main._overlay_filter()):
            ys, xs = thick_outline_mask(
                Ly, Lx, row["roi"]["ypix"], row["roi"]["xpix"], thickness=1
            )
            if not ys.size:
                continue
            highlight = not self._batch and int(row["roi_id"]) == active
            color = OUTLINE_ACTIVE_RGB if highlight else OUTLINE_RGB
            rgba[ys, xs, 0] = color[0]
            rgba[ys, xs, 1] = color[1]
            rgba[ys, xs, 2] = color[2]
            rgba[ys, xs, 3] = 255
        self.map_outlines.setImage(rgba, autoLevels=False)
        self.map_outlines.setRect(QtCore.QRectF(-0.5, -0.5, float(Lx), float(Ly)))

    def _redraw_map(self) -> None:
        arr = self._preview_map
        if arr is None:
            doc = self._doc()
            if doc is not None and self._editing_id is not None:
                hm = get_heatmap(doc, self._editing_id)
                if hm is not None and hm.get("map") is not None:
                    arr = np.asarray(hm["map"], dtype=np.float64)
        if arr is not None:
            self._show_map(np.asarray(arr, dtype=np.float64))

    def _show_map(self, arr: np.ndarray) -> None:
        img = np.asarray(arr, dtype=np.float64)
        finite = img[np.isfinite(img)]
        if finite.size:
            lo, hi = np.percentile(finite, [1, 99])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(finite.min()), float(finite.max())
            if hi <= lo:
                hi = lo + 1.0
        else:
            lo, hi = 0.0, 1.0
        self.map_img.setLookupTable(make_lut(self.cmb_map_lut.currentText()))
        self.map_img.setImage(img, autoLevels=False, levels=(float(lo), float(hi)))
        ly, lx = img.shape
        self.map_img.setRect(QtCore.QRectF(-0.5, -0.5, float(lx), float(ly)))
        self.plot_map.setTitle(f"{ly}×{lx} — levels {lo:.3g} … {hi:.3g}")
        self._refresh_map_overlay()

    def _on_cancel_compute(self) -> None:
        self._cancel = True

    def _on_compute(self) -> None:
        doc = self._doc()
        if doc is None or self.main.suite2p_dir is None:
            return
        params = self._form_params()
        if not params["ranges"]:
            QMessageBox.warning(self, "HeatMap", "Set at least one range first.")
            return
        mask = led_shutter_nan_mask(doc, self._nframes())
        self._cancel = False
        self._computing = True
        self._update_buttons()
        self.btn_cancel.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)

        def progress(stage: str, fraction: float) -> None:
            self.progress.setValue(int(round(100.0 * fraction)))
            self.lbl_status.setText(stage)
            QtWidgets.QApplication.processEvents()

        arr: np.ndarray | None
        try:
            arr = compute_heatmap_map(
                self.main.suite2p_dir,
                params,
                mask,
                progress=progress,
                should_cancel=lambda: self._cancel,
            )
        except HeatmapCancelled:
            self.lbl_status.setText("Compute cancelled")
            arr = None
        except Exception as exc:
            QMessageBox.warning(self, "HeatMap failed", str(exc))
            arr = None
        finally:
            self._computing = False
            self.btn_cancel.setEnabled(False)
            self.progress.setVisible(False)
            self._update_buttons()
        if arr is None:
            return
        self._preview_map = arr
        self._show_map(arr)
        self.lbl_status.setText("Preview ready (unsaved)")

    def _stored_map(self, heatmap_id: str | None) -> np.ndarray | None:
        doc = self._doc()
        if doc is None or heatmap_id is None:
            return None
        hm = get_heatmap(doc, heatmap_id)
        if hm is None or hm.get("map") is None:
            return None
        return np.asarray(hm["map"])

    def _on_save(self) -> None:
        doc = self._doc()
        if doc is None:
            return
        arr = self._preview_map
        if arr is None:
            arr = self._stored_map(self._editing_id)
        if arr is None:
            QMessageBox.information(self, "Save", "Compute a heatmap before saving.")
            return
        params = self._form_params()
        label = self._form_label()
        if self._editing_id is None:
            hm = make_heatmap(doc, label=label, params=params, heatmap_map=arr)
            ensure_heatmaps(doc).append(hm)
            self._editing_id = str(hm["id"])
        else:
            hm = get_heatmap(doc, self._editing_id)
            if hm is None:
                QMessageBox.warning(self, "Save", "That heatmap is no longer in the pickle.")
                return
            apply_heatmap_result(
                hm,
                label=label,
                params=params,
                heatmap_map=arr,
                nframes=self._nframes(),
            )
        self.main.dirty = True
        self.main._heatmaps_changed()
        self.reload_list(load_form=True)
        self.lbl_status.setText(f"Saved {self._editing_id}")
        self.main.statusBar().showMessage(f"Saved heatmap {self._editing_id}")

    def _on_save_as(self) -> None:
        doc = self._doc()
        if doc is None:
            return
        default = self._form_label()
        if self._editing_id is not None:
            default = f"{default} copy"
        text, ok = QInputDialog.getText(
            self, "Save as", "Label for the new heatmap:", text=default
        )
        if not ok:
            return
        arr = self._preview_map
        if arr is None:
            arr = self._stored_map(self._editing_id)
        if arr is None:
            QMessageBox.information(self, "Save as", "Compute a heatmap before saving.")
            return
        hm = make_heatmap(
            doc,
            label=str(text).strip() or "Untitled",
            params=self._form_params(),
            heatmap_map=arr,
            heatmap_id=next_heatmap_id(doc),
        )
        ensure_heatmaps(doc).append(hm)
        self._editing_id = str(hm["id"])
        self.main.dirty = True
        self.main._heatmaps_changed()
        self.reload_list(load_form=True)
        self.main.statusBar().showMessage(f"Saved heatmap {hm['id']}")

    def _on_delete(self) -> None:
        doc = self._doc()
        if doc is None or self._editing_id is None:
            return
        hid = self._editing_id
        reply = QMessageBox.question(
            self,
            "Delete heatmap",
            f"Delete heatmap {hid} from the pickle?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        doc["heatmaps"] = [h for h in ensure_heatmaps(doc) if str(h.get("id")) != hid]
        self._editing_id = None
        self._preview_map = None
        self.main.dirty = True
        self.main._heatmaps_changed()
        self.reload_list(load_form=True)
        self.main.statusBar().showMessage(f"Deleted heatmap {hid}")
