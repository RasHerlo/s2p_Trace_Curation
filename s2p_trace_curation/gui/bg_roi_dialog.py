"""Pick BG-ROI traces and threshold their raw sum into BG-motion ranges."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from s2p_trace_curation.bg_rois import (
    BG_FIELD_F,
    BG_FIELD_SM,
    BG_FIELD_SM_BC,
    BG_TRACE_FIELDS,
    BG_TRACE_LABELS,
    apply_processed_to_bg,
    bg_roi_label,
    bg_trace,
    ensure_bg_rois,
    sum_bg_traces,
)
from s2p_trace_curation.pmt_noise import (
    ranges_above_threshold,
    suggest_threshold,
)

Qt = QtCore.Qt
QCheckBox = QtWidgets.QCheckBox
QComboBox = QtWidgets.QComboBox
QDialog = QtWidgets.QDialog
QDialogButtonBox = QtWidgets.QDialogButtonBox
QDoubleSpinBox = QtWidgets.QDoubleSpinBox
QHBoxLayout = QtWidgets.QHBoxLayout
QLabel = QtWidgets.QLabel
QPushButton = QtWidgets.QPushButton
QScrollArea = QtWidgets.QScrollArea
QSpinBox = QtWidgets.QSpinBox
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget

TRACE_COLORS = (
    "#2ecc71",
    "#3498db",
    "#9b59b6",
    "#e67e22",
    "#1abc9c",
    "#e74c3c",
    "#f1c40f",
    "#34495e",
)
SUM_PEN = "#1a5276"
THRESHOLD_PEN = "#16a085"
BAND_BRUSH = (22, 160, 133, 55)


class BgRoiThresholdDialog(QDialog):
    """Overlay selected BG traces; threshold the sum of their raw F."""

    def __init__(
        self,
        doc: dict[str, Any],
        *,
        parent: QtWidgets.QWidget | None = None,
        fs: float | None = None,
        seconds: bool = False,
    ) -> None:
        super().__init__(parent)
        self._doc = doc
        self._rows = list(ensure_bg_rois(doc))
        self._nframes = int(doc["meta"]["nframes"])
        self._fs = fs if (fs and np.isfinite(fs) and fs > 0) else None
        self._seconds = bool(seconds and self._fs)
        self._ranges: list[list[int]] = []
        self._updating = False
        self._checks: dict[int, QCheckBox] = {}
        self._trace_curves: list[pg.PlotDataItem] = []

        for row in self._rows:
            if row.get(BG_FIELD_SM) is None or row.get(BG_FIELD_SM_BC) is None:
                apply_processed_to_bg(doc, row)

        self.setWindowTitle("BG-motion from BG ROIs")
        self.setModal(True)
        self.setMinimumSize(860, 640)

        layout = QVBoxLayout(self)

        pick = QHBoxLayout()
        pick.addWidget(QLabel("BG ROIs"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMaximumHeight(44)
        host = QWidget()
        host_row = QHBoxLayout(host)
        host_row.setContentsMargins(0, 0, 0, 0)
        for row in self._rows:
            bid = int(row["bg_id"])
            chk = QCheckBox(bg_roi_label(row))
            chk.setChecked(True)
            chk.toggled.connect(self._on_selection_changed)
            host_row.addWidget(chk)
            self._checks[bid] = chk
        host_row.addStretch(1)
        scroll.setWidget(host)
        pick.addWidget(scroll, stretch=1)
        pick.addWidget(QLabel("Show"))
        self.cmb_field = QComboBox()
        for field in BG_TRACE_FIELDS:
            self.cmb_field.addItem(BG_TRACE_LABELS[field], field)
        self.cmb_field.setToolTip(
            "Overlay field for the upper plot. Thresholding always uses "
            "the sum of the raw BG-ROI traces."
        )
        self.cmb_field.currentIndexChanged.connect(self._redraw_traces)
        pick.addWidget(self.cmb_field)
        layout.addLayout(pick)

        self.plot_traces = pg.PlotWidget(title="Selected BG-ROI traces")
        self.plot_traces.showGrid(x=True, y=True, alpha=0.2)
        self.plot_traces.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot_traces.addLegend(offset=(10, 10))
        layout.addWidget(self.plot_traces, stretch=1)

        self.plot_sum = pg.PlotWidget(title="Sum of raw BG-ROI traces")
        self.plot_sum.showGrid(x=True, y=True, alpha=0.2)
        self.plot_sum.setLabel("left", "sum F")
        self.plot_sum.getAxis("bottom").enableAutoSIPrefix(False)
        self.band = pg.PlotDataItem(
            pen=None, brush=QtGui.QColor(*BAND_BRUSH), fillLevel=0.0
        )
        self.band.setZValue(-10)
        self.plot_sum.addItem(self.band)
        self.curve_sum = self.plot_sum.plot(pen=pg.mkPen(SUM_PEN, width=1.3))
        self.line = pg.InfiniteLine(
            angle=0,
            movable=True,
            pen=pg.mkPen(THRESHOLD_PEN, width=2),
        )
        self.line.setZValue(20)
        self.plot_sum.addItem(self.line)
        self.line.sigPositionChanged.connect(self._on_line_moved)
        layout.addWidget(self.plot_sum, stretch=1)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Threshold"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setDecimals(4)
        self.spin_threshold.setRange(-1e12, 1e12)
        self.spin_threshold.setKeyboardTracking(False)
        self.spin_threshold.setToolTip(
            "Frames where the raw sum is strictly above this value become BG-motion."
        )
        controls.addWidget(self.spin_threshold)

        self.btn_auto = QPushButton("Auto")
        self.btn_auto.setToolTip("Suggest median + 5 robust sigma of the sum")
        self.btn_auto.clicked.connect(self._on_auto)
        controls.addWidget(self.btn_auto)

        controls.addSpacing(16)
        controls.addWidget(QLabel("Merge gaps \u2264"))
        self.spin_gap = QSpinBox()
        self.spin_gap.setRange(0, 10**6)
        self.spin_gap.setToolTip(
            "Join two above-threshold runs separated by at most this many "
            "frames. 0 keeps every run separate."
        )
        controls.addWidget(self.spin_gap)

        controls.addSpacing(16)
        controls.addWidget(QLabel("Min length"))
        self.spin_min = QSpinBox()
        self.spin_min.setRange(1, 10**6)
        self.spin_min.setValue(1)
        self.spin_min.setToolTip(
            "Drop runs shorter than this many frames (applied after merging)."
        )
        controls.addWidget(self.spin_min)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setWordWrap(True)
        layout.addWidget(self.lbl_summary)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.spin_threshold.valueChanged.connect(self._on_spin_changed)
        self.spin_gap.valueChanged.connect(lambda _=0: self._recompute())
        self.spin_min.valueChanged.connect(lambda _=0: self._recompute())

        self._apply_x_units()
        self._redraw_traces()
        self._refresh_sum_curve()
        self._set_threshold(suggest_threshold(self._sum_values()))

    # ------------------------------------------------------------------ public
    def ranges(self) -> list[list[int]]:
        return [[int(a), int(b)] for a, b in self._ranges]

    def threshold(self) -> float:
        return float(self.spin_threshold.value())

    def selected_ids(self) -> list[int]:
        return [
            bid
            for bid, chk in self._checks.items()
            if chk.isChecked()
        ]

    # ----------------------------------------------------------------- private
    def _selected_rows(self) -> list[dict[str, Any]]:
        want = set(self.selected_ids())
        return [r for r in self._rows if int(r["bg_id"]) in want]

    def _display_field(self) -> str:
        data = self.cmb_field.currentData()
        return str(data) if data else BG_FIELD_F

    def _sum_values(self) -> np.ndarray:
        return sum_bg_traces(self._selected_rows(), BG_FIELD_F, self._nframes)

    def _apply_x_units(self) -> None:
        scale = 1.0 / self._fs if self._seconds and self._fs else 1.0
        label = "time (s)" if self._seconds else "frame"
        for plot in (self.plot_traces, self.plot_sum):
            plot.getAxis("bottom").setScale(scale)
            plot.setLabel("bottom", label)

    def _on_selection_changed(self) -> None:
        self._redraw_traces()
        self._refresh_sum_curve()
        self._recompute()

    def _redraw_traces(self) -> None:
        for curve in self._trace_curves:
            try:
                self.plot_traces.removeItem(curve)
            except Exception:
                pass
        self._trace_curves.clear()
        field = self._display_field()
        xs = np.arange(self._nframes, dtype=np.float64)
        for i, row in enumerate(self._selected_rows()):
            color = TRACE_COLORS[i % len(TRACE_COLORS)]
            curve = self.plot_traces.plot(
                xs,
                bg_trace(row, field, self._nframes),
                pen=pg.mkPen(color, width=1.2),
                name=bg_roi_label(row),
                connect="finite",
            )
            self._trace_curves.append(curve)
        n = len(self._selected_rows())
        self.plot_traces.setTitle(
            f"Selected BG-ROI traces — {BG_TRACE_LABELS[field]}  ({n})"
        )

    def _refresh_sum_curve(self) -> None:
        xs = np.arange(self._nframes, dtype=np.float64)
        values = self._sum_values()
        self.curve_sum.setData(xs, values, connect="finite")
        finite = values[np.isfinite(values)]
        if finite.size:
            lo = float(finite.min())
            hi = float(finite.max())
            span = hi - lo if hi > lo else max(abs(hi), 1.0)
            self.spin_threshold.setRange(lo - 10.0 * span, hi + 10.0 * span)
            self.spin_threshold.setSingleStep(span / 100.0 if span else 0.01)
            self.spin_threshold.setDecimals(self._decimals_for(span))
        n = len(self._selected_rows())
        self.plot_sum.setTitle(f"Sum of raw BG-ROI traces  ({n})")

    @staticmethod
    def _decimals_for(span: float) -> int:
        if span <= 0 or not np.isfinite(span):
            return 4
        return int(min(max(4, 3 - int(np.floor(np.log10(span)))), 9))

    def _set_threshold(self, value: float) -> None:
        self._updating = True
        try:
            self.spin_threshold.setValue(float(value))
            self.line.setPos(float(value))
        finally:
            self._updating = False
        self._recompute()

    def _on_spin_changed(self, value: float) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self.line.setPos(float(value))
        finally:
            self._updating = False
        self._recompute()

    def _on_line_moved(self) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            self.spin_threshold.setValue(float(self.line.value()))
        finally:
            self._updating = False
        self._recompute()

    def _on_auto(self) -> None:
        self._set_threshold(suggest_threshold(self._sum_values()))

    def _recompute(self) -> None:
        values = self._sum_values()
        if not self._selected_rows() or not np.any(np.isfinite(values)):
            self._ranges = []
            self.band.setData([], [])
            self.lbl_summary.setText("Select at least one BG ROI")
            ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
            if ok is not None:
                ok.setEnabled(False)
            return
        self._ranges = ranges_above_threshold(
            values,
            self.threshold(),
            merge_gap=int(self.spin_gap.value()),
            min_frames=int(self.spin_min.value()),
        )
        self._refresh_band(values)
        n_frames = sum(b - a + 1 for a, b in self._ranges)
        n_ranges = len(self._ranges)
        total = int(np.isfinite(values).sum())
        pct = (100.0 * n_frames / total) if total else 0.0
        if n_ranges:
            text = (
                f"{n_frames} frame(s) above threshold ({pct:.2f}% of "
                f"{total} scored frames) \u2192 {n_ranges} range(s)"
            )
            if n_ranges > 50:
                text += "  \u2014 raise the threshold or merge gaps to get fewer"
        else:
            text = "No frames above this threshold \u2014 lower it to select motion"
        self.lbl_summary.setText(text)
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setEnabled(n_ranges > 0)

    def _refresh_band(self, values: np.ndarray) -> None:
        n = values.shape[0]
        if not self._ranges or n == 0:
            self.band.setData([], [])
            return
        finite = values[np.isfinite(values)]
        lo = float(finite.min()) if finite.size else 0.0
        hi = float(finite.max()) if finite.size else 1.0
        if hi <= lo:
            hi = lo + 1.0
        mask = np.zeros(n, dtype=bool)
        for a, b in self._ranges:
            mask[a : b + 1] = True
        edges = np.arange(n + 1, dtype=np.float64) - 0.5
        ys = np.where(mask, hi, lo)
        self.band.setData(edges, ys, stepMode="center", fillLevel=lo)
