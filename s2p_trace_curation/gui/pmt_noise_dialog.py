"""Threshold picker for PMT-noise: removed_rms vs frame with a live range preview."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from s2p_trace_curation.pmt_noise import (
    PerFrameRms,
    ranges_above_threshold,
    suggest_threshold,
)

Qt = QtCore.Qt
QDialog = QtWidgets.QDialog
QDialogButtonBox = QtWidgets.QDialogButtonBox
QDoubleSpinBox = QtWidgets.QDoubleSpinBox
QHBoxLayout = QtWidgets.QHBoxLayout
QLabel = QtWidgets.QLabel
QPushButton = QtWidgets.QPushButton
QSpinBox = QtWidgets.QSpinBox
QVBoxLayout = QtWidgets.QVBoxLayout

RMS_PEN = "#2c7fb8"
THRESHOLD_PEN = "#d35400"
BAND_BRUSH = (211, 84, 0, 55)


class PmtNoiseThresholdDialog(QDialog):
    """Pick a removed_rms threshold; every frame above it becomes PMT-noise."""

    def __init__(
        self,
        rms: PerFrameRms,
        *,
        parent: QtWidgets.QWidget | None = None,
        fs: float | None = None,
        seconds: bool = False,
    ) -> None:
        super().__init__(parent)
        self._rms = rms
        self._values = np.asarray(rms.values, dtype=np.float64)
        self._fs = fs if (fs and np.isfinite(fs) and fs > 0) else None
        self._seconds = bool(seconds and self._fs)
        self._ranges: list[list[int]] = []
        self._updating = False

        self.setWindowTitle("PMT-noise from removed_rms")
        self.setModal(True)
        self.setMinimumSize(760, 480)

        layout = QVBoxLayout(self)

        self.lbl_source = QLabel(rms.summary())
        self.lbl_source.setWordWrap(True)
        layout.addWidget(self.lbl_source)

        self.plot = pg.PlotWidget(title="removed_rms")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("left", "removed_rms")
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)
        # One filled step curve for the whole preview: a LinearRegionItem per
        # detected range would be thousands of items at a low threshold.
        self.band = pg.PlotDataItem(
            pen=None, brush=QtGui.QColor(*BAND_BRUSH), fillLevel=0.0
        )
        self.band.setZValue(-10)
        self.plot.addItem(self.band)
        self.curve = self.plot.plot(pen=pg.mkPen(RMS_PEN, width=1.2))
        layout.addWidget(self.plot, stretch=1)

        finite = self._rms.finite
        lo = float(finite.min()) if finite.size else 0.0
        hi = float(finite.max()) if finite.size else 1.0
        span = hi - lo if hi > lo else max(abs(hi), 1.0)

        self.line = pg.InfiniteLine(
            angle=0,
            movable=True,
            pen=pg.mkPen(THRESHOLD_PEN, width=2),
        )
        self.line.setZValue(20)
        self.plot.addItem(self.line)
        self.line.sigPositionChanged.connect(self._on_line_moved)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Threshold"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setDecimals(self._decimals_for(span))
        self.spin_threshold.setRange(lo - 10.0 * span, hi + 10.0 * span)
        self.spin_threshold.setSingleStep(span / 100.0 if span else 0.01)
        self.spin_threshold.setKeyboardTracking(False)
        self.spin_threshold.setToolTip(
            "Frames with removed_rms strictly above this value become PMT-noise."
        )
        controls.addWidget(self.spin_threshold)

        self.btn_auto = QPushButton("Auto")
        self.btn_auto.setToolTip("Suggest median + 5 robust sigma")
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

        self._draw_curve()
        self._set_threshold(suggest_threshold(self._values))
        self._apply_x_units()

    # ------------------------------------------------------------------ public
    def ranges(self) -> list[list[int]]:
        """Inclusive [start, end] frame pairs above the chosen threshold."""
        return [[int(a), int(b)] for a, b in self._ranges]

    def threshold(self) -> float:
        return float(self.spin_threshold.value())

    def merge_gap(self) -> int:
        return int(self.spin_gap.value())

    def min_frames(self) -> int:
        return int(self.spin_min.value())

    # ----------------------------------------------------------------- private
    @staticmethod
    def _decimals_for(span: float) -> int:
        if span <= 0 or not np.isfinite(span):
            return 4
        # Enough resolution to step through the data range meaningfully.
        return int(min(max(4, 3 - int(np.floor(np.log10(span)))), 9))

    def _draw_curve(self) -> None:
        xs = np.arange(self._values.shape[0], dtype=np.float64)
        self.curve.setData(xs, self._values, connect="finite")

    def _apply_x_units(self) -> None:
        scale = 1.0 / self._fs if self._seconds and self._fs else 1.0
        self.plot.getAxis("bottom").setScale(scale)
        self.plot.setLabel("bottom", "time (s)" if self._seconds else "frame")

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
        self._set_threshold(suggest_threshold(self._values))

    def _recompute(self) -> None:
        self._ranges = ranges_above_threshold(
            self._values,
            self.threshold(),
            merge_gap=self.merge_gap(),
            min_frames=self.min_frames(),
        )
        self._refresh_band()
        n_frames = sum(b - a + 1 for a, b in self._ranges)
        n_ranges = len(self._ranges)
        total = int(np.isfinite(self._values).sum())
        pct = (100.0 * n_frames / total) if total else 0.0
        if n_ranges:
            text = (
                f"{n_frames} frame(s) above threshold ({pct:.2f}% of "
                f"{total} scored frames) \u2192 {n_ranges} range(s)"
            )
            if n_ranges > 50:
                text += "  \u2014 raise the threshold or merge gaps to get fewer"
        else:
            text = "No frames above this threshold \u2014 lower it to select noise"
        self.lbl_summary.setText(text)
        ok = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok is not None:
            ok.setEnabled(n_ranges > 0)

    def _refresh_band(self) -> None:
        n = self._values.shape[0]
        if not self._ranges or n == 0:
            self.band.setData([], [])
            return
        finite = self._rms.finite
        lo = float(finite.min()) if finite.size else 0.0
        hi = float(finite.max()) if finite.size else 1.0
        if hi <= lo:
            hi = lo + 1.0
        mask = np.zeros(n, dtype=bool)
        for a, b in self._ranges:
            mask[a : b + 1] = True
        # stepMode="center" wants one more x edge than y sample, which puts the
        # band boundaries exactly on frame edges.
        edges = np.arange(n + 1, dtype=np.float64) - 0.5
        ys = np.where(mask, hi, lo)
        self.band.setData(edges, ys, stepMode="center", fillLevel=lo)
