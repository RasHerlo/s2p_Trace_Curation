"""Non-modal Trace Processing window: SG smooth + bleach correction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from s2p_trace_curation.raster import led_shutter_nan_mask
from s2p_trace_curation.trace_processing import (
    DEFAULT_SG_POLY,
    DEFAULT_SG_WINDOW,
    TAU_INDEPENDENT,
    TAU_SHARED,
    TRACE_FIELD_SM,
    TRACE_FIELD_SM_BC,
    bleach_fit_curve,
    ensure_trace_processing,
    estimate_shared_taus,
    field_has_any,
    rebuild_all_tc_norm_sm,
    rebuild_all_tc_norm_sm_bc,
    tc_norm_sm_bc_is_stale,
    tc_norm_sm_is_stale,
)

Qt = QtCore.Qt
QCheckBox = QtWidgets.QCheckBox
QComboBox = QtWidgets.QComboBox
QDialog = QtWidgets.QDialog
QDoubleSpinBox = QtWidgets.QDoubleSpinBox
QFormLayout = QtWidgets.QFormLayout
QGroupBox = QtWidgets.QGroupBox
QHBoxLayout = QtWidgets.QHBoxLayout
QLabel = QtWidgets.QLabel
QMessageBox = QtWidgets.QMessageBox
QPushButton = QtWidgets.QPushButton
QSpinBox = QtWidgets.QSpinBox
QVBoxLayout = QtWidgets.QVBoxLayout


class TraceProcessingWindow(QDialog):
    def __init__(self, main: Any) -> None:
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Trace Processing")
        self.setModal(False)
        self.setMinimumSize(640, 520)
        self._loading = False

        layout = QVBoxLayout(self)

        sg = QGroupBox("Savitzky–Golay (all ROIs)")
        sg_form = QFormLayout(sg)
        self.spin_sg_window = QSpinBox()
        self.spin_sg_window.setRange(3, 999)
        self.spin_sg_window.setSingleStep(2)
        self.spin_sg_window.setValue(DEFAULT_SG_WINDOW)
        self.spin_sg_window.setToolTip("Odd window length (even values are rounded up)")
        self.spin_sg_poly = QSpinBox()
        self.spin_sg_poly.setRange(1, 7)
        self.spin_sg_poly.setValue(DEFAULT_SG_POLY)
        self.btn_rebuild_sm = QPushButton("Rebuild tc_norm_sm")
        self.btn_rebuild_sm.setToolTip(
            "SG on trace_comp (shutter frames excised), then min–max → tc_norm_sm"
        )
        self.btn_rebuild_sm.clicked.connect(self._on_rebuild_sm)
        self.lbl_sm_status = QLabel("")
        self.lbl_sm_status.setWordWrap(True)
        sg_form.addRow("Window", self.spin_sg_window)
        sg_form.addRow("Poly order", self.spin_sg_poly)
        sg_form.addRow(self.btn_rebuild_sm)
        sg_form.addRow(self.lbl_sm_status)
        layout.addWidget(sg)

        bc = QGroupBox("Bleach correction")
        bc_form = QFormLayout(bc)
        self.chk_bleach_on = QCheckBox("Apply bleach correction")
        self.chk_bleach_on.setToolTip(
            "Off: conservative (constant) fit, so tc_norm_sm_bc matches tc_norm_sm after min–max"
        )
        self.chk_bleach_on.toggled.connect(self._on_bleach_toggled)
        self.cmb_tau_mode = QComboBox()
        self.cmb_tau_mode.addItem("Shared τ (all ROIs)", TAU_SHARED)
        self.cmb_tau_mode.addItem("Independent fits", TAU_INDEPENDENT)
        self.spin_tau1 = QDoubleSpinBox()
        self.spin_tau2 = QDoubleSpinBox()
        for spin in (self.spin_tau1, self.spin_tau2):
            spin.setRange(0.1, 1e7)
            spin.setDecimals(2)
            spin.setValue(100.0)
        self.btn_estimate_tau = QPushButton("Estimate shared τ")
        self.btn_estimate_tau.setToolTip("Fit τ on the mean of selected (iscell) tc_norm_sm")
        self.btn_estimate_tau.clicked.connect(self._on_estimate_tau)
        self.btn_rebuild_bc = QPushButton("Rebuild tc_norm_sm_bc")
        self.btn_rebuild_bc.clicked.connect(self._on_rebuild_bc)
        self.lbl_bc_status = QLabel("")
        self.lbl_bc_status.setWordWrap(True)
        bc_form.addRow(self.chk_bleach_on)
        bc_form.addRow("τ mode", self.cmb_tau_mode)
        bc_form.addRow("τ1 (frames)", self.spin_tau1)
        bc_form.addRow("τ2 (frames)", self.spin_tau2)
        bc_form.addRow(self.btn_estimate_tau)
        bc_form.addRow(self.btn_rebuild_bc)
        bc_form.addRow(self.lbl_bc_status)
        layout.addWidget(bc)

        self.plot = pg.PlotWidget(title="Current ROI")
        self.plot.showGrid(x=True, y=False, alpha=0.2)
        self.plot.addLegend(offset=(10, 10))
        self.curve_sm = self.plot.plot(pen=pg.mkPen("#1f77b4", width=1.2), name="tc_norm_sm")
        self.curve_fit = self.plot.plot(
            pen=pg.mkPen("#ff7f0e", width=1.2, style=Qt.PenStyle.DashLine), name="fit"
        )
        self.curve_bc = self.plot.plot(pen=pg.mkPen("#2ca02c", width=1.5), name="tc_norm_sm_bc")
        layout.addWidget(self.plot, stretch=1)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        close_row.addWidget(btn_close)
        layout.addLayout(close_row)

        self.cmb_tau_mode.currentIndexChanged.connect(self._sync_tau_enabled)
        self.refresh_from_doc()

    def _doc(self) -> dict[str, Any] | None:
        return self.main.doc

    def refresh_from_doc(self) -> None:
        doc = self._doc()
        self._loading = True
        try:
            if doc is None:
                return
            tp = ensure_trace_processing(doc)
            self.spin_sg_window.setValue(int(tp.get("sg_window") or DEFAULT_SG_WINDOW))
            self.spin_sg_poly.setValue(int(tp.get("sg_poly") or DEFAULT_SG_POLY))
            self.chk_bleach_on.setChecked(bool(tp.get("bleach_enabled")))
            mode = str(tp.get("tau_mode") or TAU_SHARED)
            idx = self.cmb_tau_mode.findData(mode)
            self.cmb_tau_mode.setCurrentIndex(idx if idx >= 0 else 0)
            if tp.get("shared_tau1") is not None:
                self.spin_tau1.setValue(float(tp["shared_tau1"]))
            if tp.get("shared_tau2") is not None:
                self.spin_tau2.setValue(float(tp["shared_tau2"]))
        finally:
            self._loading = False
        self._sync_tau_enabled()
        self._update_status()
        self._refresh_preview()

    def _sync_tau_enabled(self) -> None:
        on = self.chk_bleach_on.isChecked()
        shared = str(self.cmb_tau_mode.currentData() or TAU_SHARED) == TAU_SHARED
        self.cmb_tau_mode.setEnabled(on)
        self.spin_tau1.setEnabled(on and shared)
        self.spin_tau2.setEnabled(on and shared)
        self.btn_estimate_tau.setEnabled(on and shared)

    def _on_bleach_toggled(self, _checked: bool) -> None:
        if self._loading:
            return
        self._sync_tau_enabled()

    def _write_params_to_doc(self) -> dict[str, Any] | None:
        doc = self._doc()
        if doc is None:
            return None
        tp = ensure_trace_processing(doc)
        tp["sg_window"] = int(self.spin_sg_window.value())
        tp["sg_poly"] = int(self.spin_sg_poly.value())
        tp["bleach_enabled"] = bool(self.chk_bleach_on.isChecked())
        tp["tau_mode"] = str(self.cmb_tau_mode.currentData() or TAU_SHARED)
        if tp["bleach_enabled"] and tp["tau_mode"] == TAU_SHARED:
            tp["shared_tau1"] = float(self.spin_tau1.value())
            tp["shared_tau2"] = float(self.spin_tau2.value())
        elif not tp["bleach_enabled"]:
            tp["shared_tau1"] = None
            tp["shared_tau2"] = None
        return doc

    def _update_status(self) -> None:
        doc = self._doc()
        if doc is None:
            self.lbl_sm_status.setText("No session loaded")
            self.lbl_bc_status.setText("")
            return
        if not field_has_any(doc, TRACE_FIELD_SM):
            self.lbl_sm_status.setText("tc_norm_sm not built yet")
        elif tc_norm_sm_is_stale(doc):
            self.lbl_sm_status.setText("tc_norm_sm is stale — rebuild")
        else:
            self.lbl_sm_status.setText("tc_norm_sm is up to date")
        if not field_has_any(doc, TRACE_FIELD_SM_BC):
            self.lbl_bc_status.setText("tc_norm_sm_bc not built yet")
        elif tc_norm_sm_bc_is_stale(doc):
            self.lbl_bc_status.setText("tc_norm_sm_bc is stale — rebuild")
        else:
            self.lbl_bc_status.setText("tc_norm_sm_bc is up to date")

    def _refresh_preview(self) -> None:
        doc = self._doc()
        if doc is None:
            self.curve_sm.setData([], [])
            self.curve_fit.setData([], [])
            self.curve_bc.setData([], [])
            return
        nframes = int(doc["meta"]["nframes"])
        xs = np.arange(nframes)
        try:
            row = self.main._row()
        except Exception:
            row = None
        if row is None:
            return
        sm = row.get(TRACE_FIELD_SM)
        bc = row.get(TRACE_FIELD_SM_BC)
        mask = led_shutter_nan_mask(doc, nframes)
        fit = bleach_fit_curve(row, mask, nframes)
        self.curve_sm.setData(xs, np.asarray(sm, dtype=np.float64) if sm is not None else xs * np.nan)
        self.curve_fit.setData(xs, fit if fit is not None else xs * np.nan)
        self.curve_bc.setData(xs, np.asarray(bc, dtype=np.float64) if bc is not None else xs * np.nan)
        rid = int(row.get("roi_id", self.main.active_roi_id))
        cons = bool((row.get("bleach") or {}).get("conservative", False))
        extra = " — conservative" if cons else ""
        self.plot.setTitle(f"ROI {rid}{extra}")

    def _on_rebuild_sm(self) -> None:
        doc = self._write_params_to_doc()
        if doc is None:
            return
        rebuild_all_tc_norm_sm(doc)
        self.main.dirty = True
        self.main._refresh_analysis_stale_ui()
        self.main._refresh_traces(autoscale=False)
        if self.main._raster_mode:
            self.main._refresh_raster()
        self._update_status()
        self._refresh_preview()
        self.main.statusBar().showMessage("Rebuilt tc_norm_sm")

    def _on_estimate_tau(self) -> None:
        doc = self._doc()
        if doc is None:
            return
        if not field_has_any(doc, TRACE_FIELD_SM) or tc_norm_sm_is_stale(doc):
            reply = QMessageBox.question(
                self,
                "tc_norm_sm missing",
                "Shared τ needs tc_norm_sm. Rebuild with current SG defaults?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self._on_rebuild_sm()
            doc = self._doc()
            if doc is None:
                return
        nframes = int(doc["meta"]["nframes"])
        taus = estimate_shared_taus(doc, led_shutter_nan_mask(doc, nframes))
        if taus is None:
            QMessageBox.warning(self, "Estimate τ", "Could not fit shared τ; using conservative bleach.")
            return
        self.spin_tau1.setValue(float(taus[0]))
        self.spin_tau2.setValue(float(taus[1]))
        tp = ensure_trace_processing(doc)
        tp["shared_tau1"] = float(taus[0])
        tp["shared_tau2"] = float(taus[1])
        self.main.dirty = True

    def _on_rebuild_bc(self) -> None:
        doc = self._write_params_to_doc()
        if doc is None:
            return
        if not field_has_any(doc, TRACE_FIELD_SM) or tc_norm_sm_is_stale(doc):
            reply = QMessageBox.question(
                self,
                "Need smoothed traces",
                "tc_norm_sm is missing or stale. Rebuild with current SG parameters first?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self._on_rebuild_sm()
            doc = self._write_params_to_doc()
            if doc is None:
                return
        if not self.chk_bleach_on.isChecked():
            QMessageBox.information(
                self,
                "Conservative bleach",
                "Bleach is off: tc_norm_sm_bc will use a constant fit "
                "(no decay removed; traces stay equivalent to tc_norm_sm after min–max).",
            )
        rebuild_all_tc_norm_sm_bc(doc)
        tp = ensure_trace_processing(doc)
        if tp.get("shared_tau1") is not None:
            self.spin_tau1.setValue(float(tp["shared_tau1"]))
        if tp.get("shared_tau2") is not None:
            self.spin_tau2.setValue(float(tp["shared_tau2"]))
        self.main.dirty = True
        self.main._refresh_analysis_stale_ui()
        self.main._refresh_traces(autoscale=False)
        if self.main._raster_mode:
            self.main._refresh_raster()
        self._update_status()
        self._refresh_preview()
        self.main.statusBar().showMessage("Rebuilt tc_norm_sm_bc")
