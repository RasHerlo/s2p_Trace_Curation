"""Non-modal Analysis Tools window: named runs, preview, save / rebuild."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from scipy.cluster.hierarchy import dendrogram

from s2p_trace_curation.analyses import (
    KIND_HAC,
    KIND_LABELS,
    KIND_PLACEHOLDER,
    PICKLE_SORT_ID,
    apply_run_result,
    compute_run,
    dropdown_label,
    ensure_analyses,
    get_analysis,
    kind_label,
    make_analysis_run,
    next_analysis_id,
    raster_sort_id,
    set_raster_sort,
)
from s2p_trace_curation.gui.colormaps import make_lut
from s2p_trace_curation.trace_processing import (
    TRACE_FIELD_LABELS,
    TRACE_FIELD_NORM,
    TRACE_FIELD_SM_BC,
    TRACE_FIELDS,
)
from s2p_trace_curation.hac import (
    DEFAULT_HAC_PARAMS,
    HAC_LINKAGE_LABELS,
    HAC_METRIC_LABELS,
    LINKAGE_AVERAGE,
    LINKAGE_WARD,
    METRIC_EUCLIDEAN,
    METRIC_RUZICKA,
    normalize_hac_params,
    run_hac,
)

Qt = QtCore.Qt
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
QPushButton = QtWidgets.QPushButton
QSplitter = QtWidgets.QSplitter
QVBoxLayout = QtWidgets.QVBoxLayout
QStackedWidget = QtWidgets.QStackedWidget
QWidget = QtWidgets.QWidget


class AnalysisToolsWindow(QDialog):
    def __init__(self, main: Any) -> None:
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Analysis Tools")
        self.setModal(False)
        self.setMinimumSize(900, 560)
        self._editing_id: str | None = None
        self._preview: dict[str, Any] | None = None
        self._loading = False

        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Saved runs"))
        self.list_runs = QListWidget()
        self.list_runs.itemSelectionChanged.connect(self._on_run_selected)
        left_layout.addWidget(self.list_runs, stretch=1)

        btn_row = QHBoxLayout()
        self.btn_new = QPushButton("New")
        self.btn_new.setToolTip("Start a draft; Save writes it to the pickle")
        self.btn_new.clicked.connect(self._on_new)
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self._on_delete)
        self.btn_rebuild = QPushButton("Rebuild")
        self.btn_rebuild.setToolTip(
            "Recompute this saved run from current iscell / chosen traces using saved params"
        )
        self.btn_rebuild.clicked.connect(self._on_rebuild)
        btn_row.addWidget(self.btn_new)
        btn_row.addWidget(self.btn_delete)
        btn_row.addWidget(self.btn_rebuild)
        left_layout.addLayout(btn_row)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self.edit_label = QLineEdit()
        self.edit_label.setPlaceholderText("Untitled")
        self.cmb_kind = QComboBox()
        for kind, text in KIND_LABELS.items():
            self.cmb_kind.addItem(text, kind)
        self.cmb_trace_field = QComboBox()
        for field in TRACE_FIELDS:
            self.cmb_trace_field.addItem(TRACE_FIELD_LABELS[field], field)
        self.cmb_trace_field.setToolTip(
            "Trace stored in the pickle used for this run (HAC and fingerprint)"
        )
        self.lbl_status = QLabel("New draft — Run, then Save")
        self.lbl_status.setWordWrap(True)
        form.addRow("Label", self.edit_label)
        form.addRow("Kind", self.cmb_kind)
        form.addRow("Trace", self.cmb_trace_field)
        form.addRow("Status", self.lbl_status)
        right_layout.addLayout(form)

        method = QGroupBox("Method")
        method_layout = QVBoxLayout(method)
        self.method_stack = QStackedWidget()

        page_ph = QWidget()
        ph_layout = QVBoxLayout(page_ph)
        ph_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_method = QLabel(
            "Placeholder: Run uses pickle order of currently selected "
            "(iscell=True) traces. Switch Kind to HAC for clustering."
        )
        self.lbl_method.setWordWrap(True)
        ph_layout.addWidget(self.lbl_method)
        ph_layout.addStretch(1)
        self.method_stack.addWidget(page_ph)

        page_hac = QWidget()
        hac_layout = QVBoxLayout(page_hac)
        hac_layout.setContentsMargins(0, 0, 0, 0)
        hac_form = QFormLayout()
        self.cmb_hac_metric = QComboBox()
        for key, text in HAC_METRIC_LABELS.items():
            self.cmb_hac_metric.addItem(text, key)
        self.cmb_hac_metric.setToolTip(
            "Ružička: shared positive mass on the chosen trace field.\n"
            "Euclidean: L2 of finite frames (required for Ward)."
        )
        self.cmb_hac_linkage = QComboBox()
        for key, text in HAC_LINKAGE_LABELS.items():
            self.cmb_hac_linkage.addItem(text, key)
        self.cmb_hac_linkage.setToolTip(
            "Average (UPGMA) with Ružička.\nWard only with Euclidean."
        )
        hac_form.addRow("Metric", self.cmb_hac_metric)
        hac_form.addRow("Linkage", self.cmb_hac_linkage)
        hac_layout.addLayout(hac_form)

        viz_split = QSplitter(Qt.Orientation.Horizontal)
        self.plot_hac_tree = pg.PlotWidget(title="Dendrogram")
        self.plot_hac_tree.setMinimumHeight(180)
        self.plot_hac_tree.invertY(True)
        self.plot_hac_tree.setLabel("bottom", "distance")
        self.plot_hac_tree.setLabel("left", "leaf")
        self.plot_hac_tree.showGrid(x=True, y=False, alpha=0.2)
        viz_split.addWidget(self.plot_hac_tree)

        self.plot_hac_mat = pg.PlotWidget(title="Pairwise (leaf order)")
        self.plot_hac_mat.setMinimumHeight(180)
        self.plot_hac_mat.invertY(True)
        self.hac_img = pg.ImageItem()
        self.plot_hac_mat.addItem(self.hac_img)
        viz_split.addWidget(self.plot_hac_mat)
        viz_split.setStretchFactor(0, 1)
        viz_split.setStretchFactor(1, 1)
        hac_layout.addWidget(viz_split, stretch=1)
        self.method_stack.addWidget(page_hac)

        method_layout.addWidget(self.method_stack)
        right_layout.addWidget(method, stretch=1)

        self.cmb_hac_metric.currentIndexChanged.connect(self._on_hac_metric_changed)
        self.cmb_hac_linkage.currentIndexChanged.connect(self._on_hac_linkage_changed)
        self.cmb_kind.currentIndexChanged.connect(self._on_kind_changed)
        self.cmb_trace_field.currentIndexChanged.connect(self._on_trace_field_changed)

        self.lbl_preview = QLabel("No preview yet.")
        self.lbl_preview.setWordWrap(True)
        right_layout.addWidget(self.lbl_preview)

        action_row = QHBoxLayout()
        self.btn_run = QPushButton("Run")
        self.btn_run.setToolTip("Recompute in this window (does not write the pickle)")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_save = QPushButton("Save")
        self.btn_save.setToolTip("Overwrite this run in trc_curation.pkl")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save_as = QPushButton("Save as")
        self.btn_save_as.setToolTip("Write a new named run")
        self.btn_save_as.clicked.connect(self._on_save_as)
        action_row.addWidget(self.btn_run)
        action_row.addWidget(self.btn_save)
        action_row.addWidget(self.btn_save_as)
        action_row.addStretch(1)
        right_layout.addLayout(action_row)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        close_row.addWidget(btn_close)
        layout.addLayout(close_row)

        self.refresh_from_doc()

    def _doc(self) -> dict[str, Any] | None:
        return self.main.doc

    def refresh_from_doc(self) -> None:
        self._preview = None
        self.reload_run_list(load_form=True)

    def reload_run_list(self, *, load_form: bool = False) -> None:
        doc = self._doc()
        self._loading = True
        self.list_runs.clear()
        if doc is None:
            self._loading = False
            if load_form:
                self._set_draft_form()
            return
        ensure_analyses(doc)
        keep = self._editing_id
        for run in ensure_analyses(doc):
            item = QListWidgetItem(dropdown_label(run))
            item.setData(Qt.ItemDataRole.UserRole, str(run["id"]))
            self.list_runs.addItem(item)
        if keep:
            self._select_id(keep)
        elif load_form or self.list_runs.count() == 0:
            self._set_draft_form()
        self._loading = False
        if load_form and keep:
            self._on_run_selected()
        elif keep and doc is not None:
            run = get_analysis(doc, keep)
            if run is not None:
                n = len(run.get("roi_ids") or [])
                stale = "stale" if run.get("stale") else "valid"
                self.lbl_status.setText(
                    f"{run['id']} — {stale} — {n} selected ROI(s) at last run"
                )
        self._update_buttons()

    def _select_id(self, analysis_id: str) -> None:
        for i in range(self.list_runs.count()):
            item = self.list_runs.item(i)
            if item is not None and str(item.data(Qt.ItemDataRole.UserRole)) == analysis_id:
                self.list_runs.setCurrentRow(i)
                return
        self.list_runs.clearSelection()
        self._set_draft_form()

    def _set_draft_form(self) -> None:
        self._editing_id = None
        self._preview = None
        self.edit_label.setText("")
        idx = self.cmb_kind.findData(KIND_HAC)
        if idx >= 0:
            self.cmb_kind.setCurrentIndex(idx)
        self._apply_hac_params(DEFAULT_HAC_PARAMS)
        idx_f = self.cmb_trace_field.findData(TRACE_FIELD_SM_BC)
        if idx_f >= 0:
            self.cmb_trace_field.setCurrentIndex(idx_f)
        self._sync_kind_page()
        self._clear_hac_plots()
        self.lbl_status.setText("New draft — Run, then Save")
        self.lbl_preview.setText("No preview yet.")
        self._update_buttons()

    def _on_new(self) -> None:
        self._loading = True
        self.list_runs.clearSelection()
        self._loading = False
        self._set_draft_form()

    def _on_run_selected(self) -> None:
        if self._loading:
            return
        items = self.list_runs.selectedItems()
        if not items:
            return
        doc = self._doc()
        if doc is None:
            return
        aid = str(items[0].data(Qt.ItemDataRole.UserRole))
        run = get_analysis(doc, aid)
        if run is None:
            return
        self._editing_id = aid
        self._preview = None
        self.edit_label.setText(str(run.get("label") or ""))
        kind = str(run.get("kind") or KIND_PLACEHOLDER)
        idx = self.cmb_kind.findData(kind)
        if idx < 0:
            self.cmb_kind.addItem(kind_label(kind), kind)
            idx = self.cmb_kind.findData(kind)
        self.cmb_kind.setCurrentIndex(max(0, idx))
        self._apply_hac_params(run.get("params") or {})
        field = str((run.get("params") or {}).get("trace_field") or TRACE_FIELD_NORM)
        fi = self.cmb_trace_field.findData(field)
        if fi >= 0:
            self.cmb_trace_field.setCurrentIndex(fi)
        self._sync_kind_page()
        self._clear_hac_plots()
        n = len(run.get("roi_ids") or [])
        stale = "stale" if run.get("stale") else "valid"
        self.lbl_status.setText(
            f"{run['id']} — {stale} — {n} selected ROI(s) at last run"
        )
        self.lbl_preview.setText(
            f"Saved order: {len(run.get('order') or [])} ROI(s). Run to preview changes."
        )
        self._update_buttons()

    def _form_kind(self) -> str:
        kind = self.cmb_kind.currentData()
        return str(kind) if kind else KIND_PLACEHOLDER

    def _form_params(self) -> dict[str, Any]:
        field = str(self.cmb_trace_field.currentData() or TRACE_FIELD_SM_BC)
        if self._form_kind() != KIND_HAC:
            return {"trace_field": field}
        return {
            "metric": str(self.cmb_hac_metric.currentData() or METRIC_RUZICKA),
            "linkage": str(self.cmb_hac_linkage.currentData() or LINKAGE_AVERAGE),
            "trace_field": field,
        }

    def _on_trace_field_changed(self) -> None:
        if self._loading:
            return
        self._preview = None

    def _on_kind_changed(self) -> None:
        if self._loading:
            return
        self._preview = None
        self._sync_kind_page()
        if self._form_kind() == KIND_HAC:
            self._apply_hac_linkage_constraint()

    def _sync_kind_page(self) -> None:
        hac = self._form_kind() == KIND_HAC
        self.method_stack.setCurrentIndex(1 if hac else 0)

    def _apply_hac_params(self, params: dict[str, Any]) -> None:
        try:
            norm = normalize_hac_params(params) if params else dict(DEFAULT_HAC_PARAMS)
        except ValueError:
            norm = dict(DEFAULT_HAC_PARAMS)
        was = self._loading
        self._loading = True
        mi = self.cmb_hac_metric.findData(norm["metric"])
        if mi >= 0:
            self.cmb_hac_metric.setCurrentIndex(mi)
        li = self.cmb_hac_linkage.findData(norm["linkage"])
        if li >= 0:
            self.cmb_hac_linkage.setCurrentIndex(li)
        field = str(norm.get("trace_field") or TRACE_FIELD_SM_BC)
        fi = self.cmb_trace_field.findData(field)
        if fi >= 0:
            self.cmb_trace_field.setCurrentIndex(fi)
        self._loading = was
        self._apply_hac_linkage_constraint()

    def _apply_hac_linkage_constraint(self) -> None:
        metric = str(self.cmb_hac_metric.currentData() or METRIC_RUZICKA)
        model = self.cmb_hac_linkage.model()
        ward_idx = self.cmb_hac_linkage.findData(LINKAGE_WARD)
        if model is not None and ward_idx >= 0:
            item_fn = getattr(model, "item", None)
            item = item_fn(ward_idx) if callable(item_fn) else None
            if item is not None:
                item.setEnabled(metric == METRIC_EUCLIDEAN)
        if metric == METRIC_RUZICKA:
            avg_idx = self.cmb_hac_linkage.findData(LINKAGE_AVERAGE)
            if (
                avg_idx >= 0
                and str(self.cmb_hac_linkage.currentData()) == LINKAGE_WARD
            ):
                was = self._loading
                self._loading = True
                self.cmb_hac_linkage.setCurrentIndex(avg_idx)
                self._loading = was

    def _on_hac_metric_changed(self) -> None:
        if self._loading:
            return
        self._preview = None
        self._apply_hac_linkage_constraint()

    def _on_hac_linkage_changed(self) -> None:
        if self._loading:
            return
        self._preview = None
        if str(self.cmb_hac_linkage.currentData()) == LINKAGE_WARD:
            eu_idx = self.cmb_hac_metric.findData(METRIC_EUCLIDEAN)
            if eu_idx >= 0 and str(self.cmb_hac_metric.currentData()) != METRIC_EUCLIDEAN:
                self._loading = True
                self.cmb_hac_metric.setCurrentIndex(eu_idx)
                self._loading = False
                self._apply_hac_linkage_constraint()

    def _clear_hac_plots(self) -> None:
        self.plot_hac_tree.clear()
        self.hac_img.setImage(np.zeros((1, 1), dtype=np.float64), autoLevels=False)
        self.plot_hac_mat.setTitle("Pairwise (leaf order)")
        self.plot_hac_tree.setTitle("Dendrogram")

    def _render_hac(self, result: dict[str, Any]) -> None:
        Z = result.get("Z")
        matrix = result.get("matrix")
        if Z is None or matrix is None:
            self._clear_hac_plots()
            return
        matrix = np.asarray(matrix, dtype=np.float64)
        n = int(matrix.shape[0])
        params = result.get("params") or {}
        kind = str(result.get("display_kind") or "similarity")
        title = (
            "Ružička similarity (leaf order)"
            if kind == "similarity"
            else "Euclidean distance (leaf order)"
        )
        self.plot_hac_mat.setTitle(title)
        self.plot_hac_tree.setTitle(
            f"Dendrogram — {HAC_METRIC_LABELS.get(params.get('metric', ''), params.get('metric'))} / "
            f"{HAC_LINKAGE_LABELS.get(params.get('linkage', ''), params.get('linkage'))}"
        )

        lut = make_lut("magma")
        self.hac_img.setLookupTable(lut)
        self.hac_img.setImage(matrix, autoLevels=False)
        if kind == "similarity":
            self.hac_img.setLevels((0.0, 1.0))
        else:
            lo = float(np.nanmin(matrix)) if matrix.size else 0.0
            hi = float(np.nanmax(matrix)) if matrix.size else 1.0
            if hi <= lo:
                hi = lo + 1.0
            self.hac_img.setLevels((lo, hi))
        self.hac_img.setRect(QtCore.QRectF(-0.5, -0.5, float(n), float(n)))
        self.plot_hac_mat.setXRange(-0.5, n - 0.5, padding=0)
        self.plot_hac_mat.setYRange(-0.5, n - 0.5, padding=0)

        self.plot_hac_tree.clear()
        tree = dendrogram(Z, no_plot=True)
        pen = pg.mkPen("#dddddd", width=1)
        ymax = 0.0
        for xs, ys in zip(tree["dcoord"], tree["icoord"]):
            y = (np.asarray(ys, dtype=np.float64) - 5.0) / 10.0
            x = np.asarray(xs, dtype=np.float64)
            ymax = max(ymax, float(np.max(x)))
            self.plot_hac_tree.plot(x, y, pen=pen, connect="all")
        self.plot_hac_tree.setYRange(-0.5, n - 0.5, padding=0)
        self.plot_hac_tree.setXRange(0.0, max(ymax * 1.05, 1e-6), padding=0.02)


    def _form_label(self) -> str:
        return self.edit_label.text().strip() or "Untitled"

    def _update_buttons(self) -> None:
        has_doc = self._doc() is not None
        saved = self._editing_id is not None
        self.btn_run.setEnabled(has_doc)
        self.btn_save.setEnabled(has_doc)
        self.btn_save_as.setEnabled(has_doc)
        self.btn_delete.setEnabled(has_doc and saved)
        self.btn_rebuild.setEnabled(has_doc and saved)
        self.btn_new.setEnabled(has_doc)

    def _ensure_inputs(self) -> bool:
        field = str(self.cmb_trace_field.currentData() or TRACE_FIELD_SM_BC)
        return bool(self.main._ensure_analysis_inputs(field))

    def _on_run(self) -> None:
        doc = self._doc()
        if doc is None or not self._ensure_inputs():
            return
        kind = self._form_kind()
        params = self._form_params()
        try:
            result = self._execute(kind, params)
        except ValueError as exc:
            QMessageBox.warning(self, "Run failed", str(exc))
            return
        self._preview = result
        self.lbl_preview.setText(
            f"Preview: {len(result['order'])} selected ROI(s) — {kind_label(kind)}. Save to keep."
        )
        self.lbl_status.setText("Preview ready (unsaved)")

    def _result_from_preview_or_run(self) -> dict[str, Any] | None:
        if self._preview is not None:
            return self._preview
        doc = self._doc()
        if doc is None or not self._ensure_inputs():
            return None
        kind = self._form_kind()
        params = self._form_params()
        try:
            result = self._execute(kind, params)
        except ValueError as exc:
            QMessageBox.warning(self, "Run failed", str(exc))
            return None
        self._preview = result
        return self._preview

    def _execute(self, kind: str, params: dict[str, Any]) -> dict[str, Any]:
        doc = self._doc()
        if doc is None:
            raise ValueError("No session loaded")
        if kind == KIND_HAC:
            result = run_hac(doc, params)
            self._render_hac(result)
            return result
        roi_ids, order = compute_run(doc, kind, params)
        self._clear_hac_plots()
        return {
            "kind": kind,
            "params": deepcopy(params),
            "roi_ids": roi_ids,
            "order": order,
        }

    def _on_save(self) -> None:
        doc = self._doc()
        if doc is None:
            return
        result = self._result_from_preview_or_run()
        if result is None:
            return
        label = self._form_label()
        if self._editing_id is None:
            run = make_analysis_run(
                doc,
                label=label,
                kind=result["kind"],
                params=result["params"],
                roi_ids=result["roi_ids"],
                order=result["order"],
            )
            ensure_analyses(doc).append(run)
            self._editing_id = str(run["id"])
        else:
            run = get_analysis(doc, self._editing_id)
            if run is None:
                QMessageBox.warning(self, "Save", "That run is no longer in the pickle.")
                return
            apply_run_result(
                run,
                doc,
                label=label,
                kind=result["kind"],
                params=result["params"],
                roi_ids=result["roi_ids"],
                order=result["order"],
            )
        self.main._analysis_runs_changed()
        self.refresh_from_doc()
        self.lbl_status.setText(f"Saved {self._editing_id}")
        self.main.statusBar().showMessage(f"Saved analysis {self._editing_id}")

    def _on_save_as(self) -> None:
        doc = self._doc()
        if doc is None:
            return
        default = self._form_label()
        if self._editing_id is not None:
            default = f"{default} copy"
        text, ok = QInputDialog.getText(
            self, "Save as", "Label for the new run:", text=default
        )
        if not ok:
            return
        result = self._result_from_preview_or_run()
        if result is None:
            return
        run = make_analysis_run(
            doc,
            label=str(text).strip() or "Untitled",
            kind=result["kind"],
            params=result["params"],
            roi_ids=result["roi_ids"],
            order=result["order"],
            analysis_id=next_analysis_id(doc),
        )
        ensure_analyses(doc).append(run)
        self._editing_id = str(run["id"])
        self.main._analysis_runs_changed()
        self.refresh_from_doc()
        self.main.statusBar().showMessage(f"Saved analysis {run['id']}")

    def _on_rebuild(self) -> None:
        doc = self._doc()
        if doc is None or self._editing_id is None:
            return
        run = get_analysis(doc, self._editing_id)
        if run is None:
            return
        if not self._ensure_inputs():
            return
        kind = str(run.get("kind") or KIND_PLACEHOLDER)
        params = deepcopy(run.get("params") or {})
        try:
            result = self._execute(kind, params)
        except ValueError as exc:
            QMessageBox.warning(self, "Rebuild failed", str(exc))
            return
        apply_run_result(
            run,
            doc,
            roi_ids=result["roi_ids"],
            order=result["order"],
        )
        self._preview = result
        idx = self.cmb_kind.findData(kind)
        if idx >= 0:
            self.cmb_kind.setCurrentIndex(idx)
        self.edit_label.setText(str(run.get("label") or ""))
        self.main._analysis_runs_changed()
        self.refresh_from_doc()
        self.main.statusBar().showMessage(f"Rebuilt analysis {run['id']}")

    def _on_delete(self) -> None:
        doc = self._doc()
        if doc is None or self._editing_id is None:
            return
        aid = self._editing_id
        reply = QMessageBox.question(
            self,
            "Delete run",
            f"Delete analysis {aid} from the pickle?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        runs = ensure_analyses(doc)
        doc["analyses"] = [r for r in runs if str(r.get("id")) != aid]
        if raster_sort_id(doc) == aid:
            set_raster_sort(doc, PICKLE_SORT_ID)
        self._editing_id = None
        self._preview = None
        self.main._analysis_runs_changed()
        self.refresh_from_doc()
        self.main.statusBar().showMessage(f"Deleted analysis {aid}")
