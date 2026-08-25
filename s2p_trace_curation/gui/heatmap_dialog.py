"""Non-modal Edit HeatMaps window: named maps from data.bin."""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

from s2p_trace_curation.annotations import PROPERTY_SPEC, ensure_annotations
from s2p_trace_curation.heatmaps import (
    HeatmapCancelled,
    apply_heatmap_result,
    compute_heatmap_map,
    default_heatmap_params,
    ensure_heatmaps,
    format_start_frames,
    get_heatmap,
    heatmap_combo_label,
    make_heatmap,
    next_heatmap_id,
    normalize_heatmap_params,
    parse_start_frames,
)
from s2p_trace_curation.raster import led_shutter_nan_mask
from s2p_trace_curation.trace_processing import (
    TRACE_FIELD_NORM,
    TRACE_FIELD_SM,
    TRACE_FIELD_SM_BC,
    raster_trace_field,
)

Qt = QtCore.Qt
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
QSpinBox = QtWidgets.QSpinBox
QSplitter = QtWidgets.QSplitter
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget


class HeatmapEditorWindow(QDialog):
    def __init__(self, main: Any) -> None:
        super().__init__(main)
        self.main = main
        self.setWindowTitle("Edit HeatMaps")
        self.setModal(False)
        self.setMinimumSize(980, 620)
        self._editing_id: str | None = None
        self._preview_map: np.ndarray | None = None
        self._loading = False
        self._cancel = False
        self._computing = False
        self._ann_spans: list[pg.LinearRegionItem] = []
        self._start_lines: list[pg.InfiniteLine] = []

        layout = QVBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Saved heatmaps"))
        self.list_maps = QListWidget()
        self.list_maps.itemSelectionChanged.connect(self._on_selected)
        left_layout.addWidget(self.list_maps, stretch=1)
        btn_row = QHBoxLayout()
        self.btn_new = QPushButton("New")
        self.btn_new.clicked.connect(self._on_new)
        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self._on_delete)
        btn_row.addWidget(self.btn_new)
        btn_row.addWidget(self.btn_delete)
        left_layout.addLayout(btn_row)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        form = QFormLayout()
        self.edit_label = QLineEdit()
        self.edit_label.setPlaceholderText("Untitled")
        self.spin_sg_window = QSpinBox()
        self.spin_sg_window.setRange(3, 999)
        self.spin_sg_window.setSingleStep(2)
        self.spin_sg_poly = QSpinBox()
        self.spin_sg_poly.setRange(1, 7)
        self.edit_starts = QLineEdit()
        self.edit_starts.setPlaceholderText("e.g. 200, 400, 600")
        self.spin_extension = QSpinBox()
        self.spin_extension.setRange(1, 100000)
        self.spin_area_l = QSpinBox()
        self.spin_area_r = QSpinBox()
        for s in (self.spin_area_l, self.spin_area_r):
            s.setRange(1, 100000)
        self.lbl_status = QLabel("New draft — Compute, then Save")
        self.lbl_status.setWordWrap(True)
        form.addRow("Label", self.edit_label)
        form.addRow("SG window", self.spin_sg_window)
        form.addRow("SG poly", self.spin_sg_poly)
        form.addRow("Starts (frames)", self.edit_starts)
        form.addRow("Extension", self.spin_extension)
        form.addRow("Area L", self.spin_area_l)
        form.addRow("Area R", self.spin_area_r)
        form.addRow("Status", self.lbl_status)
        right_layout.addLayout(form)

        self.edit_starts.editingFinished.connect(self._refresh_guide)
        self.spin_extension.valueChanged.connect(self._refresh_guide)

        viz = QSplitter(Qt.Orientation.Vertical)
        self.plot_map = pg.PlotWidget(title="Heatmap preview")
        self.plot_map.invertY(True)
        self.map_img = pg.ImageItem()
        self.plot_map.addItem(self.map_img)
        viz.addWidget(self.plot_map)

        self.plot_guide = pg.PlotWidget(title="Guide trace + annotations")
        self.plot_guide.setLabel("bottom", "frame")
        self.plot_guide.showGrid(x=True, y=False, alpha=0.2)
        self.curve_guide = self.plot_guide.plot(pen=pg.mkPen("#2ca02c", width=1.2))
        viz.addWidget(self.plot_guide)
        viz.setStretchFactor(0, 2)
        viz.setStretchFactor(1, 1)
        right_layout.addWidget(viz, stretch=1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        right_layout.addWidget(self.progress)

        action_row = QHBoxLayout()
        self.btn_compute = QPushButton("Compute / Update")
        self.btn_compute.setToolTip("Read data.bin and build the area map (shutter frames skipped)")
        self.btn_compute.clicked.connect(self._on_compute)
        self.btn_cancel = QPushButton("Cancel compute")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._on_cancel_compute)
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_save_as = QPushButton("Save as")
        self.btn_save_as.clicked.connect(self._on_save_as)
        action_row.addWidget(self.btn_compute)
        action_row.addWidget(self.btn_cancel)
        action_row.addWidget(self.btn_save)
        action_row.addWidget(self.btn_save_as)
        action_row.addStretch(1)
        right_layout.addLayout(action_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.close)
        close_row.addWidget(btn_close)
        layout.addLayout(close_row)

        hint = QLabel(
            "Starts / extension / Area L–R are independent of annotations. "
            "Annotation spans are shown on the guide so you can avoid shutter and align to events. "
            "Area L/R are relative frames on the aligned segment (1-based)."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.refresh_from_doc()

    def _doc(self) -> dict[str, Any] | None:
        return self.main.doc

    def refresh_from_doc(self) -> None:
        self._preview_map = None
        self.reload_list(load_form=True)
        self._refresh_guide()

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

    def _default_params(self) -> dict[str, Any]:
        doc = self._doc()
        nframes = int(doc["meta"]["nframes"]) if doc is not None else 1000
        return default_heatmap_params(nframes)

    def _set_draft_form(self) -> None:
        self._editing_id = None
        self._preview_map = None
        self.edit_label.setText("")
        params = self._default_params()
        self._apply_params(params)
        self._clear_map()
        self.lbl_status.setText("New draft — Compute, then Save")
        self._update_buttons()
        self._refresh_guide()

    def _apply_params(self, params: dict[str, Any]) -> None:
        p = normalize_heatmap_params(params)
        self.spin_sg_window.setValue(int(p["sg_window"]))
        self.spin_sg_poly.setValue(int(p["sg_poly"]))
        self.edit_starts.setText(format_start_frames(list(p["starts"])))
        self.spin_extension.setValue(int(p["extension"]))
        self.spin_area_l.setValue(int(p["area_left"]))
        self.spin_area_r.setValue(int(p["area_right"]))

    def _form_params(self) -> dict[str, Any]:
        doc = self._doc()
        nframes = int(doc["meta"]["nframes"]) if doc is not None else 10**9
        p = self._default_params()
        p["sg_window"] = int(self.spin_sg_window.value())
        p["sg_poly"] = int(self.spin_sg_poly.value())
        p["starts"] = parse_start_frames(self.edit_starts.text(), nframes)
        p["extension"] = int(self.spin_extension.value())
        p["area_left"] = int(self.spin_area_l.value())
        p["area_right"] = int(self.spin_area_r.value())
        return normalize_heatmap_params(p)

    def _form_label(self) -> str:
        return self.edit_label.text().strip() or "Untitled"

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
        self._apply_params(hm.get("params") or {})
        stored = hm.get("map")
        if stored is not None:
            self._show_map(np.asarray(stored, dtype=np.float64))
        else:
            self._clear_map()
        self.lbl_status.setText(f"{hm['id']} — saved. Compute to replace the map.")
        self._update_buttons()
        self._refresh_guide()

    def _update_buttons(self) -> None:
        has = self._doc() is not None and not self._computing
        self.btn_new.setEnabled(has)
        self.btn_delete.setEnabled(has and self._editing_id is not None)
        self.btn_compute.setEnabled(has)
        self.btn_save.setEnabled(has)
        self.btn_save_as.setEnabled(has)

    def _clear_map(self) -> None:
        self.map_img.setImage(np.zeros((1, 1), dtype=np.float64), autoLevels=True)
        self.plot_map.setTitle("Heatmap preview")

    def _show_map(self, arr: np.ndarray) -> None:
        img = np.asarray(arr, dtype=np.float64)
        self.map_img.setImage(img, autoLevels=True)
        ly, lx = img.shape
        self.map_img.setRect(QtCore.QRectF(-0.5, -0.5, float(lx), float(ly)))
        self.plot_map.setTitle(f"Heatmap preview — {ly}×{lx}")

    def _guide_trace(self) -> np.ndarray | None:
        doc = self._doc()
        if doc is None or not doc.get("rois"):
            return None
        nframes = int(doc["meta"]["nframes"])
        field = raster_trace_field(doc)
        if not any(r.get(field) is not None for r in doc["rois"]):
            for cand in (TRACE_FIELD_SM_BC, TRACE_FIELD_SM, TRACE_FIELD_NORM):
                if any(r.get(cand) is not None for r in doc["rois"]):
                    field = cand
                    break
        acc = []
        for row in doc["rois"]:
            if not bool(row.get("iscell", True)):
                continue
            tr = row.get(field)
            if tr is None:
                continue
            acc.append(np.asarray(tr, dtype=np.float64))
        if not acc:
            return None
        stacked = np.vstack(acc)
        with np.errstate(all="ignore"):
            mean = np.nanmean(stacked, axis=0)
        out = np.full(nframes, np.nan, dtype=np.float64)
        n_copy = min(mean.shape[0], nframes)
        out[:n_copy] = mean[:n_copy]
        return out

    def _clear_guide_overlays(self) -> None:
        for item in self._ann_spans + self._start_lines:
            try:
                self.plot_guide.removeItem(item)
            except Exception:
                pass
        self._ann_spans.clear()
        self._start_lines.clear()

    def _refresh_guide(self) -> None:
        if self._loading:
            return
        doc = self._doc()
        self._clear_guide_overlays()
        y = self._guide_trace()
        nframes = int(doc["meta"]["nframes"]) if doc is not None else 1
        xs = np.arange(nframes)
        if y is None:
            self.curve_guide.setData([], [])
        else:
            self.curve_guide.setData(xs, y)
        if doc is None:
            return
        for ann in ensure_annotations(doc):
            prop = str(ann["property"])
            color = PROPERTY_SPEC.get(prop, {}).get("color", "#888888")
            s = float(ann["start_frame"])
            e = float(ann["end_frame"]) + 1.0
            c = QtGui.QColor(color)
            c.setAlpha(50)
            region = pg.LinearRegionItem(
                values=(s, e), movable=False, brush=c, pen=pg.mkPen(color, width=1)
            )
            region.setZValue(-10)
            self.plot_guide.addItem(region)
            self._ann_spans.append(region)
        params = self._form_params()
        ext = int(params["extension"])
        for start in params["starts"]:
            line_a = pg.InfiniteLine(
                pos=float(start),
                angle=90,
                movable=False,
                pen=pg.mkPen("#f1c40f", width=1.5),
            )
            line_b = pg.InfiniteLine(
                pos=float(start + ext),
                angle=90,
                movable=False,
                pen=pg.mkPen("#f1c40f", width=1, style=Qt.PenStyle.DotLine),
            )
            self.plot_guide.addItem(line_a)
            self.plot_guide.addItem(line_b)
            self._start_lines.extend([line_a, line_b])

    def _on_cancel_compute(self) -> None:
        self._cancel = True

    def _on_compute(self) -> None:
        doc = self._doc()
        if doc is None or self.main.suite2p_dir is None:
            return
        params = self._form_params()
        if not params["starts"]:
            QMessageBox.warning(self, "Heatmap", "Enter at least one start frame.")
            return
        nframes = int(doc["meta"]["nframes"])
        mask = led_shutter_nan_mask(doc, nframes)
        self._cancel = False
        self._computing = True
        self._update_buttons()
        self.btn_cancel.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setValue(0)
        QWidget.repaint(self.progress)

        def progress(stage: str, fraction: float) -> None:
            self.progress.setValue(int(round(100.0 * fraction)))
            self.lbl_status.setText(stage)
            QtWidgets.QApplication.processEvents()

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
            QMessageBox.warning(self, "Heatmap failed", str(exc))
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

    def _save_with_id(self, heatmap_id: str | None) -> None:
        doc = self._doc()
        if doc is None:
            return
        arr = self._preview_map
        if arr is None and heatmap_id is not None:
            hm = get_heatmap(doc, heatmap_id)
            if hm is not None and hm.get("map") is not None:
                arr = np.asarray(hm["map"])
        if arr is None:
            QMessageBox.information(self, "Save", "Compute a heatmap before saving.")
            return
        params = self._form_params()
        label = self._form_label()
        if heatmap_id is None:
            hm = make_heatmap(doc, label=label, params=params, heatmap_map=arr)
            ensure_heatmaps(doc).append(hm)
            self._editing_id = str(hm["id"])
        else:
            hm = get_heatmap(doc, heatmap_id)
            if hm is None:
                QMessageBox.warning(self, "Save", "That heatmap is no longer in the pickle.")
                return
            apply_heatmap_result(hm, label=label, params=params, heatmap_map=arr)
        self.main.dirty = True
        self.main._heatmaps_changed()
        self.reload_list(load_form=True)
        self.lbl_status.setText(f"Saved {self._editing_id}")
        self.main.statusBar().showMessage(f"Saved heatmap {self._editing_id}")

    def _on_save(self) -> None:
        self._save_with_id(self._editing_id)

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
        if arr is None and self._editing_id is not None:
            hm = get_heatmap(doc, self._editing_id)
            if hm is not None and hm.get("map") is not None:
                arr = np.asarray(hm["map"])
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
