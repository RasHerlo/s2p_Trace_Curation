"""Main curation GUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

Qt = QtCore.Qt
QTimer = QtCore.QTimer
QAction = QtGui.QAction
QKeySequence = QtGui.QKeySequence
QAbstractSpinBox = QtWidgets.QAbstractSpinBox
QApplication = QtWidgets.QApplication
QButtonGroup = QtWidgets.QButtonGroup
QCheckBox = QtWidgets.QCheckBox
QComboBox = QtWidgets.QComboBox
QDoubleSpinBox = QtWidgets.QDoubleSpinBox
QFileDialog = QtWidgets.QFileDialog
QFormLayout = QtWidgets.QFormLayout
QGroupBox = QtWidgets.QGroupBox
QHBoxLayout = QtWidgets.QHBoxLayout
QLabel = QtWidgets.QLabel
QLineEdit = QtWidgets.QLineEdit
QMainWindow = QtWidgets.QMainWindow
QMessageBox = QtWidgets.QMessageBox
QProgressBar = QtWidgets.QProgressBar
QPushButton = QtWidgets.QPushButton
QRadioButton = QtWidgets.QRadioButton
QShortcut = QtWidgets.QShortcut
QSpinBox = QtWidgets.QSpinBox
QSplitter = QtWidgets.QSplitter
QStatusBar = QtWidgets.QStatusBar
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget

from copy import deepcopy

from s2p_trace_curation.curation import (
    open_suite2p_session,
    reextract_after_mask_edit,
    reset_roi_from_suite2p,
    save_curation,
    set_compensation_x,
)
from s2p_trace_curation.gui.colormaps import LUT_NAMES, apply_lut, make_lut
from s2p_trace_curation.gui.overlays import (
    OverlayFilter,
    build_fov_overlay,
    compose_rgb_with_overlay,
    rois_at_pixel,
    thick_outline_mask,
    zoom_masks_rgba,
)
from s2p_trace_curation.mask_edit import (
    MODE_LABELS,
    ExtractCancelled,
    MaskEditMode,
    apply_brush,
)
from s2p_trace_curation.suite2p_io import BinaryStack, plane_dir, zoom_square_window
from s2p_trace_curation.user_settings import (
    last_open_start_dir,
    load_settings,
    save_settings,
)

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

FILTER_LABELS = {
    "noncell": "non-selected (iscell=0)",
    "cell": "selected (iscell=1)",
    "both": "both",
}

# Analysis cursor colors (dotted)
C_COLORS = ["#ff8c00", "#da70d6", "#7fffd4", "#ffa07a"]
C0_COLOR = "#ffff66"  # solid movie cursor


class ClickableImageView(pg.ImageView):
    """ImageView that reports left-clicks in image row/col coordinates."""

    def __init__(self, *args, on_click=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_click = on_click
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.getView().scene().sigMouseClicked.connect(self._scene_clicked)

    def _scene_clicked(self, event) -> None:
        if self._on_click is None or event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.scenePos()
        view = self.getView()
        if not view.sceneBoundingRect().contains(pos):
            return
        mouse = view.mapSceneToView(pos)
        self._on_click(int(mouse.y()), int(mouse.x()))


class PaintImageView(pg.ImageView):
    """ImageView with optional left-drag paint callback in image row/col coords."""

    def __init__(self, *args, on_paint=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_paint = on_paint
        self.paint_enabled = False
        self._painting = False
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.ui.graphicsView.viewport().installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if (
            not self.paint_enabled
            or self._on_paint is None
            or obj is not self.ui.graphicsView.viewport()
        ):
            return super().eventFilter(obj, event)

        et = event.type()
        if et == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                self._painting = True
                self._emit_paint(event.pos())
                return True
        elif et == QtCore.QEvent.Type.MouseMove:
            if self._painting and event.buttons() & Qt.MouseButton.LeftButton:
                self._emit_paint(event.pos())
                return True
        elif et == QtCore.QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton and self._painting:
                self._painting = False
                return True
        return super().eventFilter(obj, event)

    def _emit_paint(self, viewport_pos) -> None:
        assert self._on_paint is not None
        scene_pos = self.ui.graphicsView.mapToScene(viewport_pos)
        view = self.getView()
        if not view.sceneBoundingRect().contains(scene_pos):
            return
        mouse = view.mapSceneToView(scene_pos)
        self._on_paint(int(round(mouse.y())), int(round(mouse.x())))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("s2p Trace Curation")
        self.resize(1920, 1100)

        self.suite2p_dir: Path | None = None
        self.doc: dict[str, Any] | None = None
        self.stack: BinaryStack | None = None
        self.dirty = False
        self.active_roi_id = 0
        self._updating = False

        self._mask_edit_active = False
        self._mask_edit_snapshot: dict[str, Any] | None = None
        self._mask_edit_mode: MaskEditMode = "add_f"
        self._mask_traces_stale = False
        self._mask_roi_changed = False
        self._mask_neu_changed = False
        self._extract_cancel = False
        self._extracting = False
        self._w3_y0 = 0
        self._w3_x0 = 0
        self._w3_side = 1

        self._lut_cache = {name: make_lut(name) for name in LUT_NAMES}
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(500)
        self._debounce.timeout.connect(self._update_thumbnails)

        self._build_menu()
        self._build_ui()
        self._build_shortcuts()
        self._apply_saved_settings()
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Open a suite2p folder to begin.")

    # ------------------------------------------------------------------ UI
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        act_open = QAction("Open suite2p folder…", self)
        act_open.triggered.connect(self.open_suite2p)
        file_menu.addAction(act_open)

        act_merge = QAction("Merge s2p folders…", self)
        act_merge.triggered.connect(self.merge_suite2p_folders)
        file_menu.addAction(act_merge)

        act_save = QAction("Save", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self.save_session)
        file_menu.addAction(act_save)

        act_reset = QAction("Reset current ROI", self)
        act_reset.triggered.connect(self.reset_current_roi)
        file_menu.addAction(act_reset)

        file_menu.addSeparator()
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

    def _build_shortcuts(self) -> None:
        sc_up = QShortcut(QKeySequence(Qt.Key.Key_Up), self)
        sc_down = QShortcut(QKeySequence(Qt.Key.Key_Down), self)
        sc_space = QShortcut(QKeySequence(Qt.Key.Key_Space), self)
        for sc in (sc_up, sc_down, sc_space):
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc_up.activated.connect(lambda: self._roi_step(+1))
        sc_down.activated.connect(lambda: self._roi_step(-1))
        sc_space.activated.connect(self._toggle_iscell_shortcut)

    def _focus_owns_arrows_or_space(self) -> bool:
        w = QApplication.focusWidget()
        return isinstance(w, (QAbstractSpinBox, QComboBox, QLineEdit))

    def _roi_step(self, delta: int) -> None:
        if self.doc is None or self._focus_owns_arrows_or_space() or self._mask_edit_active:
            return
        self.spin_roi.setValue(self.spin_roi.value() + delta)

    def _toggle_iscell_shortcut(self) -> None:
        if self.doc is None or self._focus_owns_arrows_or_space() or self._mask_edit_active:
            return
        # Checkbox already toggles on Space when focused; avoid a double toggle.
        if QApplication.focusWidget() is self.chk_iscell:
            return
        self.chk_iscell.toggle()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(6, 6, 6, 6)
        root_layout.setSpacing(6)

        root_layout.addWidget(self._build_left_panel(), stretch=0)

        # Center column: W1–W3, then traces + C1–C4 aligned to those three windows
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)
        root_layout.addWidget(center, stretch=1)

        top_split = QSplitter(Qt.Orientation.Horizontal)
        center_layout.addWidget(top_split, stretch=3)

        self.w1 = self._make_image_panel("FOV (W1)", clickable=True)
        self.w2 = self._make_image_panel("Movie (W2)")
        self.w3 = self._make_image_panel("ROI zoom (W3)", paintable=True)
        top_split.addWidget(self.w1)
        top_split.addWidget(self.w2)
        top_split.addWidget(self.w3)
        top_split.setStretchFactor(0, 1)
        top_split.setStretchFactor(1, 1)
        top_split.setStretchFactor(2, 1)

        # Traces
        self.trace_widget = pg.GraphicsLayoutWidget()
        center_layout.addWidget(self.trace_widget, stretch=2)
        self.plot_f = self.trace_widget.addPlot(row=0, col=0, title="F / x·Fneu")
        self.plot_comp = self.trace_widget.addPlot(row=1, col=0, title="trace_comp = F − x·Fneu")
        self.plot_bleach = self.trace_widget.addPlot(row=2, col=0, title="Bleach-corrected (placeholder)")
        self.plot_comp.setXLink(self.plot_f)
        self.plot_bleach.setXLink(self.plot_f)
        self.plot_bleach.setLabel("bottom", "frame")
        self.plot_f.addLegend(offset=(10, 10))
        self.curve_f = self.plot_f.plot(pen=pg.mkPen("#1f77b4", width=1), name="F")
        self.curve_fneu = self.plot_f.plot(pen=pg.mkPen("#ff7f0e", width=1), name="x·Fneu")
        self.curve_comp = self.plot_comp.plot(pen=pg.mkPen("#2ca02c", width=1.5))

        self.cursors: list[pg.InfiniteLine] = []
        self.cursor_labels: list[pg.TextItem] = []
        # C0 movie cursor
        self.cursor_c0 = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=True,
            pen=pg.mkPen(C0_COLOR, width=2, style=Qt.PenStyle.SolidLine),
        )
        self.cursor_c0.sigPositionChanged.connect(self._on_c0_moved)
        # Use one set of lines on plot_comp for labels; sync clones on other plots
        self._cursor_clones: dict[str, list[pg.InfiniteLine]] = {"f": [], "bleach": []}

        self.plot_comp.addItem(self.cursor_c0)
        line_f0 = pg.InfiniteLine(
            pos=0, angle=90, movable=False, pen=pg.mkPen(C0_COLOR, width=2)
        )
        line_b0 = pg.InfiniteLine(
            pos=0, angle=90, movable=False, pen=pg.mkPen(C0_COLOR, width=2)
        )
        self.plot_f.addItem(line_f0)
        self.plot_bleach.addItem(line_b0)
        self._cursor_clones["f"].append(line_f0)
        self._cursor_clones["bleach"].append(line_b0)

        for i, color in enumerate(C_COLORS):
            line = pg.InfiniteLine(
                pos=0,
                angle=90,
                movable=True,
                pen=pg.mkPen(color, width=1.5, style=Qt.PenStyle.DotLine),
            )
            line.sigPositionChanged.connect(self._on_analysis_cursor_moved)
            self.plot_comp.addItem(line)
            self.cursors.append(line)
            label = pg.TextItem(color=color, anchor=(0, 1))
            self.plot_comp.addItem(label)
            self.cursor_labels.append(label)
            for key, plot in (("f", self.plot_f), ("bleach", self.plot_bleach)):
                clone = pg.InfiniteLine(
                    pos=0,
                    angle=90,
                    movable=False,
                    pen=pg.mkPen(color, width=1.5, style=Qt.PenStyle.DotLine),
                )
                plot.addItem(clone)
                self._cursor_clones[key].append(clone)

        # Thumbnails (aligned with W1–W3 via center column)
        thumb_row = QHBoxLayout()
        center_layout.addLayout(thumb_row)
        self.thumb_views: list[pg.ImageView] = []
        self.thumb_labels: list[QLabel] = []
        for i in range(4):
            box = QVBoxLayout()
            lab = QLabel(f"C{i + 1}")
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            view = pg.ImageView()
            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()
            view.ui.histogram.hide()
            view.setMinimumHeight(120)
            box.addWidget(lab)
            box.addWidget(view)
            wrap = QWidget()
            wrap.setLayout(box)
            thumb_row.addWidget(wrap)
            self.thumb_views.append(view)
            self.thumb_labels.append(lab)

        root_layout.addWidget(self._build_right_panel(), stretch=0)

    def _make_image_panel(
        self, title: str, clickable: bool = False, paintable: bool = False
    ) -> QWidget:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        if paintable:
            view = PaintImageView(on_paint=self._on_w3_paint)
        elif clickable:
            view = ClickableImageView(on_click=self._on_fov_click)
        else:
            view = pg.ImageView()
            view.ui.roiBtn.hide()
            view.ui.menuBtn.hide()
        view.ui.histogram.hide()
        layout.addWidget(view)
        box.image_view = view  # type: ignore[attr-defined]
        return box

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(320)
        layout = QVBoxLayout(panel)

        fov = QGroupBox("FOV (W1)")
        fov_form = QFormLayout(fov)
        self.cmb_fov_src = QComboBox()
        self.cmb_fov_src.addItems(["meanImg", "meanImgE", "VCorr"])
        self.cmb_fov_lut = QComboBox()
        self.cmb_fov_lut.addItems(list(LUT_NAMES))
        self.spin_fov_lo = QDoubleSpinBox()
        self.spin_fov_hi = QDoubleSpinBox()
        for s in (self.spin_fov_lo, self.spin_fov_hi):
            s.setRange(-1e9, 1e9)
            s.setDecimals(2)
        self.cmb_overlay = QComboBox()
        for key, label in FILTER_LABELS.items():
            self.cmb_overlay.addItem(label, key)
        self.cmb_overlay.setCurrentIndex(2)
        fov_form.addRow("Image", self.cmb_fov_src)
        fov_form.addRow("LUT", self.cmb_fov_lut)
        fov_form.addRow("Lower", self.spin_fov_lo)
        fov_form.addRow("Upper", self.spin_fov_hi)
        fov_form.addRow("Show ROIs", self.cmb_overlay)
        layout.addWidget(fov)

        mov = QGroupBox("Movie (W2)")
        mov_form = QFormLayout(mov)
        self.cmb_mov_lut = QComboBox()
        self.cmb_mov_lut.addItems(list(LUT_NAMES))
        self.spin_mov_lo = QDoubleSpinBox()
        self.spin_mov_hi = QDoubleSpinBox()
        for s in (self.spin_mov_lo, self.spin_mov_hi):
            s.setRange(-1e9, 1e9)
            s.setDecimals(2)
        mov_form.addRow("LUT", self.cmb_mov_lut)
        mov_form.addRow("Lower", self.spin_mov_lo)
        mov_form.addRow("Upper", self.spin_mov_hi)
        layout.addWidget(mov)

        roi = QGroupBox("ROI / curation")
        roi_form = QFormLayout(roi)
        roi_nav = QHBoxLayout()
        self.spin_roi = QSpinBox()
        self.spin_roi.setMinimum(0)
        self.btn_roi_up = QPushButton("▲")
        self.btn_roi_down = QPushButton("▼")
        self.btn_roi_up.setFixedWidth(32)
        self.btn_roi_down.setFixedWidth(32)
        roi_nav.addWidget(self.spin_roi)
        roi_nav.addWidget(self.btn_roi_up)
        roi_nav.addWidget(self.btn_roi_down)
        self.chk_iscell = QCheckBox("iscell")
        self.spin_x = QDoubleSpinBox()
        self.spin_x.setRange(-5.0, 5.0)
        self.spin_x.setSingleStep(0.05)
        self.spin_x.setDecimals(3)
        self.spin_x.setValue(1.0)
        roi_form.addRow("ROI #", roi_nav)
        roi_form.addRow(self.chk_iscell)
        roi_form.addRow("x (F−x·Fneu)", self.spin_x)
        layout.addWidget(roi)
        layout.addStretch(1)

        # connections
        self.cmb_fov_src.currentIndexChanged.connect(self._on_fov_src_changed)
        self.cmb_fov_lut.currentIndexChanged.connect(self._refresh_fov)
        self.spin_fov_lo.valueChanged.connect(self._refresh_fov)
        self.spin_fov_hi.valueChanged.connect(self._refresh_fov)
        self.cmb_overlay.currentIndexChanged.connect(self._refresh_fov)
        self.cmb_mov_lut.currentIndexChanged.connect(self._refresh_movie_views)
        self.spin_mov_lo.valueChanged.connect(self._refresh_movie_views)
        self.spin_mov_hi.valueChanged.connect(self._refresh_movie_views)
        self.spin_roi.valueChanged.connect(self._on_roi_spin)
        self.btn_roi_up.clicked.connect(lambda: self.spin_roi.setValue(self.spin_roi.value() + 1))
        self.btn_roi_down.clicked.connect(lambda: self.spin_roi.setValue(self.spin_roi.value() - 1))
        self.chk_iscell.toggled.connect(self._on_iscell_toggled)
        self.spin_x.valueChanged.connect(self._on_x_changed)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(320)
        layout = QVBoxLayout(panel)

        zoom = QGroupBox("ROI zoom (W3)")
        zoom_form = QFormLayout(zoom)
        self.cmb_w3_src = QComboBox()
        self.cmb_w3_src.addItems(["movie", "meanImg", "meanImgE", "VCorr"])
        self.cmb_w3_lut = QComboBox()
        self.cmb_w3_lut.addItems(list(LUT_NAMES))
        self.spin_w3_lo = QDoubleSpinBox()
        self.spin_w3_hi = QDoubleSpinBox()
        for s in (self.spin_w3_lo, self.spin_w3_hi):
            s.setRange(-1e9, 1e9)
            s.setDecimals(2)
        zoom_form.addRow("Image", self.cmb_w3_src)
        zoom_form.addRow("LUT", self.cmb_w3_lut)
        zoom_form.addRow("Lower", self.spin_w3_lo)
        zoom_form.addRow("Upper", self.spin_w3_hi)
        layout.addWidget(zoom)

        actions = QGroupBox("Mask tools")
        actions_layout = QVBoxLayout(actions)
        self.btn_modify = QPushButton("Modify Mask")
        self.btn_modify.setEnabled(False)
        self.btn_modify.setToolTip("Edit F / Fneu pixels in W3")
        self.btn_modify.clicked.connect(self._start_mask_edit)
        actions_layout.addWidget(self.btn_modify)

        self.mask_edit_panel = QWidget()
        edit_layout = QVBoxLayout(self.mask_edit_panel)
        edit_layout.setContentsMargins(0, 0, 0, 0)

        self.mask_mode_group = QButtonGroup(self)
        self.mask_mode_buttons: dict[MaskEditMode, QRadioButton] = {}
        for i, (mode, label) in enumerate(MODE_LABELS.items()):
            rb = QRadioButton(label)
            self.mask_mode_group.addButton(rb, i)
            self.mask_mode_buttons[mode] = rb
            edit_layout.addWidget(rb)
            rb.toggled.connect(self._on_mask_mode_toggled)
        self.mask_mode_buttons["add_f"].setChecked(True)

        brush_row = QFormLayout()
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(0, 30)
        self.spin_brush.setValue(2)
        self.spin_brush.setToolTip("Brush radius in pixels (0 = single pixel)")
        brush_row.addRow("Brush radius", self.spin_brush)
        edit_layout.addLayout(brush_row)

        self.btn_recalc = QPushButton("Re-calculate Traces")
        self.btn_recalc.clicked.connect(self._recalculate_traces)
        edit_layout.addWidget(self.btn_recalc)

        self.extract_progress = QProgressBar()
        self.extract_progress.setRange(0, 100)
        self.extract_progress.setValue(0)
        self.extract_progress.setVisible(False)
        edit_layout.addWidget(self.extract_progress)

        self.btn_cancel_job = QPushButton("Cancel job")
        self.btn_cancel_job.setEnabled(False)
        self.btn_cancel_job.clicked.connect(self._cancel_extract_job)
        edit_layout.addWidget(self.btn_cancel_job)

        self.btn_apply_mask = QPushButton("Apply Mask")
        self.btn_apply_mask.clicked.connect(self._apply_mask_edit)
        edit_layout.addWidget(self.btn_apply_mask)

        self.btn_cancel_mask = QPushButton("Cancel")
        self.btn_cancel_mask.clicked.connect(self._cancel_mask_edit)
        edit_layout.addWidget(self.btn_cancel_mask)

        self.mask_edit_panel.setVisible(False)
        actions_layout.addWidget(self.mask_edit_panel)
        actions_layout.addStretch(1)
        layout.addWidget(actions)

        layout.addStretch(1)

        self.btn_save = QPushButton("Save")
        self.btn_save.setEnabled(False)
        self.btn_save.setToolTip("Save trc_curation.pkl (Ctrl+S)")
        self.btn_save.clicked.connect(self.save_session)
        layout.addWidget(self.btn_save)

        self.cmb_w3_src.currentIndexChanged.connect(self._on_w3_src_changed)
        self.cmb_w3_lut.currentIndexChanged.connect(self._refresh_w3_and_thumbs)
        self.spin_w3_lo.valueChanged.connect(self._refresh_w3_and_thumbs)
        self.spin_w3_hi.valueChanged.connect(self._refresh_w3_and_thumbs)
        return panel

    def _apply_saved_settings(self) -> None:
        s = load_settings()
        self._updating = True
        try:
            if src := s.get("fov_src"):
                idx = self.cmb_fov_src.findText(str(src))
                if idx >= 0:
                    self.cmb_fov_src.setCurrentIndex(idx)
            if lut := s.get("fov_lut"):
                idx = self.cmb_fov_lut.findText(str(lut))
                if idx >= 0:
                    self.cmb_fov_lut.setCurrentIndex(idx)
            if filt := s.get("overlay_filter"):
                idx = self.cmb_overlay.findData(str(filt))
                if idx >= 0:
                    self.cmb_overlay.setCurrentIndex(idx)
            if lut := s.get("mov_lut"):
                idx = self.cmb_mov_lut.findText(str(lut))
                if idx >= 0:
                    self.cmb_mov_lut.setCurrentIndex(idx)
            if src := s.get("w3_src"):
                idx = self.cmb_w3_src.findText(str(src))
                if idx >= 0:
                    self.cmb_w3_src.setCurrentIndex(idx)
            if lut := s.get("w3_lut"):
                idx = self.cmb_w3_lut.findText(str(lut))
                if idx >= 0:
                    self.cmb_w3_lut.setCurrentIndex(idx)
        finally:
            self._updating = False

    def _persist_ui_settings(self, *, suite2p_dir: Path | None = None) -> None:
        updates: dict[str, Any] = {
            "fov_src": self.cmb_fov_src.currentText(),
            "fov_lut": self.cmb_fov_lut.currentText(),
            "overlay_filter": self.cmb_overlay.currentData(),
            "mov_lut": self.cmb_mov_lut.currentText(),
            "w3_src": self.cmb_w3_src.currentText(),
            "w3_lut": self.cmb_w3_lut.currentText(),
        }
        if suite2p_dir is not None:
            updates["last_suite2p_dir"] = str(suite2p_dir)
        save_settings(updates)

    # --------------------------------------------------------------- session
    def open_suite2p(self) -> None:
        start = last_open_start_dir()
        path = QFileDialog.getExistingDirectory(
            self, "Select suite2p folder", start
        )
        if not path:
            return
        try:
            self._load_suite2p(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))

    def merge_suite2p_folders(self) -> None:
        from s2p_trace_curation.gui.merge_dialog import MergeSuite2pDialog

        dlg = MergeSuite2pDialog(self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        if dlg.merged_dir is not None:
            try:
                self._load_suite2p(dlg.merged_dir)
            except Exception as exc:
                QMessageBox.critical(
                    self,
                    "Open merged folder failed",
                    f"Merge wrote files, but opening failed:\n{exc}",
                )

    def _load_suite2p(self, path: Path) -> None:
        if self._mask_edit_active:
            self._cancel_mask_edit()
        if self.stack is not None:
            self.stack.close()
            self.stack = None
        suite2p_dir, doc, created = open_suite2p_session(path)
        self.suite2p_dir = suite2p_dir
        self.doc = doc
        self.dirty = created
        self.stack = BinaryStack(plane_dir(suite2p_dir))
        n = len(doc["rois"])
        self._updating = True
        self.spin_roi.setMaximum(max(0, n - 1))
        self.spin_roi.setValue(0)
        self.active_roi_id = 0
        self._init_display_levels()
        self._init_cursors()
        self._updating = False
        self._select_roi(0, force=True)
        self._persist_ui_settings(suite2p_dir=suite2p_dir)
        self.btn_modify.setEnabled(True)
        self.btn_save.setEnabled(True)
        msg = "Created" if created else "Loaded"
        self.statusBar().showMessage(
            f"{msg} {suite2p_dir / 'trc_curation.pkl'} — {n} ROIs, "
            f"{doc['meta']['nframes']} frames"
        )
        self.setWindowTitle(f"s2p Trace Curation — {suite2p_dir}")

    def _init_display_levels(self) -> None:
        assert self.doc is not None
        img = self._fov_source_image()
        if img is not None:
            self._set_levels_from_image(img, self.spin_fov_lo, self.spin_fov_hi)
        if self.stack is not None:
            sample = self.stack.read_frame(0)
            self._set_levels_from_image(sample, self.spin_mov_lo, self.spin_mov_hi)
        w3_img = self._w3_source_image(frame_index=0)
        if w3_img is not None:
            self._set_levels_from_image(w3_img, self.spin_w3_lo, self.spin_w3_hi)

    def _set_levels_from_image(
        self, img: np.ndarray, spin_lo: QDoubleSpinBox, spin_hi: QDoubleSpinBox
    ) -> None:
        """Set Lower/Upper spins from 1–99% of an image."""
        flat = np.asarray(img, dtype=np.float64)
        flat = flat[np.isfinite(flat)]
        if flat.size == 0:
            lo, hi = 0.0, 1.0
        else:
            lo, hi = np.percentile(flat, [1, 99])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo = float(flat.min())
                hi = float(flat.max())
            if hi <= lo:
                hi = lo + 1.0
        spin_lo.blockSignals(True)
        spin_hi.blockSignals(True)
        spin_lo.setValue(float(lo))
        spin_hi.setValue(float(hi))
        spin_lo.blockSignals(False)
        spin_hi.blockSignals(False)

    def _set_fov_levels_from_image(self, img: np.ndarray) -> None:
        self._set_levels_from_image(img, self.spin_fov_lo, self.spin_fov_hi)

    def _on_fov_src_changed(self) -> None:
        if self.doc is None:
            return
        img = self._fov_source_image()
        if img is not None:
            self._set_fov_levels_from_image(img)
        self._refresh_fov()

    def _on_w3_src_changed(self) -> None:
        if self.doc is None:
            return
        img = self._w3_source_image()
        if img is not None:
            self._set_levels_from_image(img, self.spin_w3_lo, self.spin_w3_hi)
        self._refresh_w3_and_thumbs()

    def _init_cursors(self) -> None:
        assert self.doc is not None
        n = int(self.doc["meta"]["nframes"])
        fracs = (0.2, 0.4, 0.6, 0.8)
        self._updating = True
        self.cursor_c0.setValue(0)
        for line, frac in zip(self.cursors, fracs):
            line.setValue(frac * max(n - 1, 0))
        self._sync_cursor_clones()
        self._updating = False

    def save_session(self) -> bool:
        if self.doc is None or self.suite2p_dir is None:
            return False
        if self._mask_edit_active:
            QMessageBox.information(
                self,
                "Mask edit active",
                "Finish with Apply Mask or Cancel before saving.",
            )
            return False
        path = save_curation(self.doc, self.suite2p_dir)
        self.dirty = False
        self.statusBar().showMessage(f"Saved {path}")
        return True

    def reset_current_roi(self) -> None:
        if self.doc is None or self.suite2p_dir is None:
            return
        if self._mask_edit_active:
            QMessageBox.information(
                self,
                "Mask edit active",
                "Finish with Apply Mask or Cancel before resetting the ROI.",
            )
            return
        reset_roi_from_suite2p(self.doc, self.suite2p_dir, self.active_roi_id)
        self.dirty = True
        self._select_roi(self.active_roi_id, force=True)
        self.statusBar().showMessage(f"Reset ROI {self.active_roi_id} from suite2p")

    def closeEvent(self, event) -> None:
        if self._mask_edit_active:
            reply = QMessageBox.question(
                self,
                "Mask edit in progress",
                "A mask edit is still open. Cancel the edit and continue closing?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._cancel_mask_edit()

        if self.dirty:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Unsaved changes")
            box.setText("You have unsaved changes to trc_curation.pkl.")
            box.setInformativeText("Save before closing?")
            btn_save = box.addButton("Save", QMessageBox.ButtonRole.AcceptRole)
            btn_discard = box.addButton("Discard", QMessageBox.ButtonRole.DestructiveRole)
            btn_cancel = box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
            box.setDefaultButton(btn_save)
            box.exec()
            clicked = box.clickedButton()
            if clicked is btn_cancel:
                event.ignore()
                return
            if clicked is btn_save:
                if not self.save_session():
                    event.ignore()
                    return

        self._persist_ui_settings(suite2p_dir=self.suite2p_dir)
        if self.stack is not None:
            self.stack.close()
        super().closeEvent(event)

    # ----------------------------------------------------------- mask edit
    def _set_mask_edit_ui(self, active: bool) -> None:
        self._mask_edit_active = active
        self.mask_edit_panel.setVisible(active)
        self.btn_modify.setEnabled(not active and self.doc is not None)
        view: PaintImageView = self.w3.image_view  # type: ignore[attr-defined]
        view.paint_enabled = active
        # Freeze ROI controls while editing
        self.spin_roi.setEnabled(not active)
        self.btn_roi_up.setEnabled(not active)
        self.btn_roi_down.setEnabled(not active)
        self.chk_iscell.setEnabled(not active)

    def _on_mask_mode_toggled(self, checked: bool) -> None:
        if not checked:
            return
        for mode, rb in self.mask_mode_buttons.items():
            if rb.isChecked():
                self._mask_edit_mode = mode
                break

    def _start_mask_edit(self) -> None:
        if self.doc is None or self._mask_edit_active:
            return
        self._mask_edit_snapshot = deepcopy(self._row())
        self._mask_traces_stale = False
        self._mask_roi_changed = False
        self._mask_neu_changed = False
        self._set_mask_edit_ui(True)
        self.statusBar().showMessage(
            f"Modify Mask: ROI {self.active_roi_id} — paint in W3, then Apply or Cancel"
        )

    def _cancel_mask_edit(self) -> None:
        if not self._mask_edit_active:
            return
        if self._extracting:
            self._extract_cancel = True
            return
        if self._mask_edit_snapshot is not None and self.doc is not None:
            snap = self._mask_edit_snapshot
            for i, r in enumerate(self.doc["rois"]):
                if int(r["roi_id"]) == int(self.active_roi_id):
                    self.doc["rois"][i] = snap
                    break
        self._mask_edit_snapshot = None
        self._mask_traces_stale = False
        self._mask_roi_changed = False
        self._mask_neu_changed = False
        self._set_mask_edit_ui(False)
        self._refresh_all()
        self.statusBar().showMessage("Mask edit cancelled — previous masks/traces restored")

    def _apply_mask_edit(self) -> None:
        if not self._mask_edit_active or self.doc is None or self.suite2p_dir is None:
            return
        if self._extracting:
            return
        if self._mask_traces_stale:
            ok = self._recalculate_traces()
            if not ok:
                return
        path = save_curation(self.doc, self.suite2p_dir)
        self.dirty = False
        self._mask_edit_snapshot = None
        self._mask_traces_stale = False
        self._mask_roi_changed = False
        self._mask_neu_changed = False
        self._set_mask_edit_ui(False)
        self._refresh_all()
        self.statusBar().showMessage(f"Applied mask edits and saved {path}")

    def _cancel_extract_job(self) -> None:
        if self._extracting:
            self._extract_cancel = True

    def _recalculate_traces(self) -> bool:
        if self.doc is None or self.suite2p_dir is None:
            return False
        if not self._mask_roi_changed and not self._mask_neu_changed and not self._mask_traces_stale:
            self.statusBar().showMessage("Traces already up to date")
            return True
        if self._extracting:
            return False
        roi_changed = bool(self._mask_roi_changed)
        neu_changed = bool(self._mask_neu_changed)
        if self._mask_traces_stale and not roi_changed and not neu_changed:
            roi_changed = True
            neu_changed = True

        self._extracting = True
        self._extract_cancel = False
        self.extract_progress.setVisible(True)
        self.extract_progress.setValue(0)
        self.btn_cancel_job.setEnabled(True)
        self.btn_recalc.setEnabled(False)
        self.btn_apply_mask.setEnabled(False)
        self.btn_cancel_mask.setEnabled(False)
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            def _progress(step: int, total: int) -> None:
                pct = int(100 * step / max(total, 1))
                self.extract_progress.setValue(pct)
                if step == 1 or step == total or step % 25 == 0:
                    QApplication.processEvents()

            reextract_after_mask_edit(
                self._row(),
                self.suite2p_dir,
                roi_changed=roi_changed,
                neuropil_changed=neu_changed,
                progress=_progress,
                should_cancel=lambda: self._extract_cancel,
            )
            self._mask_traces_stale = False
            self._mask_roi_changed = False
            self._mask_neu_changed = False
            self.dirty = True
            self._refresh_traces(autoscale=True)
            self.extract_progress.setValue(100)
            self.statusBar().showMessage("Traces re-calculated from data.bin")
            return True
        except ExtractCancelled:
            self.statusBar().showMessage("Trace re-calculation cancelled")
            return False
        except Exception as exc:
            QMessageBox.critical(self, "Re-calculate failed", str(exc))
            return False
        finally:
            QApplication.restoreOverrideCursor()
            self._extracting = False
            self._extract_cancel = False
            self.btn_cancel_job.setEnabled(False)
            self.btn_recalc.setEnabled(True)
            self.btn_apply_mask.setEnabled(True)
            self.btn_cancel_mask.setEnabled(True)
            self.extract_progress.setVisible(False)

    def _on_w3_paint(self, y_img: int, x_img: int) -> None:
        if (
            not self._mask_edit_active
            or self._extracting
            or self.doc is None
        ):
            return
        Ly = int(self.doc["meta"]["Ly"])
        Lx = int(self.doc["meta"]["Lx"])
        if y_img < 0 or x_img < 0 or y_img >= self._w3_side or x_img >= self._w3_side:
            return
        cy = self._w3_y0 + y_img
        cx = self._w3_x0 + x_img
        if cy < 0 or cx < 0 or cy >= Ly or cx >= Lx:
            return
        changed, msg = apply_brush(
            self._row(),
            self._mask_edit_mode,
            cy,
            cx,
            int(self.spin_brush.value()),
            Ly,
            Lx,
        )
        if not changed:
            if msg:
                self.statusBar().showMessage(msg)
            return
        if self._mask_edit_mode in ("add_f", "remove_f"):
            self._mask_roi_changed = True
        if self._mask_edit_mode in ("add_fneu", "remove_fneu"):
            self._mask_neu_changed = True
        # Stealing between masks dirties both
        if self._mask_edit_mode in ("add_f", "add_fneu"):
            self._mask_roi_changed = True
            self._mask_neu_changed = True
        self._mask_traces_stale = True
        self.dirty = True
        self._refresh_fov()
        self._refresh_movie_views()
        if msg:
            self.statusBar().showMessage(msg)

    # --------------------------------------------------------------- ROI sel
    def _row(self, roi_id: int | None = None) -> dict[str, Any]:
        assert self.doc is not None
        rid = self.active_roi_id if roi_id is None else roi_id
        for row in self.doc["rois"]:
            if int(row["roi_id"]) == int(rid):
                return row
        raise KeyError(rid)

    def _overlay_filter(self) -> OverlayFilter:
        return self.cmb_overlay.currentData()  # type: ignore[return-value]

    def _on_roi_spin(self, value: int) -> None:
        if self._updating or self.doc is None or self._mask_edit_active:
            return
        self._select_roi(value)

    def _select_roi(self, roi_id: int, force: bool = False) -> None:
        if self.doc is None:
            return
        if self._mask_edit_active and not force:
            return
        if not force and roi_id == self.active_roi_id and not self._updating:
            # still refresh when forced internals call
            pass
        self.active_roi_id = int(roi_id)
        row = self._row()
        self._updating = True
        self.spin_roi.setValue(self.active_roi_id)
        self.chk_iscell.setChecked(bool(row["iscell"]))
        self.spin_x.setValue(float(row["compensation"]["x"]))
        self._updating = False
        self._refresh_all()

    def _on_fov_click(self, y: int, x: int) -> None:
        if self.doc is None or self._mask_edit_active:
            return
        Ly = int(self.doc["meta"]["Ly"])
        Lx = int(self.doc["meta"]["Lx"])
        if y < 0 or x < 0 or y >= Ly or x >= Lx:
            return
        hits = rois_at_pixel(self.doc["rois"], y, x, self._overlay_filter())
        if hits:
            self._select_roi(int(hits[0]["roi_id"]))

    def _on_iscell_toggled(self, checked: bool) -> None:
        if self._updating or self.doc is None or self._mask_edit_active:
            return
        self._row()["iscell"] = bool(checked)
        self.dirty = True
        self._refresh_fov()

    def _on_x_changed(self, value: float) -> None:
        if self._updating or self.doc is None:
            return
        set_compensation_x(self._row(), float(value))
        self.dirty = True
        self._refresh_traces(autoscale=True)

    # ------------------------------------------------------------- rendering
    def _refresh_all(self) -> None:
        self._refresh_fov()
        self._refresh_movie_views()
        self._refresh_traces(autoscale=True)
        self._update_cursor_labels()
        self._debounce.start()

    def _fov_source_image(self) -> np.ndarray | None:
        assert self.doc is not None
        key = self.cmb_fov_src.currentText()
        meta = self.doc["meta"]
        mapping = {
            "meanImg": meta.get("meanImg"),
            "meanImgE": meta.get("meanImgE"),
            "VCorr": meta.get("VCorr"),
        }
        img = mapping.get(key)
        if img is None and key != "meanImg":
            img = meta.get("meanImg")
        return None if img is None else np.asarray(img)

    def _w3_source_image(self, frame_index: int | None = None) -> np.ndarray | None:
        """Raw 2D image for W3 / thumbnail backdrop (before LUT)."""
        if self.doc is None:
            return None
        key = self.cmb_w3_src.currentText()
        meta = self.doc["meta"]
        if key == "movie":
            if self.stack is None:
                return None
            t = self._c0_frame_index() if frame_index is None else int(frame_index)
            return self.stack.read_frame(t)
        mapping = {
            "meanImg": meta.get("meanImg"),
            "meanImgE": meta.get("meanImgE"),
            "VCorr": meta.get("VCorr"),
        }
        img = mapping.get(key)
        if img is None:
            img = meta.get("meanImg")
        return None if img is None else np.asarray(img)

    def _w3_rgb(self, frame_index: int | None = None) -> np.ndarray | None:
        img = self._w3_source_image(frame_index=frame_index)
        if img is None:
            return None
        lut = self._lut_cache[self.cmb_w3_lut.currentText()]
        return apply_lut(img, lut, self.spin_w3_lo.value(), self.spin_w3_hi.value())

    @staticmethod
    def _set_display_rgb(view: pg.ImageView, rgb: np.ndarray) -> None:
        """Show pre-composited RGB uint8 without ImageView re-applying levels."""
        view.setImage(rgb, autoLevels=False, levels=(0, 255))

    def _refresh_fov(self) -> None:
        if self.doc is None:
            return
        img = self._fov_source_image()
        if img is None:
            return
        lut = self._lut_cache[self.cmb_fov_lut.currentText()]
        rgb = apply_lut(img, lut, self.spin_fov_lo.value(), self.spin_fov_hi.value())
        overlay = build_fov_overlay(
            int(self.doc["meta"]["Ly"]),
            int(self.doc["meta"]["Lx"]),
            self.doc["rois"],
            self.active_roi_id,
            self._overlay_filter(),
        )
        composed = compose_rgb_with_overlay(rgb, overlay)
        self._set_display_rgb(self.w1.image_view, composed)  # type: ignore[attr-defined]

    def _movie_rgb(self, frame: np.ndarray) -> np.ndarray:
        lut = self._lut_cache[self.cmb_mov_lut.currentText()]
        return apply_lut(frame, lut, self.spin_mov_lo.value(), self.spin_mov_hi.value())

    def _c0_frame_index(self) -> int:
        if self.doc is None:
            return 0
        n = int(self.doc["meta"]["nframes"])
        return int(np.clip(round(self.cursor_c0.value()), 0, max(n - 1, 0)))

    def _zoom_geometry(self) -> tuple[int, int, int, dict[str, Any]]:
        assert self.doc is not None
        meta = self.doc["meta"]
        Ly, Lx = int(meta["Ly"]), int(meta["Lx"])
        row = self._row()
        y0, x0, side = zoom_square_window(
            row["roi"]["ypix"], row["roi"]["xpix"], row["neuropil"]["ipix"], Ly, Lx
        )
        return y0, x0, side, row

    def _refresh_movie_views(self) -> None:
        if self.doc is None or self.stack is None:
            return
        meta = self.doc["meta"]
        Ly, Lx = int(meta["Ly"]), int(meta["Lx"])
        row = self._row()
        t = self._c0_frame_index()
        frame = self.stack.read_frame(t)
        rgb = self._movie_rgb(frame)

        # W2 with thick outline
        y_out, x_out = thick_outline_mask(Ly, Lx, row["roi"]["ypix"], row["roi"]["xpix"], thickness=2)
        w2 = rgb.copy()
        if y_out.size:
            w2[y_out, x_out] = (255, 0, 0)
        self._set_display_rgb(self.w2.image_view, w2)  # type: ignore[attr-defined]
        self.w2.setTitle(f"Movie (W2) — frame {t}")  # type: ignore[attr-defined]

        self._refresh_w3()

    def _refresh_w3_and_thumbs(self) -> None:
        self._refresh_w3()
        self._debounce.start()

    def _refresh_w3(self) -> None:
        if self.doc is None:
            return
        meta = self.doc["meta"]
        Ly, Lx = int(meta["Ly"]), int(meta["Lx"])
        t = self._c0_frame_index()
        rgb = self._w3_rgb(frame_index=t if self.cmb_w3_src.currentText() == "movie" else None)
        if rgb is None:
            return
        y0, x0, side, row = self._zoom_geometry()
        self._w3_y0, self._w3_x0, self._w3_side = y0, x0, side
        zoom = zoom_masks_rgba(rgb, y0, x0, side, row, Ly, Lx)
        self._set_display_rgb(self.w3.image_view, zoom)  # type: ignore[attr-defined]
        src = self.cmb_w3_src.currentText()
        title = f"ROI zoom (W3) — {src}"
        if src == "movie":
            title += f" frame {t}"
        if self._mask_edit_active:
            title += " [editing]"
        self.w3.setTitle(title)  # type: ignore[attr-defined]

    def _refresh_traces(self, autoscale: bool = False) -> None:
        if self.doc is None:
            return
        row = self._row()
        F = np.asarray(row["roi"]["F"], dtype=np.float64)
        Fneu = np.asarray(row["neuropil"]["Fneu"], dtype=np.float64)
        x = float(row["compensation"]["x"])
        comp = np.asarray(row["compensation"]["trace_comp"], dtype=np.float64)
        xs = np.arange(F.shape[0])
        self.curve_f.setData(xs, F)
        self.curve_fneu.setData(xs, x * Fneu)
        self.curve_comp.setData(xs, comp)
        if autoscale:
            self.plot_f.enableAutoRange(axis="y")
            self.plot_comp.enableAutoRange(axis="y")
            self.plot_f.enableAutoRange(axis="x", enable=False)
            self.plot_comp.enableAutoRange(axis="x", enable=False)
            self.plot_f.setXRange(0, max(len(F) - 1, 1), padding=0.02)
        self._update_cursor_labels()

    # -------------------------------------------------------------- cursors
    def _sync_cursor_clones(self) -> None:
        positions = [self.cursor_c0.value()] + [c.value() for c in self.cursors]
        for i, pos in enumerate(positions):
            self._cursor_clones["f"][i].setValue(pos)
            self._cursor_clones["bleach"][i].setValue(pos)

    def _on_c0_moved(self) -> None:
        if self._updating or self.doc is None:
            return
        n = int(self.doc["meta"]["nframes"])
        val = float(np.clip(self.cursor_c0.value(), 0, max(n - 1, 0)))
        if val != self.cursor_c0.value():
            self._updating = True
            self.cursor_c0.setValue(val)
            self._updating = False
        self._sync_cursor_clones()
        self._refresh_movie_views()

    def _on_analysis_cursor_moved(self) -> None:
        if self._updating or self.doc is None:
            return
        n = int(self.doc["meta"]["nframes"])
        self._updating = True
        for c in self.cursors:
            c.setValue(float(np.clip(c.value(), 0, max(n - 1, 0))))
        self._updating = False
        self._sync_cursor_clones()
        self._update_cursor_labels()
        self._debounce.start()

    def _update_cursor_labels(self) -> None:
        if self.doc is None:
            return
        comp = np.asarray(self._row()["compensation"]["trace_comp"], dtype=np.float64)
        n = len(comp)
        for line, label in zip(self.cursors, self.cursor_labels):
            fi = int(np.clip(round(line.value()), 0, max(n - 1, 0)))
            val = float(comp[fi]) if n else 0.0
            label.setText(f"f={fi}\n{val:.2f}")
            label.setPos(fi, val)

    def _update_thumbnails(self) -> None:
        if self.doc is None:
            return
        meta = self.doc["meta"]
        Ly, Lx = int(meta["Ly"]), int(meta["Lx"])
        y0, x0, side, row = self._zoom_geometry()
        n = int(meta["nframes"])
        src = self.cmb_w3_src.currentText()
        for i, (line, view, lab) in enumerate(
            zip(self.cursors, self.thumb_views, self.thumb_labels)
        ):
            fi = int(np.clip(round(line.value()), 0, max(n - 1, 0)))
            rgb = self._w3_rgb(frame_index=fi if src == "movie" else None)
            if rgb is None:
                continue
            zoom = zoom_masks_rgba(rgb, y0, x0, side, row, Ly, Lx)
            self._set_display_rgb(view, zoom)
            if src == "movie":
                lab.setText(f"C{i + 1} — frame {fi}")
            else:
                lab.setText(f"C{i + 1} — {src}")
