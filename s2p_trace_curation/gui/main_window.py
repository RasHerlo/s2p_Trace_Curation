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
QGraphicsProxyWidget = QtWidgets.QGraphicsProxyWidget
QHBoxLayout = QtWidgets.QHBoxLayout
QLabel = QtWidgets.QLabel
QLineEdit = QtWidgets.QLineEdit
QListWidget = QtWidgets.QListWidget
QListWidgetItem = QtWidgets.QListWidgetItem
QMainWindow = QtWidgets.QMainWindow
QMessageBox = QtWidgets.QMessageBox
QProgressBar = QtWidgets.QProgressBar
QPushButton = QtWidgets.QPushButton
QRadioButton = QtWidgets.QRadioButton
QShortcut = QtWidgets.QShortcut
QSlider = QtWidgets.QSlider
QSpinBox = QtWidgets.QSpinBox
QSplitter = QtWidgets.QSplitter
QStackedWidget = QtWidgets.QStackedWidget
QStatusBar = QtWidgets.QStatusBar
QToolButton = QtWidgets.QToolButton
QVBoxLayout = QtWidgets.QVBoxLayout
QWidget = QtWidgets.QWidget

from copy import deepcopy

from s2p_trace_curation.analyses import (
    PICKLE_SORT_ID,
    active_sort_run,
    apply_raster_sort,
    dropdown_label,
    ensure_analyses,
    raster_sort_id,
    refresh_stale_flags,
    set_raster_sort,
)
from s2p_trace_curation.annotations import (
    apply_nan_mask,
    ensure_annotations,
    nan_mask_from_annotations,
    property_spec,
)
from s2p_trace_curation.batch_select import mean_traces_for_rois, rois_in_lasso
from s2p_trace_curation.curation import (
    append_roi,
    empty_roi_draft,
    ensure_meanimge,
    next_roi_id,
    open_suite2p_session,
    reextract_after_mask_edit,
    reset_roi_from_suite2p,
    save_curation,
    set_compensation_x,
)
from s2p_trace_curation.gui.analysis_dialog import AnalysisToolsWindow
from s2p_trace_curation.gui.annotation_dialog import AnnotationEditorWindow
from s2p_trace_curation.gui.heatmap_dialog import HeatmapEditorWindow
from s2p_trace_curation.gui.trace_processing_dialog import TraceProcessingWindow
from s2p_trace_curation.gui.colormaps import (
    LUT_NAMES,
    apply_lut,
    colorize_raster,
    lut_with_revert,
    make_lut,
    selected_row_lut,
)
from s2p_trace_curation.heatmaps import (
    ensure_heatmaps,
    get_heatmap,
    heatmap_combo_data,
    heatmap_combo_label,
    parse_heatmap_combo_data,
)
from s2p_trace_curation.raster import (
    fill_missing_tc_norm,
    led_shutter_nan_mask,
    rebuild_all_tc_norm,
    rois_for_raster,
    tc_norm_is_stale,
    tc_norm_sig,
)
from s2p_trace_curation.trace_processing import (
    TRACE_FIELD_LABELS,
    TRACE_FIELD_NORM,
    TRACE_FIELD_SM,
    TRACE_FIELD_SM_BC,
    TRACE_FIELDS,
    bleach_fit_curve,
    ensure_trace_processing,
    field_has_any,
    raster_trace_field,
    rebuild_all_tc_norm_sm,
    rebuild_all_tc_norm_sm_bc,
    set_raster_trace_field,
    stack_trace_field,
    tc_norm_sm_bc_is_stale,
    tc_norm_sm_is_stale,
)
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
from s2p_trace_curation.suite2p_io import (
    BinaryStack,
    median_zoom_side,
    plane_dir,
    zoom_square_at,
    zoom_square_window,
)
from s2p_trace_curation.user_settings import (
    last_open_start_dir,
    load_settings,
    save_settings,
)

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

# Default window size as a fraction of the current screen work area.
# Tuned to 2456×1100 on a 3440×1392 ultrawide work area.
_DEFAULT_SCREEN_W_FRAC = 2456 / 3440
_DEFAULT_SCREEN_H_FRAC = 1100 / 1392
_MIN_WINDOW_W = 960
_MIN_WINDOW_H = 700

FILTER_LABELS = {
    "noncell": "non-selected (iscell=0)",
    "cell": "selected (iscell=1)",
    "both": "both",
}

# Analysis cursor colors (dotted)
C_COLORS = ["#ff8c00", "#da70d6", "#7fffd4", "#ffa07a"]
C0_COLOR = "#ffff66"  # solid movie cursor

_NUDGE_FRAC = 0.05  # fraction of visible span per axis-arrow click

# Rows of trace_widget, and how much height one may claim when expanded alone
_TRACE_ROWS = {"f": 0, "comp": 1, "bleach": 2}
_ROW_MAX_HEIGHT = 100000.0
_INSPECT_FOCUS_NAMES = {
    "f": "F / x·Fneu",
    "comp": "trace_comp",
    "bleach": "tc_norm_sm / bleach",
    "thumbs": "ROI zooms",
}


class _NudgePair(QWidget):
    """Two tiny arrow buttons at one end of an axis."""

    def __init__(self, vertical: bool, on_plus, on_minus, tooltip: str) -> None:
        super().__init__()
        self.setToolTip(tooltip)
        layout: QVBoxLayout | QHBoxLayout
        if vertical:
            layout = QVBoxLayout(self)
            plus = self._arrow(Qt.ArrowType.UpArrow, on_plus)
            minus = self._arrow(Qt.ArrowType.DownArrow, on_minus)
            layout.addWidget(plus)
            layout.addWidget(minus)
        else:
            layout = QHBoxLayout(self)
            minus = self._arrow(Qt.ArrowType.LeftArrow, on_minus)
            plus = self._arrow(Qt.ArrowType.RightArrow, on_plus)
            layout.addWidget(minus)
            layout.addWidget(plus)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background: rgba(40, 40, 40, 180);")

    @staticmethod
    def _arrow(kind: Qt.ArrowType, slot) -> QToolButton:
        btn = QToolButton()
        btn.setArrowType(kind)
        btn.setFixedSize(14, 14)
        btn.setAutoRepeat(True)
        btn.setAutoRepeatDelay(250)
        btn.setAutoRepeatInterval(50)
        btn.clicked.connect(slot)
        return btn


class ClickableImageView(pg.ImageView):
    """ImageView: left-click select and/or freehand lasso drawing."""

    def __init__(self, *args, on_click=None, on_lasso=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_click = on_click
        self._on_lasso = on_lasso
        self.lasso_enabled = False
        self._lasso_drawing = False
        self._lasso_points: list[tuple[float, float]] = []
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.getView().scene().sigMouseClicked.connect(self._scene_clicked)
        self.ui.graphicsView.viewport().installEventFilter(self)
        self._lasso_curve = pg.PlotDataItem(
            pen=pg.mkPen("#ffff00", width=2), connect="all"
        )
        self.getView().addItem(self._lasso_curve)

    def _view_box(self) -> Any:
        view = self.getView()
        getter = getattr(view, "getViewBox", None)
        if callable(getter):
            box = getter()
            if box is not None:
                return box
        return view

    def _map_scene_to_view(self, scene_pos: Any) -> Any | None:
        box = self._view_box()
        if not box.sceneBoundingRect().contains(scene_pos):
            return None
        return box.mapSceneToView(scene_pos)

    def clear_lasso_drawing(self) -> None:
        self._lasso_points = []
        self._lasso_curve.setData([], [])

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if (
            not self.lasso_enabled
            or self._on_lasso is None
            or obj is not self.ui.graphicsView.viewport()
        ):
            return super().eventFilter(obj, event)

        et = event.type()
        if et == QtCore.QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                self._lasso_drawing = True
                self._lasso_points = []
                self._append_lasso_pos(event.pos())
                return True
        elif et == QtCore.QEvent.Type.MouseMove:
            if self._lasso_drawing and event.buttons() & Qt.MouseButton.LeftButton:
                self._append_lasso_pos(event.pos())
                return True
        elif et == QtCore.QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.LeftButton and self._lasso_drawing:
                self._lasso_drawing = False
                self._append_lasso_pos(event.pos())
                pts = list(self._lasso_points)
                if len(pts) >= 3:
                    self._on_lasso(pts)
                return True
        return super().eventFilter(obj, event)

    def _append_lasso_pos(self, viewport_pos) -> None:
        scene_pos = self.ui.graphicsView.mapToScene(viewport_pos)
        mouse = self._map_scene_to_view(scene_pos)
        if mouse is None:
            return
        # Image coords: y=row, x=col
        pt = (float(mouse.y()), float(mouse.x()))
        if self._lasso_points:
            prev = self._lasso_points[-1]
            if (prev[0] - pt[0]) ** 2 + (prev[1] - pt[1]) ** 2 < 0.25:
                return
        self._lasso_points.append(pt)
        ys = [p[0] for p in self._lasso_points]
        xs = [p[1] for p in self._lasso_points]
        self._lasso_curve.setData(xs, ys)

    def _scene_clicked(self, event) -> None:
        if self.lasso_enabled:
            return
        if self._on_click is None or event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.scenePos()
        mouse = self._map_scene_to_view(pos)
        if mouse is None:
            return
        self._on_click(int(mouse.y()), int(mouse.x()))


class PaintImageView(pg.ImageView):
    """ImageView with optional left-drag paint and wheel zoom callbacks."""

    def __init__(self, *args, on_paint=None, on_wheel=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_paint = on_paint
        self._on_wheel = on_wheel
        self.paint_enabled = False
        self._painting = False
        self.ui.roiBtn.hide()
        self.ui.menuBtn.hide()
        self.ui.graphicsView.viewport().installEventFilter(self)

    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        if obj is not self.ui.graphicsView.viewport():
            return super().eventFilter(obj, event)

        et = event.type()
        if et == QtCore.QEvent.Type.Wheel and self._on_wheel is not None:
            delta = event.angleDelta().y()
            if delta != 0:
                self._on_wheel(1 if delta > 0 else -1)
                return True

        if not self.paint_enabled or self._on_paint is None:
            return super().eventFilter(obj, event)

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
        getter = getattr(view, "getViewBox", None)
        box = getter() if callable(getter) else view
        if box is None:
            box = view
        if not box.sceneBoundingRect().contains(scene_pos):
            return
        mouse = box.mapSceneToView(scene_pos)
        self._on_paint(int(round(mouse.y())), int(round(mouse.x())))


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("s2p Trace Curation")

        self.suite2p_dir: Path | None = None
        self.doc: dict[str, Any] | None = None
        self.stack: BinaryStack | None = None
        self.dirty = False
        self.active_roi_id = 0
        self._meanimge_failed = False
        self._meanimge_busy = False
        self._updating = False
        self._batch_mode = False
        self._batch_roi_ids: list[int] = []
        self._raster_mode = False
        self._raster_batch = False
        self._inspect_focus: str | None = None
        self._tc_norm_stale = False
        self._raster_row_ids: list[int] = []
        self._raster_box_items: list[Any] = []
        self._raster_last_shape: tuple[int, int] | None = None
        self._analysis_window: AnalysisToolsWindow | None = None
        self._trace_proc_window: TraceProcessingWindow | None = None
        self._heatmap_window: HeatmapEditorWindow | None = None
        self._annotation_window: AnnotationEditorWindow | None = None

        self._mask_edit_active = False
        self._mask_edit_kind: str | None = None  # "modify" | "add"
        self._add_mask_draft: dict[str, Any] | None = None
        self._add_mask_center: tuple[float, float] | None = None
        self._add_mask_side = 64
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

        self._ann_selected_ids: set[int] = set()
        self._scale_panel_open = False
        self._ann_span_items: list[pg.LinearRegionItem] = []
        self._trace_x_user_zoomed = False
        self._trace_y_locked = {"f": False, "comp": False, "bleach": False}
        self._scale_nudges: dict[str, dict[str, QGraphicsProxyWidget]] = {}

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
        self._apply_default_window_size()

    def _apply_default_window_size(self) -> None:
        """Size and center the window from the current screen work area."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(2456, 1100)
            return
        avail = screen.availableGeometry()
        w = int(round(avail.width() * _DEFAULT_SCREEN_W_FRAC))
        h = int(round(avail.height() * _DEFAULT_SCREEN_H_FRAC))
        w = max(_MIN_WINDOW_W, min(w, avail.width() - 16))
        h = max(_MIN_WINDOW_H, min(h, avail.height() - 16))
        self.resize(w, h)
        frame = self.frameGeometry()
        frame.moveCenter(avail.center())
        self.move(frame.topLeft())

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
        return isinstance(w, (QAbstractSpinBox, QComboBox, QLineEdit, QToolButton))

    def _roi_step(self, delta: int) -> None:
        if (
            self.doc is None
            or self._focus_owns_arrows_or_space()
            or self._mask_edit_active
            or self._batch_mode
        ):
            return
        if self._raster_mode and self._raster_row_ids:
            ids = self._raster_row_ids
        else:
            ids = self._filtered_roi_ids()
        if not ids:
            return
        try:
            idx = ids.index(self.active_roi_id)
        except ValueError:
            idx = 0 if delta > 0 else len(ids) - 1
            self._select_roi(ids[idx])
            return
        new_idx = idx + int(delta)
        if 0 <= new_idx < len(ids):
            self._select_roi(ids[new_idx])

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

        self.center_split = QSplitter(Qt.Orientation.Vertical)
        self.center_split.setChildrenCollapsible(False)
        center_layout.addWidget(self.center_split, stretch=1)

        top_split = QSplitter(Qt.Orientation.Horizontal)
        self.center_split.addWidget(top_split)

        self.w1 = self._make_image_panel("FOV (W1)", clickable=True)
        self.w2 = self._make_image_panel("Movie (W2)")
        self.w3 = self._make_image_panel("ROI zoom (W3)", paintable=True)
        top_split.addWidget(self.w1)
        top_split.addWidget(self.w2)
        top_split.addWidget(self.w3)
        top_split.setStretchFactor(0, 1)
        top_split.setStretchFactor(1, 1)
        top_split.setStretchFactor(2, 1)

        self.lower_stack = QStackedWidget()
        self.center_split.addWidget(self.lower_stack)
        self.center_split.setStretchFactor(0, 3)
        self.center_split.setStretchFactor(1, 2)

        inspect_panel = QWidget()
        self.inspect_layout = QVBoxLayout(inspect_panel)
        inspect_layout = self.inspect_layout
        inspect_layout.setContentsMargins(0, 0, 0, 0)
        inspect_layout.setSpacing(6)

        # Traces
        self.trace_widget = pg.GraphicsLayoutWidget()
        inspect_layout.addWidget(self.trace_widget, stretch=2)
        self.plot_f = self.trace_widget.addPlot(row=0, col=0, title="F / x·Fneu")
        self.plot_comp = self.trace_widget.addPlot(row=1, col=0, title="trace_comp = F − x·Fneu")
        self.plot_bleach = self.trace_widget.addPlot(row=2, col=0, title="tc_norm_sm / bleach")
        self.plot_comp.setXLink(self.plot_f)
        self.plot_bleach.setXLink(self.plot_f)
        for plot in (self.plot_f, self.plot_comp, self.plot_bleach):
            plot.titleLabel.setToolTip(
                "Click this title to expand the plot; click again to show all"
            )
        # Only the bottom-most visible plot may have a bottom axis. setLabel on
        # the X-linked upper plots makes pyqtgraph paint those axes over the data.
        self._sync_trace_bottom_axes("bleach")
        for plot in (self.plot_f, self.plot_comp, self.plot_bleach):
            plot.setMouseEnabled(x=True, y=True)
            plot.showGrid(x=True, y=False, alpha=0.2)
            plot.hideButtons()
            vb = plot.getViewBox()
            vb.setMouseMode(pg.ViewBox.PanMode)
            vb.sigResized.connect(self._reposition_scale_nudges)
        self.plot_f.sigXRangeChanged.connect(self._on_trace_x_range_changed)
        self.plot_f.sigYRangeChanged.connect(lambda *_: self._on_trace_y_range_changed("f"))
        self.plot_comp.sigYRangeChanged.connect(lambda *_: self._on_trace_y_range_changed("comp"))
        self.plot_bleach.sigYRangeChanged.connect(lambda *_: self._on_trace_y_range_changed("bleach"))
        self.trace_widget.scene().sigMouseClicked.connect(self._on_trace_scene_clicked)
        self._install_scale_nudges()
        self.plot_f.addLegend(offset=(10, 10))
        self.curve_f = self.plot_f.plot(pen=pg.mkPen("#1f77b4", width=1), name="F")
        self.curve_fneu = self.plot_f.plot(pen=pg.mkPen("#ff7f0e", width=1), name="x·Fneu")
        self.curve_comp = self.plot_comp.plot(pen=pg.mkPen("#2ca02c", width=1.5))
        self.plot_bleach.addLegend(offset=(10, 10))
        self.curve_bleach_sm = self.plot_bleach.plot(
            pen=pg.mkPen("#1f77b4", width=1.0), name="tc_norm_sm"
        )
        self.curve_bleach_fit = self.plot_bleach.plot(
            pen=pg.mkPen("#ff7f0e", width=1.0, style=Qt.PenStyle.DashLine), name="fit"
        )
        self.curve_bleach_bc = self.plot_bleach.plot(
            pen=pg.mkPen("#2ca02c", width=1.5), name="tc_norm_sm_bc"
        )

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
        self.thumb_wrap = QWidget()
        thumb_row = QHBoxLayout(self.thumb_wrap)
        thumb_row.setContentsMargins(0, 0, 0, 0)
        inspect_layout.addWidget(self.thumb_wrap)
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
            view.setToolTip(
                "Click to expand the four ROI zooms; click again to show all"
            )
            view.scene.sigMouseClicked.connect(self._on_thumb_clicked)
            box.addWidget(lab)
            box.addWidget(view)
            wrap = QWidget()
            wrap.setLayout(box)
            thumb_row.addWidget(wrap)
            self.thumb_views.append(view)
            self.thumb_labels.append(lab)

        raster_panel = QWidget()
        raster_layout = QVBoxLayout(raster_panel)
        raster_layout.setContentsMargins(0, 0, 0, 0)
        raster_layout.setSpacing(6)

        self.raster_view = ClickableImageView(on_click=self._on_raster_click)
        self.raster_view.ui.histogram.hide()
        self.raster_view.setMinimumHeight(180)
        raster_layout.addWidget(self.raster_view, stretch=3)

        self.plot_raster_trace = pg.PlotWidget(title="tc_norm")
        self.plot_raster_trace.setMinimumHeight(100)
        self.plot_raster_trace.showGrid(x=True, y=False, alpha=0.2)
        self.plot_raster_trace.setLabel("bottom", "frame")
        self.curve_raster_trace = self.plot_raster_trace.plot(
            pen=pg.mkPen("#2ca02c", width=1.5)
        )
        self.raster_c0 = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=True,
            pen=pg.mkPen(C0_COLOR, width=2, style=Qt.PenStyle.SolidLine),
        )
        self.raster_trace_c0 = pg.InfiniteLine(
            pos=0,
            angle=90,
            movable=True,
            pen=pg.mkPen(C0_COLOR, width=2, style=Qt.PenStyle.SolidLine),
        )
        self.raster_view.getView().addItem(self.raster_c0)
        self.raster_c0.setZValue(20)
        self.plot_raster_trace.addItem(self.raster_trace_c0)
        self.raster_c0.sigPositionChanged.connect(self._on_raster_c0_moved)
        self.raster_trace_c0.sigPositionChanged.connect(self._on_raster_c0_moved)
        raster_layout.addWidget(self.plot_raster_trace, stretch=1)

        self.lower_stack.addWidget(inspect_panel)
        self.lower_stack.addWidget(raster_panel)

        root_layout.addWidget(self._build_right_panel(), stretch=0)

    def _make_image_panel(
        self, title: str, clickable: bool = False, paintable: bool = False
    ) -> QWidget:
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        if paintable:
            view = PaintImageView(
                on_paint=self._on_w3_paint, on_wheel=self._on_w3_wheel
            )
        elif clickable:
            view = ClickableImageView(
                on_click=self._on_fov_click, on_lasso=self._on_fov_lasso
            )
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
        for key in ("meanImg", "meanImgE", "VCorr"):
            self.cmb_fov_src.addItem(key, key)
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
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Single"))
        self.slider_mode = QSlider(Qt.Orientation.Horizontal)
        self.slider_mode.setMinimum(0)
        self.slider_mode.setMaximum(1)
        self.slider_mode.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_mode.setTickInterval(1)
        self.slider_mode.setSingleStep(1)
        self.slider_mode.setPageStep(1)
        self.slider_mode.setFixedWidth(80)
        self.slider_mode.setToolTip("Single ROI mode ↔ Batch lasso mode")
        mode_row.addWidget(self.slider_mode)
        mode_row.addWidget(QLabel("Batch"))
        mode_row.addStretch(1)
        self.spin_x = QDoubleSpinBox()
        self.spin_x.setRange(-5.0, 5.0)
        self.spin_x.setSingleStep(0.05)
        self.spin_x.setDecimals(3)
        self.spin_x.setValue(1.0)
        roi_form.addRow("ROI #", roi_nav)
        roi_form.addRow("Mode", mode_row)
        roi_form.addRow(self.chk_iscell)
        roi_form.addRow("x (F−x·Fneu)", self.spin_x)
        layout.addWidget(roi)

        raster = QGroupBox("Raster Tools")
        raster_form = QFormLayout(raster)
        raster_on_row = QHBoxLayout()
        raster_on_row.addWidget(QLabel("OFF"))
        self.slider_raster_on = QSlider(Qt.Orientation.Horizontal)
        self.slider_raster_on.setMinimum(0)
        self.slider_raster_on.setMaximum(1)
        self.slider_raster_on.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_raster_on.setTickInterval(1)
        self.slider_raster_on.setSingleStep(1)
        self.slider_raster_on.setPageStep(1)
        self.slider_raster_on.setFixedWidth(80)
        self.slider_raster_on.setEnabled(False)
        self.slider_raster_on.setToolTip(
            "OFF: traces + C1–C4 zooms. ON: raster plot + example / Co-Activity trace."
        )
        raster_on_row.addWidget(self.slider_raster_on)
        raster_on_row.addWidget(QLabel("ON"))
        raster_on_row.addStretch(1)

        self.cmb_raster_show = QComboBox()
        for key, label in FILTER_LABELS.items():
            self.cmb_raster_show.addItem(label, key)
        self.cmb_raster_show.setCurrentIndex(self.cmb_overlay.currentIndex())
        self.cmb_raster_show.setEnabled(False)

        self.cmb_raster_sort = QComboBox()
        self.cmb_raster_sort.setEnabled(False)
        self.cmb_raster_sort.setToolTip(
            "Pickle: doc / roi_id order. Other entries: saved Analysis Tools runs."
        )
        self.cmb_raster_sort.addItem("Pickle", PICKLE_SORT_ID)

        self.cmb_raster_trace = QComboBox()
        self.cmb_raster_trace.setEnabled(False)
        self.cmb_raster_trace.setToolTip(
            "Which stored [0, 1] trace fills the raster (independent of HAC runs)"
        )
        for field in TRACE_FIELDS:
            self.cmb_raster_trace.addItem(TRACE_FIELD_LABELS[field], field)

        self.cmb_raster_lut = QComboBox()
        self.cmb_raster_lut.addItems(list(LUT_NAMES))
        self.cmb_raster_lut.setEnabled(False)

        self.chk_raster_revert = QCheckBox("Revert LUT")
        self.chk_raster_revert.setEnabled(False)
        self.chk_raster_revert.setToolTip("Flip the 0↔1 mapping of the current LUT")

        raster_mode_row = QHBoxLayout()
        raster_mode_row.addWidget(QLabel("Single"))
        self.slider_raster_batch = QSlider(Qt.Orientation.Horizontal)
        self.slider_raster_batch.setMinimum(0)
        self.slider_raster_batch.setMaximum(1)
        self.slider_raster_batch.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_raster_batch.setTickInterval(1)
        self.slider_raster_batch.setSingleStep(1)
        self.slider_raster_batch.setPageStep(1)
        self.slider_raster_batch.setFixedWidth(80)
        self.slider_raster_batch.setEnabled(False)
        self.slider_raster_batch.setToolTip(
            "Single: selected ROI's tc_norm. Batch: mean of rows in the raster (Co-Activity)."
        )
        raster_mode_row.addWidget(self.slider_raster_batch)
        raster_mode_row.addWidget(QLabel("Batch"))
        raster_mode_row.addStretch(1)

        self.btn_rebuild_tc_norm = QPushButton("Rebuild tc_norm")
        self.btn_rebuild_tc_norm.setEnabled(False)
        self.btn_rebuild_tc_norm.setToolTip(
            "Recompute stored [0, 1] traces from current trace_comp and LED+Shutter ranges"
        )

        self.btn_analysis_tools = QPushButton("Analysis Tools")
        self.btn_analysis_tools.setEnabled(False)
        self.btn_analysis_tools.setToolTip(
            "Open clustering / sort runs (params + order stored in the pickle)"
        )
        self.btn_analysis_tools.clicked.connect(self._open_analysis_tools)

        self.btn_edit_heatmaps = QPushButton("Edit HeatMaps")
        self.btn_edit_heatmaps.setEnabled(False)
        self.btn_edit_heatmaps.setToolTip(
            "Named FOV maps from data.bin; appear as HeatMap: <name> in Image dropdowns"
        )
        self.btn_edit_heatmaps.clicked.connect(self._open_heatmap_editor)

        raster_form.addRow("Raster-mode", raster_on_row)
        raster_form.addRow("Show", self.cmb_raster_show)
        raster_form.addRow("Sort", self.cmb_raster_sort)
        raster_form.addRow("Trace", self.cmb_raster_trace)
        raster_form.addRow("LUT", self.cmb_raster_lut)
        raster_form.addRow(self.chk_raster_revert)
        raster_form.addRow("Mode", raster_mode_row)
        raster_form.addRow(self.btn_rebuild_tc_norm)
        raster_form.addRow(self.btn_analysis_tools)
        raster_form.addRow(self.btn_edit_heatmaps)
        layout.addWidget(raster)
        layout.addStretch(1)

        # connections
        self.cmb_fov_src.currentIndexChanged.connect(self._on_fov_src_changed)
        self.cmb_fov_lut.currentIndexChanged.connect(self._refresh_fov)
        self.spin_fov_lo.valueChanged.connect(self._refresh_fov)
        self.spin_fov_hi.valueChanged.connect(self._refresh_fov)
        self.cmb_overlay.currentIndexChanged.connect(self._on_overlay_combo)
        self.cmb_mov_lut.currentIndexChanged.connect(self._refresh_movie_views)
        self.spin_mov_lo.valueChanged.connect(self._refresh_movie_views)
        self.spin_mov_hi.valueChanged.connect(self._refresh_movie_views)
        self.spin_roi.valueChanged.connect(self._on_roi_spin)
        self.btn_roi_up.clicked.connect(lambda: self._roi_step(+1))
        self.btn_roi_down.clicked.connect(lambda: self._roi_step(-1))
        self.chk_iscell.toggled.connect(self._on_iscell_toggled)
        self.spin_x.valueChanged.connect(self._on_x_changed)
        self.slider_mode.valueChanged.connect(self._on_mode_slider)
        self.slider_raster_on.valueChanged.connect(self._on_raster_mode_slider)
        self.cmb_raster_show.currentIndexChanged.connect(self._on_overlay_combo)
        self.cmb_raster_lut.currentIndexChanged.connect(self._on_raster_display_changed)
        self.chk_raster_revert.toggled.connect(self._on_raster_display_changed)
        self.slider_raster_batch.valueChanged.connect(self._on_raster_batch_slider)
        self.cmb_raster_sort.currentIndexChanged.connect(self._on_raster_sort_changed)
        self.cmb_raster_trace.currentIndexChanged.connect(self._on_raster_trace_changed)
        self.btn_rebuild_tc_norm.clicked.connect(self._rebuild_tc_norm)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(260)
        panel.setMaximumWidth(320)
        layout = QVBoxLayout(panel)

        zoom = QGroupBox("ROI zoom (W3)")
        zoom_form = QFormLayout(zoom)
        self.cmb_w3_src = QComboBox()
        for key in ("movie", "meanImg", "meanImgE", "VCorr"):
            self.cmb_w3_src.addItem(key, key)
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

        self.btn_add_mask = QPushButton("Add Mask")
        self.btn_add_mask.setEnabled(False)
        self.btn_add_mask.setToolTip("Create a new ROI: pick center on W1, paint in W3")
        self.btn_add_mask.clicked.connect(self._start_add_mask)
        actions_layout.addWidget(self.btn_add_mask)

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

        zoom_row = QHBoxLayout()
        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_out.setFixedWidth(32)
        self.btn_zoom_out.setToolTip("Zoom out (larger W3 crop)")
        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(32)
        self.btn_zoom_in.setToolTip("Zoom in (smaller W3 crop)")
        self.spin_add_side = QSpinBox()
        self.spin_add_side.setRange(8, 2048)
        self.spin_add_side.setValue(64)
        self.spin_add_side.setToolTip("W3 square side (pixels); wheel also zooms")
        self.lbl_zoom_side = QLabel("Zoom side")
        zoom_row.addWidget(self.lbl_zoom_side)
        zoom_row.addWidget(self.btn_zoom_out)
        zoom_row.addWidget(self.spin_add_side)
        zoom_row.addWidget(self.btn_zoom_in)
        edit_layout.addLayout(zoom_row)
        self.btn_zoom_out.clicked.connect(lambda: self._nudge_add_zoom(+8))
        self.btn_zoom_in.clicked.connect(lambda: self._nudge_add_zoom(-8))
        self.spin_add_side.valueChanged.connect(self._on_add_side_changed)
        for w in (
            self.lbl_zoom_side,
            self.spin_add_side,
            self.btn_zoom_in,
            self.btn_zoom_out,
        ):
            w.setVisible(False)

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
        self.btn_apply_mask.clicked.connect(self._finish_mask_edit)
        edit_layout.addWidget(self.btn_apply_mask)

        self.btn_cancel_mask = QPushButton("Cancel")
        self.btn_cancel_mask.clicked.connect(self._cancel_mask_edit)
        edit_layout.addWidget(self.btn_cancel_mask)

        self.mask_edit_panel.setVisible(False)
        actions_layout.addWidget(self.mask_edit_panel)
        actions_layout.addStretch(1)
        layout.addWidget(actions)

        # Annotation tools
        ann = QGroupBox("Annotation tools")
        ann_outer = QVBoxLayout(ann)
        self.btn_ann_tools = QPushButton("Annotation Tools")
        self.btn_ann_tools.setEnabled(False)
        self.btn_ann_tools.setToolTip(
            "Set annotation ranges on a raster/trace, same as HeatMaps"
        )
        self.btn_ann_tools.clicked.connect(self._open_annotation_editor)
        ann_outer.addWidget(self.btn_ann_tools)
        layout.addWidget(ann)

        # Scaling tools (fold-out, same pattern as Annotation tools)
        scale = QGroupBox("Scaling tools")
        scale_outer = QVBoxLayout(scale)
        self.btn_scale_tools = QPushButton("Scaling Tools")
        self.btn_scale_tools.setEnabled(False)
        self.btn_scale_tools.setToolTip("Trace X/Y limits, box zoom, and Reset")
        self.btn_scale_tools.clicked.connect(self._toggle_scale_panel)
        scale_outer.addWidget(self.btn_scale_tools)

        self.scale_panel = QWidget()
        scale_layout = QVBoxLayout(self.scale_panel)
        scale_layout.setContentsMargins(0, 0, 0, 0)

        self.chk_box_zoom = QCheckBox("Box zoom")
        self.chk_box_zoom.setToolTip(
            "On: left-drag a rectangle to zoom (shared X, that plot's Y).\n"
            "Off: left-drag pans."
        )
        self.chk_box_zoom.toggled.connect(self._on_box_zoom_toggled)
        scale_layout.addWidget(self.chk_box_zoom)

        self.btn_reset_all_traces = QPushButton("Reset all")
        self.btn_reset_all_traces.setToolTip(
            "Restore original autoscale: full time axis and Y on all traces"
        )
        self.btn_reset_all_traces.clicked.connect(self._reset_all_trace_scales)
        scale_layout.addWidget(self.btn_reset_all_traces)

        hint = QLabel(
            "Reset returns that axis to autoscale. Double-click a plot to Reset Y."
        )
        hint.setWordWrap(True)
        scale_layout.addWidget(hint)

        self._trace_y_spins: dict[str, tuple[QDoubleSpinBox, QDoubleSpinBox]] = {}

        def _limit_spin(decimals: int = 2) -> QDoubleSpinBox:
            spin = QDoubleSpinBox()
            spin.setRange(-1e12, 1e12)
            spin.setDecimals(decimals)
            spin.setMaximumWidth(90)
            spin.setKeyboardTracking(False)
            return spin

        x_row = QHBoxLayout()
        x_row.addWidget(QLabel("Time (X)"))
        self.spin_x_lo = _limit_spin(1)
        self.spin_x_hi = _limit_spin(1)
        self.spin_x_lo.setToolTip("X lower (frame)")
        self.spin_x_hi.setToolTip("X upper (frame)")
        self.spin_x_lo.valueChanged.connect(self._apply_trace_x_spins)
        self.spin_x_hi.valueChanged.connect(self._apply_trace_x_spins)
        btn_reset_x = QPushButton("Reset")
        btn_reset_x.setFixedWidth(52)
        btn_reset_x.setToolTip("Autoscale X to the full recording (plots stay linked)")
        btn_reset_x.clicked.connect(self._fit_trace_x)
        x_row.addWidget(self.spin_x_lo)
        x_row.addWidget(self.spin_x_hi)
        x_row.addWidget(btn_reset_x)
        scale_layout.addLayout(x_row)

        for key, label in (
            ("f", "F / Fneu"),
            ("comp", "trace_comp"),
            ("bleach", "bleach"),
        ):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            spin_lo = _limit_spin()
            spin_hi = _limit_spin()
            spin_lo.setToolTip(f"{label}: Y lower")
            spin_hi.setToolTip(f"{label}: Y upper")
            btn_reset = QPushButton("Reset")
            btn_reset.setFixedWidth(52)
            btn_reset.setToolTip(f"Autoscale Y for {label} only")
            row.addWidget(spin_lo)
            row.addWidget(spin_hi)
            row.addWidget(btn_reset)
            scale_layout.addLayout(row)
            self._trace_y_spins[key] = (spin_lo, spin_hi)
            spin_lo.valueChanged.connect(lambda _=0.0, k=key: self._apply_trace_y_spins(k))
            spin_hi.valueChanged.connect(lambda _=0.0, k=key: self._apply_trace_y_spins(k))
            btn_reset.clicked.connect(lambda _=False, k=key: self._fit_trace_y(k))

        self.scale_panel.setVisible(False)
        scale_outer.addWidget(self.scale_panel)
        layout.addWidget(scale)

        proc = QGroupBox("Trace processing")
        proc_outer = QVBoxLayout(proc)
        self.btn_trace_proc = QPushButton("Trace Processing")
        self.btn_trace_proc.setEnabled(False)
        self.btn_trace_proc.setToolTip("Savitzky–Golay smoothing and bleach correction")
        self.btn_trace_proc.clicked.connect(self._open_trace_processing)
        proc_outer.addWidget(self.btn_trace_proc)
        layout.addWidget(proc)

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
                idx = self.cmb_fov_src.findData(str(src))
                if idx < 0:
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
                    idx2 = self.cmb_raster_show.findData(str(filt))
                    if idx2 >= 0:
                        self.cmb_raster_show.setCurrentIndex(idx2)
            if lut := s.get("mov_lut"):
                idx = self.cmb_mov_lut.findText(str(lut))
                if idx >= 0:
                    self.cmb_mov_lut.setCurrentIndex(idx)
            if src := s.get("w3_src"):
                idx = self.cmb_w3_src.findData(str(src))
                if idx < 0:
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
            "fov_src": self.cmb_fov_src.currentData() or self.cmb_fov_src.currentText(),
            "fov_lut": self.cmb_fov_lut.currentText(),
            "overlay_filter": self.cmb_overlay.currentData(),
            "mov_lut": self.cmb_mov_lut.currentText(),
            "w3_src": self.cmb_w3_src.currentData() or self.cmb_w3_src.currentText(),
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
        self._meanimge_failed = False
        self.stack = BinaryStack(plane_dir(suite2p_dir))
        n = len(doc["rois"])
        self._updating = True
        self.spin_roi.setMaximum(max(0, n - 1))
        self.spin_roi.setValue(0)
        self.active_roi_id = 0
        self._init_display_levels()
        self._init_cursors()
        self._updating = False
        self._raster_mode = False
        self._raster_batch = False
        self._raster_last_shape = None
        self._updating = True
        self.slider_raster_on.setValue(0)
        self.slider_raster_batch.setValue(0)
        self._updating = False
        self.lower_stack.setCurrentIndex(0)
        self._select_roi(0, force=True)
        self._ensure_active_roi_in_filter()
        self._persist_ui_settings(suite2p_dir=suite2p_dir)
        self.btn_modify.setEnabled(True)
        self.btn_add_mask.setEnabled(True)
        self.btn_ann_tools.setEnabled(True)
        self.btn_scale_tools.setEnabled(True)
        self.btn_trace_proc.setEnabled(True)
        self.btn_save.setEnabled(True)
        ensure_annotations(doc)
        self._ann_selected_ids = set()
        self._batch_mode = False
        self._batch_roi_ids = []
        self._updating = True
        self.slider_mode.setValue(0)
        self._updating = False
        w1: ClickableImageView = self.w1.image_view  # type: ignore[attr-defined]
        w1.lasso_enabled = False
        w1.clear_lasso_drawing()
        self._apply_mode_controls()
        self._tc_norm_stale = tc_norm_is_stale(doc)
        ensure_analyses(doc)
        ensure_trace_processing(doc)
        ensure_heatmaps(doc)
        refresh_stale_flags(doc)
        self._fill_image_source_combos()
        self._fill_raster_sort_combo()
        self._fill_raster_trace_combo()
        self._update_rebuild_button()
        self._set_raster_controls_enabled(True)
        if self._analysis_window is not None:
            self._analysis_window.refresh_from_doc()
        if self._trace_proc_window is not None:
            self._trace_proc_window.refresh_from_doc()
        if self._heatmap_window is not None:
            self._heatmap_window.refresh_from_doc()
        if self._annotation_window is not None:
            self._annotation_window.refresh_from_doc()
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
                "Finish with Apply/Save Mask or Cancel before saving.",
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
                "Finish with Apply/Save Mask or Cancel before resetting the ROI.",
            )
            return
        reset_roi_from_suite2p(self.doc, self.suite2p_dir, self.active_roi_id)
        self.dirty = True
        self._mark_tc_norm_stale()
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
        can_start = not active and self.doc is not None and not self._batch_mode
        self.btn_modify.setEnabled(can_start)
        self.btn_add_mask.setEnabled(can_start)
        view: PaintImageView = self.w3.image_view  # type: ignore[attr-defined]
        paint_ok = active and (
            self._mask_edit_kind == "modify"
            or (self._mask_edit_kind == "add" and self._add_mask_center is not None)
        )
        view.paint_enabled = paint_ok
        single_ok = not active and not self._batch_mode
        self.spin_roi.setEnabled(single_ok)
        self.btn_roi_up.setEnabled(single_ok)
        self.btn_roi_down.setEnabled(single_ok)
        self.chk_iscell.setEnabled(not active and not self._batch_mode)
        self.spin_x.setEnabled(single_ok)
        self.slider_mode.setEnabled(
            not active and self.doc is not None and not self._raster_mode
        )
        self.slider_raster_on.setEnabled(not active and self.doc is not None)
        is_add = self._mask_edit_kind == "add"
        for w in (
            self.lbl_zoom_side,
            self.spin_add_side,
            self.btn_zoom_in,
            self.btn_zoom_out,
        ):
            w.setVisible(bool(active and is_add))
        if active and is_add:
            self.btn_apply_mask.setText("Save Mask")
        else:
            self.btn_apply_mask.setText("Apply Mask")

    def _on_mask_mode_toggled(self, checked: bool) -> None:
        if not checked:
            return
        for mode, rb in self.mask_mode_buttons.items():
            if rb.isChecked():
                self._mask_edit_mode = mode
                break

    def _start_mask_edit(self) -> None:
        if self.doc is None or self._mask_edit_active or self._batch_mode:
            return
        self._mask_edit_kind = "modify"
        self._add_mask_draft = None
        self._add_mask_center = None
        self._mask_edit_snapshot = deepcopy(self._row())
        self._mask_traces_stale = False
        self._mask_roi_changed = False
        self._mask_neu_changed = False
        self._set_mask_edit_ui(True)
        self.statusBar().showMessage(
            f"Modify Mask: ROI {self.active_roi_id} — paint in W3, then Apply or Cancel"
        )

    def _start_add_mask(self) -> None:
        if self.doc is None or self._mask_edit_active or self._batch_mode:
            return
        Ly = int(self.doc["meta"]["Ly"])
        Lx = int(self.doc["meta"]["Lx"])
        side = median_zoom_side(self.doc["rois"], Ly, Lx)
        self._mask_edit_kind = "add"
        self._add_mask_center = None
        self._add_mask_side = side
        self._add_mask_draft = empty_roi_draft(
            next_roi_id(self.doc), int(self.doc["meta"]["nframes"])
        )
        self._mask_edit_snapshot = None
        self._mask_traces_stale = False
        self._mask_roi_changed = False
        self._mask_neu_changed = False
        self._updating = True
        self.spin_add_side.setMaximum(min(Ly, Lx))
        self.spin_add_side.setValue(side)
        self._updating = False
        self._set_mask_edit_ui(True)
        view: PaintImageView = self.w3.image_view  # type: ignore[attr-defined]
        view.paint_enabled = False
        self.statusBar().showMessage(
            "Add Mask: click W1 to place the red cross (click again to move before painting)"
        )
        self._refresh_fov()
        self._refresh_w3()

    def _nudge_add_zoom(self, delta: int) -> None:
        if self._mask_edit_kind != "add" or self.doc is None:
            return
        Ly = int(self.doc["meta"]["Ly"])
        Lx = int(self.doc["meta"]["Lx"])
        new_side = int(np.clip(self._add_mask_side + delta, 8, min(Ly, Lx)))
        self.spin_add_side.setValue(new_side)

    def _on_add_side_changed(self, value: int) -> None:
        if self._updating or self._mask_edit_kind != "add":
            return
        self._add_mask_side = int(value)
        if self._add_mask_center is not None:
            self._refresh_w3()
            self._refresh_fov()

    def _on_w3_wheel(self, direction: int) -> None:
        if self._mask_edit_kind != "add" or not self._mask_edit_active:
            return
        self._nudge_add_zoom(-8 if direction > 0 else 8)

    def _cancel_mask_edit(self) -> None:
        if not self._mask_edit_active:
            return
        if self._extracting:
            self._extract_cancel = True
            return
        if self._mask_edit_kind == "modify" and self._mask_edit_snapshot is not None:
            if self.doc is not None:
                snap = self._mask_edit_snapshot
                for i, r in enumerate(self.doc["rois"]):
                    if int(r["roi_id"]) == int(self.active_roi_id):
                        self.doc["rois"][i] = snap
                        break
        self._mask_edit_snapshot = None
        self._add_mask_draft = None
        self._add_mask_center = None
        self._mask_edit_kind = None
        self._mask_traces_stale = False
        self._mask_roi_changed = False
        self._mask_neu_changed = False
        self._set_mask_edit_ui(False)
        self._refresh_all()
        self.statusBar().showMessage("Mask edit cancelled")

    def _finish_mask_edit(self) -> None:
        if self._mask_edit_kind == "add":
            self._save_new_mask()
        else:
            self._apply_mask_edit()

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
        self._mask_edit_kind = None
        self._mask_traces_stale = False
        self._mask_roi_changed = False
        self._mask_neu_changed = False
        self._set_mask_edit_ui(False)
        self._refresh_all()
        self.statusBar().showMessage(f"Applied mask edits and saved {path}")

    def _save_new_mask(self) -> None:
        if (
            not self._mask_edit_active
            or self._mask_edit_kind != "add"
            or self.doc is None
            or self.suite2p_dir is None
            or self._add_mask_draft is None
        ):
            return
        if self._extracting:
            return
        if self._add_mask_center is None:
            QMessageBox.warning(self, "Save Mask", "Click W1 to place the ROI center first.")
            return
        row = self._add_mask_draft
        if len(row["roi"]["ypix"]) == 0 or len(row["neuropil"]["ipix"]) == 0:
            QMessageBox.warning(
                self,
                "Save Mask",
                "Both F-ROI and Fneu-ROI must be non-empty before saving.",
            )
            return
        self._mask_roi_changed = True
        self._mask_neu_changed = True
        self._mask_traces_stale = True
        ok = self._recalculate_traces()
        if not ok:
            return
        append_roi(self.doc, row)
        new_id = int(row["roi_id"])
        if any(r.get("tc_norm") is not None for r in self.doc["rois"]):
            was_stale = self._tc_norm_stale or tc_norm_is_stale(self.doc)
            fill_missing_tc_norm(self.doc)
            if was_stale:
                self._tc_norm_stale = True
            else:
                self.doc.setdefault("meta", {})["tc_norm_sig"] = tc_norm_sig(self.doc)
                self._tc_norm_stale = False
            self._update_rebuild_button()
        self._add_mask_draft = None
        self._add_mask_center = None
        self._mask_edit_kind = None
        self._mask_traces_stale = False
        self._mask_roi_changed = False
        self._mask_neu_changed = False
        path = save_curation(self.doc, self.suite2p_dir)
        self.dirty = False
        self._set_mask_edit_ui(False)
        n = len(self.doc["rois"])
        self.spin_roi.setMaximum(max(0, n - 1))
        self._select_roi(new_id, force=True)
        self._refresh_analysis_stale_ui()
        self.statusBar().showMessage(f"Saved new ROI {new_id} to {path}")

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
            if self._mask_edit_kind != "add":
                self._mark_tc_norm_stale()
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
        if not self._mask_edit_active or self._extracting or self.doc is None:
            return
        if self._mask_edit_kind == "add" and self._add_mask_center is None:
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
        if (
            roi_id is None
            and self._mask_edit_kind == "add"
            and self._add_mask_draft is not None
        ):
            return self._add_mask_draft
        rid = self.active_roi_id if roi_id is None else roi_id
        for row in self.doc["rois"]:
            if int(row["roi_id"]) == int(rid):
                return row
        raise KeyError(rid)

    def _overlay_filter(self) -> OverlayFilter:
        return self.cmb_overlay.currentData()  # type: ignore[return-value]

    def _filtered_roi_ids(self) -> list[int]:
        if self.doc is None:
            return []
        return [int(r["roi_id"]) for r in rois_for_raster(self.doc["rois"], self._overlay_filter())]

    def _on_overlay_combo(self) -> None:
        if self._updating:
            return
        sender = self.sender()
        key = None
        if sender is self.cmb_overlay or sender is self.cmb_raster_show:
            key = sender.currentData()  # type: ignore[union-attr]
        if key is None:
            key = self.cmb_overlay.currentData()
        self._updating = True
        for cmb in (self.cmb_overlay, self.cmb_raster_show):
            idx = cmb.findData(key)
            if idx >= 0 and cmb.currentIndex() != idx:
                cmb.setCurrentIndex(idx)
        self._updating = False
        jumped = self._ensure_active_roi_in_filter()
        if not jumped:
            self._refresh_fov()
            if self._raster_mode:
                self._refresh_raster()
        self._notify_heatmap_raster()

    def _ensure_active_roi_in_filter(self) -> bool:
        """If the active ROI is hidden by Show ROIs, jump to the nearest visible one."""
        if self.doc is None or self._mask_edit_active or self._batch_mode:
            return False
        ids = self._filtered_roi_ids()
        if not ids:
            return False
        if self.active_roi_id in ids:
            return False
        arr = np.asarray(ids, dtype=np.int64)
        nearest = int(arr[int(np.argmin(np.abs(arr - self.active_roi_id)))])
        self._select_roi(nearest)
        return True

    def _on_roi_spin(self, value: int) -> None:
        if self._updating or self.doc is None or self._mask_edit_active or self._batch_mode:
            return
        ids = self._filtered_roi_ids()
        if value in ids or not ids:
            self._select_roi(value)
            return
        prev = self.active_roi_id
        direction = 1 if value > prev else -1
        if direction > 0:
            candidates = [i for i in ids if i > prev]
            nxt = min(candidates) if candidates else prev
        else:
            candidates = [i for i in ids if i < prev]
            nxt = max(candidates) if candidates else prev
        self._updating = True
        self.spin_roi.setValue(nxt)
        self._updating = False
        if nxt != self.active_roi_id:
            self._select_roi(nxt)

    def _select_roi(self, roi_id: int, force: bool = False) -> None:
        if self.doc is None:
            return
        if self._mask_edit_active and not force:
            return
        if self._batch_mode and not force:
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
        if self._trace_proc_window is not None:
            self._trace_proc_window._refresh_preview()
        if self._heatmap_window is not None:
            self._heatmap_window.on_active_roi_changed()
        if self._annotation_window is not None:
            self._annotation_window.on_active_roi_changed()

    def _on_fov_click(self, y: int, x: int) -> None:
        if self.doc is None or self._batch_mode:
            return
        Ly = int(self.doc["meta"]["Ly"])
        Lx = int(self.doc["meta"]["Lx"])
        if y < 0 or x < 0 or y >= Ly or x >= Lx:
            return

        # Add Mask: place / move center (only before painting starts)
        if self._mask_edit_active and self._mask_edit_kind == "add":
            painted = self._mask_roi_changed or self._mask_neu_changed
            if painted:
                self.statusBar().showMessage(
                    "Center locked after painting — Cancel to start over"
                )
                return
            self._add_mask_center = (float(y), float(x))
            view: PaintImageView = self.w3.image_view  # type: ignore[attr-defined]
            view.paint_enabled = True
            self._refresh_fov()
            self._refresh_w3()
            self.statusBar().showMessage(
                f"Add Mask center at (y={y}, x={x}) — paint in W3, then Save Mask"
            )
            return

        if self._mask_edit_active:
            return
        hits = rois_at_pixel(self.doc["rois"], y, x, self._overlay_filter())
        if hits:
            self._select_roi(int(hits[0]["roi_id"]))

    def _on_mode_slider(self, value: int) -> None:
        if self._updating:
            return
        if self._raster_mode and int(value) == 1:
            self._updating = True
            self.slider_mode.setValue(0)
            self._updating = False
            return
        batch = int(value) == 1
        if batch == self._batch_mode:
            return
        if batch and self._mask_edit_active:
            self._cancel_mask_edit()
        self._batch_mode = batch
        self._batch_roi_ids = []
        w1: ClickableImageView = self.w1.image_view  # type: ignore[attr-defined]
        w1.lasso_enabled = batch
        w1.clear_lasso_drawing()
        self._apply_mode_controls()
        if batch:
            self.statusBar().showMessage(
                "Batch mode: draw a closed lasso on W1 (visible ROIs only, >50% pixels)"
            )
            self._updating = True
            self.spin_x.setValue(1.0)
            self.chk_iscell.setChecked(True)
            self._updating = False
            self._refresh_all()
        else:
            self.statusBar().showMessage("Single ROI mode")
            self._select_roi(self.active_roi_id, force=True)

    def _apply_mode_controls(self) -> None:
        single = not self._batch_mode and not self._mask_edit_active
        self.spin_roi.setEnabled(single)
        self.btn_roi_up.setEnabled(single)
        self.btn_roi_down.setEnabled(single)
        self.spin_x.setEnabled(single)
        self.btn_modify.setEnabled(single and self.doc is not None)
        self.btn_add_mask.setEnabled(single and self.doc is not None)
        self.chk_iscell.setEnabled(not self._mask_edit_active and self.doc is not None)
        self.slider_mode.setEnabled(
            not self._mask_edit_active and self.doc is not None and not self._raster_mode
        )
        # Grey out W3 chrome in batch mode
        self.w3.setEnabled(not self._batch_mode)
        for w in (
            self.cmb_w3_src,
            self.cmb_w3_lut,
            self.spin_w3_lo,
            self.spin_w3_hi,
        ):
            w.setEnabled(not self._batch_mode)

    def _on_fov_lasso(self, points_yx: list[tuple[float, float]]) -> None:
        if not self._batch_mode or self.doc is None:
            return
        ids = rois_in_lasso(
            self.doc["rois"], points_yx, self._overlay_filter(), min_fraction=0.5
        )
        self._batch_roi_ids = ids
        # Snap all selected ROIs to iscell=True (default on)
        for row in self.doc["rois"]:
            if int(row["roi_id"]) in ids:
                row["iscell"] = True
        self.dirty = True
        self._updating = True
        self.chk_iscell.setChecked(True)
        self.spin_x.setValue(1.0)
        self._updating = False
        self._refresh_all()
        self._refresh_analysis_stale_ui()
        self.statusBar().showMessage(
            f"Batch: {len(ids)} ROI(s) — iscell ON for all; uncheck to clear them"
        )

    def _on_iscell_toggled(self, checked: bool) -> None:
        if self._updating or self.doc is None or self._mask_edit_active:
            return
        if self._batch_mode:
            if not self._batch_roi_ids:
                return
            id_set = set(self._batch_roi_ids)
            for row in self.doc["rois"]:
                if int(row["roi_id"]) in id_set:
                    row["iscell"] = bool(checked)
            self.dirty = True
            self._refresh_fov()
            self._refresh_movie_views()
            self._refresh_analysis_stale_ui()
            return
        self._row()["iscell"] = bool(checked)
        self.dirty = True
        self._refresh_analysis_stale_ui()
        jumped = self._ensure_active_roi_in_filter()
        if not jumped:
            self._refresh_fov()
            if self._raster_mode:
                self._refresh_raster()

    def _on_x_changed(self, value: float) -> None:
        if self._updating or self.doc is None or self._batch_mode:
            return
        set_compensation_x(self._row(), float(value))
        self.dirty = True
        self._mark_tc_norm_stale()
        self._refresh_traces(autoscale=True)

    def _set_raster_controls_enabled(self, enabled: bool) -> None:
        self.slider_raster_on.setEnabled(enabled and not self._mask_edit_active)
        self.cmb_raster_show.setEnabled(enabled)
        self.cmb_raster_lut.setEnabled(enabled)
        self.chk_raster_revert.setEnabled(enabled)
        self.slider_raster_batch.setEnabled(enabled)
        self.cmb_raster_sort.setEnabled(enabled)
        self.cmb_raster_trace.setEnabled(enabled)
        self.btn_analysis_tools.setEnabled(enabled)
        self.btn_edit_heatmaps.setEnabled(enabled)
        self._update_rebuild_button()

    def _update_rebuild_button(self) -> None:
        self.btn_rebuild_tc_norm.setEnabled(
            self.doc is not None and self._tc_norm_stale
        )

    def _refresh_analysis_stale_ui(self) -> None:
        if self.doc is None:
            return
        refresh_stale_flags(self.doc)
        self._fill_raster_sort_combo()
        if self._analysis_window is not None:
            self._analysis_window.reload_run_list()
        if self._trace_proc_window is not None:
            self._trace_proc_window.refresh_from_doc()

    def _fill_raster_sort_combo(self) -> None:
        if self.doc is None:
            return
        current = raster_sort_id(self.doc)
        was_updating = self._updating
        self._updating = True
        self.cmb_raster_sort.clear()
        self.cmb_raster_sort.addItem("Pickle", PICKLE_SORT_ID)
        for run in ensure_analyses(self.doc):
            self.cmb_raster_sort.addItem(dropdown_label(run), str(run["id"]))
        idx = self.cmb_raster_sort.findData(current)
        self.cmb_raster_sort.setCurrentIndex(idx if idx >= 0 else 0)
        self._updating = was_updating

    def _on_raster_sort_changed(self) -> None:
        if self._updating or self.doc is None:
            return
        data = self.cmb_raster_sort.currentData()
        sid = str(data) if data else PICKLE_SORT_ID
        if raster_sort_id(self.doc) == sid:
            return
        set_raster_sort(self.doc, sid)
        self.dirty = True
        if self._raster_mode:
            self._refresh_raster()
        self._notify_heatmap_raster()

    def _fill_raster_trace_combo(self) -> None:
        if self.doc is None:
            return
        current = raster_trace_field(self.doc)
        was = self._updating
        self._updating = True
        idx = self.cmb_raster_trace.findData(current)
        self.cmb_raster_trace.setCurrentIndex(idx if idx >= 0 else 0)
        self._updating = was

    def _on_raster_trace_changed(self) -> None:
        if self._updating or self.doc is None:
            return
        data = self.cmb_raster_trace.currentData()
        field = str(data) if data else TRACE_FIELD_NORM
        if raster_trace_field(self.doc) == field:
            return
        if not self._ensure_analysis_inputs(field):
            self._fill_raster_trace_combo()
            return
        set_raster_trace_field(self.doc, field)
        self.dirty = True
        if self._raster_mode:
            self._refresh_raster()
        self._notify_heatmap_raster()

    def _open_analysis_tools(self) -> None:
        if self.doc is None:
            return
        if self._analysis_window is None:
            self._analysis_window = AnalysisToolsWindow(self)

            def _clear(_obj: object = None) -> None:
                self._analysis_window = None

            self._analysis_window.destroyed.connect(_clear)
        self._analysis_window.refresh_from_doc()
        self._analysis_window.show()
        self._analysis_window.raise_()
        self._analysis_window.activateWindow()

    def _open_trace_processing(self) -> None:
        if self.doc is None:
            return
        if self._trace_proc_window is None:
            self._trace_proc_window = TraceProcessingWindow(self)

            def _clear(_obj: object = None) -> None:
                self._trace_proc_window = None

            self._trace_proc_window.destroyed.connect(_clear)
        self._trace_proc_window.refresh_from_doc()
        self._trace_proc_window.show()
        self._trace_proc_window.raise_()
        self._trace_proc_window.activateWindow()

    def _open_heatmap_editor(self) -> None:
        if self.doc is None:
            return
        if self._heatmap_window is None:
            self._heatmap_window = HeatmapEditorWindow(self)

            def _clear(_obj: object = None) -> None:
                self._heatmap_window = None

            self._heatmap_window.destroyed.connect(_clear)
        self._heatmap_window.refresh_from_doc()
        self._heatmap_window.show()
        self._heatmap_window.raise_()
        self._heatmap_window.activateWindow()

    def _open_annotation_editor(self) -> None:
        if self.doc is None:
            return
        if self._annotation_window is None:
            self._annotation_window = AnnotationEditorWindow(self)

            def _clear(_obj: object = None) -> None:
                self._annotation_window = None

            self._annotation_window.destroyed.connect(_clear)
        self._annotation_window.refresh_from_doc()
        self._annotation_window.show()
        self._annotation_window.raise_()
        self._annotation_window.activateWindow()

    def _heatmaps_changed(self) -> None:
        if self.doc is None:
            return
        self._fill_image_source_combos()
        self._refresh_fov()
        self._refresh_w3_and_thumbs()

    def _notify_heatmap_raster(self) -> None:
        """Keep the HeatMap editor's mirrored raster in sync with Raster Tools."""
        if self._heatmap_window is not None:
            self._heatmap_window.refresh_raster()
        if self._annotation_window is not None:
            self._annotation_window.refresh_raster()

    def _fill_image_source_combos(self) -> None:
        if self.doc is None:
            return
        self._fill_one_image_combo(self.cmb_fov_src, ("meanImg", "meanImgE", "VCorr"))
        self._fill_one_image_combo(
            self.cmb_w3_src, ("movie", "meanImg", "meanImgE", "VCorr")
        )

    def _fill_one_image_combo(self, combo: QComboBox, keys: tuple[str, ...]) -> None:
        current = combo.currentData() or combo.currentText()
        was = self._updating
        self._updating = True
        combo.blockSignals(True)
        combo.clear()
        for key in keys:
            combo.addItem(key, key)
        if self.doc is not None:
            for hm in ensure_heatmaps(self.doc):
                combo.addItem(heatmap_combo_label(hm), heatmap_combo_data(hm))
        idx = combo.findData(current)
        if idx < 0:
            idx = combo.findText(str(current))
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)
        self._updating = was

    def _analysis_runs_changed(self) -> None:
        if self.doc is None:
            return
        self.dirty = True
        refresh_stale_flags(self.doc)
        self._fill_raster_sort_combo()
        if self._raster_mode:
            self._refresh_raster()

    def _ensure_analysis_inputs(self, field: str | None = None) -> bool:
        if self.doc is None:
            return False
        if not self.doc["rois"]:
            QMessageBox.information(self, "Analysis Tools", "No ROIs loaded.")
            return False
        field = str(field or TRACE_FIELD_NORM)
        if field not in TRACE_FIELDS:
            field = TRACE_FIELD_NORM

        if field in (TRACE_FIELD_SM, TRACE_FIELD_SM_BC):
            if not field_has_any(self.doc, TRACE_FIELD_SM) or tc_norm_sm_is_stale(self.doc):
                reply = QMessageBox.question(
                    self,
                    "Need smoothed traces",
                    "tc_norm_sm is missing or stale. Rebuild with default SG parameters?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return False
                ensure_trace_processing(self.doc)
                rebuild_all_tc_norm_sm(self.doc)
                self.dirty = True
            if field == TRACE_FIELD_SM_BC:
                missing = not field_has_any(self.doc, TRACE_FIELD_SM_BC)
                stale = tc_norm_sm_bc_is_stale(self.doc)
                if missing or stale:
                    if missing:
                        msg = (
                            "tc_norm_sm_bc is missing. Build with conservative "
                            "(no decay removed) bleach?"
                        )
                    else:
                        msg = (
                            "tc_norm_sm_bc is stale. Rebuild with current "
                            "Trace Processing settings?"
                        )
                    reply = QMessageBox.question(
                        self,
                        "Need bleach-corrected traces",
                        msg,
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes,
                    )
                    if reply != QMessageBox.StandardButton.Yes:
                        return False
                    tp = ensure_trace_processing(self.doc)
                    if missing:
                        tp["bleach_enabled"] = False
                    rebuild_all_tc_norm_sm_bc(self.doc)
                    self.dirty = True
            self._refresh_analysis_stale_ui()
            if self._trace_proc_window is not None:
                self._trace_proc_window.refresh_from_doc()
            return True

        has_any = any(row.get("tc_norm") is not None for row in self.doc["rois"])
        if self._tc_norm_stale or (has_any and tc_norm_is_stale(self.doc)):
            reply = QMessageBox.question(
                self,
                "tc_norm is stale",
                "Rebuild tc_norm first?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return False
            self._rebuild_tc_norm()
            return True
        if not has_any:
            rebuild_all_tc_norm(self.doc)
            self.dirty = True
            self._tc_norm_stale = False
            self._update_rebuild_button()
            return True
        if fill_missing_tc_norm(self.doc):
            self.dirty = True
            self._tc_norm_stale = True
            self._update_rebuild_button()
        return True

    def _mark_tc_norm_stale(self) -> None:
        if self.doc is None:
            return
        has_any = any(row.get("tc_norm") is not None for row in self.doc["rois"])
        if not has_any:
            self._tc_norm_stale = False
            self._update_rebuild_button()
            return
        self._tc_norm_stale = True
        self._update_rebuild_button()
        self._refresh_analysis_stale_ui()

    def _on_raster_mode_slider(self, value: int) -> None:
        if self._updating:
            return
        self._set_raster_mode(int(value) == 1)

    def _on_raster_batch_slider(self, value: int) -> None:
        if self._updating:
            return
        self._raster_batch = int(value) == 1
        if self._raster_mode:
            self._refresh_raster(auto_range=False)

    def _on_raster_display_changed(self) -> None:
        if self._updating:
            return
        self._notify_heatmap_raster()
        if not self._raster_mode:
            return
        self._refresh_raster(auto_range=False)

    def _set_raster_mode(self, on: bool) -> None:
        on = bool(on)
        self._raster_mode = on
        self._updating = True
        self.slider_raster_on.setValue(1 if on else 0)
        self._updating = False
        if on:
            if self._batch_mode:
                self._updating = True
                self.slider_mode.setValue(0)
                self._updating = False
                self._batch_mode = False
                self._batch_roi_ids = []
                w1: ClickableImageView = self.w1.image_view  # type: ignore[attr-defined]
                w1.lasso_enabled = False
                w1.clear_lasso_drawing()
            if self.doc is not None:
                has = [row.get("tc_norm") is not None for row in self.doc["rois"]]
                if self.doc["rois"] and not any(has):
                    rebuild_all_tc_norm(self.doc)
                    self.dirty = True
                    self._tc_norm_stale = False
                elif any(has) and not all(has):
                    if fill_missing_tc_norm(self.doc):
                        self.dirty = True
                    self._tc_norm_stale = True
                else:
                    self._tc_norm_stale = tc_norm_is_stale(self.doc)
                self._update_rebuild_button()
            self.lower_stack.setCurrentIndex(1)
            self._ensure_active_roi_in_filter()
            self._refresh_raster()
            self.statusBar().showMessage("Raster-mode ON")
        else:
            self.lower_stack.setCurrentIndex(0)
            if self.doc is not None:
                self._refresh_traces()
                self._debounce.start()
            self.statusBar().showMessage("Raster-mode OFF")
        self._apply_mode_controls()

    def _rebuild_tc_norm(self) -> None:
        if self.doc is None:
            return
        rebuild_all_tc_norm(self.doc)
        self.dirty = True
        self._tc_norm_stale = False
        self._update_rebuild_button()
        self._refresh_analysis_stale_ui()
        if self._raster_mode:
            self._refresh_raster()
        self.statusBar().showMessage("Rebuilt tc_norm from trace_comp and LED+Shutter")

    def _clear_raster_boxes(self) -> None:
        view = self.raster_view.getView()
        for item in self._raster_box_items:
            try:
                view.removeItem(item)
            except Exception:
                pass
        self._raster_box_items.clear()

    def _add_raster_group_box(
        self, x: float, y: float, w: float, h: float, color: str
    ) -> None:
        if w <= 0 or h <= 0:
            return
        rect = QtWidgets.QGraphicsRectItem(x, y, w, h)
        rect.setPen(pg.mkPen(color, width=1))
        rect.setBrush(QtGui.QBrush(Qt.BrushStyle.NoBrush))
        rect.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        rect.setZValue(10)
        self.raster_view.getView().addItem(rect)
        self._raster_box_items.append(rect)

    def _refresh_raster(self, auto_range: bool = False) -> None:
        if self.doc is None or not self._raster_mode:
            return
        nframes = int(self.doc["meta"]["nframes"])
        if fill_missing_tc_norm(self.doc):
            self.dirty = True
            self._tc_norm_stale = True
            self._update_rebuild_button()
        rows = rois_for_raster(self.doc["rois"], self._overlay_filter())
        run = active_sort_run(self.doc)
        rows = apply_raster_sort(rows, run)
        self._raster_row_ids = [int(r["roi_id"]) for r in rows]
        n_sel = sum(1 for r in rows if bool(r.get("iscell", True)))
        n_unsel = len(rows) - n_sel
        field = raster_trace_field(self.doc)
        if rows:
            matrix = stack_trace_field(rows, nframes, field)
        else:
            matrix = np.full((1, nframes), np.nan, dtype=np.float64)
        lut = lut_with_revert(self.cmb_raster_lut.currentText(), self.chk_raster_revert.isChecked())
        highlight_row: int | None = None
        highlight_lut = None
        if not self._raster_batch and rows:
            try:
                highlight_row = self._raster_row_ids.index(self.active_roi_id)
            except ValueError:
                highlight_row = None
            if highlight_row is not None:
                highlight_lut = selected_row_lut(self.chk_raster_revert.isChecked())
        rgb = colorize_raster(
            matrix, lut, highlight_row=highlight_row, highlight_lut=highlight_lut
        )
        shape = (rgb.shape[0], rgb.shape[1])
        vb = self.raster_view.getView()
        prev_range = vb.viewRange() if self._raster_last_shape is not None else None
        do_range = auto_range or self._raster_last_shape != shape
        self.raster_view.setImage(
            rgb, autoLevels=False, levels=(0, 255), autoRange=do_range
        )
        if not do_range and prev_range is not None:
            vb.setRange(xRange=prev_range[0], yRange=prev_range[1], padding=0)
        self._raster_last_shape = shape
        self._clear_raster_boxes()
        filt = self._overlay_filter()
        if rows and (run is not None or filt != "both"):
            if run is not None:
                self._add_raster_group_box(
                    -0.5, -0.5, float(nframes), float(n_sel), "#e74c3c"
                )
                self._add_raster_group_box(
                    -0.5,
                    float(n_sel) - 0.5,
                    float(nframes),
                    float(n_unsel),
                    "#3498db",
                )
            elif filt == "cell":
                self._add_raster_group_box(
                    -0.5, -0.5, float(nframes), float(len(rows)), "#e74c3c"
                )
            elif filt == "noncell":
                self._add_raster_group_box(
                    -0.5, -0.5, float(nframes), float(len(rows)), "#3498db"
                )
        self.raster_c0.setZValue(20)
        self._updating = True
        self.raster_c0.setValue(self.cursor_c0.value())
        self.raster_trace_c0.setValue(self.cursor_c0.value())
        self._updating = False
        self._refresh_raster_trace(rows, matrix, reset_view=do_range)

    def _refresh_raster_trace(
        self,
        rows: list[dict[str, Any]],
        matrix: np.ndarray,
        *,
        reset_view: bool = True,
    ) -> None:
        nframes = matrix.shape[1]
        xs = np.arange(nframes)
        if self._raster_batch:
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
                y = np.full(nframes, np.nan, dtype=np.float64)
            self.plot_raster_trace.setTitle("Co-Activity")
        else:
            y = np.full(nframes, np.nan, dtype=np.float64)
            try:
                row = self._row()
                field = raster_trace_field(self.doc)
                tr = row.get(field)
                if tr is not None:
                    arr = np.asarray(tr, dtype=np.float64)
                    n_copy = min(arr.shape[0], nframes)
                    y[:n_copy] = arr[:n_copy]
            except Exception:
                pass
            field = raster_trace_field(self.doc) if self.doc is not None else "tc_norm"
            self.plot_raster_trace.setTitle(f"{field} — ROI {self.active_roi_id}")
        self.curve_raster_trace.setData(xs, y)
        if reset_view:
            self._updating = True
            self.plot_raster_trace.enableAutoRange(axis="y")
            self.plot_raster_trace.setXRange(0, max(nframes - 1, 1), padding=0.02)
            self._updating = False

    def _on_raster_click(self, y: int, x: int) -> None:
        if not self._raster_mode or self.doc is None or self._mask_edit_active:
            return
        if not self._raster_row_ids:
            return
        row = int(y)
        if 0 <= row < len(self._raster_row_ids):
            self._select_roi(self._raster_row_ids[row])

    def _on_raster_c0_moved(self) -> None:
        if self._updating or self.doc is None:
            return
        sender = self.sender()
        pos = float(sender.value()) if sender is not None else float(self.raster_c0.value())
        n = int(self.doc["meta"]["nframes"])
        val = float(np.clip(pos, 0, max(n - 1, 0)))
        self._updating = True
        self.cursor_c0.setValue(val)
        self.raster_c0.setValue(val)
        self.raster_trace_c0.setValue(val)
        self._updating = False
        self._sync_cursor_clones()
        self._refresh_movie_views()

    # ------------------------------------------------------------- rendering
    def _refresh_all(self) -> None:
        self._refresh_fov()
        self._refresh_movie_views()
        self._refresh_traces(autoscale=True)
        if self._raster_mode:
            self._refresh_raster(auto_range=False)
        self._update_cursor_labels()
        self._debounce.start()

    def _w3_source_is_movie(self) -> bool:
        return str(self.cmb_w3_src.currentData() or "") == "movie"

    def _image_from_combo(self, combo: QComboBox, *, movie_frame: int | None = None) -> np.ndarray | None:
        if self.doc is None:
            return None
        data = combo.currentData() or combo.currentText()
        hid = parse_heatmap_combo_data(data)
        if hid is not None:
            hm = get_heatmap(self.doc, hid)
            if hm is None or hm.get("map") is None:
                return None
            return np.asarray(hm["map"])
        key = str(data)
        meta = self.doc["meta"]
        if key == "movie":
            if self.stack is None or movie_frame is None:
                return None
            return self.stack.read_frame(int(movie_frame))
        mapping = {
            "meanImg": meta.get("meanImg"),
            "meanImgE": meta.get("meanImgE"),
            "VCorr": meta.get("VCorr"),
        }
        img = mapping.get(key)
        if img is None and key == "meanImgE":
            img = self._build_meanimge()
        if img is None and key != "meanImg":
            img = meta.get("meanImg")
        return None if img is None else np.asarray(img)

    def _build_meanimge(self) -> np.ndarray | None:
        """Derive meanImgE once for ops.npy files that never stored one."""
        if self.doc is None or self.suite2p_dir is None or self._meanimge_failed:
            return None
        if self._meanimge_busy:  # processEvents below can re-enter through a refresh
            return None
        self._meanimge_busy = True
        self.statusBar().showMessage(
            "ops.npy has no meanImgE — computing the enhanced mean image…"
        )
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            img = ensure_meanimge(self.doc, self.suite2p_dir)
        except Exception as exc:
            self._meanimge_failed = True
            self.statusBar().showMessage(f"Could not build meanImgE: {exc}")
            return None
        finally:
            QApplication.restoreOverrideCursor()
            self._meanimge_busy = False
        if img is None:
            self._meanimge_failed = True
            self.statusBar().showMessage("ops.npy has no meanImg — cannot build meanImgE")
            return None
        self.dirty = True
        self.statusBar().showMessage(
            "meanImgE was missing from ops.npy — computed here and cached in the pickle"
        )
        return img

    def _fov_source_image(self) -> np.ndarray | None:
        return self._image_from_combo(self.cmb_fov_src)

    def _w3_source_image(self, frame_index: int | None = None) -> np.ndarray | None:
        t = self._c0_frame_index() if frame_index is None else int(frame_index)
        return self._image_from_combo(self.cmb_w3_src, movie_frame=t)

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
        batch_ids = set(self._batch_roi_ids) if self._batch_mode else None
        active_id = self.active_roi_id
        rois = list(self.doc["rois"])
        if (
            self._mask_edit_kind == "add"
            and self._add_mask_draft is not None
            and len(self._add_mask_draft["roi"]["ypix"]) > 0
        ):
            rois = rois + [self._add_mask_draft]
            active_id = int(self._add_mask_draft["roi_id"])
        overlay = build_fov_overlay(
            int(self.doc["meta"]["Ly"]),
            int(self.doc["meta"]["Lx"]),
            rois,
            active_id,
            self._overlay_filter(),
            batch_roi_ids=batch_ids,
        )
        composed = compose_rgb_with_overlay(rgb, overlay)
        if self._mask_edit_kind == "add" and self._add_mask_center is not None:
            cy, cx = self._add_mask_center
            self._draw_red_cross(composed, int(round(cy)), int(round(cx)))
        self._set_display_rgb(self.w1.image_view, composed)  # type: ignore[attr-defined]

    @staticmethod
    def _draw_red_cross(rgb: np.ndarray, cy: int, cx: int, arm: int = 6) -> None:
        Ly, Lx = rgb.shape[:2]
        for d in range(-arm, arm + 1):
            y = cy + d
            x = cx + d
            if 0 <= cy < Ly and 0 <= cx + d < Lx:
                rgb[cy, cx + d] = (255, 0, 0)
            if 0 <= cy + d < Ly and 0 <= cx < Lx:
                rgb[cy + d, cx] = (255, 0, 0)

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
        if self._mask_edit_kind == "add" and self._add_mask_center is not None:
            cy, cx = self._add_mask_center
            y0, x0, side = zoom_square_at(cy, cx, self._add_mask_side, Ly, Lx)
            return y0, x0, side, row
        if self._mask_edit_kind == "add":
            cy, cx = (Ly - 1) / 2.0, (Lx - 1) / 2.0
            y0, x0, side = zoom_square_at(cy, cx, self._add_mask_side, Ly, Lx)
            return y0, x0, side, row
        y0, x0, side = zoom_square_window(
            row["roi"]["ypix"], row["roi"]["xpix"], row["neuropil"]["ipix"], Ly, Lx
        )
        return y0, x0, side, row

    def _refresh_movie_views(self) -> None:
        if self.doc is None or self.stack is None:
            return
        meta = self.doc["meta"]
        Ly, Lx = int(meta["Ly"]), int(meta["Lx"])
        t = self._c0_frame_index()
        frame = self.stack.read_frame(t)
        rgb = self._movie_rgb(frame)

        w2 = rgb.copy()
        if self._batch_mode and self._batch_roi_ids:
            id_set = set(self._batch_roi_ids)
            for row in self.doc["rois"]:
                if int(row["roi_id"]) not in id_set:
                    continue
                y_out, x_out = thick_outline_mask(
                    Ly, Lx, row["roi"]["ypix"], row["roi"]["xpix"], thickness=2
                )
                if y_out.size:
                    w2[y_out, x_out] = (255, 0, 0)
        else:
            row = self._row()
            y_out, x_out = thick_outline_mask(
                Ly, Lx, row["roi"]["ypix"], row["roi"]["xpix"], thickness=2
            )
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
        if self._batch_mode:
            # Placeholder grey panel while batch-selecting
            grey = np.full((64, 64, 3), 80, dtype=np.uint8)
            self._set_display_rgb(self.w3.image_view, grey)  # type: ignore[attr-defined]
            self.w3.setTitle("ROI zoom (W3) — batch mode")  # type: ignore[attr-defined]
            return
        meta = self.doc["meta"]
        Ly, Lx = int(meta["Ly"]), int(meta["Lx"])
        t = self._c0_frame_index()
        rgb = self._w3_rgb(frame_index=t if self._w3_source_is_movie() else None)
        if rgb is None:
            return
        y0, x0, side, row = self._zoom_geometry()
        self._w3_y0, self._w3_x0, self._w3_side = y0, x0, side
        zoom = zoom_masks_rgba(rgb, y0, x0, side, row, Ly, Lx)
        self._set_display_rgb(self.w3.image_view, zoom)  # type: ignore[attr-defined]
        src = self.cmb_w3_src.currentText()
        title = f"ROI zoom (W3) — {src}"
        if self._w3_source_is_movie():
            title += f" frame {t}"
        if self._mask_edit_active:
            title += " [editing]"
        self.w3.setTitle(title)  # type: ignore[attr-defined]

    def _refresh_traces(self, autoscale: bool = False) -> None:
        if self.doc is None:
            return
        if self._batch_mode:
            if self._batch_roi_ids:
                F, Fneu, comp = mean_traces_for_rois(self.doc["rois"], self._batch_roi_ids)
                x_disp = 1.0
            else:
                F = Fneu = comp = np.zeros(int(self.doc["meta"]["nframes"]), dtype=np.float64)
                x_disp = 1.0
            xs = np.arange(F.shape[0])
            F_d = self._display_trace(F)
            Fneu_d = self._display_trace(x_disp * Fneu)
            comp_d = self._display_trace(comp)
            self.curve_f.setData(xs, F_d)
            self.curve_fneu.setData(xs, Fneu_d)
            self.curve_comp.setData(xs, comp_d)
            self.plot_f.setTitle(
                f"mean F / mean Fneu (batch n={len(self._batch_roi_ids)}, display x=1)"
            )
            self.plot_comp.setTitle("mean(F − 1·Fneu) — batch display")
            self.curve_bleach_sm.setData(xs, np.full_like(xs, np.nan, dtype=float))
            self.curve_bleach_fit.setData(xs, np.full_like(xs, np.nan, dtype=float))
            nframes = int(self.doc["meta"]["nframes"])
            bc_acc = []
            sm_acc = []
            for row in self.doc["rois"]:
                if int(row["roi_id"]) not in set(self._batch_roi_ids):
                    continue
                if row.get(TRACE_FIELD_SM) is not None:
                    sm_acc.append(np.asarray(row[TRACE_FIELD_SM], dtype=np.float64))
                if row.get(TRACE_FIELD_SM_BC) is not None:
                    bc_acc.append(np.asarray(row[TRACE_FIELD_SM_BC], dtype=np.float64))
            if sm_acc:
                with np.errstate(all="ignore"):
                    sm_mean = np.nanmean(np.vstack(sm_acc), axis=0)
                self.curve_bleach_sm.setData(xs[: sm_mean.shape[0]], sm_mean[: xs.shape[0]])
            if bc_acc:
                with np.errstate(all="ignore"):
                    bc_mean = np.nanmean(np.vstack(bc_acc), axis=0)
                self.curve_bleach_bc.setData(xs[: bc_mean.shape[0]], bc_mean[: xs.shape[0]])
            else:
                self.curve_bleach_bc.setData(xs, np.full_like(xs, np.nan, dtype=float))
            self.plot_bleach.setTitle("mean tc_norm_sm / tc_norm_sm_bc — batch")
        else:
            row = self._row()
            F = np.asarray(row["roi"]["F"], dtype=np.float64)
            Fneu = np.asarray(row["neuropil"]["Fneu"], dtype=np.float64)
            x = float(row["compensation"]["x"])
            comp = np.asarray(row["compensation"]["trace_comp"], dtype=np.float64)
            xs = np.arange(F.shape[0])
            self.curve_f.setData(xs, self._display_trace(F))
            self.curve_fneu.setData(xs, self._display_trace(x * Fneu))
            self.curve_comp.setData(xs, self._display_trace(comp))
            self.plot_f.setTitle("F / Fneu")
            self.plot_comp.setTitle("trace_comp = F − x·Fneu")
            nframes = int(self.doc["meta"]["nframes"])
            mask = led_shutter_nan_mask(self.doc, nframes)
            sm = row.get(TRACE_FIELD_SM)
            bc = row.get(TRACE_FIELD_SM_BC)
            fit = bleach_fit_curve(row, mask, nframes)
            nan_y = np.full(xs.shape, np.nan, dtype=np.float64)
            self.curve_bleach_sm.setData(
                xs, self._display_trace(sm) if sm is not None else nan_y
            )
            self.curve_bleach_fit.setData(xs, fit if fit is not None else nan_y)
            self.curve_bleach_bc.setData(
                xs, self._display_trace(bc) if bc is not None else nan_y
            )
            cons = bool((row.get("bleach") or {}).get("conservative", False)) if bc is not None else False
            extra = " (conservative)" if cons else ""
            self.plot_bleach.setTitle(f"tc_norm_sm / tc_norm_sm_bc{extra}")
        self._refresh_ann_spans()
        if autoscale:
            for k in self._trace_y_locked:
                self._trace_y_locked[k] = False
            self._autoscale_trace_y("f")
            self._autoscale_trace_y("comp")
            self._autoscale_trace_y("bleach")
            self.plot_f.enableAutoRange(axis="x", enable=False)
            self.plot_comp.enableAutoRange(axis="x", enable=False)
            if not self._trace_x_user_zoomed:
                n = int(self.doc["meta"]["nframes"])
                self.plot_f.setXRange(0, max(n - 1, 1), padding=0.02)
        else:
            # Respect per-plot Y locks; only autoscale unlocked plots (e.g. after LED NaNs)
            for key in ("f", "comp", "bleach"):
                if not self._trace_y_locked[key]:
                    self._autoscale_trace_y(key)
        self._sync_trace_y_spins_from_plots()
        self._update_cursor_labels()

    def _trace_plot(self, key: str) -> pg.PlotItem:
        return {"f": self.plot_f, "comp": self.plot_comp, "bleach": self.plot_bleach}[key]

    def _on_trace_y_range_changed(self, which: str) -> None:
        if self._updating or self.doc is None:
            return
        self._trace_y_locked[which] = True
        self._sync_trace_y_spins_from_plots(which)

    def _autoscale_trace_y(self, key: str) -> None:
        self._updating = True
        self._trace_plot(key).enableAutoRange(axis="y")
        self._updating = False
        self._trace_y_locked[key] = False
        self._sync_trace_y_spins_from_plots(key)

    def _fit_trace_y(self, key: str) -> None:
        self._autoscale_trace_y(key)

    def _apply_trace_y_spins(self, key: str) -> None:
        if self._updating or key not in self._trace_y_spins:
            return
        spin_lo, spin_hi = self._trace_y_spins[key]
        lo = float(spin_lo.value())
        hi = float(spin_hi.value())
        if hi <= lo:
            hi = lo + 1.0
            spin_hi.blockSignals(True)
            spin_hi.setValue(hi)
            spin_hi.blockSignals(False)
        self._updating = True
        self._trace_plot(key).setYRange(lo, hi, padding=0)
        self._updating = False
        self._trace_y_locked[key] = True

    def _sync_trace_y_spins_from_plots(self, only: str | None = None) -> None:
        if not hasattr(self, "_trace_y_spins"):
            return
        keys = (only,) if only is not None else ("f", "comp", "bleach")
        for key in keys:
            if key not in self._trace_y_spins:
                continue
            _, (ymin, ymax) = self._trace_plot(key).viewRange()
            spin_lo, spin_hi = self._trace_y_spins[key]
            spin_lo.blockSignals(True)
            spin_hi.blockSignals(True)
            spin_lo.setValue(float(ymin))
            spin_hi.setValue(float(ymax))
            spin_lo.blockSignals(False)
            spin_hi.blockSignals(False)
        if only is None:
            self._sync_trace_x_spins_from_plots()

    def _apply_trace_x_spins(self, *_args: object) -> None:
        if self._updating or self.doc is None:
            return
        lo = float(self.spin_x_lo.value())
        hi = float(self.spin_x_hi.value())
        if hi <= lo:
            hi = lo + 1.0
            self.spin_x_hi.blockSignals(True)
            self.spin_x_hi.setValue(hi)
            self.spin_x_hi.blockSignals(False)
        self._updating = True
        self.plot_f.setXRange(lo, hi, padding=0)
        self._updating = False
        self._trace_x_user_zoomed = True

    def _sync_trace_x_spins_from_plots(self) -> None:
        if not hasattr(self, "spin_x_lo"):
            return
        (xmin, xmax), _ = self.plot_f.viewRange()
        self.spin_x_lo.blockSignals(True)
        self.spin_x_hi.blockSignals(True)
        self.spin_x_lo.setValue(float(xmin))
        self.spin_x_hi.setValue(float(xmax))
        self.spin_x_lo.blockSignals(False)
        self.spin_x_hi.blockSignals(False)

    def _toggle_scale_panel(self) -> None:
        self._scale_panel_open = not self._scale_panel_open
        self.scale_panel.setVisible(self._scale_panel_open)
        if self._scale_panel_open:
            self._sync_trace_y_spins_from_plots()
            self._sync_trace_x_spins_from_plots()

    def _on_box_zoom_toggled(self, checked: bool) -> None:
        mode = pg.ViewBox.RectMode if checked else pg.ViewBox.PanMode
        for plot in (self.plot_f, self.plot_comp, self.plot_bleach):
            plot.getViewBox().setMouseMode(mode)
        if checked:
            self.statusBar().showMessage(
                "Box zoom on — left-drag a rectangle on a trace"
            )
        else:
            self.statusBar().showMessage("Box zoom off — left-drag pans")

    def _on_trace_scene_clicked(self, ev) -> None:
        pos = ev.scenePos()
        if not ev.double():
            # Title bar click toggles focus; the plot area keeps its own gestures
            for key in ("f", "comp", "bleach"):
                label = self._trace_plot(key).titleLabel
                if label.isVisible() and label.sceneBoundingRect().contains(pos):
                    self._toggle_inspect_focus(key)
                    ev.accept()
                    return
            return
        for key in ("f", "comp", "bleach"):
            vb = self._trace_plot(key).getViewBox()
            if vb.sceneBoundingRect().contains(pos):
                self._fit_trace_y(key)
                ev.accept()
                return

    def _on_thumb_clicked(self, ev) -> None:
        if ev.double():
            return
        self._toggle_inspect_focus("thumbs")
        ev.accept()

    def _toggle_inspect_focus(self, key: str) -> None:
        self._inspect_focus = None if self._inspect_focus == key else key
        self._apply_inspect_focus()
        if self._inspect_focus is None:
            self.statusBar().showMessage("Trace panel: showing all")
        else:
            self.statusBar().showMessage(
                f"Trace panel: {_INSPECT_FOCUS_NAMES[key]} expanded — "
                "click it again to show all"
            )

    def _apply_inspect_focus(self) -> None:
        """Give one of the four inspect items the whole lower panel height."""
        was = self._updating
        self._updating = True
        try:
            self._apply_inspect_focus_layout()
        finally:
            self._updating = was
        QTimer.singleShot(0, self._reposition_scale_nudges)

    def _apply_inspect_focus_layout(self) -> None:
        focus = self._inspect_focus
        show_traces = focus is None or focus in _TRACE_ROWS
        show_thumbs = focus is None or focus == "thumbs"
        self.trace_widget.setVisible(show_traces)
        self.thumb_wrap.setVisible(show_thumbs)

        layout = self.trace_widget.ci.layout
        bottom_most = "bleach" if focus is None else focus
        for key, row in _TRACE_ROWS.items():
            plot = self._trace_plot(key)
            on = focus is None or focus == key
            plot.setVisible(on and show_traces)
            # Hidden rows need a zero max height: an invisible graphics item
            # still claims its share of a QGraphicsGridLayout.
            layout.setRowMinimumHeight(row, 0.0)
            layout.setRowPreferredHeight(row, 0.0)
            layout.setRowMaximumHeight(row, _ROW_MAX_HEIGHT if on else 0.0)
            layout.setRowStretchFactor(row, 1 if on else 0)
        layout.invalidate()
        layout.activate()
        self._sync_trace_bottom_axes(bottom_most if show_traces else None)

        self.inspect_layout.setStretchFactor(self.trace_widget, 2)
        self.inspect_layout.setStretchFactor(
            self.thumb_wrap, 1 if focus == "thumbs" else 0
        )

    def _sync_trace_bottom_axes(self, bottom_key: str | None) -> None:
        """Give only the bottom-most visible plot an X-axis.

        setLabel on extra X-linked plots makes pyqtgraph stretch those axes
        over the ViewBox, which paints permanent horizontal lines on the data.
        """
        for key in _TRACE_ROWS:
            plot = self._trace_plot(key)
            if key == bottom_key:
                plot.showAxis("bottom", True)
                plot.setLabel("bottom", "frame")
            else:
                plot.hideAxis("bottom")

    def _install_scale_nudges(self) -> None:
        self._scale_nudges = {}
        for key, plot in (
            ("f", self.plot_f),
            ("comp", self.plot_comp),
            ("bleach", self.plot_bleach),
        ):
            pairs = {
                "y_hi": _NudgePair(
                    True,
                    lambda _=False, k=key: self._nudge_y(k, "hi", +1),
                    lambda _=False, k=key: self._nudge_y(k, "hi", -1),
                    "Nudge Y upper limit",
                ),
                "y_lo": _NudgePair(
                    True,
                    lambda _=False, k=key: self._nudge_y(k, "lo", +1),
                    lambda _=False, k=key: self._nudge_y(k, "lo", -1),
                    "Nudge Y lower limit",
                ),
                "x_lo": _NudgePair(
                    False,
                    lambda _=False, k=key: self._nudge_x("lo", +1),
                    lambda _=False, k=key: self._nudge_x("lo", -1),
                    "Nudge X lower limit (all traces)",
                ),
                "x_hi": _NudgePair(
                    False,
                    lambda _=False, k=key: self._nudge_x("hi", +1),
                    lambda _=False, k=key: self._nudge_x("hi", -1),
                    "Nudge X upper limit (all traces)",
                ),
            }
            proxies: dict[str, QGraphicsProxyWidget] = {}
            left = plot.getAxis("left")
            bottom = plot.getAxis("bottom")
            for name, widget in pairs.items():
                proxy = QGraphicsProxyWidget()
                proxy.setWidget(widget)
                proxy.setZValue(1000)
                proxy.setParentItem(left if name.startswith("y") else bottom)
                proxies[name] = proxy
            self._scale_nudges[key] = proxies
        QTimer.singleShot(0, self._reposition_scale_nudges)

    def _reposition_scale_nudges(self, *_args: object) -> None:
        if not self._scale_nudges:
            return
        for key, plot in (
            ("f", self.plot_f),
            ("comp", self.plot_comp),
            ("bleach", self.plot_bleach),
        ):
            nudges = self._scale_nudges[key]
            left = plot.getAxis("left")
            bottom = plot.getAxis("bottom")
            # Not boundingRect(): a gridded AxisItem reports the whole ViewBox,
            # which would park these widgets in the middle of the data.
            lr = left.mapRectFromParent(left.geometry())
            br = bottom.mapRectFromParent(bottom.geometry())

            y_hi = nudges["y_hi"]
            wh = y_hi.widget()
            if wh is not None:
                wh.adjustSize()
                y_hi.setPos(lr.center().x() - wh.width() / 2, lr.top())
            y_lo = nudges["y_lo"]
            wl = y_lo.widget()
            if wl is not None:
                wl.adjustSize()
                y_lo.setPos(lr.center().x() - wl.width() / 2, lr.bottom() - wl.height())
            if not plot.getAxis("bottom").isVisible():
                continue
            x_lo = nudges["x_lo"]
            wxl = x_lo.widget()
            if wxl is not None:
                wxl.adjustSize()
                x_lo.setPos(br.left(), br.center().y() - wxl.height() / 2)
            x_hi = nudges["x_hi"]
            wxh = x_hi.widget()
            if wxh is not None:
                wxh.adjustSize()
                x_hi.setPos(br.right() - wxh.width(), br.center().y() - wxh.height() / 2)

    def _nudge_y(self, key: str, end: str, direction: int) -> None:
        plot = self._trace_plot(key)
        _, (ymin, ymax) = plot.viewRange()
        span = max(ymax - ymin, 1e-6)
        step = span * _NUDGE_FRAC
        if end == "hi":
            ymax = ymax + direction * step
        else:
            ymin = ymin + direction * step
        if ymax <= ymin:
            return
        self._updating = True
        plot.setYRange(ymin, ymax, padding=0)
        self._updating = False
        self._trace_y_locked[key] = True
        self._sync_trace_y_spins_from_plots(key)

    def _nudge_x(self, end: str, direction: int) -> None:
        (xmin, xmax), _ = self.plot_f.viewRange()
        span = max(xmax - xmin, 1.0)
        step = span * _NUDGE_FRAC
        if end == "hi":
            xmax = xmax + direction * step
        else:
            xmin = xmin + direction * step
        if self.doc is not None:
            n = int(self.doc["meta"]["nframes"])
            xmin = max(0.0, xmin)
            xmax = min(float(max(n - 1, 1)), xmax)
        if xmax <= xmin:
            return
        self._updating = True
        self.plot_f.setXRange(xmin, xmax, padding=0)
        self._updating = False
        self._trace_x_user_zoomed = True
        self._sync_trace_x_spins_from_plots()

    def _display_trace(self, trace: np.ndarray) -> np.ndarray:
        """Copy of trace with NaNs for selected LED+Shutter annotations (display only)."""
        if self.doc is None:
            return np.asarray(trace, dtype=np.float64)
        anns = ensure_annotations(self.doc)
        active = self._selected_ann_ids()
        if not active:
            return np.asarray(trace, dtype=np.float64)
        mask = nan_mask_from_annotations(len(trace), anns, active)
        if not mask.any():
            return np.asarray(trace, dtype=np.float64)
        return apply_nan_mask(trace, mask)

    def _selected_ann_ids(self) -> set[int]:
        return set(self._ann_selected_ids)

    def _set_ann_selection(self, ids: set[int]) -> None:
        self._ann_selected_ids = {int(i) for i in ids}
        self._refresh_traces(autoscale=False)

    def _annotations_changed(self, *, led_changed: bool = False) -> None:
        self.dirty = True
        self._refresh_ann_spans()
        self._refresh_traces(autoscale=False)
        if led_changed:
            self._mark_tc_norm_stale()
        if self._heatmap_window is not None:
            self._heatmap_window.refresh_raster()

    def _on_trace_x_range_changed(self, *_args: object) -> None:
        if self._updating or self.doc is None:
            return
        self._trace_x_user_zoomed = True
        self._sync_trace_x_spins_from_plots()

    def _fit_trace_x(self) -> None:
        if self.doc is None:
            return
        n = int(self.doc["meta"]["nframes"])
        self._updating = True
        self._trace_x_user_zoomed = False
        self.plot_f.setXRange(0, max(n - 1, 1), padding=0.02)
        self._updating = False
        self._sync_trace_x_spins_from_plots()

    def _reset_all_trace_scales(self) -> None:
        if self.doc is None:
            return
        self._fit_trace_x()
        for key in ("f", "comp", "bleach"):
            self._fit_trace_y(key)
        self.statusBar().showMessage("Trace scales reset to autoscale")

    def _clear_ann_spans(self) -> None:
        for item in self._ann_span_items:
            try:
                item.scene().removeItem(item)  # type: ignore[union-attr]
            except Exception:
                for plot in (self.plot_f, self.plot_comp, self.plot_bleach):
                    try:
                        plot.removeItem(item)
                    except Exception:
                        pass
        self._ann_span_items.clear()

    def _refresh_ann_spans(self) -> None:
        self._clear_ann_spans()
        if self.doc is None:
            return
        selected = self._selected_ann_ids()
        for ann in ensure_annotations(self.doc):
            prop = str(ann["property"])
            color = property_spec(prop).get("color", "#888888")
            s = float(ann["start_frame"])
            e = float(ann["end_frame"]) + 1.0  # through inclusive end
            alpha = 90 if int(ann["ann_id"]) in selected else 45
            c = QtGui.QColor(color)
            c.setAlpha(alpha)
            for plot in (self.plot_f, self.plot_comp, self.plot_bleach):
                region = pg.LinearRegionItem(
                    values=(s, e),
                    movable=False,
                    brush=c,
                    pen=pg.mkPen(color, width=1),
                )
                region.setZValue(-10)
                plot.addItem(region)
                self._ann_span_items.append(region)

    # -------------------------------------------------------------- cursors
    def _sync_cursor_clones(self) -> None:
        positions = [self.cursor_c0.value()] + [c.value() for c in self.cursors]
        for i, pos in enumerate(positions):
            self._cursor_clones["f"][i].setValue(pos)
            self._cursor_clones["bleach"][i].setValue(pos)
        if hasattr(self, "raster_c0"):
            was = self._updating
            self._updating = True
            try:
                self.raster_c0.setValue(self.cursor_c0.value())
                self.raster_trace_c0.setValue(self.cursor_c0.value())
            finally:
                self._updating = was

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
        if self._batch_mode:
            if self._batch_roi_ids:
                _, _, comp = mean_traces_for_rois(self.doc["rois"], self._batch_roi_ids)
            else:
                comp = np.zeros(int(self.doc["meta"]["nframes"]), dtype=np.float64)
        else:
            comp = np.asarray(self._row()["compensation"]["trace_comp"], dtype=np.float64)
        n = len(comp)
        for line, label in zip(self.cursors, self.cursor_labels):
            fi = int(np.clip(round(line.value()), 0, max(n - 1, 0)))
            val = float(comp[fi]) if n else 0.0
            label.setText(f"f={fi}\n{val:.2f}")
            label.setPos(fi, val)

    def _update_thumbnails(self) -> None:
        if self.doc is None or self._raster_mode:
            return
        if self._batch_mode:
            grey = np.full((64, 64, 3), 80, dtype=np.uint8)
            for i, (view, lab) in enumerate(zip(self.thumb_views, self.thumb_labels)):
                self._set_display_rgb(view, grey)
                lab.setText(f"C{i + 1} — batch")
            return
        meta = self.doc["meta"]
        Ly, Lx = int(meta["Ly"]), int(meta["Lx"])
        y0, x0, side, row = self._zoom_geometry()
        n = int(meta["nframes"])
        src = self.cmb_w3_src.currentText()
        movie = self._w3_source_is_movie()
        for i, (line, view, lab) in enumerate(
            zip(self.cursors, self.thumb_views, self.thumb_labels)
        ):
            fi = int(np.clip(round(line.value()), 0, max(n - 1, 0)))
            rgb = self._w3_rgb(frame_index=fi if movie else None)
            if rgb is None:
                continue
            zoom = zoom_masks_rgba(rgb, y0, x0, side, row, Ly, Lx)
            self._set_display_rgb(view, zoom)
            if movie:
                lab.setText(f"C{i + 1} — frame {fi}")
            else:
                lab.setText(f"C{i + 1} — {src}")
