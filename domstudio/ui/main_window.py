"""Controller-neutral desktop shell for DOMStudio.

This module intentionally imports no detection or tracing implementation.  A
controller can connect to the public signals and update the view through the
``set_*`` methods, making UI work independent from model/runtime choices.
"""

from pathlib import Path
from typing import Iterable, Mapping, Optional

from PyQt5.QtCore import QPointF, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .canvas import ImageCanvas, ImageSource, InteractionMode
from .panels import InspectorPanel, WorkflowSidebar
from .theme import COLORS


class MainWindow(QMainWindow):
    """Primary application window and stable boundary for an external controller."""

    imageOpenRequested = pyqtSignal(str)
    detectionRequested = pyqtSignal(dict)
    traceRequested = pyqtSignal(QPointF, QPointF, dict)
    settingsChanged = pyqtSignal(dict)
    undoRequested = pyqtSignal()
    clearRequested = pyqtSignal()
    exportRequested = pyqtSignal(str)
    workflowStepChanged = pyqtSignal(str)

    IMAGE_FILTER = (
        "Geological imagery (*.tif *.tiff *.jp2 *.j2k *.png *.jpg *.jpeg *.bmp *.webp);;"
        "GeoTIFF (*.tif *.tiff);;All files (*)"
    )
    EXPORT_FILTER = (
        "GeoPackage (*.gpkg);;GeoJSON (*.geojson);;Shapefile (*.shp);;"
        "Comma-separated values (*.csv)"
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("DOMStudioMainWindow")
        self.setWindowTitle("DOMStudio — Fracture Network Studio")
        self.setWindowIcon(
            QIcon(str(Path(__file__).with_name("assets") / "domstudio.svg"))
        )
        self.setMinimumSize(1080, 680)
        self.resize(1460, 900)
        self._source_path: Optional[Path] = None
        self._current_step = "import"
        self._busy = False
        self._traces_available = False

        self._build_actions()
        self._build_ui()
        self._connect_signals()
        self.set_status("Open an image to begin", "ready")

    def _build_actions(self) -> None:
        self.open_action = QAction("Open image…", self)
        self.open_action.setShortcut(QKeySequence.Open)
        self.open_action.setToolTip("Open image (Ctrl+O)")

        self.detect_action = QAction("Run detection", self)
        self.detect_action.setShortcut(QKeySequence("Ctrl+R"))
        self.detect_action.setEnabled(False)

        self.export_action = QAction("Export network…", self)
        self.export_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
        self.export_action.setEnabled(False)

        self.fit_action = QAction("Fit image", self)
        self.fit_action.setShortcut(QKeySequence("F"))
        self.actual_pixels_action = QAction("Actual pixels", self)
        self.actual_pixels_action.setShortcut(QKeySequence("1"))
        self.zoom_in_action = QAction("Zoom in", self)
        self.zoom_in_action.setShortcut(QKeySequence.ZoomIn)
        self.zoom_out_action = QAction("Zoom out", self)
        self.zoom_out_action.setShortcut(QKeySequence.ZoomOut)

        self.trace_mode_action = QAction("Trace mode", self)
        self.trace_mode_action.setShortcut(QKeySequence("T"))
        self.pan_mode_action = QAction("Pan mode", self)
        self.pan_mode_action.setShortcut(QKeySequence("H"))

        file_menu = self.menuBar().addMenu("File")
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.export_action)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close, QKeySequence.Quit)

        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self.fit_action)
        view_menu.addAction(self.actual_pixels_action)
        view_menu.addAction(self.zoom_in_action)
        view_menu.addAction(self.zoom_out_action)

        workflow_menu = self.menuBar().addMenu("Workflow")
        workflow_menu.addAction(self.detect_action)
        workflow_menu.addAction(self.pan_mode_action)
        workflow_menu.addAction(self.trace_mode_action)

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("AppRoot")
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        self.sidebar = WorkflowSidebar()
        root_layout.addWidget(self.sidebar)

        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)
        root_layout.addWidget(workspace, 1)

        self._build_header(workspace_layout)
        self._build_canvas_area(workspace_layout)

        self.inspector = InspectorPanel()
        root_layout.addWidget(self.inspector)

        self._build_status_bar()

    def _build_header(self, parent_layout: QVBoxLayout) -> None:
        header = QFrame()
        header.setObjectName("Header")
        header.setFixedHeight(66)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(18, 10, 17, 10)
        layout.setSpacing(9)

        titles = QVBoxLayout()
        titles.setSpacing(1)
        self.page_title = QLabel("Import imagery")
        self.page_title.setObjectName("PageTitle")
        self.page_subtitle = QLabel("Full-resolution raster workspace")
        self.page_subtitle.setObjectName("MutedLabel")
        titles.addWidget(self.page_title)
        titles.addWidget(self.page_subtitle)
        layout.addLayout(titles)
        layout.addStretch(1)

        self.open_button = QPushButton("Open")
        self.open_button.setObjectName("GhostButton")
        self.open_button.setMinimumWidth(76)
        self.detect_button = QPushButton("Run detection")
        self.detect_button.setObjectName("PrimaryButton")
        self.detect_button.setMinimumWidth(116)
        self.detect_button.setEnabled(False)
        layout.addWidget(self.open_button)
        layout.addWidget(self.detect_button)
        parent_layout.addWidget(header)

    def _build_canvas_area(self, parent_layout: QVBoxLayout) -> None:
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(9)

        toolbar = QFrame()
        toolbar.setObjectName("CanvasToolbar")
        toolbar.setFixedHeight(44)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(7, 5, 7, 5)
        toolbar_layout.setSpacing(5)

        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)
        self.mode_buttons = {}
        for mode, label, shortcut in (
            (InteractionMode.PAN, "Pan", "H"),
            (InteractionMode.TRACE, "Trace", "T"),
            (InteractionMode.INSPECT, "Inspect", "I"),
        ):
            button = QToolButton()
            button.setObjectName("ModeButton")
            button.setText(label)
            button.setCheckable(True)
            button.setToolTip("{} mode ({})".format(label, shortcut))
            button.setEnabled(mode == InteractionMode.PAN)
            mode_group.addButton(button)
            self.mode_buttons[mode] = button
            toolbar_layout.addWidget(button)
        self.mode_buttons[InteractionMode.PAN].setChecked(True)

        toolbar_layout.addSpacing(7)
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet("color: {};".format(COLORS["border"]))
        toolbar_layout.addWidget(separator)
        toolbar_layout.addSpacing(3)

        fit_button = QToolButton()
        fit_button.setObjectName("ModeButton")
        fit_button.setText("Fit")
        fit_button.setToolTip("Fit image to view (F)")
        actual_button = QToolButton()
        actual_button.setObjectName("ModeButton")
        actual_button.setText("1:1")
        actual_button.setToolTip("Show actual pixels (1)")
        zoom_out_button = QToolButton()
        zoom_out_button.setObjectName("ModeButton")
        zoom_out_button.setText("−")
        zoom_out_button.setToolTip("Zoom out")
        zoom_in_button = QToolButton()
        zoom_in_button.setObjectName("ModeButton")
        zoom_in_button.setText("+")
        zoom_in_button.setToolTip("Zoom in")
        toolbar_layout.addWidget(fit_button)
        toolbar_layout.addWidget(actual_button)
        toolbar_layout.addWidget(zoom_out_button)
        toolbar_layout.addWidget(zoom_in_button)
        toolbar_layout.addStretch(1)

        legend = QLabel("●  evidence / error cue    ━  vector trace")
        legend.setObjectName("MutedLabel")
        legend.setToolTip("Visible scientific layers")
        toolbar_layout.addWidget(legend)
        layout.addWidget(toolbar)

        self.canvas = ImageCanvas()
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.canvas, 1)
        parent_layout.addWidget(area, 1)

        for mode, button in self.mode_buttons.items():
            button.clicked.connect(
                lambda checked=False, selected=mode: self.canvas.set_interaction_mode(selected)
            )
        fit_button.clicked.connect(self.canvas.fit_to_image)
        actual_button.clicked.connect(self.canvas.actual_pixels)
        zoom_out_button.clicked.connect(self.canvas.zoom_out)
        zoom_in_button.clicked.connect(self.canvas.zoom_in)

    def _build_status_bar(self) -> None:
        status = QStatusBar()
        status.setSizeGripEnabled(False)
        status.setFixedHeight(28)
        self.setStatusBar(status)

        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: {}; padding-left: 7px;".format(COLORS["primary"]))
        self.status_label = QLabel()
        self.status_label.setObjectName("MutedLabel")
        status.addWidget(self.status_dot)
        status.addWidget(self.status_label, 1)

        self.progress = QProgressBar()
        self.progress.setFixedWidth(140)
        self.progress.setTextVisible(False)
        self.progress.hide()
        status.addPermanentWidget(self.progress)

        self.coordinate_label = QLabel("x —   y —")
        self.coordinate_label.setObjectName("CoordinateLabel")
        self.coordinate_label.setMinimumWidth(130)
        self.coordinate_label.setAlignment(Qt.AlignCenter)
        status.addPermanentWidget(self.coordinate_label)

        self.zoom_label = QLabel("—")
        self.zoom_label.setObjectName("ZoomLabel")
        self.zoom_label.setMinimumWidth(62)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        status.addPermanentWidget(self.zoom_label)

    def _connect_signals(self) -> None:
        self.open_action.triggered.connect(self._choose_image)
        self.open_button.clicked.connect(self._choose_image)
        self.sidebar.openRequested.connect(self._choose_image)
        self.detect_action.triggered.connect(self._request_detection)
        self.detect_button.clicked.connect(self._request_detection)
        self.inspector.detectionRequested.connect(self._forward_detection)
        self.inspector.settingsChanged.connect(self.settingsChanged)
        self.inspector.settingsChanged.connect(self._sync_detection_controls)

        self.export_action.triggered.connect(self._choose_export_path)
        self.inspector.exportRequested.connect(self._choose_export_path)
        self.sidebar.stepSelected.connect(self._workflow_step_selected)

        self.fit_action.triggered.connect(self.canvas.fit_to_image)
        self.actual_pixels_action.triggered.connect(self.canvas.actual_pixels)
        self.zoom_in_action.triggered.connect(self.canvas.zoom_in)
        self.zoom_out_action.triggered.connect(self.canvas.zoom_out)
        self.trace_mode_action.triggered.connect(
            lambda: self.set_interaction_mode(InteractionMode.TRACE)
        )
        self.pan_mode_action.triggered.connect(
            lambda: self.set_interaction_mode(InteractionMode.PAN)
        )

        self.canvas.traceRequested.connect(self._forward_trace)
        self.canvas.cursorPositionChanged.connect(self._show_coordinate)
        self.canvas.cursorLeftImage.connect(lambda: self.coordinate_label.setText("x —   y —"))
        self.canvas.zoomChanged.connect(
            lambda value: self.zoom_label.setText("{:.0f}%".format(value))
        )

        self.inspector.clearAnchorsRequested.connect(self.canvas.clear_anchors)
        self.inspector.clearTracesRequested.connect(self.clearRequested)
        self.inspector.undoRequested.connect(self.undoRequested)
        self.inspector.overlayVisibilityChanged.connect(self.canvas.set_overlay_visible)
        self.inspector.tracesVisibilityChanged.connect(self.canvas.set_traces_visible)
        self.inspector.overlayOpacityChanged.connect(self.canvas.set_overlay_opacity)

    # ------------------------------------------------------------------
    # Controller-facing API
    # ------------------------------------------------------------------
    def set_image(
        self,
        source: ImageSource,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> bool:
        """Set the current source image and unlock analysis controls."""

        if not self.canvas.set_image(source):
            self.set_status("The selected image could not be displayed", "error")
            return False

        if isinstance(source, (str, Path)):
            self._source_path = Path(source)
            display_name = self._source_path.name
        else:
            source_path = (metadata or {}).get("source_path")
            self._source_path = Path(str(source_path)) if source_path else None
            default_name = self._source_path.name if self._source_path else "Untitled image"
            display_name = str((metadata or {}).get("name", default_name))

        size = self.canvas.image_size()
        full_metadata = dict(metadata or {})
        full_metadata.setdefault("size", (size.width(), size.height()))
        full_metadata.setdefault("crs", "Not specified")
        full_metadata.setdefault("resolution", "1 px")

        self.sidebar.set_image_name(display_name)
        self.sidebar.set_image_available(True)
        self.inspector.set_image_available(True)
        self.inspector.set_metadata(full_metadata)
        self._traces_available = False
        self._sync_detection_controls()
        self._sync_export_controls()
        for button in self.mode_buttons.values():
            button.setEnabled(not self._busy)
        self.trace_mode_action.setEnabled(not self._busy)
        self.set_workflow_step("detect")
        self.set_status(
            "Loaded {} × {} pixels".format(size.width(), size.height()), "success"
        )
        return True

    def clear_image(self) -> None:
        self.canvas.clear_image()
        self._source_path = None
        self.sidebar.set_image_name("")
        self.sidebar.set_image_available(False)
        self.inspector.set_image_available(False)
        self.inspector.set_metadata()
        self.inspector.set_metrics()
        self.inspector.set_export_available(False)
        self._traces_available = False
        self._sync_detection_controls()
        self._sync_export_controls()
        self.trace_mode_action.setEnabled(False)
        for mode, button in self.mode_buttons.items():
            button.setEnabled(mode == InteractionMode.PAN)
        self.set_interaction_mode(InteractionMode.PAN)
        self.set_workflow_step("import")
        self.set_status("Open an image to begin", "ready")

    def set_probability_overlay(
        self,
        source: ImageSource,
        opacity: float = 0.55,
        *,
        activate_workflow: bool = True,
    ) -> bool:
        applied = self.canvas.set_probability_overlay(source, opacity)
        if applied:
            self.inspector.opacity_slider.setValue(round(opacity * 100))
            self.inspector.overlay_check.setChecked(True)
        if applied and activate_workflow:
            self.set_workflow_step("trace")
            self.set_interaction_mode(InteractionMode.TRACE)
            self.set_status("Evidence ready — place two trace anchors", "success")
        return applied

    def set_traces(self, traces: Iterable[Iterable[object]]) -> None:
        traces = list(traces)
        self.canvas.set_traces(traces)
        self.canvas.set_traces_visible(self.inspector.traces_check.isChecked())
        self._traces_available = bool(traces)
        self._sync_export_controls()
        if self._traces_available:
            self.set_status("{} vector trace(s) ready".format(len(traces)), "success")

    def append_trace(self, points: Iterable[object], selected: bool = False) -> None:
        self.canvas.append_trace(points, selected)
        self.canvas.set_traces_visible(self.inspector.traces_check.isChecked())
        self._traces_available = True
        self._sync_export_controls()
        self.set_status("Trace added to network", "success")

    def set_metrics(self, metrics: Optional[Mapping[str, object]] = None) -> None:
        self.inspector.set_metrics(metrics)

    def set_model_pack_path(self, path: str) -> None:
        """Select an already-installed local model pack without network access."""

        self.inspector.set_model_pack_path(path)

    def set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = bool(busy)
        self.inspector.set_processing(busy)
        self.inspector.setEnabled(not busy)
        self.sidebar.setEnabled(not busy)
        self.canvas.set_anchor_input_enabled(not busy)
        for mode, button in self.mode_buttons.items():
            button.setEnabled(
                self.canvas.has_image and (not busy or mode == InteractionMode.PAN)
            )
        self.trace_mode_action.setEnabled(self.canvas.has_image and not busy)
        self.pan_mode_action.setEnabled(self.canvas.has_image)
        self.open_button.setEnabled(not busy)
        self.open_action.setEnabled(not busy)
        self._sync_detection_controls()
        self._sync_export_controls()
        if busy:
            self.progress.setRange(0, 0)
            self.progress.show()
            self.set_status(message or "Processing image…", "working")
        else:
            self.progress.hide()
            self.progress.setRange(0, 100)
            if message:
                self.set_status(message, "success")

    def _sync_detection_controls(self, *args) -> None:
        available = (
            self.canvas.has_image
            and not self._busy
            and self.inspector.detection_available
        )
        self.detect_button.setEnabled(available)
        self.detect_action.setEnabled(available)

    def _sync_export_controls(self) -> None:
        available = self._traces_available and not self._busy
        self.inspector.set_export_available(available)
        self.export_action.setEnabled(available)

    def set_progress(self, value: Optional[int], message: str = "") -> None:
        """Show determinate progress, or an activity bar when value is ``None``."""

        self.progress.show()
        if value is None:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(max(0, min(100, int(value))))
        if message:
            self.set_status(message, "working")

    def set_status(self, message: str, level: str = "ready") -> None:
        colours = {
            "ready": COLORS["text_muted"],
            "working": COLORS["blue"],
            "success": COLORS["primary"],
            "warning": COLORS["warning"],
            "error": COLORS["danger"],
        }
        self.status_dot.setStyleSheet(
            "color: {}; padding-left: 7px;".format(colours.get(level, colours["ready"]))
        )
        self.status_label.setText(message)

    def set_workflow_step(self, key: str) -> None:
        titles = {
            "import": ("Import imagery", "Full-resolution raster workspace"),
            "detect": ("Detect fractures", "Evidence, orientation, and error cues"),
            "trace": ("Trace network", "Direction-guided paths between anchors"),
            "review": ("Review network", "Inspect connectivity and error cues"),
            "export": ("Export results", "Georeferenced vector network"),
        }
        if key not in titles:
            raise ValueError("Unknown workflow step: {}".format(key))
        self._current_step = key
        self.sidebar.set_active_step(key)
        self.page_title.setText(titles[key][0])
        self.page_subtitle.setText(titles[key][1])
        self.workflowStepChanged.emit(key)

    def set_interaction_mode(self, mode) -> None:
        mode = InteractionMode(mode)
        button = self.mode_buttons[mode]
        button.setChecked(True)
        self.canvas.set_interaction_mode(mode)

    # ------------------------------------------------------------------
    # Local UI actions
    # ------------------------------------------------------------------
    def _choose_image(self) -> None:
        if self._busy:
            return
        initial = str(self._source_path.parent) if self._source_path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open geological image", initial, self.IMAGE_FILTER
        )
        if not path:
            return
        # Decoding belongs to the controller's worker. Loading a second QPixmap
        # here would duplicate full-resolution memory and block the UI thread.
        self.set_status("Loading full-resolution raster…", "working")
        self.imageOpenRequested.emit(path)

    def _request_detection(self) -> None:
        if not self.canvas.has_image or self._busy:
            return
        settings = self.inspector.detection_settings()
        if settings["profile"] == "model_pack" and not settings["model_pack_path"]:
            self.set_status("Choose a local model pack folder first", "warning")
            return
        self._forward_detection(settings)

    def _forward_detection(self, settings: dict) -> None:
        if not self.canvas.has_image or self._busy:
            return
        if settings.get("profile") == "model_pack" and not settings.get("model_pack_path"):
            self.set_status("Choose a local model pack folder first", "warning")
            return
        self.set_workflow_step("detect")
        self.set_status("Detection requested", "working")
        self.detectionRequested.emit(dict(settings))

    def _forward_trace(self, start: QPointF, end: QPointF) -> None:
        if self._busy:
            return
        self.set_status("Finding direction-aware path…", "working")
        self.traceRequested.emit(start, end, self.inspector.trace_settings())

    def _workflow_step_selected(self, key: str) -> None:
        if key == "export":
            self._choose_export_path()
            return
        self.set_workflow_step(key)
        if key == "trace":
            self.set_interaction_mode(InteractionMode.TRACE)
        elif key in ("detect", "import"):
            self.set_interaction_mode(InteractionMode.PAN)
        elif key == "review":
            self.set_interaction_mode(InteractionMode.INSPECT)

    def _choose_export_path(self) -> None:
        if not self.export_action.isEnabled() and not self.inspector.export_button.isEnabled():
            self.set_status("Run tracing before exporting a network", "warning")
            return
        initial_name = "fracture-network.gpkg"
        if self._source_path:
            initial_name = "{}-fractures.gpkg".format(self._source_path.stem)
            initial_name = str(self._source_path.with_name(initial_name))
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export fracture network", initial_name, self.EXPORT_FILTER
        )
        if not path:
            return
        suffixes = {
            "GeoPackage": ".gpkg",
            "GeoJSON": ".geojson",
            "Shapefile": ".shp",
            "Comma-separated": ".csv",
        }
        if not Path(path).suffix:
            suffix = next(
                (
                    value
                    for name, value in suffixes.items()
                    if selected_filter.startswith(name)
                ),
                ".gpkg",
            )
            path += suffix
        self.exportRequested.emit(path)

    def _show_coordinate(self, point: QPointF) -> None:
        self.coordinate_label.setText("x {:,.1f}   y {:,.1f}".format(point.x(), point.y()))


def create_window(application: Optional[QApplication] = None) -> MainWindow:
    """Construct a themed window for launchers and small integration tests."""

    from .theme import apply_theme

    app = application or QApplication.instance()
    if app is None:
        raise RuntimeError("Create a QApplication before constructing the main window")
    apply_theme(app)
    return MainWindow()
