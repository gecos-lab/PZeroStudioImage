"""Sidebar and inspector components for the DOMStudio main window."""

from collections import OrderedDict
from typing import Dict, Mapping, Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


WORKFLOW_STEPS = OrderedDict(
    (
        ("import", ("01", "Import", "Source imagery")),
        ("detect", ("02", "Detect", "Evidence field")),
        ("trace", ("03", "Trace", "Network paths")),
        ("review", ("04", "Review", "Topology & quality")),
        ("export", ("05", "Export", "Vectors & report")),
    )
)


class WorkflowSidebar(QFrame):
    """Compact task navigation; intentionally not a collection of filter tabs."""

    openRequested = pyqtSignal()
    stepSelected = pyqtSignal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Sidebar")
        self.setFixedWidth(224)
        self._buttons: Dict[str, QToolButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 20, 16, 18)
        layout.setSpacing(7)

        brand = QLabel("DOMSTUDIO")
        brand.setObjectName("BrandMark")
        subtitle = QLabel("Fracture network studio")
        subtitle.setObjectName("BrandSub")
        layout.addWidget(brand)
        layout.addWidget(subtitle)
        layout.addSpacing(23)

        project_caption = QLabel("WORKFLOW")
        project_caption.setObjectName("SectionCaption")
        layout.addWidget(project_caption)
        layout.addSpacing(3)

        for key, (number, title, caption) in WORKFLOW_STEPS.items():
            button = QToolButton()
            button.setObjectName("WorkflowStep")
            button.setText("{}   {}\n      {}".format(number, title, caption))
            button.setCheckable(True)
            button.setAutoExclusive(True)
            button.setToolButtonStyle(Qt.ToolButtonTextOnly)
            button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            button.setFixedHeight(52)
            button.clicked.connect(lambda checked=False, step=key: self.stepSelected.emit(step))
            self._buttons[key] = button
            layout.addWidget(button)

        layout.addStretch(1)

        self._image_name = QLabel("No image loaded")
        self._image_name.setObjectName("MutedLabel")
        self._image_name.setWordWrap(True)
        layout.addWidget(self._image_name)

        open_button = QPushButton("Open image")
        open_button.setObjectName("PrimaryButton")
        open_button.setMinimumHeight(36)
        open_button.clicked.connect(self.openRequested)
        layout.addWidget(open_button)

        footer = QLabel("Research workspace")
        footer.setObjectName("SectionCaption")
        footer.setAlignment(Qt.AlignCenter)
        layout.addSpacing(7)
        layout.addWidget(footer)

        self.set_image_available(False)
        self.set_active_step("import")

    def set_active_step(self, key: str) -> None:
        button = self._buttons.get(key)
        if button is not None:
            button.setChecked(True)

    def set_image_available(self, available: bool) -> None:
        for key, button in self._buttons.items():
            button.setEnabled(key == "import" or available)

    def set_image_name(self, name: str) -> None:
        self._image_name.setText(name or "No image loaded")
        self._image_name.setToolTip(name or "")


class InspectorCard(QFrame):
    """Consistent bordered group used throughout the property inspector."""

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("InspectorCard")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(12, 11, 12, 12)
        self.layout.setSpacing(9)
        label = QLabel(title)
        label.setObjectName("PanelTitle")
        self.layout.addWidget(label)


class ValueSlider(QWidget):
    """A labelled integer slider with a compact live value."""

    valueChanged = pyqtSignal(int)

    def __init__(
        self,
        label: str,
        minimum: int,
        maximum: int,
        value: int,
        suffix: str = "%",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._suffix = suffix
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        title = QLabel(label)
        title.setObjectName("MutedLabel")
        self.value_label = QLabel()
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(self.value_label)
        layout.addLayout(header)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(self._value_changed)
        layout.addWidget(self.slider)
        self._value_changed(value)

    def _value_changed(self, value: int) -> None:
        self.value_label.setText("{}{}".format(value, self._suffix))
        self.valueChanged.emit(value)

    def value(self) -> int:
        return self.slider.value()

    def setValue(self, value: int) -> None:
        self.slider.setValue(value)


class InspectorPanel(QFrame):
    """Model, tracing, layer, and result controls for a selected image."""

    detectionRequested = pyqtSignal(dict)
    settingsChanged = pyqtSignal(dict)
    clearAnchorsRequested = pyqtSignal()
    clearTracesRequested = pyqtSignal()
    undoRequested = pyqtSignal()
    exportRequested = pyqtSignal()
    overlayVisibilityChanged = pyqtSignal(bool)
    tracesVisibilityChanged = pyqtSignal(bool)
    overlayOpacityChanged = pyqtSignal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("Inspector")
        self.setFixedWidth(292)
        self._image_available = False
        self._processing = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(16, 17, 16, 13)
        header_layout.setSpacing(3)
        title = QLabel("Inspector")
        title.setObjectName("PageTitle")
        description = QLabel("Detection and network controls")
        description.setObjectName("MutedLabel")
        header_layout.addWidget(title)
        header_layout.addWidget(description)
        outer.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll, 1)

        content = QWidget()
        self._content_layout = QVBoxLayout(content)
        self._content_layout.setContentsMargins(12, 4, 12, 14)
        self._content_layout.setSpacing(10)
        scroll.setWidget(content)

        self._build_source_card()
        self._build_detection_card()
        self._build_vectorization_card()
        self._build_trace_card()
        self._build_layer_card()
        self._build_results_card()
        self._content_layout.addStretch(1)

        footer = QWidget()
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(12, 10, 12, 12)
        self.export_button = QPushButton("Export network…")
        self.export_button.setMinimumHeight(35)
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self.exportRequested)
        footer_layout.addWidget(self.export_button)
        outer.addWidget(footer)

        self.set_image_available(False)

    def _build_source_card(self) -> None:
        card = InspectorCard("Source image")
        grid = QGridLayout()
        grid.setHorizontalSpacing(9)
        grid.setVerticalSpacing(6)
        self._metadata_labels: Dict[str, QLabel] = {}
        fields = (("size", "Dimensions"), ("crs", "Reference"), ("resolution", "Resolution"))
        for row, (key, title) in enumerate(fields):
            name = QLabel(title)
            name.setObjectName("MutedLabel")
            value = QLabel("—")
            value.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._metadata_labels[key] = value
            grid.addWidget(name, row, 0)
            grid.addWidget(value, row, 1)
        grid.setColumnStretch(1, 1)
        card.layout.addLayout(grid)
        self._content_layout.addWidget(card)

    def _build_detection_card(self) -> None:
        card = InspectorCard("Fracture detection")
        model_label = QLabel("Inference profile")
        model_label.setObjectName("MutedLabel")
        self.model_combo = QComboBox()
        self.model_combo.addItem("Fast preview", "preview")
        self.model_combo.addItem("Analytical detector", "analytical")
        self.model_combo.addItem("Local model pack", "model_pack")
        self.model_combo.currentIndexChanged.connect(self._profile_changed)
        card.layout.addWidget(model_label)
        card.layout.addWidget(self.model_combo)

        self.model_pack_row = QWidget()
        model_pack_layout = QVBoxLayout(self.model_pack_row)
        model_pack_layout.setContentsMargins(0, 0, 0, 0)
        model_pack_layout.setSpacing(5)
        pack_label = QLabel("Model pack folder")
        pack_label.setObjectName("MutedLabel")
        model_pack_layout.addWidget(pack_label)
        pack_picker = QHBoxLayout()
        pack_picker.setContentsMargins(0, 0, 0, 0)
        pack_picker.setSpacing(5)
        self.model_pack_edit = QLineEdit()
        self.model_pack_edit.setReadOnly(True)
        self.model_pack_edit.setPlaceholderText("No pack selected")
        self.model_pack_edit.setToolTip("Path to a local, versioned model pack")
        browse_button = QPushButton("Browse…")
        browse_button.setObjectName("GhostButton")
        browse_button.clicked.connect(self._choose_model_pack)
        pack_picker.addWidget(self.model_pack_edit, 1)
        pack_picker.addWidget(browse_button)
        model_pack_layout.addLayout(pack_picker)
        local_only = QLabel("Uses local files only; nothing is downloaded.")
        local_only.setObjectName("SectionCaption")
        local_only.setWordWrap(True)
        model_pack_layout.addWidget(local_only)
        self.model_pack_row.setVisible(False)
        card.layout.addWidget(self.model_pack_row)

        self.evidence_threshold_slider = ValueSlider("Evidence threshold", 5, 95, 45)
        self.evidence_threshold_slider.valueChanged.connect(self._emit_settings)
        card.layout.addWidget(self.evidence_threshold_slider)

        self.detect_button = QPushButton("Run detection")
        self.detect_button.setObjectName("PrimaryButton")
        self.detect_button.setMinimumHeight(34)
        self.detect_button.clicked.connect(
            lambda: self.detectionRequested.emit(self.detection_settings())
        )
        card.layout.addWidget(self.detect_button)
        self._content_layout.addWidget(card)

    def _build_vectorization_card(self) -> None:
        card = InspectorCard("Automatic vectorization")
        self.auto_vectorize_check = QCheckBox("Build network after detection")
        self.auto_vectorize_check.setChecked(True)
        self.auto_vectorize_check.setToolTip(
            "Convert the evidence field directly into editable vector traces; "
            "rerun detection to apply changes."
        )
        card.layout.addWidget(self.auto_vectorize_check)

        self.vectorization_options = QWidget()
        options = QVBoxLayout(self.vectorization_options)
        options.setContentsMargins(0, 0, 0, 0)
        options.setSpacing(9)

        self.weak_threshold_slider = ValueSlider(
            "Weak continuation threshold", 5, 45, 20
        )
        self.weak_threshold_slider.setToolTip(
            "Lower-evidence pixels may continue, but cannot seed, a component. "
            "This is image-scale dependent; rerun detection to apply."
        )
        self.weak_threshold_slider.valueChanged.connect(self._emit_settings)
        options.addWidget(self.weak_threshold_slider)

        bridge_row = QHBoxLayout()
        bridge_label = QLabel("Maximum bridge gap")
        bridge_label.setObjectName("MutedLabel")
        self.max_bridge_gap_spin = QSpinBox()
        self.max_bridge_gap_spin.setRange(0, 128)
        self.max_bridge_gap_spin.setValue(18)
        self.max_bridge_gap_spin.setSuffix(" px")
        self.max_bridge_gap_spin.setFixedWidth(91)
        self.max_bridge_gap_spin.setToolTip(
            "Maximum evidence-supported gap joined between compatible endpoints. "
            "This is image-scale dependent; rerun detection to apply."
        )
        self.max_bridge_gap_spin.valueChanged.connect(self._emit_settings)
        bridge_row.addWidget(bridge_label)
        bridge_row.addStretch(1)
        bridge_row.addWidget(self.max_bridge_gap_spin)
        options.addLayout(bridge_row)

        length_row = QHBoxLayout()
        length_label = QLabel("Minimum vector length")
        length_label.setObjectName("MutedLabel")
        self.vector_min_length_spin = QSpinBox()
        self.vector_min_length_spin.setRange(1, 10000)
        self.vector_min_length_spin.setValue(24)
        self.vector_min_length_spin.setSuffix(" px")
        self.vector_min_length_spin.setFixedWidth(91)
        self.vector_min_length_spin.setToolTip(
            "Shorter automatic vectors are removed. This is image-scale dependent; "
            "rerun detection to apply."
        )
        self.vector_min_length_spin.valueChanged.connect(self._emit_settings)
        length_row.addWidget(length_label)
        length_row.addStretch(1)
        length_row.addWidget(self.vector_min_length_spin)
        options.addLayout(length_row)

        self.reject_small_loops_check = QCheckBox("Reject small closed loops")
        self.reject_small_loops_check.setChecked(True)
        options.addWidget(self.reject_small_loops_check)

        self.loop_cutoff_controls = QWidget()
        loop_row = QHBoxLayout(self.loop_cutoff_controls)
        loop_row.setContentsMargins(0, 0, 0, 0)
        loop_label = QLabel("Loop length cutoff")
        loop_label.setObjectName("MutedLabel")
        self.loop_cutoff_spin = QSpinBox()
        self.loop_cutoff_spin.setRange(1, 10000)
        self.loop_cutoff_spin.setValue(60)
        self.loop_cutoff_spin.setSuffix(" px")
        self.loop_cutoff_spin.setFixedWidth(91)
        self.loop_cutoff_spin.setToolTip(
            "Closed loops shorter than this are removed. This is image-scale "
            "dependent; rerun detection to apply."
        )
        self.loop_cutoff_spin.valueChanged.connect(self._emit_settings)
        loop_row.addWidget(loop_label)
        loop_row.addStretch(1)
        loop_row.addWidget(self.loop_cutoff_spin)
        options.addWidget(self.loop_cutoff_controls)

        card.layout.addWidget(self.vectorization_options)
        self._content_layout.addWidget(card)

        self.auto_vectorize_check.toggled.connect(self._vectorization_enabled_changed)
        self.reject_small_loops_check.toggled.connect(self._loop_rejection_changed)
        self.evidence_threshold_slider.valueChanged.connect(
            self._limit_weak_continuation_threshold
        )

    def _build_trace_card(self) -> None:
        card = InspectorCard("Guided tracing")
        hint = QLabel("Choose Trace, then place start and end anchors on the image.")
        hint.setObjectName("MutedLabel")
        hint.setWordWrap(True)
        card.layout.addWidget(hint)

        min_length_row = QHBoxLayout()
        min_length_label = QLabel("Minimum accepted length")
        min_length_label.setObjectName("MutedLabel")
        self.min_length_spin = QSpinBox()
        self.min_length_spin.setRange(0, 10000)
        self.min_length_spin.setValue(24)
        self.min_length_spin.setSuffix(" px")
        self.min_length_spin.setFixedWidth(91)
        self.min_length_spin.valueChanged.connect(self._emit_settings)
        min_length_row.addWidget(min_length_label)
        min_length_row.addStretch(1)
        min_length_row.addWidget(self.min_length_spin)
        card.layout.addLayout(min_length_row)

        corridor_row = QHBoxLayout()
        corridor_label = QLabel("Search half-width")
        corridor_label.setObjectName("MutedLabel")
        self.corridor_spin = QSpinBox()
        self.corridor_spin.setRange(8, 512)
        self.corridor_spin.setValue(64)
        self.corridor_spin.setSuffix(" px")
        self.corridor_spin.setToolTip(
            "Smaller corridors are faster; increase this for strongly curved fractures."
        )
        self.corridor_spin.setFixedWidth(91)
        self.corridor_spin.valueChanged.connect(self._emit_settings)
        corridor_row.addWidget(corridor_label)
        corridor_row.addStretch(1)
        corridor_row.addWidget(self.corridor_spin)
        card.layout.addLayout(corridor_row)

        self.orientation_slider = ValueSlider("Orientation adherence", 0, 100, 70)
        self.orientation_slider.valueChanged.connect(self._emit_settings)
        card.layout.addWidget(self.orientation_slider)

        self.curvature_slider = ValueSlider("Curvature penalty", 0, 100, 35)
        self.curvature_slider.valueChanged.connect(self._emit_settings)
        card.layout.addWidget(self.curvature_slider)

        actions = QHBoxLayout()
        clear = QPushButton("Clear points")
        clear.setObjectName("GhostButton")
        clear.clicked.connect(self.clearAnchorsRequested)
        undo = QPushButton("Undo trace")
        undo.setObjectName("GhostButton")
        undo.clicked.connect(self.undoRequested)
        actions.addWidget(clear)
        actions.addWidget(undo)
        card.layout.addLayout(actions)
        self._content_layout.addWidget(card)

    def _build_layer_card(self) -> None:
        card = InspectorCard("Layers")
        self.overlay_check = QCheckBox("Evidence / error-proxy overlay")
        self.overlay_check.setChecked(True)
        self.overlay_check.toggled.connect(self.overlayVisibilityChanged)
        self.traces_check = QCheckBox("Vector traces")
        self.traces_check.setChecked(True)
        self.traces_check.toggled.connect(self.tracesVisibilityChanged)
        card.layout.addWidget(self.overlay_check)
        card.layout.addWidget(self.traces_check)

        self.opacity_slider = ValueSlider("Overlay opacity", 0, 100, 55)
        self.opacity_slider.valueChanged.connect(
            lambda value: self.overlayOpacityChanged.emit(value / 100.0)
        )
        card.layout.addWidget(self.opacity_slider)

        clear = QPushButton("Clear vector traces")
        clear.setObjectName("DangerButton")
        clear.clicked.connect(self.clearTracesRequested)
        card.layout.addWidget(clear)
        self._content_layout.addWidget(card)

    def _build_results_card(self) -> None:
        self.results_card = InspectorCard("Network summary")
        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(3)
        self._metric_labels: Dict[str, QLabel] = {}
        fields = (
            ("traces", "EDGES"),
            ("length", "TOTAL LENGTH"),
            ("junctions", "JUNCTIONS"),
            ("endpoints", "ENDPOINTS"),
            ("components", "COMPONENTS"),
            ("cycles", "CYCLES"),
        )
        for index, (key, title) in enumerate(fields):
            column = index % 3
            row = (index // 3) * 2
            value = QLabel("—")
            value.setObjectName("MetricValue")
            value.setAlignment(Qt.AlignCenter)
            name = QLabel(title)
            name.setObjectName("MetricLabel")
            name.setAlignment(Qt.AlignCenter)
            self._metric_labels[key] = value
            grid.addWidget(value, row, column)
            grid.addWidget(name, row + 1, column)
        grid.setVerticalSpacing(6)
        self.results_card.layout.addLayout(grid)
        self._content_layout.addWidget(self.results_card)

    def detection_settings(self) -> dict:
        return {
            "profile": self.model_combo.currentData(),
            "model_pack_path": self.model_pack_edit.text().strip() or None,
            "evidence_threshold": self.evidence_threshold_slider.value() / 100.0,
            "minimum_trace_length": self.min_length_spin.value(),
            "automatic_vectorization": self.auto_vectorize_check.isChecked(),
            "weak_threshold": self.weak_threshold_slider.value() / 100.0,
            "max_bridge_gap": self.max_bridge_gap_spin.value(),
            "min_vector_length": self.vector_min_length_spin.value(),
            "reject_small_loops": self.reject_small_loops_check.isChecked(),
            "max_loop_length": self.loop_cutoff_spin.value(),
        }

    def trace_settings(self) -> dict:
        return {
            "corridor_radius": self.corridor_spin.value(),
            "orientation_weight": self.orientation_slider.value() / 100.0,
            "curvature_weight": self.curvature_slider.value() / 100.0,
        }

    def settings(self) -> dict:
        values = self.detection_settings()
        values.update(self.trace_settings())
        return values

    def _emit_settings(self, *args) -> None:
        self.settingsChanged.emit(self.settings())

    def _profile_changed(self, *args) -> None:
        is_model_pack = self.model_combo.currentData() == "model_pack"
        self.model_pack_row.setVisible(is_model_pack)
        self._update_detection_availability()
        self._emit_settings()

    def _vectorization_enabled_changed(self, enabled: bool) -> None:
        self._sync_vectorization_options()
        self._emit_settings()

    def _loop_rejection_changed(self, enabled: bool) -> None:
        self._sync_vectorization_options()
        self._emit_settings()

    def _sync_vectorization_options(self) -> None:
        options_available = (
            self.auto_vectorize_check.isChecked()
            and self._image_available
            and not self._processing
        )
        self.vectorization_options.setEnabled(options_available)
        self.loop_cutoff_controls.setEnabled(
            options_available and self.reject_small_loops_check.isChecked()
        )

    def _limit_weak_continuation_threshold(self, strong_percent: int) -> None:
        maximum = max(5, int(strong_percent))
        self.weak_threshold_slider.slider.setMaximum(maximum)
        if self.weak_threshold_slider.value() > maximum:
            self.weak_threshold_slider.setValue(maximum)

    def _choose_model_pack(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select local model pack")
        if path:
            self.set_model_pack_path(path)

    def set_model_pack_path(self, path: str) -> None:
        self.model_pack_edit.setText(path or "")
        self.model_pack_edit.setToolTip(path or "Path to a local, versioned model pack")
        profile_changed = False
        if path:
            index = self.model_combo.findData("model_pack")
            if index >= 0 and index != self.model_combo.currentIndex():
                profile_changed = True
                self.model_combo.setCurrentIndex(index)
        self._update_detection_availability()
        if not profile_changed:
            self._emit_settings()

    def _update_detection_availability(self) -> None:
        controls_available = self._image_available and not self._processing
        pack_is_valid = (
            self.model_combo.currentData() != "model_pack"
            or bool(self.model_pack_edit.text().strip())
        )
        self.detect_button.setEnabled(controls_available and pack_is_valid)
        self.model_combo.setEnabled(controls_available)
        self.model_pack_row.setEnabled(controls_available)
        self.evidence_threshold_slider.setEnabled(controls_available)
        self.auto_vectorize_check.setEnabled(controls_available)
        vector_controls_available = (
            controls_available and self.auto_vectorize_check.isChecked()
        )
        self.vectorization_options.setEnabled(vector_controls_available)
        self.loop_cutoff_controls.setEnabled(
            vector_controls_available and self.reject_small_loops_check.isChecked()
        )
        self.min_length_spin.setEnabled(controls_available)
        self.corridor_spin.setEnabled(controls_available)
        self.orientation_slider.setEnabled(controls_available)
        self.curvature_slider.setEnabled(controls_available)

    @property
    def detection_available(self) -> bool:
        """Whether the current profile has everything needed to run."""

        return bool(
            self._image_available
            and not self._processing
            and (
                self.model_combo.currentData() != "model_pack"
                or self.model_pack_edit.text().strip()
            )
        )

    def set_image_available(self, available: bool) -> None:
        self._image_available = bool(available)
        self._update_detection_availability()

    def set_processing(self, processing: bool) -> None:
        self._processing = bool(processing)
        self._update_detection_availability()
        self.detect_button.setText("Analysing…" if processing else "Run detection")

    def set_export_available(self, available: bool) -> None:
        self.export_button.setEnabled(available)

    def set_metadata(self, metadata: Optional[Mapping[str, object]] = None) -> None:
        metadata = metadata or {}
        aliases = {
            "size": ("size", "dimensions"),
            "crs": ("crs", "reference"),
            "resolution": ("resolution", "pixel_size"),
        }
        for destination, keys in aliases.items():
            value = next((metadata[key] for key in keys if key in metadata), "—")
            if destination == "size" and isinstance(value, (tuple, list)) and len(value) >= 2:
                value = "{} × {} px".format(value[0], value[1])
            self._metadata_labels[destination].setText(str(value))

    def set_metrics(self, metrics: Optional[Mapping[str, object]] = None) -> None:
        metrics = metrics or {}
        aliases = {
            "traces": ("traces", "trace_count"),
            "length": ("length", "total_length"),
            "junctions": ("junctions", "junction_count"),
            "endpoints": ("endpoints", "endpoint_count"),
            "components": (
                "components",
                "component_count",
                "connected_component_count",
            ),
            "cycles": ("cycles", "cycle_count", "closed_cycle_count"),
        }
        for destination, keys in aliases.items():
            value = next((metrics[key] for key in keys if key in metrics), "—")
            self._metric_labels[destination].setText(str(value))
