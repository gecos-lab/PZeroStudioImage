"""Qt application controller for the redesigned DOMStudio workflow."""

from __future__ import annotations

import logging
import math
from pathlib import Path
import sys
import threading
from typing import Callable, Optional

import cv2
import numpy as np

# PyQt5 and ONNX Runtime ship native libraries with colliding Windows DLL
# initialization order in some environments. Loading the optional runtime first
# avoids that conflict; failure remains non-fatal for the standard analytical app.
try:  # pragma: no cover - depends on an optional native runtime
    import onnxruntime as _onnxruntime_preload
except ImportError:  # base installations intentionally omit ONNX Runtime
    _onnxruntime_preload = None

from PyQt5.QtCore import QPointF, QThread, QThreadPool, QTimer, Qt
from PyQt5.QtWidgets import QApplication, QMessageBox

from domstudio import __version__
from domstudio.core import (
    CorridorAStarTracer,
    DetectorConfig,
    EvidenceMap,
    MultiscaleFractureDetector,
    RasterDocument,
    TraceConfig,
    TraceRequest,
    TraceResult,
    VectorizationResult,
    FractureVectorizer,
    VectorizationSettings,
)
from domstudio.evaluation import polyline_length
from domstudio.network import FractureNetwork
from domstudio.presentation import array_to_qimage, evidence_to_qimage
from domstudio.services import export_traces, load_raster
from domstudio.tasks import BackgroundTask, CancellationToken
from domstudio.ui import MainWindow, create_window


LOGGER = logging.getLogger("domstudio")


def _preview_evidence(
    document: RasterDocument,
    maximum_side: int = 1600,
    *,
    cancel: Optional[Callable[[], bool]] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> EvidenceMap:
    """Run a quick multiscale pass and restore evidence to source resolution."""

    height, width = document.shape
    scale = min(1.0, maximum_side / max(height, width))
    detector = MultiscaleFractureDetector(
        DetectorConfig(scales=(0.8, 1.4, 2.2), probability_gamma=0.8)
    )
    if scale >= 1.0:
        result = detector.predict(document, cancel=cancel, progress=progress)
        result.metadata["profile"] = "preview"
        return result

    target = (max(2, round(width * scale)), max(2, round(height * scale)))
    normalized = document.normalized_grayscale()
    reduced = cv2.resize(normalized, target, interpolation=cv2.INTER_AREA)
    coarse = detector.predict(
        RasterDocument(reduced), cancel=cancel, progress=progress
    )

    probability = cv2.resize(coarse.probability, (width, height), interpolation=cv2.INTER_LINEAR)
    uncertainty = cv2.resize(coarse.uncertainty, (width, height), interpolation=cv2.INTER_LINEAR)
    # Interpolate doubled-angle vectors so axial orientations do not wrap at pi.
    axis_x = cv2.resize(
        np.cos(2.0 * coarse.orientation),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    axis_y = cv2.resize(
        np.sin(2.0 * coarse.orientation),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    orientation = np.mod(0.5 * np.arctan2(axis_y, axis_x), np.pi).astype(np.float32)
    metadata = dict(coarse.metadata)
    metadata.update({"profile": "preview", "analysis_scale": scale})
    return EvidenceMap(
        probability=probability,
        orientation=orientation,
        uncertainty=uncertainty,
        valid_mask=document.valid_mask(),
        metadata=metadata,
    )


class AppController:
    """Coordinate UI, scientific core, and file services without UI blocking."""

    def __init__(self, window: MainWindow) -> None:
        self.window = window
        self.document: Optional[RasterDocument] = None
        self.evidence: Optional[EvidenceMap] = None
        self.vectorization_result: Optional[VectorizationResult] = None
        self.traces: list[np.ndarray] = []
        self.settings: dict[str, object] = {
            "profile": "preview",
            "evidence_threshold": 0.45,
            "minimum_trace_length": 24,
            "automatic_vectorization": True,
            "weak_threshold": 0.20,
            "max_bridge_gap": 18,
            "min_vector_length": 24,
            "reject_small_loops": True,
            "max_loop_length": 60,
            "corridor_radius": 64,
            "orientation_weight": 0.70,
            "curvature_weight": 0.35,
        }
        self.pool = QThreadPool.globalInstance()
        ideal = max(1, QThread.idealThreadCount() - 1)
        self.pool.setMaxThreadCount(min(4, ideal))
        self._active_task: Optional[BackgroundTask] = None
        self._task_generation = 0
        self._overlay_task: Optional[BackgroundTask] = None
        self._overlay_generation = 0
        self._overlay_timer = QTimer(window)
        self._overlay_timer.setSingleShot(True)
        self._overlay_timer.setInterval(120)
        self._overlay_timer.timeout.connect(self._render_overlay)
        self._model_detector_lock = threading.Lock()
        self._model_detector_cache: Optional[
            tuple[Path, tuple[tuple[int, int], tuple[int, int]], object]
        ] = None

        window.imageOpenRequested.connect(self.open_image)
        window.detectionRequested.connect(self.run_detection)
        window.traceRequested.connect(self.run_trace)
        window.settingsChanged.connect(self._settings_changed)
        window.undoRequested.connect(self.undo_trace)
        window.clearRequested.connect(self.clear_traces)
        window.exportRequested.connect(self.export_network)

    def _start_task(
        self,
        operation: Callable[[CancellationToken, Callable[[int, str], None]], object],
        completed: Callable[[object], None],
        message: str,
    ) -> None:
        if self._active_task is not None:
            self._active_task.token.cancel()
        self._task_generation += 1
        generation = self._task_generation

        task: BackgroundTask

        def execute():
            return operation(task.token, task.signals.progress.emit)

        task = BackgroundTask(execute)
        self._active_task = task
        task.signals.completed.connect(
            lambda result: completed(result) if generation == self._task_generation else None
        )
        task.signals.failed.connect(
            lambda summary, details: self._task_failed(generation, summary, details)
        )
        task.signals.progress.connect(
            lambda value, label: self.window.set_progress(value, label)
            if generation == self._task_generation
            else None
        )
        task.signals.finished.connect(lambda: self._task_finished(generation))
        self.window.set_busy(True, message)
        self.pool.start(task)

    def _task_failed(self, generation: int, summary: str, details: str) -> None:
        if generation != self._task_generation:
            return
        LOGGER.error("Background operation failed: %s\n%s", summary, details)
        self.window.set_status(summary, "error")
        dialog = QMessageBox(QMessageBox.Critical, "DOMStudio", summary, parent=self.window)
        dialog.setDetailedText(details)
        dialog.exec_()

    def _task_finished(self, generation: int) -> None:
        if generation != self._task_generation:
            return
        self._active_task = None
        self.window.set_busy(False)

    def open_image(self, path: str) -> None:
        self._cancel_overlay_render()

        def read(token: CancellationToken, progress: Callable[[int, str], None]):
            document = load_raster(path)
            return document, array_to_qimage(document.image), self._metadata(document)

        self._start_task(
            read,
            self._image_loaded,
            "Loading full-resolution raster…",
        )

    def _image_loaded(self, result: object) -> None:
        if not isinstance(result, tuple) or len(result) != 3:
            raise TypeError("Raster reader returned an invalid result")
        document, display_image, metadata = result
        if not isinstance(document, RasterDocument):
            raise TypeError("Raster reader returned an invalid document")
        self.document = document
        self.evidence = None
        self.vectorization_result = None
        self.traces.clear()
        self.window.set_image(display_image, metadata)
        self.window.set_traces([])
        self._update_metrics()
        self.window.set_status(
            f"Loaded {document.width:,} × {document.height:,} pixels at native resolution",
            "success",
        )

    @staticmethod
    def _metadata(document: RasterDocument) -> dict[str, object]:
        resolution = "1 × 1 pixel"
        if document.transform is not None:
            transform = document.transform
            try:
                x_size = math.hypot(float(transform.a), float(transform.d))
                y_size = math.hypot(float(transform.b), float(transform.e))
                resolution = f"{x_size:.6g} × {y_size:.6g} map units"
            except (AttributeError, TypeError, ValueError):
                resolution = "Affine transform available"
        return {
            "name": Path(document.source_path).name if document.source_path else "Untitled image",
            "source_path": str(document.source_path) if document.source_path else "",
            "size": (document.width, document.height),
            "crs": str(document.crs) if document.crs else "Pixel coordinates",
            "resolution": resolution,
        }

    def run_detection(self, requested: dict) -> None:
        if self.document is None:
            self.window.set_status("Open an image before running detection", "warning")
            return
        self.settings.update(requested)
        document = self.document
        profile = str(requested.get("profile", "preview"))
        evidence_threshold = float(requested.get("evidence_threshold", 0.45))
        automatic_vectorization = bool(
            requested.get("automatic_vectorization", True)
        )
        self.window.canvas.clear_anchors()
        self._cancel_overlay_render()

        def detect(token: CancellationToken, progress: Callable[[int, str], None]):
            def detector_progress(completed: int, total: int) -> None:
                progress(
                    round(90.0 * completed / max(total, 1)),
                    f"Computing multiscale evidence — scale {completed}/{total}",
                )

            if profile == "preview":
                evidence = _preview_evidence(
                    document,
                    cancel=lambda: token.cancelled,
                    progress=detector_progress,
                )
            elif profile == "model_pack":
                pack_path = requested.get("model_pack_path")
                if not pack_path:
                    raise ValueError("Select a local model pack before inference.")
                detector = self._model_pack_detector(str(pack_path))
                evidence = detector.predict(
                    document,
                    cancel=lambda: token.cancelled,
                    progress=lambda completed, total: progress(
                        round(90.0 * completed / max(total, 1)),
                        f"Running model tiles — {completed}/{total}",
                    ),
                )
            elif profile == "analytical":
                evidence = MultiscaleFractureDetector().predict(
                    document,
                    cancel=lambda: token.cancelled,
                    progress=detector_progress,
                )
            else:
                raise ValueError(f"Unknown inference profile: {profile}")
            vectorization_result = None
            vectorization_error = None
            if automatic_vectorization and not token.cancelled:
                progress(92, "Building automatic fracture network")
                try:
                    vector_settings = VectorizationSettings(
                        strong_threshold=evidence_threshold,
                        weak_threshold=float(requested.get("weak_threshold", 0.20)),
                        max_bridge_gap=float(requested.get("max_bridge_gap", 18.0)),
                        min_vector_length=float(
                            requested.get("min_vector_length", 24.0)
                        ),
                        reject_small_loops=bool(
                            requested.get("reject_small_loops", True)
                        ),
                        max_loop_length=float(
                            requested.get("max_loop_length", 60.0)
                        ),
                    )
                    vectorization_result = FractureVectorizer(
                        vector_settings
                    ).vectorize(evidence)
                except Exception as exc:  # evidence remains useful for manual tracing
                    LOGGER.exception("Automatic vectorization failed")
                    vectorization_error = str(exc).strip() or type(exc).__name__

            progress(98, "Rendering evidence overlay")
            return (
                evidence,
                evidence_to_qimage(evidence, evidence_threshold),
                vectorization_result,
                vectorization_error,
            )

        self._start_task(detect, self._detection_ready, f"Running {profile} detection…")

    @staticmethod
    def _pack_fingerprint(
        root: Path,
        model_path: Path,
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        manifest_stat = (root / "model.json").stat()
        model_stat = model_path.stat()
        return (
            (manifest_stat.st_size, manifest_stat.st_mtime_ns),
            (model_stat.st_size, model_stat.st_mtime_ns),
        )

    def _model_pack_detector(self, pack_path: str):
        """Reuse a verified ORT session until either pack file changes."""

        from domstudio.learning.model_pack import OnnxFractureDetector

        root = Path(pack_path).expanduser().resolve()
        with self._model_detector_lock:
            cached = self._model_detector_cache
            if cached is not None and cached[0] == root:
                cached_detector = cached[2]
                try:
                    fingerprint = self._pack_fingerprint(
                        root, cached_detector.manifest.model_path
                    )
                except OSError:
                    fingerprint = None
                if fingerprint == cached[1]:
                    return cached_detector
            detector = OnnxFractureDetector(root)
            fingerprint = self._pack_fingerprint(root, detector.manifest.model_path)
            self._model_detector_cache = (root, fingerprint, detector)
            return detector

    def _detection_ready(self, result: object) -> None:
        if not isinstance(result, tuple) or len(result) not in (2, 3, 4):
            raise TypeError("Detector returned an invalid result")
        evidence, overlay_image = result[:2]
        vectorization_result = result[2] if len(result) >= 3 else None
        vectorization_error = result[3] if len(result) >= 4 else None
        if not isinstance(evidence, EvidenceMap):
            raise TypeError("Detector returned invalid evidence")
        if self.document is None or evidence.shape != self.document.shape:
            raise ValueError("Detector evidence is not aligned to the source raster")
        self.evidence = evidence
        automatic_requested = bool(
            self.settings.get("automatic_vectorization", True)
        )
        self.vectorization_result = (
            vectorization_result
            if automatic_requested and isinstance(vectorization_result, VectorizationResult)
            else None
        )
        traces, invalid_polylines = self._vector_polylines(vectorization_result)
        self.traces = traces if automatic_requested else []
        self.window.set_traces(self.traces)
        self.window.canvas.clear_anchors()
        self._update_metrics()
        self.window.set_probability_overlay(overlay_image, activate_workflow=False)
        detector_name = str(evidence.metadata.get("detector", "fracture detector"))
        analysis_scale = float(evidence.metadata.get("analysis_scale", 1.0))
        scale_note = (
            f" at {analysis_scale:.0%} analysis scale" if analysis_scale < 1.0 else ""
        )
        if vectorization_error:
            message = str(vectorization_error).replace("\n", " ")[:180]
            self.window.set_workflow_step("trace")
            self.window.set_interaction_mode("trace")
            self.window.set_status(
                f"Evidence ready{scale_note}; automatic vectorization failed: "
                f"{message}. Manual Trace remains available.",
                "warning",
            )
        elif automatic_requested and self.traces:
            self.window.set_workflow_step("review")
            self.window.set_interaction_mode("inspect")
            suffix = (
                f"; skipped {invalid_polylines} invalid path(s)"
                if invalid_polylines
                else ""
            )
            diagnostics = getattr(vectorization_result, "diagnostics", {})
            bridge_count = int(diagnostics.get("bridges_accepted", 0))
            bridge_note = (
                f", {bridge_count} supported gap(s) repaired"
                if bridge_count
                else ""
            )
            self.window.set_status(
                f"{detector_name}: {len(self.traces)} automatic network vector(s) "
                f"ready{bridge_note}{scale_note}{suffix}. Use Trace for corrections.",
                "warning" if invalid_polylines else "success",
            )
        elif automatic_requested:
            self.window.set_workflow_step("trace")
            self.window.set_interaction_mode("trace")
            self.window.set_status(
                f"{detector_name} evidence ready{scale_note}; no automatic vectors "
                "met the settings. Adjust them or use manual Trace.",
                "warning",
            )
        else:
            self.window.set_workflow_step("trace")
            self.window.set_interaction_mode("trace")
            self.window.set_status(
                f"{detector_name} evidence ready{scale_note} — place two anchors",
                "success",
            )

    @staticmethod
    def _vector_polylines(result: object) -> tuple[list[np.ndarray], int]:
        """Copy valid vectorizer output into the controller's editable network."""

        if result is None:
            return [], 0
        polylines = getattr(result, "polylines", None)
        if polylines is None:
            LOGGER.warning("Vectorization result has no polylines field")
            return [], 1
        accepted: list[np.ndarray] = []
        invalid = 0
        for points in polylines:
            try:
                array = np.asarray(points, dtype=np.float64)
            except (TypeError, ValueError):
                invalid += 1
                continue
            if (
                array.ndim != 2
                or array.shape[1] != 2
                or len(array) < 2
                or not np.all(np.isfinite(array))
            ):
                invalid += 1
                continue
            accepted.append(np.ascontiguousarray(array))
        return accepted, invalid

    def _settings_changed(self, settings: dict) -> None:
        old_threshold = float(self.settings.get("evidence_threshold", 0.45))
        self.settings.update(settings)
        threshold_changed = (
            float(self.settings.get("evidence_threshold", 0.45)) != old_threshold
        )
        if self.evidence is not None and threshold_changed:
            self._overlay_timer.start()

    def _render_overlay(self) -> None:
        if self.evidence is None:
            return
        self._cancel_overlay_render()
        self._overlay_generation += 1
        generation = self._overlay_generation
        evidence = self.evidence
        evidence_threshold = float(self.settings.get("evidence_threshold", 0.45))
        task = BackgroundTask(evidence_to_qimage, evidence, evidence_threshold)
        self._overlay_task = task
        task.signals.completed.connect(
            lambda image: self.window.set_probability_overlay(
                image, activate_workflow=False
            )
            if generation == self._overlay_generation
            else None
        )
        task.signals.failed.connect(
            lambda summary, details: self._overlay_failed(generation, summary, details)
        )
        task.signals.finished.connect(lambda: self._overlay_finished(generation))
        self.pool.start(task)

    def _cancel_overlay_render(self) -> None:
        self._overlay_timer.stop()
        if self._overlay_task is not None:
            self._overlay_task.token.cancel()
        self._overlay_generation += 1
        self._overlay_task = None

    def _overlay_failed(self, generation: int, summary: str, details: str) -> None:
        if generation != self._overlay_generation:
            return
        LOGGER.error("Overlay rendering failed: %s\n%s", summary, details)
        self.window.set_status(f"Overlay refresh failed: {summary}", "error")

    def _overlay_finished(self, generation: int) -> None:
        if generation == self._overlay_generation:
            self._overlay_task = None

    def run_trace(self, start: QPointF, end: QPointF, requested: dict) -> None:
        if self.evidence is None:
            self.window.set_status("Run detection before tracing", "warning")
            return
        self.settings.update(requested)
        evidence = self.evidence
        radius = float(requested.get("corridor_radius", 64.0))
        request = TraceRequest(
            (start.x(), start.y()),
            (end.x(), end.y()),
            corridor_radius=radius,
            max_expansions=2_000_000,
        )
        config = TraceConfig(
            orientation_weight=float(requested.get("orientation_weight", 0.70)),
            curvature_weight=float(requested.get("curvature_weight", 0.35)),
            uncertainty_weight=0.35,
            gap_threshold=max(
                0.05,
                min(0.65, float(self.settings.get("evidence_threshold", 0.45))),
            ),
        )
        tracer = CorridorAStarTracer(config)

        def trace(token: CancellationToken, progress: Callable[[int, str], None]):
            def report(snapshot) -> None:
                value = min(95, round(snapshot.fraction * 100.0))
                progress(value, f"Tracing path — {snapshot.expanded_states:,} states explored")

            return tracer.trace(evidence, request, cancel=lambda: token.cancelled, progress=report)

        self._start_task(trace, self._trace_ready, "Finding direction-aware path…")

    def _trace_ready(self, result: object) -> None:
        trace = result
        if not isinstance(trace, TraceResult):
            raise TypeError("Tracer returned an invalid result")
        if not trace.succeeded:
            self.window.set_status(trace.message or "No trace was found", "warning")
            return
        points = trace.simplified_xy(tolerance=0.75).astype(np.float64)
        minimum = float(self.settings.get("minimum_trace_length", 24))
        if len(points) < 2 or polyline_length(points) < minimum:
            self.window.set_status(
                f"Trace rejected because it is shorter than {minimum:g} pixels", "warning"
            )
            return
        self.traces.append(points)
        self.window.append_trace(points, selected=False)
        self.window.canvas.clear_anchors()
        self.window.set_workflow_step("review")
        self.window.set_interaction_mode("inspect")
        self._update_metrics()

    def undo_trace(self) -> None:
        if not self.traces:
            self.window.set_status("There is no trace to undo", "warning")
            return
        self.traces.pop()
        self.window.set_traces(self.traces)
        self._update_metrics()

    def clear_traces(self) -> None:
        self.traces.clear()
        self.window.set_traces([])
        self.window.canvas.clear_anchors()
        self._update_metrics()
        self.window.set_status("Derived vector traces cleared", "ready")

    def _update_metrics(self) -> None:
        summary = FractureNetwork.from_traces(self.traces).summary()
        self.window.set_metrics(
            {
                "traces": summary.trace_count,
                "length": f"{summary.total_length:,.1f} px",
                "junctions": summary.junction_count,
                "endpoints": summary.endpoint_count,
                "components": summary.connected_component_count,
                "cycles": summary.closed_cycle_count,
            }
        )

    def export_network(self, path: str) -> None:
        if self.document is None or not self.traces:
            self.window.set_status("There are no vector traces to export", "warning")
            return
        document = self.document
        traces = [trace.copy() for trace in self.traces]

        def write(token: CancellationToken, progress: Callable[[int, str], None]):
            return export_traces(path, traces, document)

        self._start_task(write, self._export_ready, "Writing georeferenced vector network…")

    def _export_ready(self, result: object) -> None:
        destination = Path(result)
        self.window.set_workflow_step("export")
        self.window.set_status(f"Exported {destination.name}", "success")

    def shutdown(self) -> None:
        """Stop accepting background results while Qt is shutting down."""

        self._overlay_timer.stop()
        self._cancel_overlay_render()
        if self._active_task is not None:
            self._active_task.token.cancel()
            self._task_generation += 1


def main(argv: Optional[list[str]] = None) -> int:
    """Launch DOMStudio and return the Qt event-loop exit status."""

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    app = QApplication.instance()
    if app is None:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
        app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("DOMStudio")
    app.setApplicationDisplayName("DOMStudio — Geological Fracture Tracing")
    app.setApplicationVersion(__version__)
    window = create_window(app)
    controller = AppController(window)
    # QApplication owns the window; this reference keeps the Python controller alive.
    setattr(app, "_domstudio_controller", controller)
    app.aboutToQuit.connect(controller.shutdown)
    window.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
