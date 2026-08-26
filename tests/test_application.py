from __future__ import annotations

import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

# Import the real launcher boundary first. On Windows it intentionally preloads
# the optional ONNX runtime before Qt's native DLLs when that extra is installed.
from domstudio.application import AppController
from domstudio.core import RasterDocument
from domstudio.presentation import array_to_qimage
from domstudio.ui import create_window
from PyQt5.QtCore import QPointF
from PyQt5.QtWidgets import QApplication


class ApplicationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self) -> None:
        self.window = create_window(self.application)
        self.controller = AppController(self.window)

    def tearDown(self) -> None:
        self.window.close()
        self.application.processEvents()

    def wait_for_task(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while self.controller._active_task is not None and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(0.005)
        self.application.processEvents()
        self.assertIsNone(self.controller._active_task, "background task timed out")

    def wait_for_overlay(self, timeout: float = 5.0) -> None:
        deadline = time.monotonic() + timeout
        while (
            self.controller._overlay_timer.isActive()
            or self.controller._overlay_task is not None
        ) and time.monotonic() < deadline:
            self.application.processEvents()
            time.sleep(0.005)
        self.application.processEvents()
        self.assertFalse(self.controller._overlay_timer.isActive())
        self.assertIsNone(self.controller._overlay_task, "overlay render timed out")

    def test_load_detect_trace_workflow_runs_off_the_ui_thread(self) -> None:
        image = np.full((96, 128), 225, dtype=np.uint8)
        cv2.line(image, (8, 28), (118, 62), 20, 3, lineType=cv2.LINE_AA)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "synthetic.png"
            self.assertTrue(cv2.imwrite(str(source), image))

            self.controller.open_image(str(source))
            self.wait_for_task()
            self.assertEqual(self.controller.document.shape, image.shape)

            self.controller.run_detection(
                {
                    "profile": "preview",
                    "evidence_threshold": 0.35,
                    "minimum_trace_length": 10,
                    "automatic_vectorization": False,
                }
            )
            self.wait_for_task()
            self.assertEqual(self.controller.evidence.shape, image.shape)

            self.controller.run_trace(
                QPointF(8.0, 28.0),
                QPointF(118.0, 62.0),
                {"orientation_weight": 0.7, "curvature_weight": 0.35},
            )
            self.wait_for_task()

        self.assertEqual(len(self.controller.traces), 1)
        self.assertGreater(len(self.controller.traces[0]), 1)
        self.assertEqual(self.window._current_step, "review")
        self.assertEqual(self.window.canvas.interaction_mode.value, "inspect")

        self.controller._settings_changed({"evidence_threshold": 0.55})
        self.wait_for_overlay()
        self.assertEqual(self.window._current_step, "review")
        self.assertEqual(self.window.canvas.interaction_mode.value, "inspect")

    def test_uint8_display_preserves_constant_intensity_and_alpha(self) -> None:
        gray = np.full((3, 4), 220, dtype=np.uint8)
        gray_image = array_to_qimage(gray)
        self.assertEqual(gray_image.pixelColor(0, 0).red(), 220)

        rgba = np.zeros((3, 4, 4), dtype=np.uint8)
        rgba[..., :3] = (12, 34, 56)
        rgba[..., 3] = 255
        rgba_image = array_to_qimage(rgba)
        colour = rgba_image.pixelColor(0, 0)
        self.assertEqual((colour.red(), colour.green(), colour.blue()), (12, 34, 56))
        self.assertEqual(colour.alpha(), 255)

    def test_detection_automatically_populates_editable_network_vectors(self) -> None:
        image = np.full((96, 128), 225, dtype=np.uint8)
        cv2.line(image, (8, 28), (118, 62), 20, 3, lineType=cv2.LINE_AA)
        document = RasterDocument(image)
        self.controller.document = document
        self.window.set_image(array_to_qimage(image))
        main_thread = threading.get_ident()
        observed = {}

        class StubVectorizer:
            def __init__(self, settings) -> None:
                observed["settings"] = settings

            def vectorize(self, evidence):
                observed["thread"] = threading.get_ident()
                observed["shape"] = evidence.shape
                return SimpleNamespace(
                    polylines=(
                        np.asarray(((8, 28), (60, 44), (118, 62)), dtype=np.float32),
                        np.asarray(((40, 12), (42, 70)), dtype=np.float32),
                    )
                )

        requested = {
            "profile": "preview",
            "evidence_threshold": 0.45,
            "automatic_vectorization": True,
            "weak_threshold": 0.20,
            "max_bridge_gap": 18,
            "min_vector_length": 24,
            "reject_small_loops": True,
            "max_loop_length": 60,
        }
        with patch("domstudio.application.FractureVectorizer", StubVectorizer):
            self.controller.run_detection(requested)
            self.wait_for_task()

        settings = observed["settings"]
        self.assertNotEqual(observed["thread"], main_thread)
        self.assertEqual(observed["shape"], image.shape)
        self.assertEqual(settings.strong_threshold, 0.45)
        self.assertEqual(settings.weak_threshold, 0.20)
        self.assertEqual(settings.max_bridge_gap, 18.0)
        self.assertEqual(settings.min_vector_length, 24.0)
        self.assertTrue(settings.reject_small_loops)
        self.assertEqual(settings.max_loop_length, 60.0)
        self.assertEqual(len(self.controller.traces), 2)
        self.assertEqual(len(self.window.canvas._trace_items), 2)
        self.assertEqual(self.window.inspector._metric_labels["traces"].text(), "2")
        self.assertEqual(self.window.inspector._metric_labels["endpoints"].text(), "4")
        self.assertEqual(self.window.inspector._metric_labels["components"].text(), "2")
        self.assertEqual(self.window.inspector._metric_labels["cycles"].text(), "0")
        self.assertTrue(self.window.export_action.isEnabled())
        self.assertEqual(self.window._current_step, "review")
        self.assertEqual(self.window.canvas.interaction_mode.value, "inspect")
        self.assertIn("Use Trace for corrections", self.window.status_label.text())

    def test_vectorization_failure_keeps_evidence_and_manual_trace_available(self) -> None:
        image = np.full((64, 80), 180, dtype=np.uint8)
        document = RasterDocument(image)
        self.controller.document = document
        self.window.set_image(array_to_qimage(image))

        class FailingVectorizer:
            def __init__(self, settings) -> None:
                self.settings = settings

            def vectorize(self, evidence):
                raise ValueError("synthetic vectorization failure")

        with patch("domstudio.application.FractureVectorizer", FailingVectorizer):
            self.controller.run_detection(
                {
                    "profile": "preview",
                    "evidence_threshold": 0.45,
                    "automatic_vectorization": True,
                    "weak_threshold": 0.20,
                }
            )
            self.wait_for_task()

        self.assertIsNotNone(self.controller.evidence)
        self.assertFalse(self.controller.traces)
        self.assertFalse(self.window.canvas._overlay_item.pixmap().isNull())
        self.assertEqual(self.window._current_step, "trace")
        self.assertEqual(self.window.canvas.interaction_mode.value, "trace")
        self.assertIn("automatic vectorization failed", self.window.status_label.text())
        self.assertIn("Manual Trace remains available", self.window.status_label.text())


if __name__ == "__main__":
    unittest.main()
