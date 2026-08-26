"""Small offscreen smoke test for the redesigned desktop shell."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PyQt5.QtCore import QPointF
    from PyQt5.QtGui import QImage
    from PyQt5.QtWidgets import QApplication

    from domstudio.ui import InteractionMode, MainWindow
    from domstudio.ui.theme import COLORS
except ImportError:  # pragma: no cover - permits headless core-only installs
    QApplication = None  # type: ignore[assignment]
    MainWindow = None  # type: ignore[assignment]


@unittest.skipIf(QApplication is None or MainWindow is None, "PyQt5 is not installed")
class MainWindowSmokeTests(unittest.TestCase):
    application = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_constructs_modern_workflow_without_legacy_filter_tabs(self) -> None:
        window = MainWindow()
        try:
            self.assertEqual(window.objectName(), "DOMStudioMainWindow")
            self.assertIn("DOMStudio", window.windowTitle())
            self.assertFalse(window.windowIcon().isNull())
            self.assertIsNotNone(window.canvas)
            self.assertIsNotNone(window.sidebar)
            self.assertIsNotNone(window.inspector)
            actions = window.findChildren(type(window.open_action))
            action_text = " ".join(action.text().lower() for action in actions)
            self.assertNotIn("canny", action_text)
            self.assertNotIn("sobel", action_text)
        finally:
            window.close()
            self.application.processEvents()

    def test_edge_coordinates_snap_inside_the_last_pixel(self) -> None:
        window = MainWindow()
        try:
            image = QImage(160, 80, QImage.Format_RGB888)
            image.fill(0)
            self.assertTrue(window.set_image(image))
            point = QPointF(159.75, 79.75)
            self.assertTrue(window.canvas._image_contains(point))
            snapped = window.canvas._nearest_pixel(point)
            self.assertEqual((snapped.x(), snapped.y()), (159.0, 79.0))
            self.assertFalse(window.canvas._image_contains(QPointF(160.0, 79.0)))
        finally:
            window.close()

    def test_busy_state_locks_mutations_and_restores_export(self) -> None:
        window = MainWindow()
        try:
            image = QImage(64, 64, QImage.Format_RGB888)
            image.fill(0)
            window.set_image(image)
            window.set_traces([((2, 2), (40, 40))])
            self.assertTrue(window.export_action.isEnabled())
            vector_pen = window.canvas._trace_items[0].pen()
            self.assertEqual(vector_pen.color().name().lower(), COLORS["blue"].lower())
            self.assertTrue(vector_pen.isCosmetic())

            window.set_busy(True, "working")
            self.assertFalse(window.export_action.isEnabled())
            self.assertFalse(window.inspector.isEnabled())
            self.assertFalse(window.sidebar.isEnabled())

            window.set_busy(False)
            self.assertTrue(window.export_action.isEnabled())
            self.assertTrue(window.inspector.isEnabled())
        finally:
            window.close()

    def test_overlay_refresh_preserves_review_and_inspect_mode(self) -> None:
        window = MainWindow()
        try:
            image = QImage(64, 64, QImage.Format_RGB888)
            image.fill(0)
            overlay = QImage(64, 64, QImage.Format_RGBA8888)
            overlay.fill(0)
            window.set_image(image)
            window.set_workflow_step("review")
            window.set_interaction_mode(InteractionMode.INSPECT)

            window.set_probability_overlay(overlay, activate_workflow=False)

            self.assertEqual(window._current_step, "review")
            self.assertEqual(window.canvas.interaction_mode, InteractionMode.INSPECT)
        finally:
            window.close()

    def test_header_detection_respects_missing_model_pack(self) -> None:
        window = MainWindow()
        try:
            image = QImage(64, 64, QImage.Format_RGB888)
            image.fill(0)
            window.set_image(image)
            index = window.inspector.model_combo.findData("model_pack")
            window.inspector.model_combo.setCurrentIndex(index)
            self.application.processEvents()
            self.assertFalse(window.inspector.detect_button.isEnabled())
            self.assertFalse(window.detect_button.isEnabled())
            self.assertFalse(window.detect_action.isEnabled())
        finally:
            window.close()

    def test_automatic_vectorization_controls_have_practical_defaults(self) -> None:
        window = MainWindow()
        try:
            settings = window.inspector.detection_settings()
            self.assertTrue(settings["automatic_vectorization"])
            self.assertEqual(settings["weak_threshold"], 0.20)
            self.assertEqual(settings["max_bridge_gap"], 18)
            self.assertEqual(settings["min_vector_length"], 24)
            self.assertTrue(settings["reject_small_loops"])
            self.assertEqual(settings["max_loop_length"], 60)

            image = QImage(64, 64, QImage.Format_RGB888)
            image.fill(0)
            window.set_image(image)
            self.assertTrue(window.inspector.vectorization_options.isEnabled())

            window.inspector.auto_vectorize_check.setChecked(False)
            self.assertFalse(window.inspector.vectorization_options.isEnabled())
            self.assertTrue(window.detect_button.isEnabled())

            window.inspector.auto_vectorize_check.setChecked(True)
            window.inspector.reject_small_loops_check.setChecked(False)
            self.assertFalse(window.inspector.loop_cutoff_controls.isEnabled())

            window.inspector.evidence_threshold_slider.setValue(15)
            self.assertLessEqual(
                window.inspector.weak_threshold_slider.value(), 15
            )
        finally:
            window.close()

    def test_network_summary_exposes_graph_topology(self) -> None:
        window = MainWindow()
        try:
            window.set_metrics(
                {
                    "trace_count": 7,
                    "total_length": "512.0 px",
                    "junction_count": 2,
                    "endpoint_count": 5,
                    "connected_component_count": 3,
                    "closed_cycle_count": 1,
                }
            )

            labels = window.inspector._metric_labels
            self.assertEqual(labels["traces"].text(), "7")
            self.assertEqual(labels["length"].text(), "512.0 px")
            self.assertEqual(labels["junctions"].text(), "2")
            self.assertEqual(labels["endpoints"].text(), "5")
            self.assertEqual(labels["components"].text(), "3")
            self.assertEqual(labels["cycles"].text(), "1")
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
