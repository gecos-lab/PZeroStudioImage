"""Tests for vector-network summaries and geometry-first evaluation."""

from __future__ import annotations

import math
import unittest

import numpy as np

from domstudio.evaluation import evaluate_geometry, polyline_length
from domstudio.network import FractureNetwork


class FractureNetworkTests(unittest.TestCase):
    def test_summary_clusters_shared_endpoints_into_a_junction(self) -> None:
        traces = [
            np.asarray(((0.0, 10.0), (10.0, 10.0))),
            np.asarray(((10.0, 0.0), (10.2, 10.1))),
            np.asarray(((10.1, 10.2), (20.0, 20.0))),
        ]

        summary = FractureNetwork.from_traces(traces, snap_tolerance=0.5).summary()

        self.assertEqual(summary.trace_count, 3)
        self.assertEqual(summary.endpoint_count, 3)
        self.assertEqual(summary.junction_count, 1)
        expected_length = sum(polyline_length(trace) for trace in traces)
        self.assertAlmostEqual(summary.total_length, expected_length, places=10)
        self.assertEqual(summary.units, "px")


class GeometryEvaluationTests(unittest.TestCase):
    def test_identical_trace_with_reversed_vertex_order_is_perfect(self) -> None:
        reference = np.asarray(((0.0, 0.0), (5.0, 4.0), (10.0, 5.0)))
        metrics = evaluate_geometry(
            reference[::-1], reference, tolerance=0.01, sample_spacing=0.25
        )

        self.assertAlmostEqual(metrics.precision, 1.0)
        self.assertAlmostEqual(metrics.recall, 1.0)
        self.assertAlmostEqual(metrics.f1, 1.0)
        # Resampling starts from each polyline's first endpoint. When order is
        # reversed and the length is not an exact multiple of the spacing, the
        # two sample grids differ by a tiny fraction of that spacing.
        self.assertLess(metrics.mean_symmetric_distance, 0.01)
        self.assertLess(metrics.hausdorff_distance, 0.01)
        self.assertAlmostEqual(metrics.endpoint_error, 0.0, places=10)
        self.assertAlmostEqual(metrics.length_relative_error, 0.0, places=10)

    def test_one_pixel_offset_is_scored_by_tolerance_and_distance(self) -> None:
        reference = np.asarray(((0.0, 0.0), (20.0, 0.0)))
        prediction = reference + np.asarray((0.0, 1.0))

        metrics = evaluate_geometry(
            prediction, reference, tolerance=1.01, sample_spacing=0.5
        )

        self.assertAlmostEqual(metrics.precision, 1.0)
        self.assertAlmostEqual(metrics.recall, 1.0)
        self.assertAlmostEqual(metrics.f1, 1.0)
        self.assertAlmostEqual(metrics.mean_symmetric_distance, 1.0)
        self.assertAlmostEqual(metrics.hausdorff_distance, 1.0)
        self.assertAlmostEqual(metrics.endpoint_error, 1.0)
        self.assertAlmostEqual(metrics.length_relative_error, 0.0)
        self.assertTrue(math.isfinite(metrics.mean_symmetric_distance))


if __name__ == "__main__":
    unittest.main()
