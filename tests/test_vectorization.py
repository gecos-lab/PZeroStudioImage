"""Synthetic topology tests for automatic fracture vectorization."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from domstudio.core import (
    EvidenceMap,
    FractureVectorizer,
    VectorizationSettings,
)


def _blank_evidence(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    probability = np.full(shape, 0.01, dtype=np.float32)
    orientation = np.zeros(shape, dtype=np.float32)
    uncertainty = np.full(shape, 0.9, dtype=np.float32)
    return probability, orientation, uncertainty


def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    x1, y1 = start
    x2, y2 = end
    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    error = dx + dy
    points: list[tuple[int, int]] = []
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            return points
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x1 += sx
        if doubled <= dx:
            error += dx
            y1 += sy


def _paint_line(
    probability: np.ndarray,
    orientation: np.ndarray,
    uncertainty: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    value: float = 0.95,
) -> None:
    angle = float(np.mod(np.arctan2(end[1] - start[1], end[0] - start[0]), np.pi))
    for x, y in _bresenham(start, end):
        probability[y, x] = value
        orientation[y, x] = angle
        uncertainty[y, x] = 0.02


class FractureVectorizerTests(unittest.TestCase):
    def test_short_faint_gap_becomes_one_end_to_end_vector(self) -> None:
        probability, orientation, uncertainty = _blank_evidence((64, 82))
        _paint_line(probability, orientation, uncertainty, (7, 31), (35, 31))
        _paint_line(probability, orientation, uncertainty, (42, 31), (74, 31))
        # Faint oriented evidence is deliberately below the weak hysteresis
        # threshold, so only the topology-aware bridge can retain it.
        probability[31, 36:42] = 0.14
        uncertainty[31, 36:42] = 0.10

        result = FractureVectorizer().vectorize(
            EvidenceMap(probability, orientation, uncertainty)
        )

        self.assertEqual(result.polyline_count, 1, result.diagnostics)
        trace = result.polylines[0]
        self.assertLessEqual(float(trace[:, 0].min()), 7.0)
        self.assertGreaterEqual(float(trace[:, 0].max()), 74.0)
        self.assertEqual(result.diagnostics["bridges_accepted"], 1)
        self.assertTrue(np.all(result.skeleton_mask[31, 7:75]))

    def test_y_network_preserves_three_branches_and_shared_junction(self) -> None:
        probability, orientation, uncertainty = _blank_evidence((82, 82))
        junction = (40, 39)
        _paint_line(probability, orientation, uncertainty, junction, (14, 12))
        _paint_line(probability, orientation, uncertainty, junction, (67, 12))
        _paint_line(probability, orientation, uncertainty, junction, (40, 73))

        result = FractureVectorizer(
            VectorizationSettings(simplify_tolerance=0.25)
        ).vectorize(EvidenceMap(probability, orientation, uncertainty))

        self.assertEqual(result.polyline_count, 3, result.diagnostics)
        endpoint_counts: dict[tuple[float, float], int] = {}
        for trace in result.polylines:
            for endpoint in (trace[0], trace[-1]):
                key = (round(float(endpoint[0]), 3), round(float(endpoint[1]), 3))
                endpoint_counts[key] = endpoint_counts.get(key, 0) + 1
        self.assertEqual(max(endpoint_counts.values()), 3)

    def test_thick_y_and_x_are_split_at_one_shared_graph_junction(self) -> None:
        settings = VectorizationSettings(
            strong_threshold=0.5,
            weak_threshold=0.2,
            min_vector_length=5.0,
            min_spur_length=2.0,
            simplify_tolerance=0.2,
        )
        networks = (
            (
                3,
                (
                    ((64, 64), (64, 10)),
                    ((64, 64), (18, 110)),
                    ((64, 64), (110, 110)),
                ),
            ),
            (
                4,
                (
                    ((64, 64), (64, 10)),
                    ((64, 64), (64, 118)),
                    ((64, 64), (10, 64)),
                    ((64, 64), (118, 64)),
                ),
            ),
        )
        for expected_branches, segments in networks:
            with self.subTest(expected_branches=expected_branches):
                probability, orientation, uncertainty = _blank_evidence((128, 128))
                for start, end in segments:
                    cv2.line(probability, start, end, 0.9, thickness=3)
                    support = np.zeros(probability.shape, dtype=np.uint8)
                    cv2.line(support, start, end, 1, thickness=3)
                    angle = np.mod(
                        np.arctan2(end[1] - start[1], end[0] - start[0]), np.pi
                    )
                    orientation[support > 0] = angle
                    uncertainty[support > 0] = 0.02

                result = FractureVectorizer(settings).vectorize(
                    EvidenceMap(probability, orientation, uncertainty)
                )

                self.assertEqual(result.polyline_count, expected_branches, result.diagnostics)
                endpoint_counts: dict[tuple[float, float], int] = {}
                for trace in result.polylines:
                    for endpoint in (trace[0], trace[-1]):
                        key = (round(float(endpoint[0]), 3), round(float(endpoint[1]), 3))
                        endpoint_counts[key] = endpoint_counts.get(key, 0) + 1
                self.assertEqual(max(endpoint_counts.values()), expected_branches)

    def test_small_circle_and_short_fragment_are_rejected(self) -> None:
        probability, orientation, uncertainty = _blank_evidence((80, 94))
        _paint_line(probability, orientation, uncertainty, (8, 65), (84, 65))
        _paint_line(probability, orientation, uncertainty, (8, 8), (12, 8))
        centre = np.asarray((43.0, 27.0))
        circle = [
            tuple(np.rint(centre + 4.0 * np.asarray((np.cos(a), np.sin(a)))).astype(int))
            for a in np.linspace(0.0, 2.0 * np.pi, 33)
        ]
        for first, second in zip(circle[:-1], circle[1:]):
            _paint_line(probability, orientation, uncertainty, first, second)

        result = FractureVectorizer(
            VectorizationSettings(
                min_vector_length=10.0,
                reject_small_loops=True,
                max_loop_length=40.0,
            )
        ).vectorize(EvidenceMap(probability, orientation, uncertainty))

        self.assertEqual(result.polyline_count, 1, result.diagnostics)
        self.assertGreater(result.total_length, 70.0)
        self.assertGreaterEqual(result.diagnostics["small_loops_rejected"], 1)
        self.assertGreaterEqual(
            result.diagnostics["short_vectors_rejected"]
            + result.diagnostics["early_components_rejected"],
            1,
        )

    def test_close_parallel_fractures_are_not_cross_connected(self) -> None:
        probability, orientation, uncertainty = _blank_evidence((64, 86))
        for start, end in (
            ((8, 25), (35, 25)),
            ((42, 25), (77, 25)),
            ((8, 31), (43, 31)),
            ((50, 31), (77, 31)),
        ):
            _paint_line(probability, orientation, uncertainty, start, end)
        probability[25, 36:42] = 0.13
        probability[31, 44:50] = 0.13
        uncertainty[25, 36:42] = 0.1
        uncertainty[31, 44:50] = 0.1

        result = FractureVectorizer(
            VectorizationSettings(max_bridge_gap=9.0)
        ).vectorize(EvidenceMap(probability, orientation, uncertainty))

        self.assertEqual(result.polyline_count, 2, result.diagnostics)
        self.assertEqual(result.diagnostics["bridges_accepted"], 2)
        for trace in result.polylines:
            self.assertLessEqual(float(np.ptp(trace[:, 1])), 0.5)
        self.assertFalse(np.any(result.bridge_mask[26:31]))
        self.assertGreaterEqual(result.diagnostics["bridge_rejected_geometry"], 1)

    def test_all_invalid_evidence_returns_clean_empty_result(self) -> None:
        probability = np.ones((20, 24), dtype=np.float32)
        orientation = np.zeros_like(probability)
        uncertainty = np.zeros_like(probability)
        valid = np.zeros_like(probability, dtype=bool)

        result = FractureVectorizer().vectorize(
            EvidenceMap(probability, orientation, uncertainty, valid)
        )

        self.assertEqual(result.polyline_count, 0)
        self.assertFalse(np.any(result.accepted_mask))
        self.assertFalse(np.any(result.skeleton_mask))
        self.assertFalse(np.any(result.bridge_mask))


if __name__ == "__main__":
    unittest.main()
