"""Tests for full-resolution raster handling, detection, and A* tracing."""

from __future__ import annotations

import heapq
import math
import unittest

import numpy as np
from scipy.spatial import cKDTree

from domstudio.core import (
    CorridorAStarTracer,
    DetectorConfig,
    EvidenceMap,
    MultiscaleFractureDetector,
    RasterDocument,
    TraceConfig,
    TraceRequest,
    TraceStatus,
)


class RasterDocumentTests(unittest.TestCase):
    def test_preserves_full_resolution_dtype_and_affine_roundtrip(self) -> None:
        source = np.arange(37 * 53 * 3, dtype=np.uint16).reshape(37, 53, 3)
        transform = (2.25, 0.35, 420_000.0, -0.15, -3.5, 4_810_000.0)

        document = RasterDocument(
            source,
            transform=transform,
            crs="EPSG:32632",
            source_path="outcrop.tif",
        )

        self.assertEqual(document.shape, (37, 53))
        self.assertEqual(document.image.shape, source.shape)
        self.assertEqual(document.image.dtype, source.dtype)
        self.assertTrue(np.array_equal(document.image, source))
        self.assertEqual(document.band_count, 3)

        pixels = np.asarray(((0.0, 0.0), (17.25, 9.5), (52.0, 36.0)))
        for center in (False, True):
            world = document.pixel_to_world(pixels, center=center)
            recovered = document.world_to_pixel(world, center=center)
            # The inverse operates on large projected coordinates, so allow a
            # small amount of ordinary double-precision cancellation.
            np.testing.assert_allclose(recovered, pixels, rtol=0.0, atol=2.0e-10)

    def test_nodata_mask_does_not_mutate_or_resize_source(self) -> None:
        source = np.arange(35, dtype=np.float32).reshape(5, 7)
        source[2, 4] = -9999.0
        original = source.copy()

        document = RasterDocument(source, nodata=-9999.0)
        normalized = document.normalized_grayscale(0.0, 100.0)

        self.assertEqual(normalized.shape, source.shape)
        self.assertFalse(document.valid_mask()[2, 4])
        self.assertTrue(np.array_equal(document.image, original))
        self.assertTrue(np.isfinite(normalized).all())

    def test_explicit_dataset_mask_is_combined_with_finite_and_nodata_rules(self) -> None:
        source = np.arange(20, dtype=np.float32).reshape(4, 5)
        source[0, 0] = np.nan
        validity = np.ones((4, 5), dtype=bool)
        validity[2, 3] = False

        document = RasterDocument(source, validity_mask=validity)
        combined = document.valid_mask()

        self.assertFalse(combined[0, 0])
        self.assertFalse(combined[2, 3])
        self.assertEqual(int(combined.sum()), 18)


class DeterministicDetectorTests(unittest.TestCase):
    @staticmethod
    def _dark_fracture_raster() -> RasterDocument:
        image = np.ones((96, 112), dtype=np.float32)
        image[:, 54:57] = 0.0
        return RasterDocument(image)

    def test_outputs_are_repeatable_bounded_and_source_aligned(self) -> None:
        document = self._dark_fracture_raster()
        detector = MultiscaleFractureDetector(
            DetectorConfig(scales=(0.8, 1.4, 2.2), polarity="dark")
        )

        first = detector.predict(document)
        second = detector.predict(document)

        self.assertEqual(first.shape, document.shape)
        self.assertEqual(first.probability.dtype, np.float32)
        self.assertEqual(first.orientation.dtype, np.float32)
        self.assertEqual(first.uncertainty.dtype, np.float32)
        np.testing.assert_array_equal(first.probability, second.probability)
        np.testing.assert_array_equal(first.orientation, second.orientation)
        np.testing.assert_array_equal(first.uncertainty, second.uncertainty)
        self.assertTrue(np.isfinite(first.probability).all())
        self.assertGreaterEqual(float(first.probability.min()), 0.0)
        self.assertLessEqual(float(first.probability.max()), 1.0)
        self.assertGreaterEqual(float(first.orientation.min()), 0.0)
        self.assertLess(float(first.orientation.max()), float(np.pi))
        self.assertGreaterEqual(float(first.uncertainty.min()), 0.0)
        self.assertLessEqual(float(first.uncertainty.max()), 1.0)
        self.assertEqual(first.metadata["detector"], "multiscale_hessian_ridge")

        fracture_response = float(np.median(first.probability[8:-8, 55]))
        background_response = float(np.median(first.probability[8:-8, 15]))
        self.assertGreater(fracture_response, background_response + 0.25)

    def test_detection_can_be_cancelled_between_scales(self) -> None:
        detector = MultiscaleFractureDetector(DetectorConfig(scales=(0.8, 1.4)))
        with self.assertRaises(InterruptedError):
            detector.predict(self._dark_fracture_raster(), cancel=lambda: True)


class DirectionAwareTracingTests(unittest.TestCase):
    @staticmethod
    def _curved_evidence() -> tuple[EvidenceMap, np.ndarray]:
        height, width = 72, 88
        probability = np.full((height, width), 0.003, dtype=np.float32)
        uncertainty = np.full((height, width), 0.95, dtype=np.float32)
        orientation = np.zeros((height, width), dtype=np.float32)

        x = np.arange(7, 81, dtype=np.int32)
        # A connected, oblique arch: consecutive samples move by at most one
        # pixel vertically, so it is a valid 8-neighbour centerline.
        phase = (x - x[0]) / float(x[-1] - x[0])
        y = np.rint(17.0 + 22.0 * np.sin(np.pi * phase)).astype(np.int32)
        centerline = np.column_stack((x, y))

        tangent = np.gradient(centerline.astype(np.float32), axis=0)
        angles = np.mod(np.arctan2(tangent[:, 1], tangent[:, 0]), np.pi)
        for (px, py), angle in zip(centerline, angles):
            probability[py, px] = 0.999
            uncertainty[py, px] = 0.005
            orientation[py, px] = angle

        return EvidenceMap(probability, orientation, uncertainty), centerline

    def test_astar_follows_curved_oriented_fracture(self) -> None:
        evidence, centerline = self._curved_evidence()
        tracer = CorridorAStarTracer(
            TraceConfig(
                probability_weight=1.6,
                orientation_weight=0.8,
                uncertainty_weight=0.5,
                curvature_weight=0.08,
                gap_weight=2.0,
                gap_threshold=0.3,
            )
        )
        result = tracer.trace(
            evidence,
            TraceRequest(
                tuple(centerline[0]),
                tuple(centerline[-1]),
                corridor_radius=28.0,
                max_expansions=500_000,
            ),
        )

        self.assertEqual(result.status, TraceStatus.SUCCESS, result.message)
        self.assertTrue(result.succeeded)
        np.testing.assert_array_equal(result.path_xy[0], centerline[0])
        np.testing.assert_array_equal(result.path_xy[-1], centerline[-1])
        self.assertGreater(len(result.path_xy), 65)

        distances = cKDTree(centerline).query(result.path_xy, k=1)[0]
        self.assertLessEqual(float(distances.max()), np.sqrt(2.0))
        self.assertGreater(float(np.mean(distances <= 1.0)), 0.95)
        # The route must follow the arch rather than shortcutting along the
        # horizontal endpoint chord.
        self.assertGreater(int(result.path_xy[:, 1].max()), 35)

    def test_optimized_astar_matches_exhaustive_direction_state_dijkstra(self) -> None:
        """Guard the admissible bounds and incumbent pruning against regressions."""

        generator = np.random.default_rng(20260825)
        for _ in range(8):
            shape = (11, 13)
            probability = generator.uniform(0.02, 1.0, shape).astype(np.float32)
            orientation = generator.uniform(0.0, np.pi, shape).astype(np.float32)
            uncertainty = generator.uniform(0.0, 0.9, shape).astype(np.float32)
            evidence = EvidenceMap(probability, orientation, uncertainty)
            config = TraceConfig(
                probability_weight=1.25,
                orientation_weight=0.65,
                uncertainty_weight=0.4,
                curvature_weight=0.3,
                gap_weight=0.8,
                gap_threshold=0.35,
            )
            tracer = CorridorAStarTracer(config)
            request = TraceRequest((1, 1), (11, 9), corridor_radius=30.0)

            result = tracer.trace(evidence, request)
            reference_cost = self._exhaustive_cost(evidence, request, tracer)

            self.assertEqual(result.status, TraceStatus.SUCCESS)
            self.assertAlmostEqual(result.cost, reference_cost, places=5)

    @staticmethod
    def _exhaustive_cost(
        evidence: EvidenceMap,
        request: TraceRequest,
        tracer: CorridorAStarTracer,
    ) -> float:
        """Full direction-state Dijkstra without A* bounds or pruning."""

        x_offset, y_offset, allowed = tracer._make_corridor(evidence, request)
        height, width = allowed.shape
        probability = evidence.probability[
            y_offset : y_offset + height, x_offset : x_offset + width
        ]
        orientation = evidence.orientation[
            y_offset : y_offset + height, x_offset : x_offset + width
        ]
        uncertainty = evidence.uncertainty[
            y_offset : y_offset + height, x_offset : x_offset + width
        ]
        config = tracer.config
        clipped = np.clip(probability, config.probability_epsilon, 1.0)
        gap = np.maximum(
            0.0,
            (config.gap_threshold - probability) / config.gap_threshold,
        )
        traversal = (
            config.base_cost
            + config.probability_weight * -np.log(clipped)
            + config.uncertainty_weight * uncertainty
            + config.gap_weight * gap**2
        )
        direction_xy = (
            (0, -1),
            (1, -1),
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
        )
        direction_angles = np.asarray(
            [math.atan2(dy, dx) for dx, dy in direction_xy]
        )
        reliability = probability * (1.0 - uncertainty)
        start_x, start_y = request.start_pixel
        end_x, end_y = request.end_pixel
        start = (start_y - y_offset, start_x - x_offset)
        goal = (end_y - y_offset, end_x - x_offset)
        distances: dict[tuple[int, int, int], float] = {}
        frontier: list[tuple[float, int, int, int]] = []
        for incoming in range(8):
            state = (start[0], start[1], incoming)
            distances[state] = 0.0
            heapq.heappush(frontier, (0.0, *state))

        while frontier:
            cost, y, x, incoming = heapq.heappop(frontier)
            state = (y, x, incoming)
            if cost > distances[state]:
                continue
            if (y, x) == goal:
                return cost
            for next_direction, (dx, dy) in enumerate(direction_xy):
                next_x, next_y = x + dx, y + dy
                if not (0 <= next_x < width and 0 <= next_y < height):
                    continue
                if not allowed[next_y, next_x]:
                    continue
                axial_direction = next_direction % 4
                movement_angle = direction_angles[axial_direction]
                source_orientation = config.orientation_weight * reliability[y, x] * (
                    math.sin(movement_angle - float(orientation[y, x])) ** 2
                )
                target_orientation = (
                    config.orientation_weight
                    * reliability[next_y, next_x]
                    * math.sin(
                        movement_angle - float(orientation[next_y, next_x])
                    )
                    ** 2
                )
                step_length = math.sqrt(2.0) if dx and dy else 1.0
                edge = 0.5 * step_length * (
                    float(traversal[y, x])
                    + float(traversal[next_y, next_x])
                    + source_orientation
                    + target_orientation
                )
                turn_difference = abs(
                    float(direction_angles[incoming] - direction_angles[next_direction])
                )
                turn = min(turn_difference, 2.0 * math.pi - turn_difference)
                candidate = cost + edge + config.curvature_weight * turn**2
                next_state = (next_y, next_x, next_direction)
                if candidate < distances.get(next_state, math.inf):
                    distances[next_state] = candidate
                    heapq.heappush(frontier, (candidate, *next_state))
        return math.inf


if __name__ == "__main__":
    unittest.main()
