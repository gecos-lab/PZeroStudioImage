"""Topology-aware conversion of dense fracture evidence into vector traces.

This module intentionally avoids contour extraction: contours describe the two
sides of a thick mask, not a geological fracture centreline.  Instead it uses
geodesic hysteresis, deterministic thinning, conservative endpoint bridging,
and an explicit skeleton graph whose branches share junction coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_propagation, label, map_coordinates
from scipy.spatial import cKDTree

from .models import EvidenceMap
from .tracing import simplify_polyline


_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
)
_CONNECTIVITY_8 = np.ones((3, 3), dtype=np.uint8)


@dataclass(frozen=True)
class VectorizationSettings:
    """Controls hysteresis, gap repair, and graph pruning.

    Thresholds operate on ``P * (1 - uncertainty_weight * U)``.  The strong
    threshold supplies seeds and the weak threshold supplies only pixels that
    are 8-connected to a seed.  Thus low-confidence islands are not accepted
    merely because they exceed the weak threshold.
    """

    strong_threshold: float = 0.55
    weak_threshold: float = 0.25
    uncertainty_weight: float = 0.5
    ridge_nms: bool = True
    nms_radius: float = 1.0
    max_bridge_gap: float = 10.0
    max_bridge_angle_degrees: float = 30.0
    bridge_min_mean_evidence: float = 0.05
    tangent_lookback: int = 6
    min_vector_length: float = 8.0
    min_mean_vector_evidence: float = 0.65
    min_spur_length: float = 4.0
    reject_small_loops: bool = True
    max_loop_length: float = 30.0
    simplify_tolerance: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 <= self.weak_threshold <= self.strong_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= weak_threshold <= strong_threshold <= 1"
            )
        if not 0.0 <= self.uncertainty_weight <= 1.0:
            raise ValueError("uncertainty_weight must be in [0, 1]")
        if self.nms_radius <= 0.0:
            raise ValueError("nms_radius must be positive")
        if self.max_bridge_gap < 0.0:
            raise ValueError("max_bridge_gap must be non-negative")
        if not 0.0 < self.max_bridge_angle_degrees < 90.0:
            raise ValueError("max_bridge_angle_degrees must be in (0, 90)")
        if not 0.0 <= self.bridge_min_mean_evidence <= 1.0:
            raise ValueError("bridge_min_mean_evidence must be in [0, 1]")
        if self.tangent_lookback < 1:
            raise ValueError("tangent_lookback must be a positive integer")
        if self.min_vector_length < 0.0 or self.min_spur_length < 0.0:
            raise ValueError("minimum vector and spur lengths must be non-negative")
        if not 0.0 <= self.min_mean_vector_evidence <= 1.0:
            raise ValueError("min_mean_vector_evidence must be in [0, 1]")
        if self.max_loop_length < 0.0:
            raise ValueError("max_loop_length must be non-negative")
        if self.simplify_tolerance < 0.0:
            raise ValueError("simplify_tolerance must be non-negative")


@dataclass(frozen=True)
class VectorizationResult:
    """Vectorized fracture branches and auditable intermediate masks.

    Every polyline is a ``float32 (n, 2)`` array in pixel ``(x, y)`` order.
    ``accepted_mask`` is the thick hysteresis mask, ``bridge_mask`` identifies
    newly inferred gap pixels, and ``skeleton_mask`` contains only geometry
    represented by the returned polylines.
    """

    polylines: Tuple[np.ndarray, ...]
    accepted_mask: np.ndarray
    skeleton_mask: np.ndarray
    bridge_mask: np.ndarray
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        polylines: List[np.ndarray] = []
        for polyline in self.polylines:
            array = np.asarray(polyline, dtype=np.float32)
            if array.ndim != 2 or array.shape[1] != 2:
                raise ValueError("each polyline must have shape (n, 2)")
            polylines.append(np.ascontiguousarray(array))
        accepted = np.asarray(self.accepted_mask, dtype=bool)
        skeleton = np.asarray(self.skeleton_mask, dtype=bool)
        bridge = np.asarray(self.bridge_mask, dtype=bool)
        if accepted.ndim != 2 or skeleton.shape != accepted.shape or bridge.shape != accepted.shape:
            raise ValueError("all vectorization masks must be aligned two-dimensional arrays")
        object.__setattr__(self, "polylines", tuple(polylines))
        object.__setattr__(self, "accepted_mask", np.ascontiguousarray(accepted))
        object.__setattr__(self, "skeleton_mask", np.ascontiguousarray(skeleton))
        object.__setattr__(self, "bridge_mask", np.ascontiguousarray(bridge))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    @property
    def polyline_count(self) -> int:
        return len(self.polylines)

    @property
    def total_length(self) -> float:
        return float(sum(_polyline_length(polyline) for polyline in self.polylines))


class FractureVectorizer:
    """Convert an :class:`EvidenceMap` to branch-preserving vector centrelines."""

    def __init__(self, settings: Optional[VectorizationSettings] = None) -> None:
        self.settings = settings or VectorizationSettings()

    def vectorize(self, evidence: EvidenceMap) -> VectorizationResult:
        """Vectorize fracture evidence deterministically at source resolution."""

        settings = self.settings
        valid = np.asarray(evidence.valid_mask, dtype=bool)
        reliability = evidence.probability * (
            1.0 - settings.uncertainty_weight * evidence.uncertainty
        )
        reliability = np.asarray(np.clip(reliability, 0.0, 1.0), dtype=np.float32)
        reliability[~valid] = 0.0
        strong = valid & (reliability >= settings.strong_threshold)
        weak = valid & (reliability >= settings.weak_threshold)
        threshold_weak_count = int(np.count_nonzero(weak))
        if settings.ridge_nms:
            nms_support = _orientation_ridge_nms(
                reliability,
                evidence.orientation,
                valid,
                settings.nms_radius,
            )
            strong &= nms_support
            weak &= nms_support
        else:
            nms_support = valid.copy()
        diagnostics: Dict[str, Any] = {
            "strong_pixel_count": int(np.count_nonzero(strong)),
            "weak_pixel_count": int(np.count_nonzero(weak)),
            "nms_enabled": settings.ridge_nms,
            "nms_supported_pixel_count": int(np.count_nonzero(weak)),
            "nms_removed_weak_pixels": threshold_weak_count - int(np.count_nonzero(weak)),
            "hysteresis_pixel_count": 0,
            "initial_skeleton_pixel_count": 0,
            "bridge_candidates_considered": 0,
            "bridges_accepted": 0,
            "bridge_rejected_geometry": 0,
            "bridge_rejected_invalid": 0,
            "bridge_rejected_evidence": 0,
            "bridge_rejected_intersection": 0,
            "bridge_rejected_conflict": 0,
            "spur_pixels_removed": 0,
            "low_support_components_rejected": 0,
            "low_support_component_pixels_removed": 0,
            "early_components_rejected": 0,
            "early_component_pixels_removed": 0,
            "short_vectors_rejected": 0,
            "low_evidence_vectors_rejected": 0,
            "small_loops_rejected": 0,
            "polylines_accepted": 0,
        }
        if not np.any(strong):
            return self._empty_result(evidence.shape, diagnostics)

        accepted = binary_propagation(
            strong,
            structure=_CONNECTIVITY_8,
            mask=weak,
        ).astype(bool)
        accepted &= valid
        diagnostics["hysteresis_pixel_count"] = int(np.count_nonzero(accepted))
        skeleton = _zhang_suen_thinning(accepted)
        skeleton &= valid
        diagnostics["initial_skeleton_pixel_count"] = int(np.count_nonzero(skeleton))
        if not np.any(skeleton):
            return self._empty_result(evidence.shape, diagnostics, accepted)

        support_evidence = evidence.probability * (1.0 - evidence.uncertainty)
        skeleton, low_support_components, low_support_pixels = _prune_low_support_components(
            skeleton,
            support_evidence,
            settings.min_mean_vector_evidence,
        )
        diagnostics["low_support_components_rejected"] = low_support_components
        diagnostics["low_support_component_pixels_removed"] = low_support_pixels
        if not np.any(skeleton):
            return self._empty_result(evidence.shape, diagnostics, accepted)

        skeleton, bridge_mask, bridge_diagnostics = self._bridge_gaps(
            skeleton,
            evidence,
            reliability,
        )
        diagnostics.update(bridge_diagnostics)
        skeleton, removed_spurs = _prune_short_spurs(
            skeleton,
            settings.min_spur_length,
        )
        diagnostics["spur_pixels_removed"] = removed_spurs
        skeleton, rejected_components, rejected_component_pixels = _prune_short_components(
            skeleton,
            settings.min_vector_length,
        )
        diagnostics["early_components_rejected"] = rejected_components
        diagnostics["early_component_pixels_removed"] = rejected_component_pixels

        graph_paths = _skeleton_graph_paths(skeleton)
        full_paths: List[np.ndarray] = []
        output_paths: List[np.ndarray] = []
        for polyline, closed in graph_paths:
            length = _polyline_length(polyline)
            if closed and settings.reject_small_loops and length <= settings.max_loop_length:
                diagnostics["small_loops_rejected"] += 1
                continue
            if length < settings.min_vector_length:
                diagnostics["short_vectors_rejected"] += 1
                continue
            mean_evidence = _polyline_mean_evidence(polyline, support_evidence)
            if mean_evidence < settings.min_mean_vector_evidence:
                diagnostics["low_evidence_vectors_rejected"] += 1
                continue
            full_paths.append(polyline)
            simplified = polyline
            if settings.simplify_tolerance > 0.0 and len(polyline) > 2:
                simplified = np.asarray(
                    simplify_polyline(polyline, settings.simplify_tolerance),
                    dtype=np.float32,
                )
                if closed and not np.array_equal(simplified[0], simplified[-1]):
                    simplified = np.vstack((simplified, simplified[0]))
            output_paths.append(np.ascontiguousarray(simplified, dtype=np.float32))

        ordering = sorted(
            range(len(output_paths)),
            key=lambda index: _polyline_sort_key(output_paths[index]),
        )
        output_paths = [output_paths[index] for index in ordering]
        full_paths = [full_paths[index] for index in ordering]
        final_skeleton = np.zeros(evidence.shape, dtype=bool)
        for polyline in full_paths:
            _rasterize_polyline(final_skeleton, polyline)
        final_skeleton &= valid
        bridge_mask &= final_skeleton
        diagnostics["polylines_accepted"] = len(output_paths)
        diagnostics["final_skeleton_pixel_count"] = int(np.count_nonzero(final_skeleton))
        diagnostics["total_vector_length"] = float(
            sum(_polyline_length(polyline) for polyline in output_paths)
        )
        return VectorizationResult(
            tuple(output_paths),
            accepted,
            final_skeleton,
            bridge_mask,
            diagnostics,
        )

    @staticmethod
    def _empty_result(
        shape: Tuple[int, int],
        diagnostics: Mapping[str, Any],
        accepted: Optional[np.ndarray] = None,
    ) -> VectorizationResult:
        empty = np.zeros(shape, dtype=bool)
        accepted_mask = empty if accepted is None else np.asarray(accepted, dtype=bool)
        values = dict(diagnostics)
        values.setdefault("final_skeleton_pixel_count", 0)
        values.setdefault("total_vector_length", 0.0)
        return VectorizationResult((), accepted_mask, empty, empty, values)

    def _bridge_gaps(
        self,
        skeleton: np.ndarray,
        evidence: EvidenceMap,
        reliability: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
        """Greedily add mutually compatible, face-to-face endpoint bridges."""

        keys = (
            "bridge_candidates_considered",
            "bridges_accepted",
            "bridge_rejected_geometry",
            "bridge_rejected_invalid",
            "bridge_rejected_evidence",
            "bridge_rejected_intersection",
            "bridge_rejected_conflict",
        )
        counts = {key: 0 for key in keys}
        settings = self.settings
        bridge_mask = np.zeros(skeleton.shape, dtype=bool)
        if settings.max_bridge_gap <= 0.0:
            return skeleton, bridge_mask, counts

        degrees = _topological_degree(skeleton)
        endpoint_y, endpoint_x = np.nonzero(skeleton & (degrees == 1))
        if len(endpoint_y) < 2:
            return skeleton, bridge_mask, counts
        component_labels, component_count = label(skeleton, structure=_CONNECTIVITY_8)
        component_sizes = np.bincount(component_labels.ravel(), minlength=component_count + 1)
        minimum_component_size = max(2, int(math.ceil(settings.min_spur_length)))

        endpoint_records: List[Tuple[int, int, int, np.ndarray]] = []
        for y, x in zip(endpoint_y.tolist(), endpoint_x.tolist()):
            component = int(component_labels[y, x])
            if component_sizes[component] < minimum_component_size:
                continue
            tangent = _endpoint_outward_tangent(
                skeleton,
                component_labels,
                component,
                y,
                x,
                settings.tangent_lookback,
            )
            if tangent is not None:
                endpoint_records.append((y, x, component, tangent))
        if len(endpoint_records) < 2:
            return skeleton, bridge_mask, counts

        coordinates = np.asarray(
            [(record[1], record[0]) for record in endpoint_records],
            dtype=np.float64,
        )
        pairs = sorted(cKDTree(coordinates).query_pairs(settings.max_bridge_gap))
        maximum_angle = math.radians(settings.max_bridge_angle_degrees)
        minimum_facing = math.cos(maximum_angle)
        candidates: List[Tuple[float, int, int, Tuple[Tuple[int, int], ...]]] = []
        valid = evidence.valid_mask

        for first, second in pairs:
            y1, x1, component1, tangent1 = endpoint_records[first]
            y2, x2, component2, tangent2 = endpoint_records[second]
            if component1 == component2:
                continue
            counts["bridge_candidates_considered"] += 1
            delta = np.asarray((x2 - x1, y2 - y1), dtype=np.float64)
            distance = float(np.linalg.norm(delta))
            if distance <= 1.0:
                counts["bridge_rejected_geometry"] += 1
                continue
            direction = delta / distance
            facing_first = float(np.dot(tangent1, direction))
            facing_second = float(np.dot(tangent2, -direction))
            if facing_first < minimum_facing or facing_second < minimum_facing:
                counts["bridge_rejected_geometry"] += 1
                continue

            bridge_angle = math.atan2(direction[1], direction[0])
            endpoint_alignment = 0.5 * (
                _axial_mismatch(bridge_angle, float(evidence.orientation[y1, x1]))
                + _axial_mismatch(bridge_angle, float(evidence.orientation[y2, x2]))
            )
            if endpoint_alignment > math.sin(maximum_angle) ** 2:
                counts["bridge_rejected_geometry"] += 1
                continue

            line_pixels = tuple(_bresenham((x1, y1), (x2, y2)))
            interior = line_pixels[1:-1]
            if any(not valid[y, x] for x, y in line_pixels):
                counts["bridge_rejected_invalid"] += 1
                continue
            if any(skeleton[y, x] for x, y in interior):
                counts["bridge_rejected_intersection"] += 1
                continue
            samples = interior if interior else line_pixels
            mean_evidence = float(np.mean([reliability[y, x] for x, y in samples]))
            if mean_evidence < settings.bridge_min_mean_evidence:
                counts["bridge_rejected_evidence"] += 1
                continue
            weights = np.asarray(
                [max(float(reliability[y, x]), 1.0e-6) for x, y in line_pixels]
            )
            orientation_error = np.asarray(
                [
                    _axial_mismatch(bridge_angle, float(evidence.orientation[y, x]))
                    for x, y in line_pixels
                ]
            )
            mean_orientation_error = float(np.average(orientation_error, weights=weights))
            score = (
                distance / max(settings.max_bridge_gap, 1.0)
                + (1.0 - facing_first)
                + (1.0 - facing_second)
                + 0.5 * (1.0 - mean_evidence)
                + 0.5 * mean_orientation_error
            )
            candidates.append((score, first, second, line_pixels))

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        used_endpoints: set[int] = set()
        parents = np.arange(component_count + 1, dtype=np.int32)

        def find(component: int) -> int:
            root = component
            while parents[root] != root:
                root = int(parents[root])
            while parents[component] != component:
                following = int(parents[component])
                parents[component] = root
                component = following
            return root

        for _, first, second, line_pixels in candidates:
            component1 = endpoint_records[first][2]
            component2 = endpoint_records[second][2]
            interior = line_pixels[1:-1]
            if (
                first in used_endpoints
                or second in used_endpoints
                or find(component1) == find(component2)
                or any(bridge_mask[y, x] for x, y in interior)
            ):
                counts["bridge_rejected_conflict"] += 1
                continue
            root1 = find(component1)
            root2 = find(component2)
            parents[root2] = root1
            used_endpoints.add(first)
            used_endpoints.add(second)
            for x, y in line_pixels:
                if not skeleton[y, x]:
                    bridge_mask[y, x] = True
                skeleton[y, x] = True
            counts["bridges_accepted"] += 1
        return skeleton, bridge_mask, counts


def _orientation_ridge_nms(
    reliability: np.ndarray,
    orientation: np.ndarray,
    valid: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Keep local reliability maxima sampled along the predicted line normal."""

    height, width = reliability.shape
    yy, xx = np.indices((height, width), dtype=np.float32)
    normal_y = np.cos(orientation).astype(np.float32, copy=False)
    normal_x = -np.sin(orientation).astype(np.float32, copy=False)
    forward = map_coordinates(
        reliability,
        (yy + radius * normal_y, xx + radius * normal_x),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    backward = map_coordinates(
        reliability,
        (yy - radius * normal_y, xx - radius * normal_x),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    epsilon = np.float32(8.0 * np.finfo(np.float32).eps)
    return valid & (reliability + epsilon >= forward) & (reliability + epsilon >= backward)


def _zhang_suen_thinning(mask: np.ndarray) -> np.ndarray:
    """Return a deterministic one-pixel skeleton using Zhang-Suen thinning."""

    image = np.asarray(mask, dtype=bool).copy()
    if not np.any(image):
        return image
    maximum_iterations = max(image.shape)
    for _ in range(maximum_iterations):
        changed = False
        for first_step in (True, False):
            padded = np.pad(image, 1, mode="constant", constant_values=False)
            p2 = padded[:-2, 1:-1]
            p3 = padded[:-2, 2:]
            p4 = padded[1:-1, 2:]
            p5 = padded[2:, 2:]
            p6 = padded[2:, 1:-1]
            p7 = padded[2:, :-2]
            p8 = padded[1:-1, :-2]
            p9 = padded[:-2, :-2]
            neighbours = (
                p2.astype(np.uint8)
                + p3
                + p4
                + p5
                + p6
                + p7
                + p8
                + p9
            )
            transitions = (
                ((~p2) & p3).astype(np.uint8)
                + ((~p3) & p4)
                + ((~p4) & p5)
                + ((~p5) & p6)
                + ((~p6) & p7)
                + ((~p7) & p8)
                + ((~p8) & p9)
                + ((~p9) & p2)
            )
            remove = image & (neighbours >= 2) & (neighbours <= 6) & (transitions == 1)
            if first_step:
                remove &= ~(p2 & p4 & p6)
                remove &= ~(p4 & p6 & p8)
            else:
                remove &= ~(p2 & p4 & p8)
                remove &= ~(p2 & p6 & p8)
            if np.any(remove):
                image[remove] = False
                changed = True
        if not changed:
            break
    return image


def _topological_degree(mask: np.ndarray) -> np.ndarray:
    """Count neighbours while suppressing redundant diagonal corner edges."""

    image = np.asarray(mask, dtype=bool)
    padded = np.pad(image, 1, mode="constant", constant_values=False)
    centre = padded[1:-1, 1:-1]
    degree = np.zeros(image.shape, dtype=np.uint8)
    for dy, dx in _DIRECTIONS:
        neighbour = padded[1 + dy : 1 + dy + image.shape[0], 1 + dx : 1 + dx + image.shape[1]]
        connected = centre & neighbour
        if dy != 0 and dx != 0:
            horizontal = padded[1:-1, 1 + dx : 1 + dx + image.shape[1]]
            vertical = padded[1 + dy : 1 + dy + image.shape[0], 1:-1]
            connected &= ~(horizontal | vertical)
        degree += connected
    return degree


def _neighbours(mask: np.ndarray, y: int, x: int) -> List[Tuple[int, int]]:
    height, width = mask.shape
    result: List[Tuple[int, int]] = []
    for dy, dx in _DIRECTIONS:
        next_y = y + dy
        next_x = x + dx
        if not (0 <= next_y < height and 0 <= next_x < width and mask[next_y, next_x]):
            continue
        if dy != 0 and dx != 0 and (mask[y, next_x] or mask[next_y, x]):
            continue
        result.append((next_y, next_x))
    return result


def _endpoint_outward_tangent(
    skeleton: np.ndarray,
    component_labels: np.ndarray,
    component: int,
    y: int,
    x: int,
    lookback: int,
) -> Optional[np.ndarray]:
    """Estimate the directed tangent pointing away from an endpoint."""

    path = [(y, x)]
    previous: Optional[Tuple[int, int]] = None
    current = (y, x)
    for _ in range(lookback):
        candidates = [
            point
            for point in _neighbours(skeleton, current[0], current[1])
            if point != previous and component_labels[point] == component
        ]
        if not candidates:
            break
        if len(candidates) > 1 and previous is not None:
            break
        following = min(candidates)
        path.append(following)
        previous, current = current, following
    if len(path) < 2:
        return None
    inward = np.asarray((path[-1][1] - x, path[-1][0] - y), dtype=np.float64)
    norm = float(np.linalg.norm(inward))
    if norm <= np.finfo(np.float64).eps:
        return None
    return -inward / norm


def _prune_short_spurs(skeleton: np.ndarray, minimum_length: float) -> Tuple[np.ndarray, int]:
    """Remove endpoint-to-junction branches shorter than ``minimum_length``."""

    result = np.asarray(skeleton, dtype=bool).copy()
    if minimum_length <= 0.0:
        return result, 0
    removed_total = 0
    while True:
        degree = _topological_degree(result)
        endpoints = list(zip(*np.nonzero(result & (degree == 1))))
        remove = np.zeros(result.shape, dtype=bool)
        for endpoint in endpoints:
            if remove[endpoint] or not result[endpoint]:
                continue
            path = [endpoint]
            previous: Optional[Tuple[int, int]] = None
            current = endpoint
            length = 0.0
            reached_junction = False
            while True:
                candidates = [
                    point
                    for point in _neighbours(result, current[0], current[1])
                    if point != previous
                ]
                if not candidates:
                    break
                following = min(candidates)
                length += math.hypot(following[1] - current[1], following[0] - current[0])
                if degree[following] >= 3:
                    reached_junction = True
                    break
                path.append(following)
                if degree[following] <= 1:
                    break
                previous, current = current, following
            if reached_junction and length < minimum_length:
                for point in path:
                    remove[point] = True
        removed = int(np.count_nonzero(remove))
        if removed == 0:
            break
        result[remove] = False
        removed_total += removed
    return result, removed_total


def _prune_short_components(
    skeleton: np.ndarray,
    minimum_length: float,
) -> Tuple[np.ndarray, int, int]:
    """Reject components whose maximum possible 8-neighbour path is too short."""

    result = np.asarray(skeleton, dtype=bool).copy()
    if minimum_length <= 0.0 or not np.any(result):
        return result, 0, 0
    component_labels, component_count = label(result, structure=_CONNECTIVITY_8)
    sizes = np.bincount(component_labels.ravel(), minlength=component_count + 1)
    reject = np.zeros(component_count + 1, dtype=bool)
    for component in range(1, component_count + 1):
        # Any simple path visits each component pixel at most once and every
        # 8-neighbour step is at most sqrt(2), giving a safe geodesic upper bound.
        maximum_possible_length = max(0, int(sizes[component]) - 1) * math.sqrt(2.0)
        if maximum_possible_length < minimum_length:
            reject[component] = True
    removed_mask = reject[component_labels]
    removed_pixels = int(np.count_nonzero(removed_mask))
    result[removed_mask] = False
    return result, int(np.count_nonzero(reject)), removed_pixels


def _prune_low_support_components(
    skeleton: np.ndarray,
    support_evidence: np.ndarray,
    minimum_mean: float,
) -> Tuple[np.ndarray, int, int]:
    """Remove disconnected ridge components lacking mean certainty support.

    Support uses ``probability * (1 - uncertainty)``.  Applying it per connected
    component retains weak pixels within a well-supported fracture while
    rejecting isolated texture ridges before endpoint pairing becomes quadratic.
    """

    result = np.asarray(skeleton, dtype=bool).copy()
    if minimum_mean <= 0.0 or not np.any(result):
        return result, 0, 0
    component_labels, component_count = label(result, structure=_CONNECTIVITY_8)
    sizes = np.bincount(component_labels.ravel(), minlength=component_count + 1)
    support_sum = np.bincount(
        component_labels.ravel(),
        weights=np.asarray(support_evidence, dtype=np.float64).ravel(),
        minlength=component_count + 1,
    )
    mean_support = np.divide(
        support_sum,
        sizes,
        out=np.zeros_like(support_sum),
        where=sizes > 0,
    )
    reject = mean_support < minimum_mean
    reject[0] = False
    removed_mask = reject[component_labels]
    removed_pixels = int(np.count_nonzero(removed_mask))
    result[removed_mask] = False
    return result, int(np.count_nonzero(reject)), removed_pixels


def _skeleton_graph_paths(skeleton: np.ndarray) -> List[Tuple[np.ndarray, bool]]:
    """Split an 8-connected skeleton into branch polylines at true junctions."""

    if not np.any(skeleton):
        return []
    degree = _topological_degree(skeleton)
    junction_seeds = skeleton & (degree >= 3)
    # Compress the one-pixel geodesic neighbourhood around each junction.  If
    # only the central degree>=3 pixel is removed, two arms can remain diagonal
    # neighbours around its corner and be mistaken for one pass-through edge.
    junction_mask = skeleton & binary_dilation(
        junction_seeds,
        structure=_CONNECTIVITY_8,
    )
    junction_labels, junction_count = label(junction_mask, structure=_CONNECTIVITY_8)
    centroids: Dict[int, np.ndarray] = {}
    junction_coordinates = _label_coordinate_groups(junction_labels, junction_count)
    for junction in range(1, junction_count + 1):
        coordinates = np.asarray(junction_coordinates[junction], dtype=np.float32)
        centroids[junction] = np.asarray(
            (float(np.mean(coordinates[:, 1])), float(np.mean(coordinates[:, 0]))),
            np.float32,
        )

    residual = skeleton & ~junction_mask
    residual_labels, residual_count = label(residual, structure=_CONNECTIVITY_8)
    residual_coordinates = _label_coordinate_groups(residual_labels, residual_count)
    paths: List[Tuple[np.ndarray, bool]] = []
    for component in range(1, residual_count + 1):
        ordered, is_loop = _ordered_component(
            residual_labels,
            component,
            residual_coordinates[component],
            skeleton,
        )
        if not ordered:
            continue
        xy = np.asarray([(x, y) for y, x in ordered], dtype=np.float32)
        start_junctions = _adjacent_junctions(
            skeleton, junction_labels, ordered[0][0], ordered[0][1]
        )
        end_junctions = _adjacent_junctions(
            skeleton, junction_labels, ordered[-1][0], ordered[-1][1]
        )
        if len(ordered) == 1:
            combined = sorted(set(start_junctions + end_junctions))
            start_junctions = combined[:1]
            end_junctions = combined[1:2]
        if start_junctions:
            xy = np.vstack((centroids[start_junctions[0]], xy))
        if end_junctions:
            xy = np.vstack((xy, centroids[end_junctions[0]]))
        closed = is_loop or (
            bool(start_junctions)
            and bool(end_junctions)
            and start_junctions[0] == end_junctions[0]
        )
        if closed and not np.array_equal(xy[0], xy[-1]):
            xy = np.vstack((xy, xy[0]))
        paths.append((np.ascontiguousarray(xy, dtype=np.float32), closed))
    return paths


def _ordered_component(
    component_labels: np.ndarray,
    component: int,
    coordinates: Sequence[Tuple[int, int]],
    topology_mask: np.ndarray,
) -> Tuple[List[Tuple[int, int]], bool]:
    if not coordinates:
        return [], False
    degrees = {
        point: len(
            _label_neighbours(
                component_labels,
                component,
                point[0],
                point[1],
                topology_mask,
            )
        )
        for point in coordinates
    }
    endpoints = sorted(point for point in coordinates if degrees[point] <= 1)
    closed = not endpoints and len(coordinates) > 1
    start = endpoints[0] if endpoints else min(coordinates)
    ordered = [start]
    previous: Optional[Tuple[int, int]] = None
    current = start
    for _ in range(len(coordinates) + 1):
        candidates = [
            point
            for point in _label_neighbours(
                component_labels,
                component,
                current[0],
                current[1],
                topology_mask,
            )
            if point != previous
        ]
        if not candidates:
            break
        following = min(candidates)
        if following == start:
            if closed:
                ordered.append(start)
            break
        if following in ordered:
            break
        ordered.append(following)
        previous, current = current, following
    return ordered, closed


def _label_coordinate_groups(
    labels: np.ndarray,
    component_count: int,
) -> List[List[Tuple[int, int]]]:
    groups: List[List[Tuple[int, int]]] = [
        [] for _ in range(component_count + 1)
    ]
    ys, xs = np.nonzero(labels)
    for y, x in zip(ys.tolist(), xs.tolist()):
        groups[int(labels[y, x])].append((y, x))
    return groups


def _label_neighbours(
    labels: np.ndarray,
    component: int,
    y: int,
    x: int,
    topology_mask: np.ndarray,
) -> List[Tuple[int, int]]:
    height, width = labels.shape
    result: List[Tuple[int, int]] = []
    for dy, dx in _DIRECTIONS:
        next_y = y + dy
        next_x = x + dx
        if not (
            0 <= next_y < height
            and 0 <= next_x < width
            and labels[next_y, next_x] == component
        ):
            continue
        if dy != 0 and dx != 0 and (
            topology_mask[y, next_x] or topology_mask[next_y, x]
        ):
            continue
        result.append((next_y, next_x))
    return result


def _adjacent_junctions(
    skeleton: np.ndarray,
    junction_labels: np.ndarray,
    y: int,
    x: int,
) -> List[int]:
    values = {
        int(junction_labels[next_y, next_x])
        for next_y, next_x in _neighbours(skeleton, y, x)
        if junction_labels[next_y, next_x] > 0
    }
    return sorted(values)


def _axial_mismatch(first: float, second: float) -> float:
    return float(math.sin(first - second) ** 2)


def _polyline_length(polyline: np.ndarray) -> float:
    if len(polyline) < 2:
        return 0.0
    differences = np.diff(np.asarray(polyline, dtype=np.float64), axis=0)
    return float(np.sum(np.linalg.norm(differences, axis=1)))


def _polyline_mean_evidence(polyline: np.ndarray, evidence: np.ndarray) -> float:
    """Sample mean evidence once per raster pixel along a polyline."""

    if len(polyline) == 0:
        return 0.0
    rounded = np.rint(polyline).astype(np.int32)
    pixels: set[Tuple[int, int]] = set()
    if len(rounded) == 1:
        pixels.add((int(rounded[0, 0]), int(rounded[0, 1])))
    else:
        for first, second in zip(rounded[:-1], rounded[1:]):
            pixels.update(_bresenham(tuple(first), tuple(second)))
    height, width = evidence.shape
    values = [
        float(evidence[y, x])
        for x, y in pixels
        if 0 <= x < width and 0 <= y < height
    ]
    return float(np.mean(values)) if values else 0.0


def _polyline_sort_key(polyline: np.ndarray) -> Tuple[float, ...]:
    first = polyline[0]
    last = polyline[-1]
    return (
        round(float(first[1]), 6),
        round(float(first[0]), 6),
        round(float(last[1]), 6),
        round(float(last[0]), 6),
        -round(_polyline_length(polyline), 6),
    )


def _bresenham(
    start_xy: Tuple[int, int],
    end_xy: Tuple[int, int],
) -> List[Tuple[int, int]]:
    x1, y1 = start_xy
    x2, y2 = end_xy
    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    step_x = 1 if x1 < x2 else -1
    step_y = 1 if y1 < y2 else -1
    error = dx + dy
    result: List[Tuple[int, int]] = []
    while True:
        result.append((x1, y1))
        if x1 == x2 and y1 == y2:
            return result
        doubled = 2 * error
        if doubled >= dy:
            error += dy
            x1 += step_x
        if doubled <= dx:
            error += dx
            y1 += step_y


def _rasterize_polyline(mask: np.ndarray, polyline: np.ndarray) -> None:
    if len(polyline) == 0:
        return
    points = np.rint(polyline).astype(np.int32)
    height, width = mask.shape
    if len(points) == 1:
        x, y = points[0]
        if 0 <= x < width and 0 <= y < height:
            mask[y, x] = True
        return
    for first, second in zip(points[:-1], points[1:]):
        for x, y in _bresenham(tuple(first), tuple(second)):
            if 0 <= x < width and 0 <= y < height:
                mask[y, x] = True


__all__ = [
    "FractureVectorizer",
    "VectorizationResult",
    "VectorizationSettings",
]
