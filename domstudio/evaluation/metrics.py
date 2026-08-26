"""Geometry-first evaluation for thin fracture traces.

Raster overlap alone is deliberately not used here: two one-pixel traces can
represent the same geological interpretation while having zero exact overlap.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree


def _points(value: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError("A trace must have shape (n, 2).")
    return result


def polyline_length(value: Sequence[Sequence[float]] | np.ndarray) -> float:
    points = _points(value)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _resample(points: np.ndarray, spacing: float) -> np.ndarray:
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if len(points) < 2:
        return points.copy()
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segments)))
    total = cumulative[-1]
    if total <= np.finfo(np.float64).eps:
        return points[:1].copy()
    distances = np.arange(0.0, total, spacing)
    if distances.size == 0 or distances[-1] != total:
        distances = np.append(distances, total)
    x = np.interp(distances, cumulative, points[:, 0])
    y = np.interp(distances, cumulative, points[:, 1])
    return np.column_stack((x, y))


@dataclass(frozen=True, slots=True)
class GeometryMetrics:
    precision: float
    recall: float
    f1: float
    mean_symmetric_distance: float
    hausdorff_distance: float
    endpoint_error: float
    length_relative_error: float


def evaluate_geometry(
    prediction: Sequence[Sequence[float]] | np.ndarray,
    reference: Sequence[Sequence[float]] | np.ndarray,
    *,
    tolerance: float,
    sample_spacing: float = 1.0,
) -> GeometryMetrics:
    """Compare two polylines using tolerance-aware and distance metrics."""

    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    predicted = _points(prediction)
    expected = _points(reference)
    if len(predicted) < 2 or len(expected) < 2:
        raise ValueError("Both traces need at least two distinct vertices.")
    predicted_samples = _resample(predicted, sample_spacing)
    expected_samples = _resample(expected, sample_spacing)
    predicted_to_expected = cKDTree(expected_samples).query(predicted_samples, k=1)[0]
    expected_to_predicted = cKDTree(predicted_samples).query(expected_samples, k=1)[0]
    precision = float(np.mean(predicted_to_expected <= tolerance))
    recall = float(np.mean(expected_to_predicted <= tolerance))
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_symmetric = float(
        0.5 * (predicted_to_expected.mean() + expected_to_predicted.mean())
    )
    hausdorff = float(max(predicted_to_expected.max(), expected_to_predicted.max()))

    direct = np.linalg.norm(predicted[[0, -1]] - expected[[0, -1]], axis=1).mean()
    reverse = np.linalg.norm(predicted[[0, -1]] - expected[[-1, 0]], axis=1).mean()
    endpoint_error = float(min(direct, reverse))
    reference_length = polyline_length(expected)
    length_error = abs(polyline_length(predicted) - reference_length) / max(
        reference_length, np.finfo(np.float64).eps
    )
    return GeometryMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        mean_symmetric_distance=mean_symmetric,
        hausdorff_distance=hausdorff,
        endpoint_error=endpoint_error,
        length_relative_error=float(length_error),
    )
