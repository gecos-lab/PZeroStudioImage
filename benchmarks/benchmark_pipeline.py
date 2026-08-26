"""Repeatable synthetic timing benchmark for detection, vectorization, and tracing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import platform
from pathlib import Path
from statistics import median
import sys
from time import perf_counter

import numpy as np
import scipy

from domstudio import __version__
from domstudio.core import (
    CorridorAStarTracer,
    MultiscaleFractureDetector,
    RasterDocument,
    TraceConfig,
    TraceRequest,
    FractureVectorizer,
    VectorizationSettings,
)


def synthetic_outcrop(size: int, seed: int) -> RasterDocument:
    if size < 128:
        raise ValueError("size must be at least 128 pixels")
    generator = np.random.default_rng(seed)
    image = generator.normal(0.72, 0.055, (size, size)).astype(np.float32)
    xx = np.arange(size)
    centre = 0.48 * size + 0.13 * size * np.sin(xx / (0.18 * size))
    yy = np.arange(size)[:, None]
    fracture = np.abs(yy - centre[None, :]) <= max(1.0, size / 512.0)
    image[fracture] = 0.08
    return RasterDocument(np.clip(image, 0.0, 1.0))


def benchmark(size: int, repeats: int, seed: int) -> dict[str, object]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    document = synthetic_outcrop(size, seed)
    detector_times: list[float] = []
    evidence = None
    detector = MultiscaleFractureDetector()
    for _ in range(repeats):
        started = perf_counter()
        evidence = detector.predict(document)
        detector_times.append(perf_counter() - started)
    assert evidence is not None

    vectorization_config = VectorizationSettings(
        strong_threshold=0.45,
        weak_threshold=0.20,
        max_bridge_gap=18.0,
        min_vector_length=24.0,
        reject_small_loops=True,
        max_loop_length=60.0,
    )
    vectorizer = FractureVectorizer(vectorization_config)
    vectorization_times: list[float] = []
    vectorization_result = None
    for _ in range(repeats):
        started = perf_counter()
        vectorization_result = vectorizer.vectorize(evidence)
        vectorization_times.append(perf_counter() - started)
    assert vectorization_result is not None

    start = (4.0, 0.48 * size + 0.13 * size * np.sin(4.0 / (0.18 * size)))
    end_x = float(size - 5)
    end = (
        end_x,
        0.48 * size + 0.13 * size * np.sin(end_x / (0.18 * size)),
    )
    endpoint_distance = float(np.hypot(end[0] - start[0], end[1] - start[1]))
    request = TraceRequest(
        start,
        end,
        corridor_radius=min(96.0, max(24.0, endpoint_distance * 0.20)),
        max_expansions=2_000_000,
    )
    trace_config = TraceConfig(
        orientation_weight=0.70,
        curvature_weight=0.35,
        uncertainty_weight=0.35,
        gap_threshold=0.45,
    )
    tracer = CorridorAStarTracer(trace_config)
    trace_times: list[float] = []
    result = None
    for _ in range(repeats):
        started = perf_counter()
        result = tracer.trace(evidence, request)
        trace_times.append(perf_counter() - started)
    assert result is not None

    detection_seconds = median(detector_times)
    vectorization_seconds = median(vectorization_times)
    trace_seconds = median(trace_times)
    return {
        "domstudio_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "seed": seed,
        "image_shape": [size, size],
        "repeats": repeats,
        "detector_seconds_samples": detector_times,
        "detector_seconds_first": detector_times[0],
        "detector_seconds_warm_median": (
            median(detector_times[1:]) if len(detector_times) > 1 else None
        ),
        "detector_seconds_median": detection_seconds,
        "detector_megapixels_per_second": (size * size / 1_000_000) / detection_seconds,
        "detector_config": asdict(detector.config),
        "vectorization_seconds_samples": vectorization_times,
        "vectorization_seconds_first": vectorization_times[0],
        "vectorization_seconds_warm_median": (
            median(vectorization_times[1:])
            if len(vectorization_times) > 1
            else None
        ),
        "vectorization_seconds_median": vectorization_seconds,
        "vectorization_megapixels_per_second": (
            size * size / 1_000_000
        )
        / vectorization_seconds,
        "vectorization_config": asdict(vectorization_config),
        "vectorization_polyline_count": vectorization_result.polyline_count,
        "vectorization_total_length": vectorization_result.total_length,
        "vectorization_diagnostics": dict(vectorization_result.diagnostics),
        "trace_seconds_samples": trace_times,
        "trace_seconds_first": trace_times[0],
        "trace_seconds_warm_median": (
            median(trace_times[1:]) if len(trace_times) > 1 else None
        ),
        "trace_seconds_median": trace_seconds,
        "trace_status": result.status.value,
        "trace_config": asdict(trace_config),
        "trace_corridor_radius": request.corridor_radius,
        "trace_max_expansions": request.max_expansions,
        "trace_expanded_states": result.expanded_states,
        "trace_path_vertices": len(result.path_xy),
        "trace_searchable_pixels": result.searchable_pixels,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    report = benchmark(arguments.size, arguments.repeats, arguments.seed)
    encoded = json.dumps(report, indent=2)
    if arguments.output:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
