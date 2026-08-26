from __future__ import annotations

from dataclasses import replace
from time import perf_counter

import numpy as np
from scipy.ndimage import binary_propagation, label, map_coordinates

from benchmarks.benchmark_pipeline import synthetic_outcrop
from domstudio.core import (
    EvidenceMap,
    FractureVectorizer,
    MultiscaleFractureDetector,
    VectorizationSettings,
)


CONNECTIVITY = np.ones((3, 3), dtype=np.uint8)


def reliability(evidence: EvidenceMap, settings: VectorizationSettings) -> np.ndarray:
    value = evidence.probability * (
        1.0 - settings.uncertainty_weight * evidence.uncertainty
    )
    return np.asarray(np.clip(value, 0.0, 1.0), dtype=np.float32)


def normal_nms(evidence: EvidenceMap, settings: VectorizationSettings, radius: float):
    score = reliability(evidence, settings)
    orientation = evidence.orientation
    yy, xx = np.indices(score.shape, dtype=np.float32)
    normal_x = -np.sin(orientation)
    normal_y = np.cos(orientation)
    positive = map_coordinates(
        score,
        (yy + radius * normal_y, xx + radius * normal_x),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    negative = map_coordinates(
        score,
        (yy - radius * normal_y, xx - radius * normal_x),
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )
    keep = evidence.valid_mask & (score >= positive) & (score >= negative)
    probability = np.where(keep, evidence.probability, 0.0).astype(np.float32)
    return EvidenceMap(
        probability,
        evidence.orientation,
        evidence.uncertainty,
        evidence.valid_mask,
        evidence.metadata,
    ), keep


def early_filter(
    evidence: EvidenceMap,
    settings: VectorizationSettings,
    minimum_pixels: int,
):
    score = reliability(evidence, settings)
    strong = evidence.valid_mask & (score >= settings.strong_threshold)
    weak = evidence.valid_mask & (score >= settings.weak_threshold)
    accepted = binary_propagation(strong, structure=CONNECTIVITY, mask=weak)
    labels, count = label(accepted, structure=CONNECTIVITY)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    allowed_labels = sizes >= minimum_pixels
    allowed_labels[0] = False
    keep = allowed_labels[labels]
    probability = np.where(keep, evidence.probability, 0.0).astype(np.float32)
    return EvidenceMap(
        probability,
        evidence.orientation,
        evidence.uncertainty,
        evidence.valid_mask,
        evidence.metadata,
    ), int(count), int(np.count_nonzero(keep))


def quality(result, size: int) -> dict[str, float | int]:
    yy, xx = np.nonzero(result.skeleton_mask)
    if len(xx):
        centre = 0.48 * size + 0.13 * size * np.sin(xx / (0.18 * size))
        true = np.abs(yy - centre) <= max(3.0, size / 128.0)
        true_pixels = int(np.count_nonzero(true))
        false_pixels = int(len(xx) - true_pixels)
        coverage = len(np.unique(xx[true])) / size
    else:
        true_pixels = false_pixels = 0
        coverage = 0.0
    return {
        "output": result.polyline_count,
        "length": round(result.total_length, 3),
        "skeleton": int(np.count_nonzero(result.skeleton_mask)),
        "true_skeleton": true_pixels,
        "false_skeleton": false_pixels,
        "column_recall": round(float(coverage), 4),
    }


def run_variant(name, evidence, settings, size, pre_nms=0.0, minimum=0):
    prepared = evidence
    pre_at = perf_counter()
    nms_keep = None
    component_count = -1
    filtered_pixels = -1
    if pre_nms:
        prepared, nms_keep = normal_nms(evidence, settings, pre_nms)
    if minimum:
        prepared, component_count, filtered_pixels = early_filter(
            prepared, settings, minimum
        )
    pre_ms = (perf_counter() - pre_at) * 1000.0
    started = perf_counter()
    result = FractureVectorizer(settings).vectorize(prepared)
    vector_ms = (perf_counter() - started) * 1000.0
    print(
        {
            "name": name,
            "pre_ms": round(pre_ms, 3),
            "pre_nms_keep": (
                int(np.count_nonzero(nms_keep)) if nms_keep is not None else -1
            ),
            "component_count": component_count,
            "filtered_pixels": filtered_pixels,
            "vector_ms": round(vector_ms, 3),
            **quality(result, size),
            "strong": result.diagnostics.get("strong_pixel_count"),
            "weak": result.diagnostics.get("weak_pixel_count"),
            "nms_supported": result.diagnostics.get("nms_supported_pixel_count"),
            "nms_removed_weak": result.diagnostics.get("nms_removed_weak_pixels"),
            "hysteresis": result.diagnostics.get("hysteresis_pixel_count"),
            "initial_skeleton": result.diagnostics.get("initial_skeleton_pixel_count"),
            "early_rejected": result.diagnostics.get("early_components_rejected"),
            "early_removed": result.diagnostics.get("early_component_pixels_removed"),
            "short_rejected": result.diagnostics.get("short_vectors_rejected"),
        },
        flush=True,
    )


def run(size: int) -> None:
    detector = MultiscaleFractureDetector()
    detected_at = perf_counter()
    evidence = detector.predict(synthetic_outcrop(size, 20260825))
    detector_ms = (perf_counter() - detected_at) * 1000.0
    base = VectorizationSettings(
        strong_threshold=0.45,
        weak_threshold=0.20,
        max_bridge_gap=18.0,
        min_vector_length=24.0,
        reject_small_loops=True,
        max_loop_length=60.0,
    )
    print("SIZE", size, "detector_ms", round(detector_ms, 3), flush=True)
    variants = [
        ("bench_no_nms", replace(base, ridge_nms=False), 0.0, 0),
        ("bench_nms075", replace(base, nms_radius=0.75), 0.0, 0),
        ("bench_nms1", base, 0.0, 0),
        ("bench_nms15", replace(base, nms_radius=1.5), 0.0, 0),
        (
            "bench_nms1_prefilter4",
            replace(base, ridge_nms=False),
            1.0,
            4,
        ),
        (
            "bench_nms1_prefilter8",
            replace(base, ridge_nms=False),
            1.0,
            8,
        ),
        (
            "threshold_055_025_nms1",
            replace(base, strong_threshold=0.55, weak_threshold=0.25),
            0.0,
            0,
        ),
        (
            "threshold_060_030_nms1",
            replace(base, strong_threshold=0.60, weak_threshold=0.30),
            0.0,
            0,
        ),
        (
            "threshold_065_035_nms1",
            replace(base, strong_threshold=0.65, weak_threshold=0.35),
            0.0,
            0,
        ),
        (
            "u1_075_035_nms1",
            replace(
                base,
                uncertainty_weight=1.0,
                strong_threshold=0.75,
                weak_threshold=0.35,
            ),
            0.0,
            0,
        ),
        (
            "u1_080_040_nms1",
            replace(
                base,
                uncertainty_weight=1.0,
                strong_threshold=0.80,
                weak_threshold=0.40,
            ),
            0.0,
            0,
        ),
        (
            "u1_085_045_nms1",
            replace(
                base,
                uncertainty_weight=1.0,
                strong_threshold=0.85,
                weak_threshold=0.45,
            ),
            0.0,
            0,
        ),
        (
            "u1_090_050_nms1",
            replace(
                base,
                uncertainty_weight=1.0,
                strong_threshold=0.90,
                weak_threshold=0.50,
            ),
            0.0,
            0,
        ),
        (
            "best_r075",
            replace(
                base,
                uncertainty_weight=1.0,
                strong_threshold=0.90,
                weak_threshold=0.50,
                nms_radius=0.75,
            ),
            0.0,
            0,
        ),
        (
            "best_r15",
            replace(
                base,
                uncertainty_weight=1.0,
                strong_threshold=0.90,
                weak_threshold=0.50,
                nms_radius=1.5,
            ),
            0.0,
            0,
        ),
        *[
            (
                f"best_support_{minimum_support:.2f}",
                replace(
                    base,
                    uncertainty_weight=1.0,
                    strong_threshold=0.90,
                    weak_threshold=0.50,
                    min_mean_vector_evidence=minimum_support,
                ),
                0.0,
                0,
            )
            for minimum_support in (0.70, 0.75, 0.80, 0.85, 0.90)
        ],
        *[
            (
                f"best_length_{minimum_length}",
                replace(
                    base,
                    uncertainty_weight=1.0,
                    strong_threshold=0.90,
                    weak_threshold=0.50,
                    min_vector_length=float(minimum_length),
                ),
                0.0,
                0,
            )
            for minimum_length in (32, 40, 48)
        ],
    ]
    variants = [variant for variant in variants if variant[0].startswith("best_")]
    for variant in variants:
        run_variant(variant[0], evidence, variant[1], size, variant[2], variant[3])


if __name__ == "__main__":
    run(512)
