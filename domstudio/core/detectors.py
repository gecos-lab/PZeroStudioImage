"""Fracture evidence detectors with no GUI dependency.

The deterministic detector is intended as a transparent, reproducible fallback
and as a meaningful baseline for learned models.  It uses scale-normalized
Hessian ridge evidence rather than a thresholded first-derivative edge filter:
thin fractures are represented by their centreline response, tangent direction,
and cross-scale disagreement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.ndimage import gaussian_filter

from .models import EvidenceMap, RasterDocument


@dataclass(frozen=True)
class DetectorConfig:
    """Configuration for :class:`MultiscaleFractureDetector`.

    ``scales`` are Gaussian standard deviations in pixels.  Fracture polarity
    describes the expected centreline relative to its local surroundings.
    ``beta`` controls rejection of blob-like Hessian responses, while
    ``contrast_scale`` controls the soft suppression of low-contrast structure.
    """

    scales: Tuple[float, ...] = (0.8, 1.2, 1.8, 2.7, 4.0)
    polarity: str = "dark"
    beta: float = 0.5
    contrast_percentile: float = 90.0
    contrast_scale: float = 0.5
    probability_gamma: float = 0.75
    support_threshold: float = 0.15
    normalization_percentiles: Tuple[float, float] = (1.0, 99.0)

    def __post_init__(self) -> None:
        if not self.scales or any(scale <= 0.0 for scale in self.scales):
            raise ValueError("scales must contain positive values")
        if tuple(sorted(self.scales)) != self.scales:
            raise ValueError("scales must be ordered from fine to coarse")
        if self.polarity not in {"dark", "bright", "both"}:
            raise ValueError("polarity must be 'dark', 'bright', or 'both'")
        if self.beta <= 0.0 or self.contrast_scale <= 0.0:
            raise ValueError("beta and contrast_scale must be positive")
        if not 0.0 < self.contrast_percentile <= 100.0:
            raise ValueError("contrast_percentile must be in (0, 100]")
        if self.probability_gamma <= 0.0:
            raise ValueError("probability_gamma must be positive")
        if not 0.0 <= self.support_threshold <= 1.0:
            raise ValueError("support_threshold must be in [0, 1]")
        low, high = self.normalization_percentiles
        if not 0.0 <= low < high <= 100.0:
            raise ValueError("normalization_percentiles must satisfy 0 <= low < high <= 100")


class MultiscaleFractureDetector:
    """Deterministic, scale-aware ridge detector for fracture centrelines.

    At every scale, scale-normalized Hessian eigenvalues provide a line-likeness
    response and the low-curvature eigenvector provides the fracture tangent.
    Axial circular averaging combines orientations.  The returned uncertainty
    rises when evidence is weak, appears at only one scale, or has inconsistent
    orientations across scales.
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        self.config = config or DetectorConfig()

    def predict(
        self,
        document: RasterDocument,
        *,
        cancel: Optional[Callable[[], bool]] = None,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> EvidenceMap:
        """Compute probability, tangent orientation, and uncertainty maps."""

        config = self.config
        image = document.normalized_grayscale(*config.normalization_percentiles)
        valid = document.valid_mask()
        shape = document.shape

        maximum = np.zeros(shape, dtype=np.float32)
        response_sum = np.zeros(shape, dtype=np.float32)
        vector_x = np.zeros(shape, dtype=np.float32)
        vector_y = np.zeros(shape, dtype=np.float32)
        support_count = np.zeros(shape, dtype=np.float32)

        for scale_index, sigma in enumerate(config.scales, start=1):
            if cancel is not None and cancel():
                raise InterruptedError("Fracture detection was cancelled.")
            response, orientation = self._scale_response(image, valid, sigma)
            maximum = np.maximum(maximum, response)
            response_sum += response
            vector_x += response * np.cos(2.0 * orientation)
            vector_y += response * np.sin(2.0 * orientation)
            support_count += response >= config.support_threshold
            if progress is not None:
                progress(scale_index, len(config.scales))

        epsilon = np.float32(np.finfo(np.float32).eps)
        orientation = np.mod(
            0.5 * np.arctan2(vector_y, vector_x), np.float32(np.pi)
        ).astype(np.float32)
        orientation_coherence = np.clip(
            np.hypot(vector_x, vector_y) / (response_sum + epsilon), 0.0, 1.0
        )
        scale_agreement = np.clip(
            response_sum / (maximum * len(config.scales) + epsilon), 0.0, 1.0
        )
        threshold_support = support_count / np.float32(len(config.scales))

        # Gamma is an explicit monotonic calibration.  Values below one retain
        # weak but spatially coherent evidence for the interactive path solver.
        probability = np.power(np.clip(maximum, 0.0, 1.0), config.probability_gamma)
        agreement = 0.5 * scale_agreement + 0.5 * threshold_support
        confidence = probability * (0.5 + 0.5 * orientation_coherence) * (
            0.6 + 0.4 * agreement
        )
        uncertainty = np.clip(1.0 - confidence, 0.0, 1.0)

        probability[~valid] = 0.0
        orientation[~valid] = 0.0
        uncertainty[~valid] = 1.0
        metadata = {
            "detector": "multiscale_hessian_ridge",
            "detector_version": 1,
            "orientation_convention": "axial tangent radians in [0, pi)",
            "uncertainty_semantics": "heuristic_cross_scale_disagreement",
            "config": asdict(config),
        }
        return EvidenceMap(probability, orientation, uncertainty, valid, metadata)

    def _scale_response(
        self,
        image: np.ndarray,
        valid: np.ndarray,
        sigma: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return Frangi-style line evidence and tangent angle at one scale."""

        sigma_squared = np.float32(sigma * sigma)
        # Axis zero is image y and axis one is image x.
        hxx = gaussian_filter(image, sigma=sigma, order=(0, 2), mode="reflect") * sigma_squared
        hxy = gaussian_filter(image, sigma=sigma, order=(1, 1), mode="reflect") * sigma_squared
        hyy = gaussian_filter(image, sigma=sigma, order=(2, 0), mode="reflect") * sigma_squared

        trace = hxx + hyy
        discriminant = np.sqrt(np.maximum((hxx - hyy) ** 2 + 4.0 * hxy**2, 0.0))
        eigen_minus = 0.5 * (trace - discriminant)
        eigen_plus = 0.5 * (trace + discriminant)
        minus_is_small = np.abs(eigen_minus) <= np.abs(eigen_plus)
        lambda_small = np.where(minus_is_small, eigen_minus, eigen_plus)
        lambda_large = np.where(minus_is_small, eigen_plus, eigen_minus)

        # This is the eigenvector angle for the algebraically larger eigenvalue.
        plus_angle = 0.5 * np.arctan2(2.0 * hxy, hxx - hyy)
        # The line tangent follows the eigenvector with the smaller absolute
        # eigenvalue.  eigen_minus is orthogonal to eigen_plus.
        tangent = np.where(minus_is_small, plus_angle + 0.5 * np.pi, plus_angle)
        tangent = np.mod(tangent, np.pi).astype(np.float32)

        epsilon = np.finfo(np.float32).eps
        anisotropy = np.abs(lambda_small) / (np.abs(lambda_large) + epsilon)
        structure = np.sqrt(lambda_small**2 + lambda_large**2)
        structure_values = structure[valid]
        contrast_reference = float(
            np.percentile(structure_values, self.config.contrast_percentile)
        )
        contrast = max(self.config.contrast_scale * contrast_reference, epsilon)
        blob_rejection = np.exp(-(anisotropy**2) / (2.0 * self.config.beta**2))
        noise_rejection = 1.0 - np.exp(-(structure**2) / (2.0 * contrast**2))
        response = blob_rejection * noise_rejection

        if self.config.polarity == "dark":
            response = np.where(lambda_large > 0.0, response, 0.0)
        elif self.config.polarity == "bright":
            response = np.where(lambda_large < 0.0, response, 0.0)
        response = np.asarray(np.clip(response, 0.0, 1.0), dtype=np.float32)
        response[~valid] = 0.0
        return response, tangent


PredictorOutput = Union[
    EvidenceMap,
    Mapping[str, np.ndarray],
    Sequence[np.ndarray],
    np.ndarray,
]


@dataclass(frozen=True)
class ModelDetectorConfig:
    """Output conventions for :class:`ModelDetector`."""

    probability_is_logits: bool = False
    orientation_in_degrees: bool = False
    default_uncertainty: float = 0.5
    normalization_percentiles: Tuple[float, float] = (1.0, 99.0)

    def __post_init__(self) -> None:
        if not 0.0 <= self.default_uncertainty <= 1.0:
            raise ValueError("default_uncertainty must be in [0, 1]")
        low, high = self.normalization_percentiles
        if not 0.0 <= low < high <= 100.0:
            raise ValueError("normalization_percentiles must satisfy 0 <= low < high <= 100")


class ModelDetector:
    """Dependency-neutral adapter for a learned array-to-array predictor.

    The callable receives a normalized ``float32 (height, width)`` array.  It may
    return an :class:`EvidenceMap`, a mapping with ``probability`` and optional
    ``orientation``/``uncertainty`` entries, a tuple in that order, or a single
    probability array. Framework-specific loading is intentionally excluded;
    application inference uses the verified pack loader in ``domstudio.learning``.
    """

    def __init__(
        self,
        predictor: Callable[[np.ndarray], PredictorOutput],
        config: Optional[ModelDetectorConfig] = None,
        model_name: str = "external_model",
    ) -> None:
        if not callable(predictor):
            raise TypeError("predictor must be callable")
        self.predictor = predictor
        self.config = config or ModelDetectorConfig()
        self.model_name = model_name

    def predict(self, document: RasterDocument) -> EvidenceMap:
        image = document.normalized_grayscale(*self.config.normalization_percentiles)
        output = self.predictor(image)
        if isinstance(output, EvidenceMap):
            if output.shape != document.shape:
                raise ValueError("model evidence shape does not match the input raster")
            combined_valid = output.valid_mask & document.valid_mask()
            metadata = dict(output.metadata)
            metadata.setdefault("detector", self.model_name)
            return EvidenceMap(
                output.probability,
                output.orientation,
                output.uncertainty,
                combined_valid,
                metadata,
            )

        probability, orientation, uncertainty = self._unpack_output(output, document.shape)
        if self.config.probability_is_logits:
            clipped = np.clip(probability, -80.0, 80.0)
            probability = 1.0 / (1.0 + np.exp(-clipped))
        if self.config.orientation_in_degrees:
            orientation = np.deg2rad(orientation)
        metadata = {
            "detector": self.model_name,
            "detector_version": 1,
            "orientation_convention": "axial tangent radians in [0, pi)",
        }
        return EvidenceMap(
            probability,
            orientation,
            uncertainty,
            document.valid_mask(),
            metadata,
        )

    def _unpack_output(
        self,
        output: PredictorOutput,
        shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(output, Mapping):
            if "probability" not in output:
                raise ValueError("model output mapping must contain 'probability'")
            probability = np.asarray(output["probability"])
            orientation = np.asarray(output.get("orientation", np.zeros(shape, dtype=np.float32)))
            uncertainty = np.asarray(
                output.get(
                    "uncertainty",
                    np.full(shape, self.config.default_uncertainty, dtype=np.float32),
                )
            )
        elif isinstance(output, np.ndarray):
            probability = output
            orientation = np.zeros(shape, dtype=np.float32)
            uncertainty = np.full(shape, self.config.default_uncertainty, dtype=np.float32)
        else:
            values = tuple(output)
            if not 1 <= len(values) <= 3:
                raise ValueError("model tuple output must have one to three arrays")
            probability = np.asarray(values[0])
            orientation = (
                np.asarray(values[1]) if len(values) >= 2 else np.zeros(shape, dtype=np.float32)
            )
            uncertainty = (
                np.asarray(values[2])
                if len(values) >= 3
                else np.full(shape, self.config.default_uncertainty, dtype=np.float32)
            )
        probability = _squeeze_spatial_output(probability, shape, "probability")
        orientation = _squeeze_spatial_output(orientation, shape, "orientation")
        uncertainty = _squeeze_spatial_output(uncertainty, shape, "uncertainty")
        return probability, orientation, uncertainty

def _squeeze_spatial_output(array: np.ndarray, shape: Tuple[int, int], name: str) -> np.ndarray:
    """Accept HxW, 1xHxW, or 1x1xHxW model outputs."""

    result = np.asarray(array)
    while result.ndim > 2 and result.shape[0] == 1:
        result = result[0]
    if result.shape != shape:
        raise ValueError(f"model {name} output has shape {result.shape}, expected {shape}")
    return np.asarray(result, dtype=np.float32)
