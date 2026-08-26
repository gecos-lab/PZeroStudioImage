"""Typed data models shared by detection, tracing, and presentation layers.

The classes in this module deliberately have no Qt, rasterio, or machine-learning
dependency.  Rasterio ``Affine`` and CRS objects may be stored when available,
but are kept opaque so callers do not lose their original geospatial metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Tuple, Union

import numpy as np


PathLike = Union[str, Path]
PixelXY = Tuple[int, int]


def _affine_coefficients(transform: Any) -> Tuple[float, float, float, float, float, float]:
    """Return ``(a, b, c, d, e, f)`` without importing rasterio.

    Attribute access supports ``affine.Affine``/``rasterio.Affine``.  A plain
    sequence is interpreted in the same matrix order, not GDAL geotransform
    order::

        world_x = a * pixel_x + b * pixel_y + c
        world_y = d * pixel_x + e * pixel_y + f
    """

    names = ("a", "b", "c", "d", "e", "f")
    if all(hasattr(transform, name) for name in names):
        coefficients = tuple(float(getattr(transform, name)) for name in names)
        return coefficients  # type: ignore[return-value]
    try:
        values = tuple(transform)
    except TypeError as exc:
        raise TypeError("transform must be affine-like or a six-value sequence") from exc
    if len(values) < 6:
        raise ValueError("an affine sequence must contain at least six values")
    return tuple(float(value) for value in values[:6])  # type: ignore[return-value]


@dataclass
class RasterDocument:
    """An in-memory raster plus its unchanged spatial reference.

    ``image`` must be ``(height, width)`` or band-last
    ``(height, width, channels)``.  The original array dtype and the original
    ``transform`` and ``crs`` objects are retained.  Algorithms request a
    normalized floating-point view through :meth:`normalized_grayscale`
    instead of destructively rescaling the source raster.
    """

    image: np.ndarray
    transform: Optional[Any] = None
    crs: Optional[Any] = None
    nodata: Optional[float] = None
    validity_mask: Optional[np.ndarray] = None
    source_path: Optional[PathLike] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.image = np.asarray(self.image)
        if self.image.ndim not in (2, 3):
            raise ValueError("image must be a 2-D raster or a band-last 3-D raster")
        if self.image.shape[0] == 0 or self.image.shape[1] == 0:
            raise ValueError("image dimensions must be non-zero")
        if self.image.ndim == 3 and self.image.shape[2] == 0:
            raise ValueError("a multiband raster must contain at least one channel")
        if self.validity_mask is not None:
            validity = np.asarray(self.validity_mask, dtype=bool)
            if validity.shape != self.shape:
                raise ValueError("validity_mask must match the raster's spatial shape")
            self.validity_mask = np.ascontiguousarray(validity)
        if self.transform is not None:
            _affine_coefficients(self.transform)
        if self.source_path is not None:
            self.source_path = Path(self.source_path)
        self.metadata = dict(self.metadata)

    @property
    def shape(self) -> Tuple[int, int]:
        """Spatial ``(height, width)`` shape."""

        return int(self.image.shape[0]), int(self.image.shape[1])

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    @property
    def band_count(self) -> int:
        return 1 if self.image.ndim == 2 else int(self.image.shape[2])

    def valid_mask(self) -> np.ndarray:
        """Return pixels that are finite and different from the nodata value."""

        if self.image.ndim == 2:
            finite = np.isfinite(self.image)
            if self.nodata is None:
                valid = finite
            elif np.isnan(self.nodata):
                valid = finite & ~np.isnan(self.image)
            else:
                valid = finite & (self.image != self.nodata)
        else:
            finite = np.all(np.isfinite(self.image), axis=2)
            if self.nodata is None:
                valid = finite
            elif np.isnan(self.nodata):
                valid = finite & ~np.any(np.isnan(self.image), axis=2)
            else:
                valid = finite & ~np.all(self.image == self.nodata, axis=2)
        if self.validity_mask is not None:
            valid &= self.validity_mask
        return valid

    def grayscale(self) -> np.ndarray:
        """Return a float32 grayscale copy without changing its value range."""

        image = self.image.astype(np.float32, copy=False)
        if image.ndim == 2:
            return np.array(image, dtype=np.float32, copy=True)
        if image.shape[2] == 1:
            return np.array(image[..., 0], dtype=np.float32, copy=True)
        # ITU-R BT.709 weights work for RGB while avoiding an OpenCV dependency.
        gray = (
            image[..., 0] * np.float32(0.2126)
            + image[..., 1] * np.float32(0.7152)
            + image[..., 2] * np.float32(0.0722)
        )
        return np.asarray(gray, dtype=np.float32)

    def normalized_grayscale(
        self,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
    ) -> np.ndarray:
        """Return robustly normalized grayscale data in ``[0, 1]``.

        Percentiles are measured only over valid pixels.  Invalid pixels are
        filled with the valid median for stable filtering; callers should use
        :meth:`valid_mask` to exclude them from outputs.
        """

        if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
            raise ValueError("normalization percentiles must satisfy 0 <= low < high <= 100")
        gray = self.grayscale()
        valid = self.valid_mask()
        if not np.any(valid):
            raise ValueError("raster contains no valid pixels")
        values = gray[valid].astype(np.float64, copy=False)
        low, high = np.percentile(values, (lower_percentile, upper_percentile))
        if not np.isfinite(low) or not np.isfinite(high):
            raise ValueError("raster normalization produced non-finite limits")
        if high <= low:
            normalized = np.zeros(self.shape, dtype=np.float32)
        else:
            normalized = np.clip((gray - low) / (high - low), 0.0, 1.0).astype(np.float32)
        median = np.float32(np.median(values))
        if high > low:
            median = np.float32(np.clip((median - low) / (high - low), 0.0, 1.0))
        else:
            median = np.float32(0.0)
        normalized[~valid] = median
        return normalized

    def pixel_to_world(self, xy: np.ndarray, center: bool = True) -> np.ndarray:
        """Transform one or more ``(x, y)`` pixel coordinates to world space."""

        if self.transform is None:
            raise ValueError("this raster has no affine transform")
        points = np.asarray(xy, dtype=np.float64)
        original_shape = points.shape
        if points.ndim == 1:
            if points.shape[0] != 2:
                raise ValueError("a coordinate must contain x and y")
            points = points.reshape(1, 2)
        elif points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("coordinates must have shape (2,) or (n, 2)")
        a, b, c, d, e, f = _affine_coefficients(self.transform)
        offset = 0.5 if center else 0.0
        px = points[:, 0] + offset
        py = points[:, 1] + offset
        result = np.column_stack((a * px + b * py + c, d * px + e * py + f))
        return result[0] if len(original_shape) == 1 else result

    def world_to_pixel(self, xy: np.ndarray, center: bool = True) -> np.ndarray:
        """Apply the inverse affine transform to world ``(x, y)`` coordinates."""

        if self.transform is None:
            raise ValueError("this raster has no affine transform")
        points = np.asarray(xy, dtype=np.float64)
        original_shape = points.shape
        if points.ndim == 1:
            if points.shape[0] != 2:
                raise ValueError("a coordinate must contain x and y")
            points = points.reshape(1, 2)
        elif points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("coordinates must have shape (2,) or (n, 2)")
        a, b, c, d, e, f = _affine_coefficients(self.transform)
        determinant = a * e - b * d
        if abs(determinant) <= np.finfo(np.float64).eps:
            raise ValueError("affine transform is singular")
        dx = points[:, 0] - c
        dy = points[:, 1] - f
        px = (e * dx - b * dy) / determinant
        py = (-d * dx + a * dy) / determinant
        if center:
            px -= 0.5
            py -= 0.5
        result = np.column_stack((px, py))
        return result[0] if len(original_shape) == 1 else result


@dataclass
class EvidenceMap:
    """Dense fracture evidence aligned to a :class:`RasterDocument`.

    ``probability`` and ``uncertainty`` are in ``[0, 1]``.  ``orientation`` is
    an *axial* tangent angle in radians, normalized to ``[0, pi)``: a line at
    angle zero and a line at angle pi are therefore equivalent.
    """

    probability: np.ndarray
    orientation: np.ndarray
    uncertainty: np.ndarray
    valid_mask: Optional[np.ndarray] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        probability = np.asarray(self.probability, dtype=np.float32)
        orientation = np.asarray(self.orientation, dtype=np.float32)
        uncertainty = np.asarray(self.uncertainty, dtype=np.float32)
        if probability.ndim != 2:
            raise ValueError("evidence arrays must be two-dimensional")
        if orientation.shape != probability.shape or uncertainty.shape != probability.shape:
            raise ValueError("probability, orientation, and uncertainty must have equal shapes")
        if self.valid_mask is None:
            valid = np.ones(probability.shape, dtype=bool)
        else:
            valid = np.array(self.valid_mask, dtype=bool, copy=True)
            if valid.shape != probability.shape:
                raise ValueError("valid_mask must have the same shape as evidence arrays")
        finite = np.isfinite(probability) & np.isfinite(orientation) & np.isfinite(uncertainty)
        valid &= finite
        probability = np.clip(probability, 0.0, 1.0)
        uncertainty = np.clip(uncertainty, 0.0, 1.0)
        orientation = np.mod(orientation, np.float32(np.pi))
        probability[~valid] = 0.0
        orientation[~valid] = 0.0
        uncertainty[~valid] = 1.0
        self.probability = np.ascontiguousarray(probability)
        self.orientation = np.ascontiguousarray(orientation)
        self.uncertainty = np.ascontiguousarray(uncertainty)
        self.valid_mask = np.ascontiguousarray(valid)
        self.metadata = dict(self.metadata)

    @property
    def shape(self) -> Tuple[int, int]:
        return int(self.probability.shape[0]), int(self.probability.shape[1])


class EvidenceDetector(Protocol):
    """Structural interface implemented by all fracture evidence detectors."""

    def predict(self, document: RasterDocument) -> EvidenceMap:
        """Infer dense evidence aligned to ``document``."""
