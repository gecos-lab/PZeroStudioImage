"""Raster loading without resampling or discarding spatial metadata."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from domstudio.core import RasterDocument


class RasterIOError(RuntimeError):
    """Raised when an input image cannot be decoded."""


def _display_ready(array: np.ndarray) -> np.ndarray:
    """Return a contiguous 2-D or RGB image while retaining full resolution."""
    data = np.asarray(array)
    if data.ndim == 3 and data.shape[0] in (1, 3, 4) and data.shape[-1] not in (3, 4):
        data = np.moveaxis(data, 0, -1)
    if data.ndim == 3 and data.shape[-1] > 4:
        data = data[..., :3]
    if data.ndim not in (2, 3):
        raise RasterIOError(f"Expected a 2-D or RGB raster, got shape {data.shape}.")
    return np.ascontiguousarray(data)


def load_raster(path: str | Path) -> RasterDocument:
    """Load a common image or GeoTIFF without changing its dimensions.

    Rasterio is used first for TIFF input so CRS, affine transform, nodata, and
    multiband values survive the UI round trip. Other formats use OpenCV.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise RasterIOError(f"Image does not exist: {source}")

    if source.suffix.lower() in {".tif", ".tiff", ".jp2", ".j2k"}:
        try:
            import rasterio

            with rasterio.open(source) as dataset:
                if dataset.count >= 3:
                    image = np.moveaxis(dataset.read((1, 2, 3)), 0, -1)
                else:
                    image = dataset.read(1)
                image = _display_ready(image)
                validity = dataset.dataset_mask() > 0
                if np.all(validity):
                    validity = None
                return RasterDocument(
                    image=image,
                    transform=dataset.transform,
                    crs=dataset.crs,
                    nodata=dataset.nodata,
                    validity_mask=validity,
                    source_path=source,
                    metadata={
                        "driver": dataset.driver,
                        "source_band_count": dataset.count,
                        "source_dtypes": list(dataset.dtypes),
                        "color_interpretation": [
                            value.name for value in dataset.colorinterp
                        ],
                    },
                )
        except Exception as exc:
            raise RasterIOError(
                f"Unable to load spatial raster metadata: {source} ({exc})"
            ) from exc

    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RasterIOError(f"Unable to decode image: {source}")
    validity = None
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
            validity = image[..., 3] > 0
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return RasterDocument(
        image=_display_ready(image),
        validity_mask=validity,
        source_path=source,
    )
