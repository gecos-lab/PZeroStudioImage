"""Conversions between scientific arrays and detached Qt display images."""

from __future__ import annotations

import numpy as np
from PyQt5.QtGui import QImage

from domstudio.core import EvidenceMap


def _channel_to_uint8(channel: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    source = np.asarray(channel)
    values = np.asarray(source, dtype=np.float32)
    finite = np.isfinite(values) if valid is None else (np.isfinite(values) & valid)
    if not np.any(finite):
        return np.zeros(values.shape, dtype=np.uint8)
    if source.dtype == np.uint8:
        result = np.array(source, dtype=np.uint8, copy=True)
        result[~finite] = 0
        return result
    low, high = np.percentile(values[finite], (1.0, 99.0))
    if high <= low:
        constant = float(low)
        if 0.0 <= constant <= 1.0:
            display_value = round(constant * 255.0)
        elif 0.0 <= constant <= 255.0:
            display_value = round(constant)
        else:
            display_value = 127 if constant > 0.0 else 0
        result = np.full(values.shape, display_value, dtype=np.uint8)
        result[~finite] = 0
        return result
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    scaled[~finite] = 0.0
    return np.round(scaled * 255.0).astype(np.uint8)


def _alpha_to_uint8(channel: np.ndarray) -> np.ndarray:
    """Preserve ordinary alpha semantics instead of contrast-stretching them."""

    source = np.asarray(channel)
    values = np.asarray(source, dtype=np.float32)
    finite = np.isfinite(values)
    if source.dtype == np.uint8:
        result = np.array(source, dtype=np.uint8, copy=True)
    elif np.issubdtype(source.dtype, np.integer):
        maximum = float(np.iinfo(source.dtype).max)
        result = np.round(np.clip(values / maximum, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif np.any(finite) and float(np.nanmax(values)) <= 1.0:
        result = np.round(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        result = np.round(np.clip(values, 0.0, 255.0)).astype(np.uint8)
    result[~finite] = 0
    return result


def array_to_qimage(image: np.ndarray) -> QImage:
    """Create an owned QImage while leaving the source array untouched."""
    array = np.asarray(image)
    if array.ndim == 2:
        display = _channel_to_uint8(array)
        height, width = display.shape
        return QImage(
            display.data, width, height, display.strides[0], QImage.Format_Grayscale8
        ).copy()
    if array.ndim != 3 or array.shape[2] not in (1, 3, 4):
        raise ValueError(f"Unsupported display array shape {array.shape}.")
    if array.shape[2] == 1:
        return array_to_qimage(array[..., 0])
    valid = np.all(np.isfinite(array), axis=2)
    channels = [_channel_to_uint8(array[..., index], valid) for index in range(3)]
    if array.shape[2] == 4:
        channels.append(_alpha_to_uint8(array[..., 3]))
    display = np.dstack(channels)
    height, width, channel_count = display.shape
    image_format = QImage.Format_RGB888 if channel_count == 3 else QImage.Format_RGBA8888
    return QImage(display.data, width, height, display.strides[0], image_format).copy()


def evidence_to_qimage(evidence: EvidenceMap, threshold: float = 0.45) -> QImage:
    """Render fracture evidence in teal and the detector's error proxy in amber."""
    probability = np.asarray(evidence.probability, dtype=np.float32)
    uncertainty = np.asarray(evidence.uncertainty, dtype=np.float32)
    visible = np.clip(
        (probability - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0
    )
    certainty = 1.0 - uncertainty
    teal = np.array([29.0, 225.0, 168.0], dtype=np.float32)
    amber = np.array([255.0, 170.0, 64.0], dtype=np.float32)
    colour = certainty[..., None] * teal + uncertainty[..., None] * amber
    rgba = np.empty((*probability.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.round(np.clip(colour, 0.0, 255.0)).astype(np.uint8)
    rgba[..., 3] = np.round(visible * 235.0).astype(np.uint8)
    rgba[~evidence.valid_mask, 3] = 0
    height, width = probability.shape
    return QImage(
        rgba.data, width, height, rgba.strides[0], QImage.Format_RGBA8888
    ).copy()
