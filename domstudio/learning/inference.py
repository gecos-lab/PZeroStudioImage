"""Tiled, full-resolution inference for GeoTrace checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import numpy as np
import torch
from torch import Tensor

from domstudio.core import EvidenceMap, RasterDocument

from .model import GeoTraceNet, ModelOutput


@dataclass(frozen=True, slots=True)
class LearnedDetectorConfig:
    checkpoint: Path
    expected_sha256: str
    tile_size: int = 512
    overlap: int = 64
    batch_size: int = 4
    device: str = "auto"
    mixed_precision: bool = True
    allow_torchscript: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.expected_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", self.expected_sha256
        ):
            raise ValueError("expected_sha256 must be a lowercase SHA-256 digest")
        for name, value in (
            ("tile_size", self.tile_size),
            ("overlap", self.overlap),
            ("batch_size", self.batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if not 64 <= self.tile_size <= 4096:
            raise ValueError("tile_size must be between 64 and 4096 pixels")
        if not 0 <= self.overlap < self.tile_size // 2:
            raise ValueError("overlap must be non-negative and less than half a tile")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    values = list(range(0, max(1, length - tile + 1), stride))
    final = length - tile
    if values[-1] != final:
        values.append(final)
    return values


def _image_tensor(document: RasterDocument) -> Tensor:
    image = np.asarray(document.image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    else:
        image = image[..., :3]
    image = image.astype(np.float32, copy=False)
    valid = document.valid_mask()
    normalized = np.zeros_like(image, dtype=np.float32)
    for channel in range(3):
        values = image[..., channel][valid]
        if values.size == 0:
            continue
        low, high = np.percentile(values, (1.0, 99.0))
        if high > low:
            channel_values = np.clip(
                (image[..., channel] - low) / (high - low), 0.0, 1.0
            )
            fill = float(np.clip((np.median(values) - low) / (high - low), 0.0, 1.0))
            normalized[..., channel] = channel_values
            normalized[..., channel][~valid] = fill
    return torch.from_numpy(np.moveaxis(normalized, -1, 0))


def _weight_window(size: int) -> np.ndarray:
    axis = np.hanning(size).astype(np.float32)
    window = np.outer(axis, axis)
    return np.maximum(window, np.float32(0.05))


class TorchFractureDetector:
    """Verify a local checkpoint and infer seamless evidence maps.

    State dictionaries created from :class:`GeoTraceNet` are loaded through
    ``weights_only=True``. TorchScript is accepted only when the caller opts in,
    because it is executable code rather than a data-only weight container.
    """

    def __init__(self, config: LearnedDetectorConfig) -> None:
        self.config = config
        checkpoint = Path(config.checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if checkpoint.stat().st_size > 1_000_000_000:
            raise ValueError("Checkpoint exceeds the supported one-gigabyte limit.")
        self.checkpoint = checkpoint
        self.checksum = _sha256(checkpoint)
        if self.checksum != config.expected_sha256:
            raise ValueError(
                "Checkpoint checksum mismatch: "
                f"expected {config.expected_sha256}, got {self.checksum}."
            )
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        self.model = self._load_model().to(self.device).eval()

    def _load_model(self) -> torch.nn.Module:
        try:
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=True)
        except (RuntimeError, ValueError, TypeError) as exc:
            if not self.config.allow_torchscript:
                raise ValueError(
                    "Checkpoint is not a data-only state dictionary. Set "
                    "allow_torchscript=True only for a trusted, verified file."
                ) from exc
            return torch.jit.load(str(self.checkpoint), map_location="cpu")
        state_dict = (
            payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        )
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint does not contain a model state dictionary.")
        width = 32
        stem_weight = state_dict.get("stem.0.weight")
        if isinstance(stem_weight, Tensor):
            width = int(stem_weight.shape[0])
        model = GeoTraceNet(width=width)
        model.load_state_dict(state_dict, strict=True)
        return model

    @staticmethod
    def _coerce_output(raw: object) -> ModelOutput:
        if isinstance(raw, ModelOutput):
            return raw
        if isinstance(raw, dict):
            return ModelOutput(raw["probability"], raw["orientation"], raw["uncertainty"])
        if isinstance(raw, (tuple, list)) and len(raw) == 3:
            return ModelOutput(raw[0], raw[1], raw[2])
        raise TypeError("Model must return probability, orientation, and uncertainty tensors.")

    def predict(self, document: RasterDocument) -> EvidenceMap:
        source = _image_tensor(document)
        _, height, width = source.shape
        tile = self.config.tile_size
        y_starts = _starts(height, tile, self.config.overlap)
        x_starts = _starts(width, tile, self.config.overlap)
        coordinates = [(y, x) for y in y_starts for x in x_starts]

        probability = np.zeros((height, width), dtype=np.float32)
        uncertainty = np.zeros_like(probability)
        orientation_x = np.zeros_like(probability)
        orientation_y = np.zeros_like(probability)
        weights = np.zeros_like(probability)
        blend = _weight_window(tile)

        for offset in range(0, len(coordinates), self.config.batch_size):
            batch_coordinates = coordinates[offset : offset + self.config.batch_size]
            tiles: list[Tensor] = []
            sizes: list[tuple[int, int]] = []
            for y, x in batch_coordinates:
                patch = source[:, y : y + tile, x : x + tile]
                patch_height, patch_width = patch.shape[-2:]
                sizes.append((patch_height, patch_width))
                if patch_height != tile or patch_width != tile:
                    patch = torch.nn.functional.pad(
                        patch,
                        (0, tile - patch_width, 0, tile - patch_height),
                        mode="replicate",
                    )
                tiles.append(patch)
            batch = torch.stack(tiles).to(self.device, non_blocking=True)
            autocast_enabled = self.config.mixed_precision and self.device.type == "cuda"
            with torch.inference_mode(), torch.autocast(
                device_type=self.device.type,
                enabled=autocast_enabled,
            ):
                output = self._coerce_output(self.model(batch))

            expected_scalar = (len(batch_coordinates), 1, tile, tile)
            expected_orientation = (len(batch_coordinates), 2, tile, tile)
            if tuple(output.probability.shape) != expected_scalar:
                raise ValueError("Model probability output violates the tile contract.")
            if tuple(output.orientation.shape) != expected_orientation:
                raise ValueError("Model orientation output violates the tile contract.")
            if tuple(output.uncertainty.shape) != expected_scalar:
                raise ValueError("Model uncertainty output violates the tile contract.")
            values = (output.probability, output.orientation, output.uncertainty)
            if not all(bool(torch.all(torch.isfinite(value))) for value in values):
                raise ValueError("Model outputs contain non-finite values.")
            for name, value in (
                ("probability", output.probability),
                ("uncertainty", output.uncertainty),
            ):
                if bool(torch.any((value < -1.0e-5) | (value > 1.0 + 1.0e-5))):
                    raise ValueError(f"Model {name} output violates its sigmoid range.")
            orientation_norm = torch.linalg.vector_norm(output.orientation, dim=1)
            nonzero = orientation_norm > 1.0e-5
            if bool(torch.any(torch.abs(orientation_norm[nonzero] - 1.0) > 1.0e-3)):
                raise ValueError("Model orientation output is not L2-normalized.")

            prob = output.probability.float().cpu().numpy()[:, 0]
            vector = output.orientation.float().cpu().numpy()
            unc = output.uncertainty.float().cpu().numpy()[:, 0]
            for index, ((y, x), (patch_height, patch_width)) in enumerate(
                zip(batch_coordinates, sizes)
            ):
                window = blend[:patch_height, :patch_width]
                ys = slice(y, y + patch_height)
                xs = slice(x, x + patch_width)
                probability[ys, xs] += prob[index, :patch_height, :patch_width] * window
                tile_probability = prob[index, :patch_height, :patch_width]
                tile_uncertainty = unc[index, :patch_height, :patch_width]
                reliability = np.clip(tile_probability * (1.0 - tile_uncertainty), 0.0, 1.0)
                orientation_window = window * reliability
                orientation_x[ys, xs] += (
                    vector[index, 0, :patch_height, :patch_width] * orientation_window
                )
                orientation_y[ys, xs] += (
                    vector[index, 1, :patch_height, :patch_width] * orientation_window
                )
                uncertainty[ys, xs] += unc[index, :patch_height, :patch_width] * window
                weights[ys, xs] += window

        weights = np.maximum(weights, np.finfo(np.float32).eps)
        probability /= weights
        uncertainty /= weights
        orientation = 0.5 * np.arctan2(orientation_y / weights, orientation_x / weights)
        orientation = np.mod(orientation, np.pi).astype(np.float32)
        return EvidenceMap(
            probability=probability,
            orientation=orientation,
            uncertainty=uncertainty,
            valid_mask=document.valid_mask(),
            metadata={
                "detector": "GeoTraceNet",
                "checkpoint_file": self.checkpoint.name,
                "checkpoint_sha256": self.checksum,
                "uncertainty_semantics": "learned_error_proxy_uncalibrated",
                "device": str(self.device),
                "tile_size": tile,
                "overlap": self.config.overlap,
            },
        )
