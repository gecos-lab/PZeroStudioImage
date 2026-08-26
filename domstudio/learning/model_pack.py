"""Strict, self-checking ONNX model packs for reproducible inference."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterator, Mapping

import numpy as np

from domstudio.core import EvidenceMap, RasterDocument


PACK_VERSION = 2
MAX_MANIFEST_BYTES = 1_000_000
MAX_MODEL_BYTES = 1_000_000_000
MIN_TILE_SIZE = 64
MAX_TILE_SIZE = 4096
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def canonical_manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Hash canonical manifest content, excluding its derived pack digest."""

    canonical = dict(payload)
    canonical.pop("pack_sha256", None)
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def compute_pack_sha256(payload: Mapping[str, Any], model_sha256: str) -> str:
    """Bind the complete canonical manifest to its verified model bytes."""

    if not isinstance(model_sha256, str) or not _SHA256_PATTERN.fullmatch(
        model_sha256
    ):
        raise ValueError("model_sha256 must be a lowercase SHA-256 digest.")
    material = f"{canonical_manifest_sha256(payload)}\n{model_sha256}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key {key!r} is not allowed in model.json.")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Non-finite JSON number {value!r} is not allowed.")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object.")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string.")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must be a non-empty string.")
    return result


def _sha256(value: Any, name: str) -> str:
    result = _text(value, name).lower()
    if not _SHA256_PATTERN.fullmatch(result):
        raise ValueError(f"{name} must be a lowercase 64-character SHA-256 digest.")
    return result


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _shape(value: Any, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an integer array.")
    return [_integer(dimension, f"{name} dimension") for dimension in value]


@dataclass(frozen=True, slots=True)
class ModelPackManifest:
    name: str
    model_version: str
    architecture: str
    model_path: Path
    model_sha256: str
    manifest_sha256: str
    pack_sha256: str
    license: str
    input_name: str
    probability_output: str
    orientation_output: str
    uncertainty_output: str
    lower_percentile: float
    upper_percentile: float
    tile_size: int
    overlap: int
    provenance: Mapping[str, Any]
    limitations: tuple[str, ...]
    version: int = PACK_VERSION

    @property
    def sha256(self) -> str:
        """Backward-compatible alias for the verified model digest."""

        return self.model_sha256

    @classmethod
    def load(cls, directory: str | Path) -> "ModelPackManifest":
        root = Path(directory).expanduser().resolve()
        manifest_path = root / "model.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        if manifest_path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("model.json exceeds the model-pack manifest size limit.")
        payload = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
        payload = _mapping(payload, "model.json")

        version = _integer(payload.get("version"), "version")
        if version != PACK_VERSION:
            raise ValueError(
                f"Unsupported model-pack version {version}; expected {PACK_VERSION}."
            )
        relative_model = Path(_text(payload.get("model", ""), "model"))
        if relative_model.is_absolute() or relative_model.suffix.lower() != ".onnx":
            raise ValueError("model must be a relative .onnx path.")
        model_path = (root / relative_model).resolve()
        try:
            model_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("The model path escapes its model-pack directory.") from exc
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        if model_path.stat().st_size > MAX_MODEL_BYTES:
            raise ValueError("The ONNX model exceeds the supported one-gigabyte limit.")

        expected_model = _sha256(payload.get("model_sha256", ""), "model_sha256")
        actual_model = _digest(model_path)
        if actual_model != expected_model:
            raise ValueError(
                f"Model checksum mismatch: expected {expected_model}, got {actual_model}."
            )
        expected_pack = _sha256(payload.get("pack_sha256", ""), "pack_sha256")
        actual_pack = compute_pack_sha256(payload, actual_model)
        if actual_pack != expected_pack:
            raise ValueError(
                f"Pack checksum mismatch: expected {expected_pack}, got {actual_pack}."
            )

        tile_size = _integer(payload.get("tile_size"), "tile_size")
        overlap = _integer(payload.get("overlap"), "overlap")
        if not MIN_TILE_SIZE <= tile_size <= MAX_TILE_SIZE:
            raise ValueError(
                f"tile_size must be between {MIN_TILE_SIZE} and {MAX_TILE_SIZE}."
            )
        if not 0 <= overlap < tile_size // 2:
            raise ValueError("overlap must be non-negative and less than half a tile.")

        contract = _mapping(payload.get("contract"), "contract")
        input_contract = _mapping(contract.get("input"), "contract.input")
        if input_contract.get("layout") != "NCHW":
            raise ValueError("Only NCHW model input is supported.")
        if input_contract.get("dtype") != "float32":
            raise ValueError("Only float32 model input is supported.")
        if input_contract.get("color_space") != "RGB":
            raise ValueError("Only RGB model input is supported.")
        if _shape(input_contract.get("shape"), "contract.input.shape") != [
            1,
            3,
            tile_size,
            tile_size,
        ]:
            raise ValueError("contract.input.shape must match [1, 3, tile_size, tile_size].")
        input_name = _text(input_contract.get("name", ""), "contract.input.name")

        preprocessing = _mapping(contract.get("preprocessing"), "contract.preprocessing")
        if preprocessing.get("method") != "per_image_channel_percentile":
            raise ValueError("Unsupported preprocessing method.")
        if preprocessing.get("invalid_fill") != "normalized_valid_median":
            raise ValueError("Unsupported invalid-pixel fill policy.")
        lower = _number(
            preprocessing.get("lower_percentile"),
            "contract.preprocessing.lower_percentile",
        )
        upper = _number(
            preprocessing.get("upper_percentile"),
            "contract.preprocessing.upper_percentile",
        )
        if not 0.0 <= lower < upper <= 100.0:
            raise ValueError("Preprocessing percentiles must satisfy 0 <= low < high <= 100.")

        outputs = _mapping(contract.get("outputs"), "contract.outputs")
        probability = _mapping(outputs.get("probability"), "probability output")
        orientation = _mapping(outputs.get("orientation"), "orientation output")
        uncertainty = _mapping(outputs.get("uncertainty"), "uncertainty output")
        for output_name, output in (
            ("probability", probability),
            ("orientation", orientation),
            ("uncertainty", uncertainty),
        ):
            if output.get("dtype") != "float32" or output.get("layout") != "NCHW":
                raise ValueError(
                    f"{output_name.capitalize()} output must declare NCHW float32."
                )
        if probability.get("activation") != "sigmoid":
            raise ValueError("Probability output must declare sigmoid activation.")
        if probability.get("semantics") != "fracture_centreline_probability":
            raise ValueError("Unsupported probability output semantics.")
        if _shape(probability.get("shape"), "probability output shape") != [
            1,
            1,
            tile_size,
            tile_size,
        ]:
            raise ValueError("Probability output shape must match the tile contract.")
        if orientation.get("activation") != "l2_normalized":
            raise ValueError("Orientation output must declare l2_normalized activation.")
        if _shape(orientation.get("shape"), "orientation output shape") != [
            1,
            2,
            tile_size,
            tile_size,
        ]:
            raise ValueError("Orientation output shape must match the tile contract.")
        if uncertainty.get("activation") != "sigmoid":
            raise ValueError("Uncertainty output must declare sigmoid activation.")
        if _shape(uncertainty.get("shape"), "uncertainty output shape") != [
            1,
            1,
            tile_size,
            tile_size,
        ]:
            raise ValueError("Uncertainty output shape must match the tile contract.")
        if uncertainty.get("semantics") != "learned_error_proxy_uncalibrated":
            raise ValueError(
                "Uncertainty must be declared as learned_error_proxy_uncalibrated."
            )
        if orientation.get("encoding") != "axial_cos2_sin2":
            raise ValueError("Orientation output must use axial_cos2_sin2 encoding.")
        output_names = (
            _text(probability.get("name", ""), "probability output name"),
            _text(orientation.get("name", ""), "orientation output name"),
            _text(uncertainty.get("name", ""), "uncertainty output name"),
        )
        if len({input_name, *output_names}) != 4:
            raise ValueError("Input and output tensor names must be distinct.")

        provenance = _mapping(payload.get("provenance"), "provenance")
        required_provenance = (
            "code_revision",
            "training_checkpoint_sha256",
            "dataset_id",
            "split_id",
        )
        for key in required_provenance:
            _text(provenance.get(key, ""), f"provenance.{key}")
        _sha256(
            provenance.get("training_checkpoint_sha256", ""),
            "provenance.training_checkpoint_sha256",
        )
        limitations_value = payload.get("limitations")
        if not isinstance(limitations_value, list) or not limitations_value:
            raise ValueError("limitations must be a non-empty JSON string array.")
        limitations = tuple(_text(item, "limitations item") for item in limitations_value)

        return cls(
            name=_text(payload.get("name", ""), "name"),
            model_version=_text(payload.get("model_version", ""), "model_version"),
            architecture=_text(payload.get("architecture", ""), "architecture"),
            model_path=model_path,
            model_sha256=actual_model,
            manifest_sha256=canonical_manifest_sha256(payload),
            pack_sha256=actual_pack,
            license=_text(payload.get("license", ""), "license"),
            input_name=input_name,
            probability_output=output_names[0],
            orientation_output=output_names[1],
            uncertainty_output=output_names[2],
            lower_percentile=lower,
            upper_percentile=upper,
            tile_size=tile_size,
            overlap=overlap,
            provenance=dict(provenance),
            limitations=limitations,
            version=version,
        )


def _normalized_rgb(
    document: RasterDocument,
    lower_percentile: float,
    upper_percentile: float,
) -> np.ndarray:
    image = np.asarray(document.image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    elif image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    else:
        image = image[..., :3]
    image = image.astype(np.float32, copy=False)
    valid = document.valid_mask()
    result = np.zeros_like(image, dtype=np.float32)
    for channel in range(3):
        values = image[..., channel][valid]
        if not values.size:
            continue
        low, high = np.percentile(values, (lower_percentile, upper_percentile))
        if high > low:
            normalized = np.clip((image[..., channel] - low) / (high - low), 0.0, 1.0)
            fill = float(np.clip((np.median(values) - low) / (high - low), 0.0, 1.0))
            result[..., channel] = normalized
            result[..., channel][~valid] = fill
    return np.moveaxis(result, -1, 0)


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def _onnx_tensors(graph: Any) -> Iterator[Any]:
    """Yield tensors recursively so external-data references can be rejected."""

    yield from graph.initializer
    for sparse in graph.sparse_initializer:
        yield sparse.values
        yield sparse.indices
    for node in graph.node:
        for attribute in node.attribute:
            if attribute.HasField("t"):
                yield attribute.t
            yield from attribute.tensors
            if attribute.HasField("sparse_tensor"):
                yield attribute.sparse_tensor.values
                yield attribute.sparse_tensor.indices
            for sparse in attribute.sparse_tensors:
                yield sparse.values
                yield sparse.indices
            if attribute.HasField("g"):
                yield from _onnx_tensors(attribute.g)
            for child_graph in attribute.graphs:
                yield from _onnx_tensors(child_graph)


def _validate_tile_outputs(
    probability: np.ndarray,
    orientation: np.ndarray,
    uncertainty: np.ndarray,
    tile_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_scalar = (1, 1, tile_size, tile_size)
    expected_orientation = (1, 2, tile_size, tile_size)
    values = (
        np.asarray(probability, dtype=np.float32),
        np.asarray(orientation, dtype=np.float32),
        np.asarray(uncertainty, dtype=np.float32),
    )
    if values[0].shape != expected_scalar:
        raise ValueError(f"Probability output shape {values[0].shape} != {expected_scalar}.")
    if values[1].shape != expected_orientation:
        raise ValueError(
            f"Orientation output shape {values[1].shape} != {expected_orientation}."
        )
    if values[2].shape != expected_scalar:
        raise ValueError(f"Uncertainty output shape {values[2].shape} != {expected_scalar}.")
    if not all(np.all(np.isfinite(value)) for value in values):
        raise ValueError("Model outputs contain non-finite values.")
    tolerance = 1.0e-5
    for name, value in (("probability", values[0]), ("uncertainty", values[2])):
        if float(value.min()) < -tolerance or float(value.max()) > 1.0 + tolerance:
            raise ValueError(f"Model {name} output violates its declared sigmoid range.")
    orientation_norm = np.linalg.norm(values[1], axis=1)
    nonzero = orientation_norm > tolerance
    if np.any(np.abs(orientation_norm[nonzero] - 1.0) > 1.0e-3):
        raise ValueError("Model orientation output is not L2-normalized.")
    return np.clip(values[0], 0.0, 1.0), values[1], np.clip(values[2], 0.0, 1.0)


class OnnxFractureDetector:
    """Run a strict local model pack using ONNX Runtime providers."""

    def __init__(self, directory: str | Path, providers: list[str] | None = None) -> None:
        self.manifest = ModelPackManifest.load(directory)
        try:
            import onnx
            from onnx.external_data_helper import uses_external_data
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "ONNX and ONNX Runtime are required; install domstudio[inference]."
            ) from exc

        graph = onnx.load_model(str(self.manifest.model_path), load_external_data=False)
        if any(uses_external_data(tensor) for tensor in _onnx_tensors(graph.graph)):
            raise ValueError("External ONNX tensor data is not allowed in a model pack.")
        onnx.checker.check_model(graph)
        self.onnx_version = onnx.__version__
        self.runtime_version = ort.__version__

        available = ort.get_available_providers()
        requested = providers or [
            name
            for name in (
                "CUDAExecutionProvider",
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            )
            if name in available
        ]
        if not requested:
            requested = ["CPUExecutionProvider"]
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ValueError(f"Unavailable ONNX Runtime provider(s): {', '.join(unknown)}")
        self.session = ort.InferenceSession(
            str(self.manifest.model_path), providers=requested
        )
        inputs = self.session.get_inputs()
        if len(inputs) != 1 or inputs[0].name != self.manifest.input_name:
            raise ValueError("ONNX model input does not match the manifest contract.")
        if inputs[0].type != "tensor(float)":
            raise ValueError("ONNX model input must be a float32 tensor.")
        expected_input_shape = [1, 3, self.manifest.tile_size, self.manifest.tile_size]
        if list(inputs[0].shape) != expected_input_shape:
            raise ValueError(
                f"ONNX input shape {inputs[0].shape} != {expected_input_shape}."
            )
        outputs = {output.name: output for output in self.session.get_outputs()}
        available_outputs = set(outputs)
        required_outputs = {
            self.manifest.probability_output,
            self.manifest.orientation_output,
            self.manifest.uncertainty_output,
        }
        if required_outputs != available_outputs:
            missing = ", ".join(sorted(required_outputs - available_outputs))
            unexpected = ", ".join(sorted(available_outputs - required_outputs))
            details = []
            if missing:
                details.append(f"missing: {missing}")
            if unexpected:
                details.append(f"undeclared: {unexpected}")
            raise ValueError(
                "ONNX outputs do not exactly match the manifest ("
                + "; ".join(details)
                + ")."
            )
        tile_size = self.manifest.tile_size
        expected_shapes = {
            self.manifest.probability_output: [1, 1, tile_size, tile_size],
            self.manifest.orientation_output: [1, 2, tile_size, tile_size],
            self.manifest.uncertainty_output: [1, 1, tile_size, tile_size],
        }
        for name, expected_shape in expected_shapes.items():
            output = outputs[name]
            if output.type != "tensor(float)" or list(output.shape) != expected_shape:
                raise ValueError(
                    f"ONNX output {name!r} must be float32 with shape {expected_shape}."
                )

    def predict(
        self,
        document: RasterDocument,
        *,
        cancel: Callable[[], bool] | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> EvidenceMap:
        source = _normalized_rgb(
            document,
            self.manifest.lower_percentile,
            self.manifest.upper_percentile,
        )
        _, height, width = source.shape
        tile = self.manifest.tile_size
        overlap = self.manifest.overlap
        y_starts = _tile_starts(height, tile, overlap)
        x_starts = _tile_starts(width, tile, overlap)
        taper = np.maximum(
            np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32), 0.05
        )
        probability = np.zeros((height, width), np.float32)
        uncertainty = np.zeros_like(probability)
        axis_x = np.zeros_like(probability)
        axis_y = np.zeros_like(probability)
        weights = np.zeros_like(probability)
        output_names = [
            self.manifest.probability_output,
            self.manifest.orientation_output,
            self.manifest.uncertainty_output,
        ]
        completed_tiles = 0
        total_tiles = len(y_starts) * len(x_starts)
        for y in y_starts:
            for x in x_starts:
                if cancel is not None and cancel():
                    raise InterruptedError("Model inference was cancelled.")
                patch = source[:, y : y + tile, x : x + tile]
                patch_height, patch_width = patch.shape[-2:]
                pad_mode = "reflect" if min(patch_height, patch_width) > 1 else "edge"
                padded_patch = np.pad(
                    patch,
                    ((0, 0), (0, tile - patch_height), (0, tile - patch_width)),
                    mode=pad_mode,
                )
                padded = padded_patch[None, ...].astype(np.float32, copy=False)
                raw = self.session.run(output_names, {self.manifest.input_name: padded})
                raw_probability, raw_orientation, raw_uncertainty = (
                    _validate_tile_outputs(*raw, tile)
                )
                tile_probability = raw_probability[0, 0, :patch_height, :patch_width]
                tile_uncertainty = raw_uncertainty[0, 0, :patch_height, :patch_width]
                tile_orientation = raw_orientation[0, :, :patch_height, :patch_width]
                window = taper[:patch_height, :patch_width]
                reliability = np.clip(
                    tile_probability * (1.0 - tile_uncertainty), 0.0, 1.0
                )
                orientation_window = window * reliability
                ys = slice(y, y + patch_height)
                xs = slice(x, x + patch_width)
                probability[ys, xs] += tile_probability * window
                axis_x[ys, xs] += tile_orientation[0] * orientation_window
                axis_y[ys, xs] += tile_orientation[1] * orientation_window
                uncertainty[ys, xs] += tile_uncertainty * window
                weights[ys, xs] += window
                completed_tiles += 1
                if progress is not None:
                    progress(completed_tiles, total_tiles)
        weights = np.maximum(weights, np.finfo(np.float32).eps)
        orientation = np.mod(0.5 * np.arctan2(axis_y, axis_x), np.pi)
        return EvidenceMap(
            probability=probability / weights,
            orientation=orientation.astype(np.float32),
            uncertainty=uncertainty / weights,
            valid_mask=document.valid_mask(),
            metadata={
                "detector": self.manifest.name,
                "model_version": self.manifest.model_version,
                "architecture": self.manifest.architecture,
                "format": "onnx-model-pack-v2",
                "model_file": self.manifest.model_path.name,
                "model_sha256": self.manifest.model_sha256,
                "manifest_sha256": self.manifest.manifest_sha256,
                "pack_sha256": self.manifest.pack_sha256,
                "license": self.manifest.license,
                "provenance": dict(self.manifest.provenance),
                "limitations": list(self.manifest.limitations),
                "tile_size": tile,
                "overlap": overlap,
                "onnx_version": self.onnx_version,
                "onnxruntime_version": self.runtime_version,
                "providers": self.session.get_providers(),
            },
        )
