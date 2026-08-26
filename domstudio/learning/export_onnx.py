"""Export a GeoTraceNet checkpoint as a validated ONNX model pack."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import inspect
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from .model import GeoTraceNet
from .model_pack import (
    MAX_TILE_SIZE,
    MIN_TILE_SIZE,
    PACK_VERSION,
    ModelPackManifest,
    _digest,
    _onnx_tensors,
    _validate_tile_outputs,
    compute_pack_sha256,
)


ONNX_OPSET = 18
PARITY_ATOL = 1.0e-5
PARITY_RTOL = 1.0e-4
PARITY_RANDOM_SEED = 1729
MODEL_FILENAME = "geotrace.onnx"
MANIFEST_FILENAME = "model.json"


def _required_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a non-empty string.")
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must be a non-empty string.")
    return result


def _validate_export_options(
    *,
    tile_size: int,
    overlap: int,
    lower_percentile: float,
    upper_percentile: float,
) -> None:
    if isinstance(tile_size, bool) or not isinstance(tile_size, int):
        raise TypeError("tile_size must be an integer.")
    if not MIN_TILE_SIZE <= tile_size <= MAX_TILE_SIZE:
        raise ValueError(
            f"tile_size must be between {MIN_TILE_SIZE} and {MAX_TILE_SIZE}."
        )
    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise TypeError("overlap must be an integer.")
    if not 0 <= overlap < tile_size // 2:
        raise ValueError("overlap must be non-negative and less than half a tile.")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")


def _load_model(checkpoint: Path) -> tuple[GeoTraceNet, int]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = payload.get("state_dict", payload) if isinstance(payload, Mapping) else payload
    if not isinstance(state_dict, Mapping):
        raise ValueError("Checkpoint does not contain a state dictionary.")
    stem = state_dict.get("stem.0.weight")
    if not isinstance(stem, torch.Tensor) or stem.ndim != 4 or int(stem.shape[0]) <= 0:
        raise ValueError("Checkpoint is missing a valid stem.0.weight tensor.")
    width = int(stem.shape[0])
    model = GeoTraceNet(width=width).cpu().eval()
    model.load_state_dict(state_dict, strict=True)
    return model, width


def _parity_inputs(tile_size: int) -> dict[str, torch.Tensor]:
    axis = torch.linspace(0.0, 1.0, tile_size, dtype=torch.float32)
    vertical = axis[:, None].expand(tile_size, tile_size)
    horizontal = axis[None, :].expand(tile_size, tile_size)
    diagonal = (horizontal + vertical) * 0.5
    gradient = (
        torch.stack((horizontal, vertical, diagonal), dim=0)
        .unsqueeze(0)
        .contiguous()
    )

    block_size = max(tile_size // 16, 1)
    coordinates = torch.arange(tile_size)
    checker = (
        (coordinates[:, None] // block_size + coordinates[None, :] // block_size) % 2
    ).to(torch.float32)
    checkerboard = (
        torch.stack((checker, 1.0 - checker, checker), dim=0)
        .unsqueeze(0)
        .contiguous()
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(PARITY_RANDOM_SEED)
    seeded_random = torch.rand(
        (1, 3, tile_size, tile_size), dtype=torch.float32, generator=generator
    )
    return {
        "zeros": torch.zeros((1, 3, tile_size, tile_size), dtype=torch.float32),
        "checkerboard": checkerboard,
        "gradient": gradient,
        "seeded_random": seeded_random,
    }


def _export_onnx(model: GeoTraceNet, example: torch.Tensor, destination: Path) -> str:
    parameters: dict[str, Any] = {
        "input_names": ["image"],
        "output_names": ["probability", "orientation", "uncertainty"],
        "opset_version": ONNX_OPSET,
        "do_constant_folding": True,
        "export_params": True,
    }
    export_signature = inspect.signature(torch.onnx.export).parameters
    if "dynamo" in export_signature:
        # The legacy exporter is available throughout the supported torch range and
        # does not introduce an undeclared onnxscript dependency.
        parameters["dynamo"] = False
    if "external_data" in export_signature:
        parameters["external_data"] = False
    torch.onnx.export(model, example, str(destination), **parameters)
    return "torch.onnx.export(dynamo=False)" if "dynamo" in parameters else "torch.onnx.export"


def _verify_runtime_contract(session: Any, tile_size: int) -> None:
    inputs = session.get_inputs()
    expected_input_shape = [1, 3, tile_size, tile_size]
    if (
        len(inputs) != 1
        or inputs[0].name != "image"
        or inputs[0].type != "tensor(float)"
        or list(inputs[0].shape) != expected_input_shape
    ):
        raise ValueError(
            "Exported ONNX input must be float32 image with shape "
            f"{expected_input_shape}."
        )

    expected_outputs = {
        "probability": [1, 1, tile_size, tile_size],
        "orientation": [1, 2, tile_size, tile_size],
        "uncertainty": [1, 1, tile_size, tile_size],
    }
    outputs = {output.name: output for output in session.get_outputs()}
    if set(outputs) != set(expected_outputs):
        raise ValueError(
            "Exported ONNX outputs do not exactly match probability, orientation, "
            "and uncertainty."
        )
    for output_name, expected_shape in expected_outputs.items():
        output = outputs[output_name]
        if output.type != "tensor(float)" or list(output.shape) != expected_shape:
            raise ValueError(
                f"Exported ONNX output {output_name!r} must be float32 with shape "
                f"{expected_shape}."
            )


def _compare_outputs(
    model: GeoTraceNet,
    session: Any,
    cases: Mapping[str, torch.Tensor],
    tile_size: int,
) -> list[dict[str, Any]]:
    output_names = ["probability", "orientation", "uncertainty"]
    results: list[dict[str, Any]] = []
    for case_name, example in cases.items():
        with torch.inference_mode():
            torch_raw = model(example)
        torch_outputs = _validate_tile_outputs(
            *(value.detach().cpu().numpy() for value in torch_raw), tile_size
        )
        runtime_raw = session.run(output_names, {"image": example.numpy()})
        runtime_outputs = _validate_tile_outputs(*runtime_raw, tile_size)

        maximum_errors: dict[str, float] = {}
        for output_name, expected, actual in zip(
            output_names, torch_outputs, runtime_outputs, strict=True
        ):
            maximum_error = float(np.max(np.abs(expected - actual)))
            maximum_errors[output_name] = maximum_error
            if not np.allclose(expected, actual, rtol=PARITY_RTOL, atol=PARITY_ATOL):
                raise ValueError(
                    f"ONNX parity failed for {case_name}/{output_name}: maximum "
                    f"absolute error {maximum_error:.8g} "
                    f"(atol={PARITY_ATOL}, rtol={PARITY_RTOL})."
                )
        results.append(
            {
                "name": case_name,
                "passed": True,
                "max_abs_error": maximum_errors,
            }
        )
    return results


def _manifest(
    *,
    name: str,
    model_version: str,
    architecture: str,
    model_sha256: str,
    license_id: str,
    tile_size: int,
    overlap: int,
    lower_percentile: float,
    upper_percentile: float,
    code_revision: str,
    checkpoint_sha256: str,
    dataset_id: str,
    split_id: str,
    limitations: list[str],
    width: int,
    parameter_count: int,
    onnx_version: str,
    runtime_version: str,
    exporter: str,
    parity_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "version": PACK_VERSION,
        "name": name,
        "model_version": model_version,
        "architecture": architecture,
        "model": MODEL_FILENAME,
        "model_sha256": model_sha256,
        "license": license_id,
        "tile_size": tile_size,
        "overlap": overlap,
        "contract": {
            "input": {
                "name": "image",
                "layout": "NCHW",
                "dtype": "float32",
                "color_space": "RGB",
                "shape": [1, 3, tile_size, tile_size],
            },
            "preprocessing": {
                "method": "per_image_channel_percentile",
                "lower_percentile": lower_percentile,
                "upper_percentile": upper_percentile,
                "invalid_fill": "normalized_valid_median",
            },
            "outputs": {
                "probability": {
                    "name": "probability",
                    "dtype": "float32",
                    "layout": "NCHW",
                    "shape": [1, 1, tile_size, tile_size],
                    "activation": "sigmoid",
                    "semantics": "fracture_centreline_probability",
                },
                "orientation": {
                    "name": "orientation",
                    "dtype": "float32",
                    "layout": "NCHW",
                    "shape": [1, 2, tile_size, tile_size],
                    "activation": "l2_normalized",
                    "encoding": "axial_cos2_sin2",
                },
                "uncertainty": {
                    "name": "uncertainty",
                    "dtype": "float32",
                    "layout": "NCHW",
                    "shape": [1, 1, tile_size, tile_size],
                    "activation": "sigmoid",
                    "semantics": "learned_error_proxy_uncalibrated",
                },
            },
        },
        "provenance": {
            "code_revision": code_revision,
            "training_checkpoint_sha256": checkpoint_sha256,
            "dataset_id": dataset_id,
            "split_id": split_id,
            "architecture_width": width,
            "parameter_count": parameter_count,
            "export": {
                "opset": ONNX_OPSET,
                "exporter": exporter,
                "torch_version": str(torch.__version__),
                "onnx_version": onnx_version,
                "onnxruntime_version": runtime_version,
                "validation_provider": "CPUExecutionProvider",
                "parity": {
                    "atol": PARITY_ATOL,
                    "rtol": PARITY_RTOL,
                    "random_seed": PARITY_RANDOM_SEED,
                    "cases": [dict(result) for result in parity_results],
                },
            },
        },
        "limitations": limitations,
    }
    manifest["pack_sha256"] = compute_pack_sha256(manifest, model_sha256)
    return manifest


def export_model_pack(
    checkpoint: str | Path,
    output_directory: str | Path,
    *,
    model_version: str,
    license_id: str,
    code_revision: str,
    dataset_id: str,
    split_id: str,
    limitations: Sequence[str],
    name: str = "GeoTraceNet",
    architecture: str = "GeoTraceNet",
    tile_size: int = 512,
    overlap: int = 64,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> Path:
    """Export, check, execute, and package a fixed-shape GeoTraceNet model."""

    source = Path(checkpoint).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = Path(output_directory).expanduser().resolve()
    if source in {
        (destination / MODEL_FILENAME).resolve(),
        (destination / MANIFEST_FILENAME).resolve(),
    }:
        raise ValueError("The checkpoint cannot also be a model-pack output file.")
    _validate_export_options(
        tile_size=tile_size,
        overlap=overlap,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )
    name = _required_text(name, "name")
    model_version = _required_text(model_version, "model_version")
    architecture = _required_text(architecture, "architecture")
    if architecture != GeoTraceNet.__name__:
        raise ValueError(
            f"This exporter creates {GeoTraceNet.__name__} models; "
            f"architecture cannot be declared as {architecture!r}."
        )
    license_id = _required_text(license_id, "license")
    code_revision = _required_text(code_revision, "code_revision")
    dataset_id = _required_text(dataset_id, "dataset_id")
    split_id = _required_text(split_id, "split_id")
    if isinstance(limitations, (str, bytes)) or not isinstance(limitations, Sequence):
        raise ValueError("limitations must be a non-empty sequence of strings.")
    normalized_limitations = [
        _required_text(limitation, "limitations item") for limitation in limitations
    ]
    if not normalized_limitations:
        raise ValueError("limitations must be a non-empty sequence of strings.")

    try:
        import onnx
        from onnx.external_data_helper import uses_external_data
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on optional packages
        raise RuntimeError(
            "ONNX export validation requires torch, ONNX, and ONNX Runtime; "
            "install domstudio[learned]."
        ) from exc

    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is required for export validation.")

    checkpoint_sha256 = _digest(source)
    model, width = _load_model(source)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parity_cases = _parity_inputs(tile_size)
    export_example = parity_cases["gradient"]
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".model-pack-export-", dir=destination) as tmp:
        temporary_root = Path(tmp)
        temporary_model = temporary_root / MODEL_FILENAME
        exporter = _export_onnx(model, export_example, temporary_model)

        graph = onnx.load_model(str(temporary_model), load_external_data=False)
        if any(uses_external_data(tensor) for tensor in _onnx_tensors(graph.graph)):
            raise ValueError("External ONNX tensor data is not allowed in a model pack.")
        onnx.checker.check_model(graph, full_check=True)

        session = ort.InferenceSession(
            str(temporary_model), providers=["CPUExecutionProvider"]
        )
        _verify_runtime_contract(session, tile_size)
        parity_results = _compare_outputs(model, session, parity_cases, tile_size)
        model_sha256 = _digest(temporary_model)
        manifest = _manifest(
            name=name,
            model_version=model_version,
            architecture=architecture,
            model_sha256=model_sha256,
            license_id=license_id,
            tile_size=tile_size,
            overlap=overlap,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            code_revision=code_revision,
            checkpoint_sha256=checkpoint_sha256,
            dataset_id=dataset_id,
            split_id=split_id,
            limitations=normalized_limitations,
            width=width,
            parameter_count=parameter_count,
            onnx_version=onnx.__version__,
            runtime_version=ort.__version__,
            exporter=exporter,
            parity_results=parity_results,
        )
        temporary_manifest = temporary_root / MANIFEST_FILENAME
        temporary_manifest.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        # Exercise the exact loader contract before publishing either file.
        ModelPackManifest.load(temporary_root)
        del session, graph
        temporary_model.replace(destination / MODEL_FILENAME)
        temporary_manifest.replace(destination / MANIFEST_FILENAME)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--name", default="GeoTraceNet")
    parser.add_argument("--architecture", default="GeoTraceNet")
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--license", dest="license_id", required=True)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--limitation", action="append", required=True, dest="limitations")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--lower-percentile", type=float, default=1.0)
    parser.add_argument("--upper-percentile", type=float, default=99.0)
    arguments = parser.parse_args()
    export_model_pack(
        arguments.checkpoint,
        arguments.output_directory,
        name=arguments.name,
        model_version=arguments.model_version,
        architecture=arguments.architecture,
        license_id=arguments.license_id,
        code_revision=arguments.code_revision,
        dataset_id=arguments.dataset_id,
        split_id=arguments.split_id,
        limitations=arguments.limitations,
        tile_size=arguments.tile_size,
        overlap=arguments.overlap,
        lower_percentile=arguments.lower_percentile,
        upper_percentile=arguments.upper_percentile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
