"""Optional learned inference components.

ONNX model packs remain usable without PyTorch. Training-only symbols are
loaded on first access so the desktop application's base dependency set stays
small.
"""

from __future__ import annotations

from typing import Any

from .model_pack import ModelPackManifest, OnnxFractureDetector

__all__ = [
    "GeoTraceNet",
    "LearnedDetectorConfig",
    "ModelOutput",
    "ModelPackManifest",
    "OnnxFractureDetector",
    "TorchFractureDetector",
]


def __getattr__(name: str) -> Any:
    if name in {"GeoTraceNet", "ModelOutput"}:
        from .model import GeoTraceNet, ModelOutput

        return {"GeoTraceNet": GeoTraceNet, "ModelOutput": ModelOutput}[name]
    if name in {"LearnedDetectorConfig", "TorchFractureDetector"}:
        from .inference import LearnedDetectorConfig, TorchFractureDetector

        return {
            "LearnedDetectorConfig": LearnedDetectorConfig,
            "TorchFractureDetector": TorchFractureDetector,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
