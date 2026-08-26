"""Focused integration coverage for the optional ONNX export toolchain."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


def _export_stack_available() -> bool:
    try:
        for package in ("torch", "onnx", "onnxruntime"):
            importlib.import_module(package)
    except (ImportError, OSError):
        return False
    return True


_EXPORT_STACK_AVAILABLE = _export_stack_available()


@unittest.skipUnless(
    _EXPORT_STACK_AVAILABLE,
    "torch, ONNX, and ONNX Runtime are required for the exporter integration test",
)
class ModelPackExportTests(unittest.TestCase):
    def test_export_emits_checked_v2_pack_with_exact_contract(self) -> None:
        import onnx
        from onnx.external_data_helper import uses_external_data
        import torch

        from domstudio.learning.export_onnx import export_model_pack
        from domstudio.learning.model import GeoTraceNet
        from domstudio.core import RasterDocument
        from domstudio.learning.model_pack import (
            ModelPackManifest,
            OnnxFractureDetector,
            _onnx_tensors,
            compute_pack_sha256,
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.pt"
            torch.save(GeoTraceNet(width=4).state_dict(), checkpoint)
            checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

            pack = export_model_pack(
                checkpoint,
                root / "pack",
                name="GeoTraceNet unit test",
                model_version="test-1",
                architecture="GeoTraceNet",
                license_id="MIT",
                code_revision="test-revision",
                dataset_id="synthetic-test-v1",
                split_id="fixed-test-split-v1",
                limitations=["Synthetic randomly initialized model; not for scientific use."],
                tile_size=64,
                overlap=8,
            )
            payload = json.loads((pack / "model.json").read_text(encoding="utf-8"))
            manifest = ModelPackManifest.load(pack)

            self.assertEqual(manifest.version, 2)
            self.assertEqual(
                set(path.name for path in pack.iterdir()),
                {"geotrace.onnx", "model.json"},
            )
            self.assertEqual(
                payload["provenance"]["training_checkpoint_sha256"], checkpoint_sha256
            )
            self.assertEqual(
                payload["pack_sha256"],
                compute_pack_sha256(payload, payload["model_sha256"]),
            )
            outputs = payload["contract"]["outputs"]
            self.assertEqual(outputs["probability"]["shape"], [1, 1, 64, 64])
            self.assertEqual(outputs["orientation"]["shape"], [1, 2, 64, 64])
            self.assertEqual(outputs["orientation"]["activation"], "l2_normalized")
            self.assertEqual(outputs["uncertainty"]["shape"], [1, 1, 64, 64])
            self.assertEqual(
                outputs["uncertainty"]["semantics"],
                "learned_error_proxy_uncalibrated",
            )
            parity = payload["provenance"]["export"]["parity"]
            self.assertEqual(parity["random_seed"], 1729)
            self.assertEqual(
                [case["name"] for case in parity["cases"]],
                ["zeros", "checkerboard", "gradient", "seeded_random"],
            )
            for case in parity["cases"]:
                self.assertTrue(case["passed"])
                self.assertEqual(
                    set(case["max_abs_error"]),
                    {"probability", "orientation", "uncertainty"},
                )

            graph = onnx.load_model(str(pack / "geotrace.onnx"), load_external_data=False)
            onnx.checker.check_model(graph, full_check=True)
            self.assertFalse(
                any(uses_external_data(tensor) for tensor in _onnx_tensors(graph.graph))
            )

            image = np.random.default_rng(99).integers(
                0, 256, size=(37, 51, 3), dtype=np.uint8
            )
            evidence = OnnxFractureDetector(pack, providers=["CPUExecutionProvider"]).predict(
                RasterDocument(image)
            )
            self.assertEqual(evidence.shape, (37, 51))
            self.assertTrue(np.all(np.isfinite(evidence.probability)))
            self.assertTrue(np.all(np.isfinite(evidence.orientation)))
            self.assertEqual(evidence.metadata["pack_sha256"], manifest.pack_sha256)
            self.assertNotIn(str(pack), json.dumps(evidence.metadata))


if __name__ == "__main__":
    unittest.main()
