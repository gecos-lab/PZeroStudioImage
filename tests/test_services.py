from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from domstudio.application import _preview_evidence
from domstudio.core import RasterDocument
from domstudio.learning.model_pack import (
    PACK_VERSION,
    ModelPackManifest,
    compute_pack_sha256,
)
from domstudio.services import export_traces, load_raster


class PreviewPipelineTests(unittest.TestCase):
    def test_downsampled_preview_returns_native_aligned_evidence(self) -> None:
        yy, xx = np.mgrid[:96, :128]
        image = np.full((96, 128), 220, dtype=np.uint8)
        image[np.abs(yy - (0.3 * xx + 20.0)) < 2.0] = 20
        document = RasterDocument(image)

        evidence = _preview_evidence(document, maximum_side=64)

        self.assertEqual(evidence.shape, document.shape)
        self.assertEqual(evidence.probability.dtype, np.float32)
        self.assertEqual(evidence.metadata["profile"], "preview")
        self.assertLess(evidence.metadata["analysis_scale"], 1.0)
        self.assertTrue(np.all(np.isfinite(evidence.orientation)))


class RasterLoadingTests(unittest.TestCase):
    def test_geotiff_preserves_native_values_spatial_reference_and_mask(self) -> None:
        import rasterio
        from rasterio.transform import from_origin

        source_values = np.arange(30, dtype=np.uint16).reshape(5, 6)
        source_values[2, 4] = 65535
        transform = from_origin(410_000.0, 5_120_000.0, 0.02, 0.03)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "outcrop.tif"
            with rasterio.open(
                source,
                "w",
                driver="GTiff",
                width=6,
                height=5,
                count=1,
                dtype="uint16",
                crs="EPSG:32632",
                transform=transform,
                nodata=65535,
            ) as dataset:
                dataset.write(source_values, 1)

            document = load_raster(source)

        self.assertEqual(document.image.dtype, np.uint16)
        np.testing.assert_array_equal(document.image, source_values)
        self.assertEqual(document.shape, (5, 6))
        self.assertEqual(str(document.crs), "EPSG:32632")
        self.assertEqual(document.transform, transform)
        self.assertEqual(document.nodata, 65535)
        self.assertFalse(document.valid_mask()[2, 4])
        np.testing.assert_allclose(
            document.pixel_to_world((0.0, 0.0)),
            (410_000.01, 5_119_999.985),
        )


class ExportTests(unittest.TestCase):
    def test_csv_exports_affine_transformed_pixel_centres(self) -> None:
        document = RasterDocument(
            np.zeros((8, 8), dtype=np.uint8),
            transform=(2.0, 0.0, 100.0, 0.0, -3.0, 200.0),
            crs="EPSG:32632",
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = export_traces(
                Path(directory) / "network.csv",
                [((0.0, 0.0), (4.0, 2.0))],
                document,
            )
            with destination.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))

        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(float(rows[0]["x"]), 101.0)
        self.assertAlmostEqual(float(rows[0]["y"]), 198.5)
        self.assertAlmostEqual(float(rows[1]["x"]), 109.0)
        self.assertAlmostEqual(float(rows[1]["y"]), 192.5)
        self.assertAlmostEqual(float(rows[0]["length_px"]), 20**0.5)
        self.assertEqual(rows[0]["edge_id"], "edge-000000")
        self.assertEqual(rows[0]["from_node_id"], "node-000000")
        self.assertEqual(rows[0]["to_node_id"], "node-000001")
        self.assertEqual(int(rows[0]["from_degree"]), 1)
        self.assertAlmostEqual(float(rows[0]["orientation_degrees"]), 26.565051, places=5)
        self.assertEqual(rows[0]["crs"], "EPSG:32632")

    def test_geojson_contains_world_geometry_and_measurements(self) -> None:
        document = RasterDocument(
            np.zeros((8, 8), dtype=np.uint8),
            transform=(2.0, 0.0, 100.0, 0.0, -3.0, 200.0),
            crs="EPSG:32632",
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = export_traces(
                Path(directory) / "network.geojson",
                [((0.0, 0.0), (4.0, 2.0))],
                document,
            )
            payload = json.loads(destination.read_text(encoding="utf-8"))

        feature = payload["features"][0]
        self.assertEqual(feature["geometry"]["type"], "LineString")
        self.assertEqual(feature["geometry"]["coordinates"][0], [101.0, 198.5])
        self.assertAlmostEqual(feature["properties"]["length_px"], 20**0.5)
        self.assertEqual(feature["properties"]["edge_id"], "edge-000000")
        self.assertEqual(feature["properties"]["from_degree"], 1)
        # Orientation is analysed in source-pixel space, not distorted by a
        # non-square geospatial affine.
        self.assertAlmostEqual(feature["properties"]["orientation_deg"], 26.565051, places=5)


def _model_pack_payload(model_name: str, model_sha256: str) -> dict[str, object]:
    tile_size = 64
    payload: dict[str, object] = {
        "version": PACK_VERSION,
        "name": "test",
        "model_version": "test-1",
        "architecture": "GeoTraceNet",
        "model": model_name,
        "model_sha256": model_sha256,
        "license": "MIT",
        "tile_size": tile_size,
        "overlap": 8,
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
                "lower_percentile": 1.0,
                "upper_percentile": 99.0,
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
            "code_revision": "test-revision",
            "training_checkpoint_sha256": "1" * 64,
            "dataset_id": "test-dataset-v1",
            "split_id": "test-split-v1",
        },
        "limitations": ["Synthetic unit-test model; not for scientific interpretation."],
    }
    payload["pack_sha256"] = compute_pack_sha256(payload, model_sha256)
    return payload


class ModelPackManifestTests(unittest.TestCase):
    def test_manifest_verifies_checksum_without_loading_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.onnx"
            model.write_bytes(b"test-model")
            checksum = hashlib.sha256(model.read_bytes()).hexdigest()
            payload = _model_pack_payload(model.name, checksum)
            (root / "model.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )

            manifest = ModelPackManifest.load(root)

        self.assertEqual(manifest.name, "test")
        self.assertEqual(manifest.version, PACK_VERSION)
        self.assertEqual(manifest.model_version, "test-1")
        self.assertEqual(manifest.sha256, checksum)
        self.assertEqual(manifest.pack_sha256, payload["pack_sha256"])
        self.assertEqual(manifest.tile_size, 64)

    def test_manifest_rejects_path_escape_before_runtime_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.json").write_text(
                json.dumps(
                    {
                        "version": PACK_VERSION,
                        "model": "../outside.onnx",
                        "model_sha256": "0" * 64,
                        "pack_sha256": "0" * 64,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                ModelPackManifest.load(root)

    def test_manifest_checksum_binds_metadata_to_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.onnx"
            model.write_bytes(b"test-model")
            checksum = hashlib.sha256(model.read_bytes()).hexdigest()
            payload = _model_pack_payload(model.name, checksum)
            payload["name"] = "tampered-after-signing"
            (root / "model.json").write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Pack checksum mismatch"):
                ModelPackManifest.load(root)

    def test_manifest_requires_exact_uncertainty_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.onnx"
            model.write_bytes(b"test-model")
            checksum = hashlib.sha256(model.read_bytes()).hexdigest()
            payload = _model_pack_payload(model.name, checksum)
            contract = payload["contract"]
            assert isinstance(contract, dict)
            outputs = contract["outputs"]
            assert isinstance(outputs, dict)
            uncertainty = outputs["uncertainty"]
            assert isinstance(uncertainty, dict)
            uncertainty["semantics"] = "calibrated_probability"
            payload["pack_sha256"] = compute_pack_sha256(payload, checksum)
            (root / "model.json").write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "learned_error_proxy_uncalibrated"):
                ModelPackManifest.load(root)


if __name__ == "__main__":
    unittest.main()
