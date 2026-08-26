"""Optional tests for the compact learned evidence model."""

from __future__ import annotations

import importlib.util
import unittest


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch learning extra is not installed")
class GeoTraceNetTests(unittest.TestCase):
    def test_torch_inference_config_requires_an_explicit_checkpoint_digest(self) -> None:
        from pathlib import Path

        from domstudio.learning.inference import LearnedDetectorConfig

        config = LearnedDetectorConfig(
            checkpoint=Path("checkpoint.pt"),
            expected_sha256="0" * 64,
            tile_size=64,
            overlap=8,
        )
        self.assertEqual(config.expected_sha256, "0" * 64)
        with self.assertRaises(ValueError):
            LearnedDetectorConfig(
                checkpoint=Path("checkpoint.pt"),
                expected_sha256="not-a-digest",
            )

    def test_three_heads_have_aligned_shapes_and_finite_loss(self) -> None:
        import torch

        from domstudio.learning.losses import geotrace_loss
        from domstudio.learning.model import GeoTraceNet

        torch.manual_seed(7)
        model = GeoTraceNet(width=8)
        image = torch.randn(2, 3, 64, 80)
        output = model(image)
        self.assertEqual(tuple(output.probability.shape), (2, 1, 64, 80))
        self.assertEqual(tuple(output.orientation.shape), (2, 2, 64, 80))
        self.assertEqual(tuple(output.uncertainty.shape), (2, 1, 64, 80))
        self.assertTrue(torch.all((0.0 <= output.probability) & (output.probability <= 1.0)))
        self.assertTrue(torch.all((0.0 <= output.uncertainty) & (output.uncertainty <= 1.0)))

        target_mask = torch.zeros_like(output.probability)
        target_mask[:, :, 20:44, 18:62] = 1.0
        target_orientation = torch.zeros_like(output.orientation)
        target_orientation[:, 0] = 1.0
        loss = geotrace_loss(output, target_mask, target_orientation)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
