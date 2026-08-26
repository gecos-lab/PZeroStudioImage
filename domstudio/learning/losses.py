"""Multi-head losses for thin fracture evidence and axial orientation."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from .model import ModelOutput


def soft_dice(probability: Tensor, target: Tensor, epsilon: float = 1e-6) -> Tensor:
    dims = tuple(range(1, probability.ndim))
    intersection = (probability * target).sum(dim=dims)
    denominator = probability.sum(dim=dims) + target.sum(dim=dims)
    return 1.0 - ((2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def orientation_loss(prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    """Axial cosine loss restricted to annotated fracture pixels."""
    similarity = (prediction * target).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
    weighted = (1.0 - similarity) * mask
    return weighted.sum() / mask.sum().clamp_min(1.0)


def geotrace_loss(
    output: ModelOutput,
    target_mask: Tensor,
    target_orientation: Tensor,
    *,
    orientation_weight: float = 0.25,
    uncertainty_weight: float = 0.05,
) -> Tensor:
    """Combine evidence, axial-orientation, and learned error-proxy losses.

    The third head estimates a bounded proxy for per-pixel prediction error.  It
    is deliberately not described as calibrated uncertainty: calibration must
    be measured, and if necessary fitted, on a held-out validation set.
    """
    bce = F.binary_cross_entropy(output.probability, target_mask, reduction="none")
    # Detaching avoids teaching the segmentation head to manufacture ambiguity.
    error_proxy_target = bce.detach().clamp(0.0, 1.0)
    error_proxy = F.l1_loss(output.uncertainty, error_proxy_target)
    segmentation = bce.mean() + soft_dice(output.probability, target_mask)
    direction = orientation_loss(output.orientation, target_orientation, target_mask)
    return segmentation + orientation_weight * direction + uncertainty_weight * error_proxy
