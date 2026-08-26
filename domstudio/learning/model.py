"""Compact multi-head network for fracture evidence, direction, and uncertainty.

The application does not silently download weights. A checkpoint must be
selected explicitly and its provenance can therefore be recorded in a session.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class ModelOutput(NamedTuple):
    probability: Tensor
    orientation: Tensor
    uncertainty: Tensor


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        return F.gelu(value + self.block(value))


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
        self.context = nn.Sequential(
            ResidualDepthwiseBlock(out_channels),
            ResidualDepthwiseBlock(out_channels),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.context(self.projection(value))


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            ResidualDepthwiseBlock(out_channels),
        )

    def forward(self, value: Tensor, skip: Tensor) -> Tensor:
        value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(torch.cat((value, skip), dim=1))


class GeoTraceNet(nn.Module):
    """Efficient encoder-decoder with geology-oriented prediction heads.

    Orientation is represented axially as ``(cos(2θ), sin(2θ))`` so a fracture
    tangent has no artificial forward/backward distinction.
    """

    def __init__(self, in_channels: int = 3, width: int = 32) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.GELU(),
            ResidualDepthwiseBlock(width),
        )
        self.enc1 = EncoderStage(width, width * 2)
        self.enc2 = EncoderStage(width * 2, width * 4)
        self.enc3 = EncoderStage(width * 4, width * 8)
        self.dec2 = DecoderStage(width * 8, width * 4, width * 4)
        self.dec1 = DecoderStage(width * 4, width * 2, width * 2)
        self.dec0 = DecoderStage(width * 2, width, width)
        self.probability_head = nn.Conv2d(width, 1, 1)
        self.orientation_head = nn.Conv2d(width, 2, 1)
        self.uncertainty_head = nn.Conv2d(width, 1, 1)

    def forward(self, image: Tensor) -> ModelOutput:
        s0 = self.stem(image)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        latent = self.enc3(s2)
        decoded = self.dec2(latent, s2)
        decoded = self.dec1(decoded, s1)
        decoded = self.dec0(decoded, s0)

        probability = torch.sigmoid(self.probability_head(decoded))
        orientation = F.normalize(self.orientation_head(decoded), dim=1, eps=1e-6)
        uncertainty = torch.sigmoid(self.uncertainty_head(decoded))
        return ModelOutput(probability, orientation, uncertainty)
