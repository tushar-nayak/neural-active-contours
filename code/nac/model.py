from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TinyNACNet(nn.Module):
    """Lightweight encoder-decoder that predicts a soft active-contour mask."""

    def __init__(self, base_channels: int = 24, feature_channels: int = 8) -> None:
        super().__init__()
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.enc1 = ConvBlock(3, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)

        self.dec2 = ConvBlock(c3 + c2, c2)
        self.dec1 = ConvBlock(c2 + c1, c1)

        self.mask_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.feature_head = nn.Sequential(
            nn.Conv2d(c1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c1, feature_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, kernel_size=2))
        e3 = self.enc3(F.max_pool2d(e2, kernel_size=2))

        up2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))
        up1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))

        return {
            "mask_logits": self.mask_head(d1),
            "features": self.feature_head(d1),
        }

