"""Feature-fusion necks.

``identity`` is a null object rather than ``None``, so the detector's forward
pass has no branch on neck presence (UML design §5.1).
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import COMPONENTS, register_component


@register_component("neck", "identity")
class IdentityNeck(nn.Module):
    """Pass the deepest feature map through untouched.

    Args:
        in_channels: Channel counts of the tapped feature maps, in tap order.
    """

    def __init__(self, in_channels: Sequence[int], **_: object) -> None:
        super().__init__()
        self.out_channels = in_channels[-1]

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        return features[-1]


@register_component("neck", "fpn_lite")
class FPNLiteNeck(nn.Module):
    """Lightweight FPN fusing an early, high-resolution tap with the deepest one.

    The deepest map is reduced, upsampled to the early tap's resolution, added
    to the reduced early map, merged by a 3x3 conv, and pooled to a fixed grid
    for the head. Which taps are used is decided by
    :meth:`~dart.models.backbone.Backbone.tap_indices`, not hardcoded here, so
    a six-block backbone can feed the same neck from positions 3 and 6.

    Args:
        in_channels: Channel counts of the tapped feature maps, in tap order.
            The first and last taps are fused.
        out_channels: Channels of the fused map; defaults to the deepest tap's.
        pool_size: Spatial size the fused map is pooled to before the head.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int | None = None,
        pool_size: int = 6,
    ) -> None:
        super().__init__()
        if len(in_channels) < 2:
            raise ValueError(
                "fpn_lite needs at least two tapped feature maps; set "
                "'backbone.tap_indices' to expose an early block output "
                "(e.g. [1, -1])"
            )
        early_ch, deep_ch = in_channels[0], in_channels[-1]
        out_ch = out_channels or deep_ch
        self.out_channels = out_ch

        self.lateral = nn.Sequential(
            nn.Conv2d(deep_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(early_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.merge = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.downsample = nn.AdaptiveAvgPool2d(pool_size)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        early, deep = features[0], features[-1]
        lateral = F.interpolate(self.lateral(deep), size=early.shape[2:], mode="nearest")
        merged = self.merge(lateral + self.reduce(early))
        return self.downsample(merged)


def build_neck(name: str, in_channels: Sequence[int], **kwargs) -> nn.Module:
    """Build a neck by registered name."""
    return COMPONENTS.build("neck", name, in_channels=list(in_channels), **kwargs)
