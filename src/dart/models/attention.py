"""Attention modules and the activation factory.

Attention is a property of *every* backbone block, not of one block type
(UML design §5). ``none`` resolves to :class:`IdentityAttention`, a null object,
so ``BackboneBlock.forward`` has no branch and no ``Optional`` to guard.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .registry import COMPONENTS, register_component


def make_activation(name: str) -> nn.Module:
    """Build an activation module by name (``relu6``, ``silu``/``swish``, ...)."""
    return COMPONENTS.build("activation", name)


class _InplaceActivation:
    """Adapter making an ``nn`` activation constructible with no arguments."""

    def __init__(self, module_cls: type) -> None:
        self._module_cls = module_cls

    def __call__(self) -> nn.Module:
        return self._module_cls(inplace=True)


COMPONENTS.register("activation", "relu6", _InplaceActivation(nn.ReLU6))
COMPONENTS.register("activation", "relu", _InplaceActivation(nn.ReLU))
# The paper writes "SiLU"; the research codebase spelled the same function
# "swish". Both names resolve to the same module so old configs keep working.
COMPONENTS.register("activation", "silu", _InplaceActivation(nn.SiLU))
COMPONENTS.register("activation", "swish", _InplaceActivation(nn.SiLU))
COMPONENTS.register("activation", "hardswish", _InplaceActivation(nn.Hardswish))


@register_component("attention", "none")
class IdentityAttention(nn.Module):
    """Null-object attention: returns its input unchanged."""

    def __init__(self, channels: int = 0, **_: object) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


@register_component("attention", "se")
class SEAttention(nn.Module):
    """Squeeze-and-Excitation (Hu et al., 2018): per-channel importance weights.

    Args:
        channels: Number of input (and output) channels.
        ratio: Channel reduction ratio of the bottleneck; floored at 4 channels.
    """

    def __init__(self, channels: int, ratio: int = 4) -> None:
        super().__init__()
        reduced = max(4, channels // ratio)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


@register_component("attention", "coord")
class CoordAttention(nn.Module):
    """Coordinate Attention (Hou et al., 2021).

    Extends SE by pooling along the H and W axes separately, so the learned
    weights carry directional information. This is what improves cx/cy
    regression in the paper's DART-X and DART-XD variants.

    Args:
        channels: Number of input (and output) channels.
        ratio: Channel reduction ratio of the shared bottleneck.
    """

    def __init__(self, channels: int, ratio: int = 4) -> None:
        super().__init__()
        reduced = max(4, channels // ratio)
        self.conv1 = nn.Conv2d(channels, reduced, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(reduced, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(reduced, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        x_h = x.mean(dim=3, keepdim=True)  # (B, C, H, 1)
        x_w = x.mean(dim=2, keepdim=True)  # (B, C, 1, W)
        # Concatenate along H so a single conv is shared by both directions.
        y = torch.cat([x_h, x_w.transpose(2, 3)], dim=2)  # (B, C, H+W, 1)
        y = self.act(self.bn1(self.conv1(y)))
        x_h2, x_w2 = y.split([height, width], dim=2)
        x_w2 = x_w2.transpose(2, 3)
        a_h = torch.sigmoid(self.conv_h(x_h2))
        a_w = torch.sigmoid(self.conv_w(x_w2))
        return x * a_h * a_w


def make_attention(name: str, channels: int) -> nn.Module:
    """Build an attention module by registered name."""
    return COMPONENTS.build("attention", name, channels=channels)
