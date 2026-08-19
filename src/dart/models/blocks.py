"""Backbone block types.

Every block subclasses :class:`BackboneBlock`, which owns the attention slot
and applies it through a template method: the public ``forward()`` calls the
subclass's ``_forward_impl()`` and then the attention module. Subclasses
implement convolution logic only and never mention attention (UML design §5).

``fuse()`` is defined here as a no-op and overridden only by
:class:`RepConvBlock`, so :meth:`~dart.models.backbone.Backbone.fuse` can
iterate all blocks unconditionally.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import make_activation, make_attention
from .registry import register_block


class BackboneBlock(nn.Module):
    """Abstract backbone block: convolution logic plus an attention slot.

    Args:
        in_ch: Input channel count.
        out_ch: Output channel count (already scaled by ``width_mult``).
        stride: Spatial stride of the block.
        attention: Registered attention name; ``"none"`` gives the null object.
        act: Registered activation name.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        attention: str = "none",
        act: str = "relu6",
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.act_name = act
        self.attention = make_attention(attention, out_ch)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention(self._forward_impl(x))

    def fuse(self) -> None:
        """Collapse training-time structure into its inference-time equivalent.

        A no-op for every block whose training and inference graphs already
        coincide. Overridden by :class:`RepConvBlock`.
        """
        return None


@register_block("repconv")
class RepConvBlock(BackboneBlock):
    """Structural reparameterization block (RepVGG, Ding et al., 2021).

    Training: three parallel branches (3x3 conv+BN, 1x1 conv+BN, identity BN).
    Inference: all branches fused analytically into a single 3x3 conv, which is
    what :meth:`fuse` performs. The fusion is exact up to floating-point error;
    the equivalence is enforced by a CI test (FR-5.3).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        attention: str = "none",
        act: str = "relu6",
    ) -> None:
        super().__init__(in_ch, out_ch, stride, attention, act)
        self.deployed = False
        self.fused_conv: nn.Conv2d | None = None

        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # The identity branch only exists when it is shape-preserving.
        self.identity = nn.BatchNorm2d(in_ch) if (in_ch == out_ch and stride == 1) else None
        self.act = make_activation(act)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        if self.deployed:
            return self.act(self.fused_conv(x))
        out = self.conv3x3(x) + self.conv1x1(x)
        if self.identity is not None:
            out = out + self.identity(x)
        return self.act(out)

    # ── Fusion ──────────────────────────────────────────────────────────

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).view(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _identity_as_conv(self):
        bn = self.identity
        channels = self.in_ch
        kernel = torch.zeros(
            channels, channels, 3, 3, device=bn.weight.device, dtype=bn.weight.dtype
        )
        for i in range(channels):
            kernel[i, i, 1, 1] = 1.0
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).view(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def _fused_kernel_bias(self):
        k3, b3 = self._fuse_conv_bn(self.conv3x3[0], self.conv3x3[1])
        k1, b1 = self._fuse_conv_bn(self.conv1x1[0], self.conv1x1[1])
        kernel = k3 + F.pad(k1, [1, 1, 1, 1])
        bias = b3 + b1
        if self.identity is not None:
            ki, bi = self._identity_as_conv()
            kernel = kernel + ki
            bias = bias + bi
        return kernel, bias

    @torch.no_grad()
    def fuse(self) -> None:
        """Collapse the three branches into one 3x3 conv. Idempotent."""
        if self.deployed:
            return
        kernel, bias = self._fused_kernel_bias()
        fused = nn.Conv2d(self.in_ch, self.out_ch, 3, stride=self.stride, padding=1, bias=True)
        fused.weight.copy_(kernel)
        fused.bias.copy_(bias)
        fused = fused.to(kernel.device)
        for attr in ("conv3x3", "conv1x1", "identity"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.fused_conv = fused
        self.deployed = True


@register_block("dw_sep")
class DepthwiseSeparableBlock(BackboneBlock):
    """Depthwise-separable convolution (MobileNetV1 style).

    Promoted to a first-class block type by the framework design: in the
    research codebase it existed only inside the inverted residual, which made
    the cheaper non-expanding alternative unavailable in other positions.

    Args:
        dw_kernel: Depthwise kernel size (3 or 5).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        attention: str = "none",
        act: str = "relu6",
        dw_kernel: int = 3,
    ) -> None:
        super().__init__(in_ch, out_ch, stride, attention, act)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_ch,
                in_ch,
                dw_kernel,
                stride=stride,
                padding=dw_kernel // 2,
                groups=in_ch,
                bias=False,
            ),
            nn.BatchNorm2d(in_ch),
            make_activation(act),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            make_activation(act),
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


@register_block("inverted_res")
class InvertedResidualBlock(BackboneBlock):
    """MobileNetV2 inverted residual with a linear bottleneck.

    Args:
        expand_ratio: Pointwise expansion factor of the hidden dimension.
        dw_kernel: Depthwise kernel size (3 or 5).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        attention: str = "none",
        act: str = "relu6",
        expand_ratio: int = 4,
        dw_kernel: int = 3,
    ) -> None:
        super().__init__(in_ch, out_ch, stride, attention, act)
        self.expand_ratio = expand_ratio
        hidden = in_ch * expand_ratio
        self.use_skip = stride == 1 and in_ch == out_ch
        self.conv = nn.Sequential(
            # 1. Pointwise expand
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            make_activation(act),
            # 2. Depthwise spatial filtering
            nn.Conv2d(
                hidden,
                hidden,
                dw_kernel,
                stride=stride,
                padding=dw_kernel // 2,
                groups=hidden,
                bias=False,
            ),
            nn.BatchNorm2d(hidden),
            make_activation(act),
            # 3. Pointwise project (linear bottleneck — no activation)
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The residual is added *after* attention, matching the research
        # codebase's InvertedResidualSE, so metrics stay comparable.
        out = self.attention(self._forward_impl(x))
        if self.use_skip:
            out = out + x
        return out


@register_block("ghost")
class GhostBlock(BackboneBlock):
    """Ghost module (Han et al., 2020): half the maps come from a cheap conv.

    Args:
        ratio: Fraction divisor for the "primary" half of the output channels.
            ``2`` (the default and the paper's setting) splits the output
            evenly between the pointwise and the cheap depthwise branch.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        attention: str = "none",
        act: str = "relu6",
        ratio: int = 2,
    ) -> None:
        super().__init__(in_ch, out_ch, stride, attention, act)
        init_ch = math.ceil(out_ch / ratio)
        cheap_ch = out_ch - init_ch
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(init_ch),
            make_activation(act),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_ch, cheap_ch, 3, stride=1, padding=1, groups=init_ch, bias=False),
            nn.BatchNorm2d(cheap_ch),
            make_activation(act),
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        primary = self.primary(x)
        return torch.cat([primary, self.cheap(primary)], dim=1)
