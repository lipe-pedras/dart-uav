"""
Backbone block types.

Every block subclasses :class:`BackboneBlock`, which owns the attention slot
and applies it through a template method: the public ``forward()`` calls the
subclass's ``_forward_impl()`` and then the attention module. Subclasses
implement convolution logic only and never mention attention (UML design §5).

``fuse()`` is defined here as a no-op and overridden only by
:class:`RepConvBlock`, so :meth:`~dart.models.backbone.Backbone.fuse` can
iterate all blocks unconditionally.

Every block takes ``(B, in_ch, H, W)`` and returns ``(B, out_ch, H', W')``,
where ``H'`` and ``W'`` are reduced by ``stride``.

Shape symbols: ``B`` batch, ``H`` and ``W`` spatial.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import make_activation, make_attention
from .registry import register_block


class BackboneBlock(nn.Module):
    """
    Abstract backbone block: convolution logic plus an attention slot.

    Subclasses implement :meth:`_forward_impl` and nothing else; the attention
    module is applied by :meth:`forward` here, so no subclass mentions
    attention. This is the template-method arrangement of UML design §5.

    Parameters
    ----------
    in_ch : int
        Input channel count.
    out_ch : int
        Output channel count, already scaled by ``width_mult`` — blocks never
        apply the width multiplier themselves.
    stride : int, default: 1
        Spatial stride of the block.
    attention : {"none", "se", "coord"}, default: "none"
        Registered attention name; ``"none"`` gives the null object.
    act : {"relu", "relu6", "silu", "swish", "hardswish"}, default: "relu6"
        Registered activation name.

    Attributes
    ----------
    in_ch : int
        Input channel count, as passed in.
    out_ch : int
        Output channel count, as passed in.
    stride : int
        Spatial stride, as passed in.
    act_name : str
        Registered activation name, kept so a block can rebuild its activation.
    attention : torch.nn.Module
        The constructed attention module, applied after :meth:`_forward_impl`.

    See Also
    --------
    :func:`~dart.models.registry.build_block` : Builds a block from a config
        spec, and is the one place the width multiplier is applied.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        attention: str = "none",
        act: str = "relu6",
    ) -> None:
        """Record the block geometry and build its attention module."""
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.stride = stride
        self.act_name = act
        self.attention = make_attention(attention, out_ch)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the block's convolution logic, without attention.

        The one method a subclass must implement. Attention is applied by
        :meth:`forward` afterwards, so an implementation must not apply it.

        Parameters
        ----------
        x : torch.Tensor, shape (B, in_ch, H, W)
            Input feature map.

        Returns
        -------
        torch.Tensor, shape (B, out_ch, H', W')
            Convolved feature map, spatially reduced by ``stride``.

        Raises
        ------
        NotImplementedError
            Always; subclasses supply the convolution logic.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the block's convolution logic, then its attention module.

        Parameters
        ----------
        x : torch.Tensor, shape (B, in_ch, H, W)
            Input feature map.

        Returns
        -------
        torch.Tensor, shape (B, out_ch, H', W')
            Attention-weighted output, spatially reduced by ``stride``.
        """
        return self.attention(self._forward_impl(x))

    def fuse(self) -> None:
        """
        Collapse training-time structure into its inference-time equivalent.

        A no-op for every block whose training and inference graphs already
        coincide. Overridden by :class:`RepConvBlock`.
        """
        return None


@register_block("repconv")
class RepConvBlock(BackboneBlock):
    """
    Structural reparameterization block.

    Training: three parallel branches (3x3 conv+BN, 1x1 conv+BN, identity BN).
    Inference: all branches fused analytically into a single 3x3 conv, which is
    what :meth:`fuse` performs. The fusion is exact up to floating-point error;
    the equivalence is enforced by a CI test (FR-5.3) [1]_.

    The multi-branch form trains better than a plain 3x3 stack; the fused form
    costs exactly one convolution at inference. Both properties are wanted, and
    reparameterization is what allows having each where it matters.

    Parameters
    ----------
    in_ch : int
        Input channel count.
    out_ch : int
        Output channel count, already scaled by ``width_mult``.
    stride : int, default: 1
        Spatial stride of the block.
    attention : {"none", "se", "coord"}, default: "none"
        Registered attention name.
    act : {"relu", "relu6", "silu", "swish", "hardswish"}, default: "relu6"
        Registered activation name.

    Attributes
    ----------
    deployed : bool
        ``False`` while the three training branches are live, ``True`` once
        :meth:`fuse` has collapsed them.
    fused_conv : torch.nn.Conv2d or None
        The single equivalent convolution; ``None`` until :meth:`fuse` runs.

    Notes
    -----
    The identity branch exists only when the block is shape-preserving
    (``in_ch == out_ch`` and ``stride == 1``); otherwise the input cannot be
    added to the output and the branch is omitted.

    References
    ----------
    .. [1] X. Ding, X. Zhang, N. Ma, J. Han, G. Ding and J. Sun, "RepVGG:
           Making VGG-style ConvNets Great Again", CVPR 2021.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        attention: str = "none",
        act: str = "relu6",
    ) -> None:
        """Build the 3x3, 1x1 and (where shape-preserving) identity branches."""
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
        """
        Sum the parallel branches, or apply the fused convolution.

        Parameters
        ----------
        x : torch.Tensor, shape (B, in_ch, H, W)
            Input feature map.

        Returns
        -------
        torch.Tensor, shape (B, out_ch, H', W')
            Activated output. Numerically equivalent before and after
            :meth:`fuse`, up to floating-point error.
        """
        if self.deployed:
            return self.act(self.fused_conv(x))
        out = self.conv3x3(x) + self.conv1x1(x)
        if self.identity is not None:
            out = out + self.identity(x)
        return self.act(out)

    # ── Fusion ──────────────────────────────────────────────────────────

    @staticmethod
    def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d):
        """
        Fold a batch-norm layer into the convolution preceding it.

        Batch norm at inference is an affine map per channel, so it can be
        absorbed into the convolution's weights and bias: each output channel's
        kernel is scaled by ``gamma / sqrt(var + eps)``, and the bias becomes
        ``beta - mean * gamma / sqrt(var + eps)``.

        Parameters
        ----------
        conv : torch.nn.Conv2d
            The convolution, which must have no bias of its own.
        bn : torch.nn.BatchNorm2d
            The batch-norm layer immediately after it, in eval statistics.

        Returns
        -------
        tuple of torch.Tensor
            The folded ``(kernel, bias)``, with kernel shaped as ``conv.weight``
            and bias shaped ``(out_channels,)``.
        """
        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).view(-1, 1, 1, 1)
        return conv.weight * t, bn.bias - bn.running_mean * bn.weight / std

    def _identity_as_conv(self):
        """
        Express the identity branch as an equivalent 3x3 convolution.

        The identity branch is a bare batch norm on the input. To be summed
        with the other branches it must first become a convolution: a 3x3
        kernel that is zero everywhere except a 1 at the centre tap of each
        channel's own input channel, which convolves to the identity. That
        kernel is then folded with the branch's batch norm exactly as in
        :meth:`_fuse_conv_bn`.

        Returns
        -------
        tuple of torch.Tensor
            The ``(kernel, bias)`` of the equivalent convolution, with kernel of
            shape ``(in_ch, in_ch, 3, 3)`` on the batch norm's device and dtype.
        """
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
        """
        Sum all branches into the parameters of one 3x3 convolution.

        Convolution is linear in its kernel, so parallel branches summed at the
        output are equivalent to a single convolution whose kernel is the sum of
        theirs. The 1x1 kernel is zero-padded to 3x3 to be added, and the
        identity branch is converted by :meth:`_identity_as_conv` when present.

        Returns
        -------
        tuple of torch.Tensor
            The combined ``(kernel, bias)``, of shapes ``(out_ch, in_ch, 3, 3)``
            and ``(out_ch,)``.
        """
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
        """
        Collapse the three branches into one 3x3 conv. Idempotent.

        Replaces the branches with a single equivalent convolution and deletes
        them, so the fused block holds one convolution's worth of parameters.
        Calling this a second time does nothing.

        Notes
        -----
        Destructive and one-way: the training branches are deleted, so a fused
        block cannot be trained further.
        :class:`~dart.metrics.efficiency.EfficiencyProfiler` therefore always
        fuses a deep copy, never the caller's model.
        """
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
    """
    Depthwise-separable convolution, MobileNetV1 style.

    A depthwise spatial convolution followed by a pointwise projection, each
    with batch norm and activation. Promoted to a first-class block type by the
    framework design: in the research codebase it existed only inside the
    inverted residual, which made the cheaper non-expanding alternative
    unavailable in other positions.

    Parameters
    ----------
    in_ch : int
        Input channel count.
    out_ch : int
        Output channel count, already scaled by ``width_mult``.
    stride : int, default: 1
        Spatial stride, applied by the depthwise convolution.
    attention : {"none", "se", "coord"}, default: "none"
        Registered attention name.
    act : {"relu", "relu6", "silu", "swish", "hardswish"}, default: "relu6"
        Registered activation name.
    dw_kernel : int, default: 3
        Depthwise kernel size (3 or 5). Padding is derived from it, so the
        spatial size depends only on ``stride``.

    See Also
    --------
    InvertedResidualBlock : The expanding variant, with a residual connection.

    References
    ----------
    .. [1] A. G. Howard et al., "MobileNets: Efficient Convolutional Neural
           Networks for Mobile Vision Applications", arXiv:1704.04861, 2017.
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
        """Build the depthwise and pointwise convolution stack."""
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
        """
        Apply the depthwise then pointwise convolutions.

        Parameters
        ----------
        x : torch.Tensor, shape (B, in_ch, H, W)
            Input feature map.

        Returns
        -------
        torch.Tensor, shape (B, out_ch, H', W')
            Output feature map, spatially reduced by ``stride``.
        """
        return self.conv(x)


@register_block("inverted_res")
class InvertedResidualBlock(BackboneBlock):
    """
    MobileNetV2 inverted residual with a linear bottleneck.

    Expands the channel count pointwise, filters spatially with a depthwise
    convolution, then projects back down with *no* activation — the linear
    bottleneck, which is what stops the projection from discarding information
    that the low-dimensional output cannot afford to lose [1]_.

    Parameters
    ----------
    in_ch : int
        Input channel count.
    out_ch : int
        Output channel count, already scaled by ``width_mult``.
    stride : int, default: 1
        Spatial stride, applied by the depthwise convolution.
    attention : {"none", "se", "coord"}, default: "none"
        Registered attention name.
    act : {"relu", "relu6", "silu", "swish", "hardswish"}, default: "relu6"
        Registered activation name.
    expand_ratio : int, default: 4
        Pointwise expansion factor of the hidden dimension.
    dw_kernel : int, default: 3
        Depthwise kernel size (3 or 5).

    Attributes
    ----------
    expand_ratio : int
        The expansion factor, as passed in.
    use_skip : bool
        Whether the residual connection is active. ``True`` only when the block
        is shape-preserving (``stride == 1`` and ``in_ch == out_ch``).

    See Also
    --------
    DepthwiseSeparableBlock : The non-expanding, residual-free alternative.

    References
    ----------
    .. [1] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov and L.-C. Chen,
           "MobileNetV2: Inverted Residuals and Linear Bottlenecks", CVPR 2018.
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
        """Build the expand, depthwise and linear-projection stack."""
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
        """
        Apply the expand-depthwise-project stack, without the residual.

        Parameters
        ----------
        x : torch.Tensor, shape (B, in_ch, H, W)
            Input feature map.

        Returns
        -------
        torch.Tensor, shape (B, out_ch, H', W')
            Projected output, with no activation applied to the projection.
        """
        return self.conv(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the block, then attention, then the residual connection.

        Overrides :meth:`BackboneBlock.forward` to add the skip connection
        *after* attention rather than before.

        Parameters
        ----------
        x : torch.Tensor, shape (B, in_ch, H, W)
            Input feature map.

        Returns
        -------
        torch.Tensor, shape (B, out_ch, H', W')
            Attention-weighted output, with ``x`` added when ``use_skip``.

        Notes
        -----
        The ordering — residual after attention — matches the research
        codebase's ``InvertedResidualSE``, so the paper's metrics stay
        comparable. Reversing it would change trained behaviour.
        """
        # The residual is added *after* attention, matching the research
        # codebase's InvertedResidualSE, so metrics stay comparable.
        out = self.attention(self._forward_impl(x))
        if self.use_skip:
            out = out + x
        return out


@register_block("ghost")
class GhostBlock(BackboneBlock):
    """
    Ghost module: half the maps come from a cheap convolution.

    Produces one part of the output channels with an ordinary pointwise
    convolution, then generates the rest by applying a cheap depthwise
    convolution to *those* maps and concatenating. The premise is that feature
    maps in a trained network are largely redundant, so the redundant half can
    be manufactured more cheaply than it can be learned [1]_.

    Parameters
    ----------
    in_ch : int
        Input channel count.
    out_ch : int
        Output channel count, already scaled by ``width_mult``.
    stride : int, default: 1
        Spatial stride, applied by the primary convolution.
    attention : {"none", "se", "coord"}, default: "none"
        Registered attention name.
    act : {"relu", "relu6", "silu", "swish", "hardswish"}, default: "relu6"
        Registered activation name.
    ratio : int, default: 2
        Fraction divisor for the "primary" half of the output channels. ``2``
        (the default and the paper's setting) splits the output evenly between
        the pointwise and the cheap depthwise branch.

    See Also
    --------
    DepthwiseSeparableBlock : The straightforward cheap-convolution block.

    References
    ----------
    .. [1] K. Han, Y. Wang, Q. Tian, J. Guo, C. Xu and C. Xu, "GhostNet: More
           Features from Cheap Operations", CVPR 2020.
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
        """Build the primary pointwise branch and the cheap depthwise branch."""
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
        """
        Concatenate the primary maps with the cheaply derived ones.

        Parameters
        ----------
        x : torch.Tensor, shape (B, in_ch, H, W)
            Input feature map.

        Returns
        -------
        torch.Tensor, shape (B, out_ch, H', W')
            The primary maps followed by the cheap maps along the channel axis,
            spatially reduced by ``stride``.
        """
        primary = self.primary(x)
        return torch.cat([primary, self.cheap(primary)], dim=1)
