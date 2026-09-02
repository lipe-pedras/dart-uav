"""
Attention modules and the activation factory.

Attention is a property of *every* backbone block, not of one block type
(UML design §5). ``none`` resolves to :class:`IdentityAttention`, a null object,
so ``BackboneBlock.forward`` has no branch and no ``Optional`` to guard.

Every attention module is channel-preserving and spatially transparent: it takes
``(B, C, H, W)`` and returns the same shape, so it can be dropped into any
position without changing the block's contract.

Shape symbols: ``B`` batch, ``C`` channels, ``H`` and ``W`` spatial.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .registry import COMPONENTS, register_component


def make_activation(name: str) -> nn.Module:
    """
    Build an activation module by name.

    Parameters
    ----------
    name : {"relu", "relu6", "silu", "swish", "hardswish"}
        Registered activation name. ``silu`` and ``swish`` are two spellings of
        the same function, both kept so the paper's and the research codebase's
        configs agree.

    Returns
    -------
    torch.nn.Module
        A freshly constructed activation, in-place where the underlying module
        supports it.

    Raises
    ------
    ~dart.errors.RegistryError
        If ``name`` is not a registered activation.
    """
    return COMPONENTS.build("activation", name)


class _InplaceActivation:
    """
    Adapter making an ``nn`` activation constructible with no arguments.

    The registry builds every component by calling it with keyword arguments
    only. Activations need ``inplace=True`` supplied at construction, which a
    bare class reference cannot carry, so each is registered as an instance of
    this adapter instead.

    Parameters
    ----------
    module_cls : type
        The :mod:`torch.nn` activation class to construct, such as
        :class:`torch.nn.ReLU6`.
    """

    def __init__(self, module_cls: type) -> None:
        """Record the activation class to construct on call."""
        self._module_cls = module_cls

    def __call__(self) -> nn.Module:
        """
        Construct the wrapped activation in in-place mode.

        Returns
        -------
        torch.nn.Module
            A new instance of the wrapped class, with ``inplace=True``.
        """
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
    """
    Null-object attention: returns its input unchanged.

    Exists so that a block configured without attention still has an attention
    module to call, which is what keeps
    :meth:`~dart.models.blocks.BackboneBlock.forward` free of a branch and of an
    ``Optional`` to guard. Holds no parameters, so it costs nothing at inference.

    Parameters
    ----------
    channels : int, default: 0
        Accepted and ignored; present so every attention module shares one
        construction signature.
    **_ : object
        Accepted and ignored, so options meant for another attention module do
        not prevent this one from building.

    See Also
    --------
    SEAttention : Channel-weighting attention.
    CoordAttention : Direction-aware attention.
    """

    def __init__(self, channels: int = 0, **_: object) -> None:
        """Build a parameterless pass-through module."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the input untouched.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, H, W)
            Feature map.

        Returns
        -------
        torch.Tensor, shape (B, C, H, W)
            The same tensor object, not a copy.
        """
        return x


@register_component("attention", "se")
class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation attention: per-channel importance weights.

    Squeezes each channel to a single number by global average pooling, learns a
    weight per channel through a bottleneck, and rescales the input by those
    weights. Spatial information is discarded in the squeeze, which is what
    :class:`CoordAttention` addresses [1]_.

    Parameters
    ----------
    channels : int
        Number of input (and output) channels.
    ratio : int, default: 4
        Channel reduction ratio of the bottleneck. The reduced width is floored
        at 4 channels, so a narrow layer cannot collapse the bottleneck to
        nothing.

    See Also
    --------
    CoordAttention : Pools the two spatial axes separately, keeping direction.
    IdentityAttention : The null object used when attention is disabled.

    References
    ----------
    .. [1] J. Hu, L. Shen and G. Sun, "Squeeze-and-Excitation Networks",
           CVPR 2018.
    """

    def __init__(self, channels: int, ratio: int = 4) -> None:
        """Build the pooling, bottleneck and gating stack."""
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
        """
        Rescale each channel by its learned importance weight.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, H, W)
            Feature map, with ``C`` equal to the ``channels`` this module was
            built with.

        Returns
        -------
        torch.Tensor, shape (B, C, H, W)
            The input scaled channel-wise by weights in ``[0, 1]``.
        """
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


@register_component("attention", "coord")
class CoordAttention(nn.Module):
    """
    Coordinate Attention: direction-aware channel weighting.

    Extends SE by pooling along the H and W axes separately, so the learned
    weights carry directional information. This is what improves cx/cy
    regression in the paper's DART-X and DART-XD variants [1]_.

    Parameters
    ----------
    channels : int
        Number of input (and output) channels.
    ratio : int, default: 4
        Channel reduction ratio of the shared bottleneck. The reduced width is
        floored at 4 channels.

    See Also
    --------
    SEAttention : The cheaper, direction-blind predecessor.

    Notes
    -----
    The two pooled axes are concatenated along H and passed through a *single*
    shared convolution, then split apart again. One convolution rather than two
    is what keeps the parameter cost close to SE's.

    References
    ----------
    .. [1] Q. Hou, D. Zhou and J. Feng, "Coordinate Attention for Efficient
           Mobile Network Design", CVPR 2021.
    """

    def __init__(self, channels: int, ratio: int = 4) -> None:
        """Build the shared bottleneck and the two per-axis gating convs."""
        super().__init__()
        reduced = max(4, channels // ratio)
        self.conv1 = nn.Conv2d(channels, reduced, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(reduced)
        self.act = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(reduced, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(reduced, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rescale the input by separate height-wise and width-wise weights.

        Parameters
        ----------
        x : torch.Tensor, shape (B, C, H, W)
            Feature map, with ``C`` equal to the ``channels`` this module was
            built with.

        Returns
        -------
        torch.Tensor, shape (B, C, H, W)
            The input scaled by the product of the two directional attention
            maps, each in ``[0, 1]``.
        """
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
    """
    Build an attention module by registered name.

    Parameters
    ----------
    name : {"none", "se", "coord"}
        Registered attention name. ``"none"`` yields
        :class:`IdentityAttention`. Any name added through
        :func:`~dart.models.registry.register_component` also works.
    channels : int
        Channel count the module must accept and return.

    Returns
    -------
    torch.nn.Module
        The constructed attention module, channel-preserving and spatially
        transparent.

    Raises
    ------
    ~dart.errors.RegistryError
        If ``name`` is not a registered attention module.
    """
    return COMPONENTS.build("attention", name, channels=channels)
