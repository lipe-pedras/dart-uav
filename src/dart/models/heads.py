"""Detection heads.

**Output contract** (resolution of UML open question 1). Every head returns a
pair of tensors with an explicit target-count dimension::

    conf  : (B, N, 1)  objectness in [0, 1]
    boxes : (B, N, 4)  [cx, cy, w, h] in [0, 1], relative to the input image

For the single-target task ``N == 1``. Multi-target support therefore becomes a
head that emits ``N > 1`` plus an assignment policy, not a change to the
signature, the loss interface, or the metric interface (NFR-8).
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn

from .attention import make_activation
from .registry import COMPONENTS, register_component


class SPPLite(nn.Module):
    """Spatial Pyramid Pooling — Lite.

    Pools at 1x1, 2x2 and 3x3 and concatenates, preserving the multi-scale
    spatial context that plain global average pooling discards, then projects
    back to ``out_dim``.

    Args:
        in_ch: Input channel count.
        out_dim: Width of the projected feature vector.
    """

    CELLS = 1 + 4 + 9

    def __init__(self, in_ch: int, out_dim: int) -> None:
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.proj = nn.Linear(in_ch * self.CELLS, out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat(
            [self.pool1(x).flatten(1), self.pool2(x).flatten(1), self.pool3(x).flatten(1)],
            dim=1,
        )
        return self.act(self.proj(pooled))


class Head(nn.Module):
    """Base class fixing the head output contract.

    Args:
        in_ch: Channels of the feature map produced by the neck.
        num_targets: Number of boxes emitted per image (``N`` above).
        act: Registered activation name.
    """

    def __init__(self, in_ch: int, num_targets: int = 1, act: str = "relu6") -> None:
        super().__init__()
        self.in_ch = in_ch
        self.num_targets = num_targets
        self.act_name = act

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def _shape(self, conf: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reshape flat ``(B, N)`` / ``(B, N*4)`` predictions to the contract."""
        batch = conf.shape[0]
        return conf.view(batch, self.num_targets, 1), boxes.view(batch, self.num_targets, 4)


@register_component("head", "gap")
class GAPHead(Head):
    """Global average pooling followed by two linear branches.

    The paper's baseline head: cheapest, but localization precision is capped
    because the spatial grid is collapsed before regression.
    """

    def __init__(self, in_ch: int, num_targets: int = 1, act: str = "relu6") -> None:
        super().__init__(in_ch, num_targets, act)
        self.pooler = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.conf_head = nn.Linear(in_ch, num_targets)
        self.bbox_head = nn.Linear(in_ch, 4 * num_targets)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.pooler(x)
        return self._shape(torch.sigmoid(self.conf_head(feat)), torch.sigmoid(self.bbox_head(feat)))


@register_component("head", "spp")
class SPPLiteHead(Head):
    """SPP-Lite pooling shared by both branches, then two linear heads.

    Args:
        feat_dim: Width of the pooled feature vector fed to both branches.
    """

    def __init__(
        self, in_ch: int, num_targets: int = 1, act: str = "relu6", feat_dim: int = 64
    ) -> None:
        super().__init__(in_ch, num_targets, act)
        self.pooler = SPPLite(in_ch, feat_dim)
        self.conf_head = nn.Linear(feat_dim, num_targets)
        self.bbox_head = nn.Linear(feat_dim, 4 * num_targets)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.pooler(x)
        return self._shape(torch.sigmoid(self.conf_head(feat)), torch.sigmoid(self.bbox_head(feat)))


@register_component("head", "decoupled_spp")
class DecoupledSPPHead(Head):
    """Fully decoupled head: independent conv stem *and* SPP pooler per branch.

    Confidence and box regression no longer share a pooled vector, which is
    what separates DART-D/DART-XD from DART-S/DART-X in the paper.

    Args:
        hidden: Channels of each branch's conv stem; defaults to ``in_ch``.
        feat_dim: Width of each branch's pooled feature vector.
    """

    def __init__(
        self,
        in_ch: int,
        num_targets: int = 1,
        act: str = "relu6",
        hidden: int | None = None,
        feat_dim: int = 64,
    ) -> None:
        super().__init__(in_ch, num_targets, act)
        hidden = hidden or in_ch
        self.conf_stem = _stem(in_ch, hidden, act)
        self.conf_spp = SPPLite(hidden, feat_dim)
        self.conf_fc = nn.Linear(feat_dim, num_targets)

        self.bbox_stem = _stem(in_ch, hidden, act)
        self.bbox_spp = SPPLite(hidden, feat_dim)
        self.bbox_fc = nn.Linear(feat_dim, 4 * num_targets)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        conf = torch.sigmoid(self.conf_fc(self.conf_spp(self.conf_stem(x))))
        boxes = torch.sigmoid(self.bbox_fc(self.bbox_spp(self.bbox_stem(x))))
        return self._shape(conf, boxes)


@register_component("head", "softargmax")
class SoftArgmaxHead(Head):
    """Decoupled head whose centre comes from a spatial soft-argmax.

    A one-channel heatmap is normalised with a spatial softmax and the centre
    is the expectation of the normalised coordinate grid under it — true
    multi-position weighted pooling, giving sub-cell centre accuracy. Width and
    height still come from a pooled vector.

    Not one of the paper's seven variants; registered because the research
    codebase validated it and the registry makes it free to keep.

    Args:
        hidden: Channels of each branch's conv stem; defaults to ``in_ch``.
        temperature: Initial softmax temperature; learned thereafter.
    """

    def __init__(
        self,
        in_ch: int,
        num_targets: int = 1,
        act: str = "relu6",
        hidden: int | None = None,
        temperature: float = 1.0,
    ) -> None:
        super().__init__(in_ch, num_targets, act)
        if num_targets != 1:
            raise ValueError("softargmax head currently emits a single centre")
        hidden = hidden or in_ch
        self.conf_stem = _stem(in_ch, hidden, act)
        self.conf_pool = nn.AdaptiveAvgPool2d(1)
        self.conf_fc = nn.Linear(hidden, 1)

        self.bbox_stem = _stem(in_ch, hidden, act)
        self.center_conv = nn.Conv2d(hidden, 1, 1)
        self.size_pool = nn.AdaptiveAvgPool2d(1)
        self.size_fc = nn.Linear(hidden, 2)
        self.log_temp = nn.Parameter(torch.tensor(float(math.log(temperature))))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = x.shape

        conf_feat = self.conf_stem(x)
        conf = torch.sigmoid(self.conf_fc(self.conf_pool(conf_feat).flatten(1)))

        bbox_feat = self.bbox_stem(x)
        heat = self.center_conv(bbox_feat).view(batch, -1)
        attn = torch.softmax(heat * torch.exp(self.log_temp), dim=1).view(batch, 1, height, width)
        xs = (torch.arange(width, device=x.device, dtype=x.dtype) + 0.5) / width
        ys = (torch.arange(height, device=x.device, dtype=x.dtype) + 0.5) / height
        cx = (attn.sum(dim=2) * xs).sum(dim=2)
        cy = (attn.sum(dim=3) * ys).sum(dim=2)
        wh = torch.sigmoid(self.size_fc(self.size_pool(bbox_feat).flatten(1)))
        return self._shape(conf, torch.cat([cx, cy, wh], dim=1))


def _stem(in_ch: int, hidden: int, act: str) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
        nn.BatchNorm2d(hidden),
        make_activation(act),
    )


def build_head(name: str, in_ch: int, **kwargs) -> Head:
    """Build a head by registered name."""
    return COMPONENTS.build("head", name, in_ch=in_ch, **kwargs)
