"""The composite detection loss (FR-3.6).

Two named presets reproduce the paper's two loss configurations:

===================  ==========================
``bce + smooth_l1``  the DART-B baseline
``focal + ciou``     every other paper variant
===================  ==========================

Both are just values of :class:`~dart.config.schema.LossConfig`, so a third
combination needs no new class.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config.schema import LossConfig
from ..errors import ConfigurationError
from .functional import ciou_loss, focal_bce


@dataclass
class LossBreakdown:
    """The scalar total plus its detached components, for logging."""

    total: torch.Tensor
    conf: torch.Tensor
    bbox: torch.Tensor

    def item(self) -> float:
        return float(self.total.detach())


class CompositeLoss(nn.Module):
    """``total = alpha * conf_loss + beta * bbox_loss``.

    The box term is computed over positive samples only: a negative image has
    no box to regress toward, and including its zero-filled placeholder would
    teach the model to predict a degenerate box.

    Args:
        cfg: Loss configuration naming the two terms and their weights.
    """

    def __init__(self, cfg: LossConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LossConfig()
        if self.cfg.conf not in ("bce", "focal"):
            raise ConfigurationError(f"unknown confidence loss '{self.cfg.conf}'")
        if self.cfg.bbox not in ("smooth_l1", "ciou"):
            raise ConfigurationError(f"unknown box loss '{self.cfg.bbox}'")

    def forward(
        self,
        pred_conf: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_conf: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> LossBreakdown:
        """Compute the loss.

        Args:
            pred_conf: ``(B, N, 1)`` predicted objectness in ``[0, 1]``.
            pred_boxes: ``(B, N, 4)`` predicted ``cxcywh`` boxes in ``[0, 1]``.
            gt_conf: ``(B, N)`` target objectness.
            gt_boxes: ``(B, N, 4)`` target boxes.
        """
        pred_conf_flat = pred_conf.reshape(-1)
        gt_conf_flat = gt_conf.reshape(-1)
        pred_boxes_flat = pred_boxes.reshape(-1, 4)
        gt_boxes_flat = gt_boxes.reshape(-1, 4)

        # Binary cross-entropy on probabilities is numerically unsafe in fp16,
        # so the confidence term always runs in fp32 even under autocast.
        with torch.autocast(device_type=pred_conf.device.type, enabled=False):
            probs = pred_conf_flat.float().clamp(1e-7, 1 - 1e-7)
            targets = gt_conf_flat.float()
            if self.cfg.conf == "focal":
                conf_loss = focal_bce(
                    probs, targets, alpha=self.cfg.focal_alpha, gamma=self.cfg.focal_gamma
                )
            else:
                conf_loss = F.binary_cross_entropy(probs, targets)

        positive = gt_conf_flat > 0.5
        if bool(positive.any()):
            pred_pos = pred_boxes_flat[positive]
            gt_pos = gt_boxes_flat[positive]
            if self.cfg.bbox == "ciou":
                bbox_loss = ciou_loss(pred_pos.float(), gt_pos.float())
            else:
                bbox_loss = F.smooth_l1_loss(pred_pos, gt_pos)
        else:
            bbox_loss = pred_boxes.sum() * 0.0

        total = self.cfg.alpha * conf_loss + self.cfg.beta * bbox_loss
        return LossBreakdown(total=total, conf=conf_loss.detach(), bbox=bbox_loss.detach())

    def __repr__(self) -> str:
        return (
            f"CompositeLoss(conf={self.cfg.conf}, bbox={self.cfg.bbox}, "
            f"alpha={self.cfg.alpha}, beta={self.cfg.beta})"
        )


#: The paper's two loss configurations, by name.
LOSS_PRESETS = {
    "bce_smooth_l1": LossConfig(conf="bce", bbox="smooth_l1"),
    "focal_ciou": LossConfig(conf="focal", bbox="ciou"),
}


def build_loss(preset_or_cfg) -> CompositeLoss:
    """Build a loss from a preset name or a :class:`LossConfig`."""
    if isinstance(preset_or_cfg, LossConfig):
        return CompositeLoss(preset_or_cfg)
    key = str(preset_or_cfg).lower()
    if key not in LOSS_PRESETS:
        raise ConfigurationError(
            f"unknown loss preset '{preset_or_cfg}'; use one of {sorted(LOSS_PRESETS)} "
            "or pass a LossConfig"
        )
    return CompositeLoss(LOSS_PRESETS[key])
