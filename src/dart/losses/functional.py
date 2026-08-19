"""Loss primitives.

These are DART's own contribution and stay in-house; the commodity pieces
(dataset parsing, metric computation) are delegated to third parties instead.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def focal_bce(
    pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """Focal loss for binary classification (Lin et al., RetinaNet 2017).

    Down-weights well-classified examples so the model spends capacity on hard
    ones. On DART's heavily negative-skewed datasets this is what prevents the
    collapse to "always predict positive" that plain BCE exhibits.

    Args:
        pred: Predicted probabilities in ``[0, 1]``.
        target: Binary targets, same shape as ``pred``.
        alpha: Weight of the positive class.
        gamma: Focusing strength; higher is more aggressive.

    Returns:
        Scalar mean loss.
    """
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    p_t = pred * target + (1 - pred) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


def ciou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete IoU loss (Zheng et al., 2020).

    Extends IoU loss with a centroid-distance penalty and an aspect-ratio
    consistency term, so gradients remain informative even when boxes do not
    overlap.

    Args:
        pred: ``(n, 4)`` predicted boxes as ``[cx, cy, w, h]`` in ``[0, 1]``.
        target: ``(n, 4)`` ground-truth boxes in the same format.
        eps: Numerical floor.

    Returns:
        Scalar mean loss, ``1 - CIoU``.
    """
    p_x1, p_y1, p_x2, p_y2 = _to_xyxy(pred)
    g_x1, g_y1, g_x2, g_y2 = _to_xyxy(target)

    inter_w = (torch.min(p_x2, g_x2) - torch.max(p_x1, g_x1)).clamp(min=0)
    inter_h = (torch.min(p_y2, g_y2) - torch.max(p_y1, g_y1)).clamp(min=0)
    inter = inter_w * inter_h
    area_p = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
    area_g = (g_x2 - g_x1).clamp(min=0) * (g_y2 - g_y1).clamp(min=0)
    union = area_p + area_g - inter + eps
    iou = inter / union

    # Diagonal of the smallest enclosing box, squared.
    enc_w = torch.max(p_x2, g_x2) - torch.min(p_x1, g_x1)
    enc_h = torch.max(p_y2, g_y2) - torch.min(p_y1, g_y1)
    c2 = (enc_w**2 + enc_h**2).clamp(min=eps)

    d2 = (pred[:, 0] - target[:, 0]) ** 2 + (pred[:, 1] - target[:, 1]) ** 2

    v = (4 / math.pi**2) * (
        torch.atan(target[:, 2] / target[:, 3].clamp(min=eps))
        - torch.atan(pred[:, 2] / pred[:, 3].clamp(min=eps))
    ) ** 2
    with torch.no_grad():
        alpha_v = v / (1 - iou + v + eps)

    return (1 - (iou - d2 / c2 - alpha_v * v)).mean()


def box_iou_pairwise(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Element-wise IoU between two aligned sets of ``cxcywh`` boxes.

    Args:
        pred: ``(n, 4)`` boxes.
        target: ``(n, 4)`` boxes, paired positionally with ``pred``.

    Returns:
        ``(n,)`` IoU values.
    """
    p_x1, p_y1, p_x2, p_y2 = _to_xyxy(pred)
    g_x1, g_y1, g_x2, g_y2 = _to_xyxy(target)

    inter_w = (torch.min(p_x2, g_x2) - torch.max(p_x1, g_x1)).clamp(min=0)
    inter_h = (torch.min(p_y2, g_y2) - torch.max(p_y1, g_y1)).clamp(min=0)
    inter = inter_w * inter_h
    area_p = (p_x2 - p_x1).clamp(min=0) * (p_y2 - p_y1).clamp(min=0)
    area_g = (g_x2 - g_x1).clamp(min=0) * (g_y2 - g_y1).clamp(min=0)
    return inter / (area_p + area_g - inter + eps)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert ``[cx, cy, w, h]`` boxes to ``[x1, y1, x2, y2]``."""
    x1, y1, x2, y2 = _to_xyxy(boxes)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _to_xyxy(boxes: torch.Tensor):
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
