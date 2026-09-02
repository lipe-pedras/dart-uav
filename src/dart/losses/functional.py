"""
Loss primitives.

These are DART's own contribution and stay in-house; the commodity pieces
(dataset parsing, metric computation) are delegated to third parties instead.

Every function here is stateless and operates on boxes in ``[cx, cy, w, h]``
format, normalised to ``[0, 1]`` relative to the input image — the same format
the detection heads emit and the datasets deliver, so no conversion is needed at
the call site.

Shape symbols used throughout: ``n`` is the number of boxes in a flat, already
paired set.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def focal_bce(
    pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0
) -> torch.Tensor:
    """
    Compute the focal binary cross-entropy of a set of predictions.

    Down-weights well-classified examples so the model spends capacity on hard
    ones. On DART's heavily negative-skewed datasets this is what prevents the
    collapse to "always predict positive" that plain BCE exhibits [1]_.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted probabilities in ``[0, 1]``. Already passed through a sigmoid;
        this function does not apply one.
    target : torch.Tensor
        Binary targets, same shape as ``pred``.
    alpha : float, default: 0.25
        Weight of the positive class. Values below ``0.5`` favour the negatives,
        which is the usual setting when background dominates.
    gamma : float, default: 2.0
        Focusing strength; higher is more aggressive. ``0`` reduces the whole
        expression to weighted BCE.

    Returns
    -------
    torch.Tensor
        Scalar mean loss over every element of ``pred``.

    See Also
    --------
    :class:`~dart.losses.composite.CompositeLoss` : Where this term is weighted
        against the box-regression term.

    References
    ----------
    .. [1] T.-Y. Lin, P. Goyal, R. Girshick, K. He and P. Dollar, "Focal Loss
           for Dense Object Detection", ICCV 2017.
    """
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    p_t = pred * target + (1 - pred) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    return (alpha_t * (1 - p_t) ** gamma * bce).mean()


def ciou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Compute the Complete-IoU regression loss between paired boxes.

    Extends IoU loss with a centroid-distance penalty and an aspect-ratio
    consistency term, so gradients remain informative even when the predicted
    and target boxes do not overlap at all — the case where plain IoU loss is
    flat and teaches the regressor nothing [1]_.

    Parameters
    ----------
    pred : torch.Tensor, shape (n, 4)
        Predicted boxes as ``[cx, cy, w, h]`` in ``[0, 1]``.
    target : torch.Tensor, shape (n, 4)
        Ground-truth boxes in the same format, paired positionally with ``pred``.
    eps : float, default: 1e-7
        Numerical floor applied to every denominator, so degenerate boxes of
        zero width or height cannot produce a division by zero.

    Returns
    -------
    torch.Tensor
        Scalar mean of ``1 - CIoU`` over the ``n`` pairs. Zero for a perfect
        match; unbounded above.

    See Also
    --------
    box_iou_pairwise : Plain element-wise IoU, without the penalty terms.
    :class:`~dart.losses.composite.CompositeLoss` : Where this term is weighted in.

    Notes
    -----
    The aspect-ratio trade-off weight is computed under :func:`torch.no_grad`,
    following the reference implementation: it scales the aspect-ratio term by
    how badly the boxes already overlap, and letting gradients flow through that
    scaling destabilises training.

    References
    ----------
    .. [1] Z. Zheng, P. Wang, W. Liu, J. Li, R. Ye and D. Ren, "Distance-IoU
           Loss: Faster and Better Learning for Bounding Box Regression",
           AAAI 2020.
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
    """
    Compute element-wise IoU between two aligned sets of boxes.

    Pairwise rather than all-against-all: box ``i`` of ``pred`` is compared only
    with box ``i`` of ``target``, which is what the single-target task needs and
    what keeps the cost linear in ``n``.

    Parameters
    ----------
    pred : torch.Tensor, shape (n, 4)
        Predicted boxes as ``[cx, cy, w, h]``.
    target : torch.Tensor, shape (n, 4)
        Ground-truth boxes in the same format, paired positionally with ``pred``.
    eps : float, default: 1e-6
        Numerical floor on the union area.

    Returns
    -------
    torch.Tensor, shape (n,)
        IoU per pair, in ``[0, 1]``.

    See Also
    --------
    ciou_loss : The regression loss built on top of this quantity.
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
    """
    Convert centre-form boxes to corner form.

    Parameters
    ----------
    boxes : torch.Tensor, shape (..., 4)
        Boxes as ``[cx, cy, w, h]``. Any number of leading dimensions is
        allowed; only the last is interpreted.

    Returns
    -------
    torch.Tensor, shape (..., 4)
        The same boxes as ``[x1, y1, x2, y2]``, in the input's coordinate scale.

    Examples
    --------
    >>> import torch
    >>> cxcywh_to_xyxy(torch.tensor([[0.5, 0.5, 0.2, 0.4]]))
    tensor([[0.4000, 0.3000, 0.6000, 0.7000]])
    """
    x1, y1, x2, y2 = _to_xyxy(boxes)
    return torch.stack([x1, y1, x2, y2], dim=-1)


def _to_xyxy(boxes: torch.Tensor):
    """
    Split centre-form boxes into four corner tensors.

    Parameters
    ----------
    boxes : torch.Tensor, shape (..., 4)
        Boxes as ``[cx, cy, w, h]``.

    Returns
    -------
    tuple of torch.Tensor
        The four corner coordinates ``(x1, y1, x2, y2)``, each of shape
        ``(...)``. Returned unstacked because every caller here consumes them
        as separate tensors.
    """
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
