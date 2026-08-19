"""Detection metrics, computed with ``torchmetrics`` (FR-4.1).

This module contains no bespoke metric arithmetic. That is a
scientific-integrity requirement, not an engineering preference: numbers
reported in a peer-reviewed paper should come from code the community has
already validated, not from project-specific code a reviewer cannot audit.
The research codebase's hand-written ``iou_batch`` and ``compute_pr_metrics``
are deliberately *not* carried over.

What this module does own is the *framing* — how the paper's single-target
task maps onto standard metrics:

``map50`` / ``map50_95``
    :class:`torchmetrics.detection.MeanAveragePrecision`, one predicted box per
    image scored by its confidence.

``precision``
    Of the images where a detection was announced (``conf >= threshold``), the
    fraction where a correct detection was actually available (object present
    **and** ``IoU >= 0.5``). A confident but badly-localised box is a false
    positive.

``recall``
    Of the images that contain an object, the fraction where a detection was
    both announced and well-localised. The denominator is the total number of
    ground-truth positives, so poor localisation cannot inflate it.

``f1``
    Harmonic mean of the two above.

``mean_iou``
    Mean IoU over ground-truth-positive images, from
    ``torchmetrics.functional.detection.intersection_over_union``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch
from torchmetrics.classification import BinaryPrecision, BinaryRecall
from torchmetrics.detection import MeanAveragePrecision
from torchmetrics.functional.detection import intersection_over_union

from ..losses.functional import cxcywh_to_xyxy

#: IoU above which a predicted box counts as correctly localised.
LOCALIZATION_IOU = 0.5


@dataclass
class MetricResult:
    """One split's metrics.

    ``map50_95`` is reported beyond the paper's own set because it is the
    standard COCO-style primary metric; without it DART is hard to compare
    against externally-published detectors (FR-4.1).
    """

    map50: float = 0.0
    map50_95: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    mean_iou: float = 0.0
    best_threshold: float = 0.5
    best_f1: float = 0.0
    loss: float = 0.0
    conf_loss: float = 0.0
    bbox_loss: float = 0.0
    extras: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        data = asdict(self)
        extras = data.pop("extras")
        data.update(extras)
        return data

    def __getitem__(self, key: str) -> float:
        data = self.to_dict()
        if key not in data:
            raise KeyError(f"unknown metric '{key}'. Available: {sorted(data)}")
        return data[key]

    def __str__(self) -> str:
        return (
            f"mAP50={self.map50:.4f} mAP50-95={self.map50_95:.4f} "
            f"P={self.precision:.4f} R={self.recall:.4f} F1={self.f1:.4f} "
            f"IoU={self.mean_iou:.4f}"
        )


class MetricSet:
    """Accumulates predictions and computes a :class:`MetricResult`.

    Args:
        conf_threshold: Confidence at which precision/recall/F1 are reported.
        imgsz: Image size used to scale normalised boxes into pixels, so
            ``MeanAveragePrecision``'s area-based breakdowns are meaningful.
        device: Device the metric states live on.
        sweep_thresholds: Confidence grid searched for the best-F1 operating
            point (the paper's ``val_best_thresh`` / ``val_best_f1``).
    """

    def __init__(
        self,
        conf_threshold: float = 0.5,
        imgsz: int = 96,
        device: Optional[torch.device] = None,
        sweep_thresholds: int = 101,
    ) -> None:
        self.conf_threshold = float(conf_threshold)
        self.imgsz = int(imgsz)
        self.device = device or torch.device("cpu")
        self.sweep = torch.linspace(0.0, 1.0, sweep_thresholds)

        self._map = MeanAveragePrecision(
            box_format="xyxy", iou_type="bbox", backend=_map_backend()
        ).to(self.device)
        self._precision = BinaryPrecision().to(self.device)
        self._recall = BinaryRecall().to(self.device)

        self._scores: List[torch.Tensor] = []
        self._correct: List[torch.Tensor] = []
        self._present: List[torch.Tensor] = []
        self._positive_ious: List[torch.Tensor] = []

    # ── Accumulation ────────────────────────────────────────────────────

    @torch.no_grad()
    def update(
        self,
        pred_conf: torch.Tensor,
        pred_boxes: torch.Tensor,
        gt_conf: torch.Tensor,
        gt_boxes: torch.Tensor,
    ) -> None:
        """Accumulate one batch.

        Args:
            pred_conf: ``(B, N, 1)`` predicted objectness.
            pred_boxes: ``(B, N, 4)`` predicted ``cxcywh`` boxes in ``[0, 1]``.
            gt_conf: ``(B, N)`` target objectness.
            gt_boxes: ``(B, N, 4)`` target boxes.
        """
        scores = pred_conf.detach().reshape(-1).float().to(self.device)
        gt_present = (gt_conf.detach().reshape(-1) > 0.5).to(self.device)
        pred_xyxy = cxcywh_to_xyxy(pred_boxes.detach().reshape(-1, 4).float()).to(self.device)
        gt_xyxy = cxcywh_to_xyxy(gt_boxes.detach().reshape(-1, 4).float()).to(self.device)

        ious = self._pairwise_iou(pred_xyxy, gt_xyxy)
        correct = gt_present & (ious >= LOCALIZATION_IOU)

        self._scores.append(scores)
        self._correct.append(correct)
        self._present.append(gt_present)
        if bool(gt_present.any()):
            self._positive_ious.append(ious[gt_present])

        announced = scores >= self.conf_threshold
        # Precision: of the announcements, how many had a correct detection
        # available. Recall: of the objects present, how many were both
        # announced and well-localised.
        self._precision.update(announced.int(), correct.int())
        self._recall.update((announced & correct).int(), gt_present.int())

        self._map.update(
            self._as_map_inputs(pred_xyxy, scores, announced),
            self._as_map_targets(gt_xyxy, gt_present),
        )

    # ── Computation ─────────────────────────────────────────────────────

    @torch.no_grad()
    def compute(self) -> MetricResult:
        """Compute every metric from what has been accumulated."""
        if not self._scores:
            return MetricResult()

        scores = torch.cat(self._scores)
        correct = torch.cat(self._correct)
        present = torch.cat(self._present)

        map_values = self._map.compute()
        precision = float(self._precision.compute())
        recall = float(self._recall.compute())
        f1 = _harmonic_mean(precision, recall)

        mean_iou = float(torch.cat(self._positive_ious).mean()) if self._positive_ious else 0.0
        best_threshold, best_f1 = self._best_f1(scores, correct, present)

        return MetricResult(
            map50=_scalar(map_values.get("map_50")),
            map50_95=_scalar(map_values.get("map")),
            precision=precision,
            recall=recall,
            f1=f1,
            mean_iou=mean_iou,
            best_threshold=best_threshold,
            best_f1=best_f1,
        )

    def reset(self) -> None:
        """Discard all accumulated state."""
        self._map.reset()
        self._precision.reset()
        self._recall.reset()
        self._scores.clear()
        self._correct.clear()
        self._present.clear()
        self._positive_ious.clear()

    # ── Internals ───────────────────────────────────────────────────────

    def _pairwise_iou(self, pred_xyxy: torch.Tensor, gt_xyxy: torch.Tensor) -> torch.Tensor:
        """Element-wise IoU of positionally-paired boxes.

        ``intersection_over_union`` returns the full pred-by-target matrix, so
        the aligned pairs are its diagonal. Done per batch, the matrix stays
        small; accumulating first and calling it once would be quadratic in the
        size of the whole split.
        """
        if pred_xyxy.numel() == 0:
            return torch.zeros(0, device=self.device)
        matrix = intersection_over_union(
            pred_xyxy * self.imgsz, gt_xyxy * self.imgsz, aggregate=False
        )
        return matrix.diagonal().clamp(min=0.0)

    def _as_map_inputs(
        self, boxes: torch.Tensor, scores: torch.Tensor, announced: torch.Tensor
    ) -> List[dict]:
        """One detection per image, dropped when nothing was announced."""
        result = []
        for i in range(boxes.shape[0]):
            keep = bool(announced[i])
            result.append(
                {
                    "boxes": (boxes[i : i + 1] * self.imgsz) if keep else _empty_boxes(self.device),
                    "scores": scores[i : i + 1] if keep else _empty(self.device),
                    "labels": torch.zeros(1 if keep else 0, dtype=torch.long, device=self.device),
                }
            )
        return result

    def _as_map_targets(self, boxes: torch.Tensor, present: torch.Tensor) -> List[dict]:
        result = []
        for i in range(boxes.shape[0]):
            keep = bool(present[i])
            result.append(
                {
                    "boxes": (boxes[i : i + 1] * self.imgsz) if keep else _empty_boxes(self.device),
                    "labels": torch.zeros(1 if keep else 0, dtype=torch.long, device=self.device),
                }
            )
        return result

    def _best_f1(self, scores: torch.Tensor, correct: torch.Tensor, present: torch.Tensor) -> tuple:
        """Sweep the confidence threshold for the best-F1 operating point.

        Uses the same precision/recall definitions as the fixed-threshold
        metrics above, evaluated on a grid.
        """
        n_present = float(present.sum())
        if n_present == 0:
            return self.conf_threshold, 0.0

        thresholds = self.sweep.to(scores.device)
        announced = scores.unsqueeze(0) >= thresholds.unsqueeze(1)  # (T, n)
        true_pos = (announced & correct.unsqueeze(0)).sum(dim=1).float()
        announced_n = announced.sum(dim=1).float()

        precision = true_pos / announced_n.clamp(min=1e-9)
        recall = true_pos / n_present
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-9)

        best = int(torch.argmax(f1))
        return float(thresholds[best]), float(f1[best])


def _harmonic_mean(a: float, b: float) -> float:
    return 2 * a * b / (a + b) if (a + b) > 0 else 0.0


def _scalar(value) -> float:
    if value is None:
        return 0.0
    value = float(value)
    # torchmetrics reports -1.0 for "no data for this breakdown".
    return max(value, 0.0)


def _empty(device: torch.device) -> torch.Tensor:
    return torch.zeros(0, device=device)


def _empty_boxes(device: torch.device) -> torch.Tensor:
    return torch.zeros((0, 4), device=device)


def _map_backend() -> str:
    """Pick whichever COCO evaluation backend is installed."""
    try:
        import pycocotools  # noqa: F401

        return "pycocotools"
    except ImportError:
        return "faster_coco_eval"
