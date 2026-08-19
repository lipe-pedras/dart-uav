"""Validation, usable with or without a trainer.

``Evaluator`` depends on the model and the data, never on ``Trainer``, which is
what makes benchmark-only evaluation (FR-4.2) cheap: load a checkpoint,
evaluate, done — no training machinery is constructed.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import DataLoader

from ..losses import CompositeLoss
from ..metrics import MetricResult, MetricSet


class Evaluator:
    """Runs a model over a validation loader and computes metrics.

    Args:
        conf_threshold: Confidence at which precision/recall/F1 are reported.
        loss: Optional loss, so validation loss is reported alongside metrics.
        device: Device to evaluate on; defaults to the model's.
        amp: Use autocast on CUDA.
    """

    def __init__(
        self,
        conf_threshold: float = 0.5,
        loss: Optional[CompositeLoss] = None,
        device: Optional[torch.device] = None,
        amp: bool = True,
    ) -> None:
        self.conf_threshold = float(conf_threshold)
        self.loss = loss
        self.device = device
        self.amp = amp

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        imgsz: Optional[int] = None,
        verbose: bool = False,
    ) -> MetricResult:
        """Evaluate ``model`` over every batch in ``loader``.

        Returns:
            A :class:`~dart.metrics.MetricResult`, including validation losses
            when a loss function was supplied.
        """
        device = self.device or next(model.parameters()).device
        imgsz = imgsz or getattr(model, "imgsz", 96)
        was_training = model.training
        model.eval()

        metrics = MetricSet(
            conf_threshold=self.conf_threshold, imgsz=imgsz, device=torch.device("cpu")
        )
        totals = {"loss": 0.0, "conf_loss": 0.0, "bbox_loss": 0.0}
        seen = 0

        use_amp = self.amp and device.type == "cuda"
        for images, gt_conf, gt_boxes in loader:
            images = images.to(device, non_blocking=True)
            gt_conf = gt_conf.to(device, non_blocking=True)
            gt_boxes = gt_boxes.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred_conf, pred_boxes = model(images)

            pred_conf = pred_conf.float()
            pred_boxes = pred_boxes.float()

            batch = images.shape[0]
            if self.loss is not None:
                breakdown = self.loss(pred_conf, pred_boxes, gt_conf, gt_boxes)
                totals["loss"] += float(breakdown.total) * batch
                totals["conf_loss"] += float(breakdown.conf) * batch
                totals["bbox_loss"] += float(breakdown.bbox) * batch
            seen += batch

            metrics.update(pred_conf.cpu(), pred_boxes.cpu(), gt_conf.cpu(), gt_boxes.cpu())

        result = metrics.compute()
        if seen and self.loss is not None:
            result.loss = totals["loss"] / seen
            result.conf_loss = totals["conf_loss"] / seen
            result.bbox_loss = totals["bbox_loss"] / seen

        if was_training:
            model.train()
        if verbose:
            print(f"  {result}")
        return result
