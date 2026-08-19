"""The training engine (FR-3.1 – FR-3.6).

The trainer orchestrates; it does not persist. Every file written during a run
goes through :class:`~dart.engine.recorder.RunRecorder`, and every stop/continue
decision goes through :class:`~dart.engine.early_stopping.EarlyStopper`. That
separation is what makes both testable on their own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..config.schema import RunConfig
from ..data import DARTDataset, build_dataloader
from ..errors import CheckpointError
from ..losses import CompositeLoss
from ..metrics import MetricResult
from ..utils import select_device, set_seed, unique_dir
from .early_stopping import EarlyStopper
from .evaluator import Evaluator
from .recorder import RunRecorder


@dataclass
class RunResult:
    """The outcome of one training run.

    Attributes:
        run_dir: Directory holding the run's artifacts.
        best_metrics: Metrics of the best epoch by the monitored criterion.
        best_epoch: Epoch number those metrics came from.
        epochs_run: How many epochs actually executed.
        stopped_early: Whether early stopping ended the run.
        stop_reason: Why the run ended.
        best_checkpoint: Path to ``weights/best.pt``.
        history: Per-epoch metric rows.
        fingerprint: Config fingerprint, tying the numbers to their config.
    """

    run_dir: Path
    best_metrics: Dict[str, float] = field(default_factory=dict)
    best_epoch: int = 0
    epochs_run: int = 0
    stopped_early: bool = False
    stop_reason: str = ""
    best_checkpoint: Optional[Path] = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    fingerprint: str = ""

    def __str__(self) -> str:
        best = " ".join(
            f"{k}={v:.4f}" for k, v in self.best_metrics.items() if isinstance(v, float)
        )
        return f"RunResult(epochs={self.epochs_run}, best_epoch={self.best_epoch}, {best})"


class Trainer:
    """Runs the train/validate/decide/record loop.

    Args:
        cfg: The fully-resolved run configuration.
        run_dir: Explicit output directory; by default
            ``<train.project>/<train.name>``, suffixed if that exists.
    """

    def __init__(self, cfg: RunConfig, run_dir: Optional[Path] = None) -> None:
        self.cfg = cfg
        self.device = select_device(cfg.train.device)
        self.run_dir = (
            Path(run_dir)
            if run_dir is not None
            else unique_dir(Path(cfg.train.project), cfg.train.name)
        )
        self.recorder = RunRecorder(self.run_dir)
        self.loss = CompositeLoss(cfg.train.loss)
        self.stopper = EarlyStopper(cfg.train.early_stop)
        self.evaluator = Evaluator(
            conf_threshold=cfg.train.conf_threshold,
            loss=self.loss,
            device=self.device,
            amp=cfg.train.amp,
        )

    # ── Public API ──────────────────────────────────────────────────────

    def fit(
        self,
        model: nn.Module,
        train_data: DARTDataset,
        val_data: Optional[DARTDataset] = None,
        verbose: bool = True,
    ) -> RunResult:
        """Train ``model`` and return its :class:`RunResult`.

        Validation falls back to the training split when ``val_data`` is
        ``None``, which is only sensible for smoke tests; the resulting metrics
        are reported as-is and are not held out.
        """
        cfg = self.cfg
        set_seed(cfg.seed)
        model = model.to(self.device)

        train_loader = build_dataloader(
            train_data,
            batch_size=cfg.train.batch_size,
            shuffle=True,
            workers=cfg.data.workers,
            seed=cfg.seed,
            drop_last=len(train_data) > cfg.train.batch_size,
            pin_memory=self.device.type == "cuda",
        )
        val_loader = build_dataloader(
            val_data if val_data is not None else train_data,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            workers=cfg.data.workers,
            pin_memory=self.device.type == "cuda",
        )

        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)
        scaler = torch.amp.GradScaler(
            self.device.type, enabled=cfg.train.amp and self.device.type == "cuda"
        )

        start_epoch = 1
        if cfg.train.resume:
            start_epoch = self._resume(model, optimizer, scheduler, cfg.train.resume)

        self.recorder.on_run_start(cfg, {"device": str(self.device)})
        if verbose:
            self._print_header(model, len(train_data))

        best_metrics: Dict[str, float] = {}
        best_epoch = 0
        best_checkpoint: Optional[Path] = None
        decision = None
        epoch = start_epoch - 1

        for epoch in range(start_epoch, cfg.train.epochs + 1):
            started = time.time()
            train_metrics = self.train_epoch(model, train_loader, optimizer, scaler)
            val_metrics = self.evaluator.evaluate(model, val_loader, imgsz=cfg.model.imgsz)
            scheduler.step()

            row = _epoch_row(
                epoch=epoch,
                lr=optimizer.param_groups[0]["lr"],
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                elapsed=time.time() - started,
            )
            decision = self.stopper.step(val_metrics.to_dict(), epoch=epoch)
            self.recorder.on_epoch_end(epoch, row)

            if decision.improved:
                best_metrics = val_metrics.to_dict()
                best_epoch = epoch
                best_checkpoint = self.recorder.save_checkpoint(
                    "best", model, cfg, epoch, metrics=best_metrics
                )
            self.recorder.save_checkpoint(
                "last",
                model,
                cfg,
                epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                stopper_state=self.stopper.state_dict(),
                metrics=val_metrics.to_dict(),
            )
            if cfg.train.save_period and epoch % cfg.train.save_period == 0:
                self.recorder.save_checkpoint(f"epoch{epoch}", model, cfg, epoch)

            if verbose:
                self._print_row(row, decision)
            if decision.stop:
                break

        if not best_metrics and decision is not None:
            # Nothing ever counted as an improvement (a one-epoch run with a
            # metric that starts at its floor); report the final epoch.
            best_metrics = val_metrics.to_dict()
            best_epoch = epoch
            best_checkpoint = self.recorder.save_checkpoint(
                "best", model, cfg, epoch, metrics=best_metrics
            )

        result = RunResult(
            run_dir=self.run_dir,
            best_metrics=best_metrics,
            best_epoch=best_epoch,
            epochs_run=epoch - start_epoch + 1,
            stopped_early=bool(decision and decision.stop),
            stop_reason=decision.reason if decision else "",
            best_checkpoint=best_checkpoint,
            history=self.recorder.rows,
            fingerprint=cfg.fingerprint(),
        )
        self.recorder.on_run_end(result, model=model)
        efficiency = self.recorder.profile(model)
        self.recorder.write_comparison_row(
            cfg,
            {
                "params": efficiency.params,
                "int8_size_kb": efficiency.int8_size_kb,
                "latency_mean_ms": efficiency.latency.mean_ms if efficiency.latency else "",
                "latency_p95_ms": efficiency.latency.p95_ms if efficiency.latency else "",
                "fps": efficiency.latency.fps if efficiency.latency else "",
                "device": str(self.device),
            },
        )
        if verbose:
            print(f"\n  Done. {result.stop_reason or 'epoch budget exhausted'}")
            print(f"  Best epoch {best_epoch}: {_format_metrics(best_metrics)}")
            print(f"  Artifacts: {self.run_dir}")
        return result

    # ── Epoch ───────────────────────────────────────────────────────────

    def train_epoch(
        self,
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler,
    ) -> Dict[str, float]:
        """Run one training epoch and return its mean losses."""
        model.train()
        totals = {"loss": 0.0, "conf_loss": 0.0, "bbox_loss": 0.0}
        seen = 0
        use_amp = self.cfg.train.amp and self.device.type == "cuda"

        for images, gt_conf, gt_boxes in loader:
            images = images.to(self.device, non_blocking=True)
            gt_conf = gt_conf.to(self.device, non_blocking=True)
            gt_boxes = gt_boxes.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=self.device.type, enabled=use_amp):
                pred_conf, pred_boxes = model(images)
                breakdown = self.loss(pred_conf, pred_boxes, gt_conf, gt_boxes)

            scaler.scale(breakdown.total).backward()
            if self.cfg.train.grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self.cfg.train.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            batch = images.shape[0]
            totals["loss"] += float(breakdown.total.detach()) * batch
            totals["conf_loss"] += float(breakdown.conf) * batch
            totals["bbox_loss"] += float(breakdown.bbox) * batch
            seen += batch

        return {k: v / max(seen, 1) for k, v in totals.items()}

    # ── Setup helpers ───────────────────────────────────────────────────

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        cfg = self.cfg.train
        # param_groups() keeps backbone and head separate so a pretrained
        # backbone can take a smaller step than a fresh head (FR-2.6).
        groups = (
            model.param_groups(cfg.lr)
            if hasattr(model, "param_groups")
            else [{"params": list(model.parameters()), "lr": cfg.lr}]
        )
        name = cfg.optimizer.lower()
        if name == "adamw":
            return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if name == "adam":
            return torch.optim.Adam(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if name == "sgd":
            return torch.optim.SGD(
                groups, lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay, nesterov=True
            )
        raise ValueError(f"unknown optimizer '{cfg.optimizer}'; use adamw, adam, or sgd")

    def _build_scheduler(self, optimizer: torch.optim.Optimizer):
        cfg = self.cfg.train
        name = cfg.scheduler.lower()
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr
            )
        if name in ("none", "constant"):
            return torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=max(1, cfg.epochs // 3), gamma=0.1
            )
        raise ValueError(f"unknown scheduler '{cfg.scheduler}'; use cosine, step, or none")

    def _resume(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        path: str,
    ) -> int:
        """Restore model, optimizer, scheduler, and stopper from a checkpoint.

        Resume granularity is the epoch boundary: mid-epoch resume would
        require checkpointing DataLoader state for no practical gain.
        """
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise CheckpointError(f"cannot resume: checkpoint not found at {checkpoint_path}")
        payload = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        model.load_state_dict(payload["model_state"])
        if "optimizer_state" in payload:
            optimizer.load_state_dict(payload["optimizer_state"])
        if "scheduler_state" in payload:
            scheduler.load_state_dict(payload["scheduler_state"])
        if "stopper_state" in payload:
            self.stopper.load_state_dict(payload["stopper_state"])
        epoch = int(payload.get("epoch", 0))
        print(f"  Resumed from {checkpoint_path} at epoch {epoch}")
        return epoch + 1

    # ── Console output ──────────────────────────────────────────────────

    def _print_header(self, model: nn.Module, n_train: int) -> None:
        info = model.model_info() if hasattr(model, "model_info") else {}
        print(f"\n  DART training — variant '{self.cfg.variant}' on {self.device}")
        if info:
            print(f"  Model: {info['params']:,} params, INT8 ~{info['est_int8_kb']} KB")
        print(f"  Data : {n_train} training images, imgsz={self.cfg.model.imgsz}")
        print(f"  Loss : {self.loss}")
        print(f"  Out  : {self.run_dir}\n")
        print(
            f"{'epoch':>6} {'train':>9} {'val':>9} {'mAP50':>8} {'IoU':>8} "
            f"{'F1':>8} {'P':>8} {'R':>8} {'time':>7}"
        )
        print("-" * 78)

    @staticmethod
    def _print_row(row: Dict[str, Any], decision) -> None:
        marker = "*" if decision.improved else " "
        print(
            f"{row['epoch']:>6} {row['train_loss']:>9.4f} {row['val_loss']:>9.4f} "
            f"{row['val_mAP50']:>8.4f} {row['val_mean_iou']:>8.4f} {row['val_f1']:>8.4f} "
            f"{row['val_precision']:>8.4f} {row['val_recall']:>8.4f} "
            f"{row['epoch_time_s']:>6.1f}s{marker}"
        )


def _epoch_row(
    epoch: int,
    lr: float,
    train_metrics: Dict[str, float],
    val_metrics: MetricResult,
    elapsed: float,
) -> Dict[str, Any]:
    return {
        "epoch": epoch,
        "lr": lr,
        "train_loss": train_metrics["loss"],
        "train_conf_loss": train_metrics["conf_loss"],
        "train_bbox_loss": train_metrics["bbox_loss"],
        "val_loss": val_metrics.loss,
        "val_conf_loss": val_metrics.conf_loss,
        "val_bbox_loss": val_metrics.bbox_loss,
        "val_mAP50": val_metrics.map50,
        "val_mAP50_95": val_metrics.map50_95,
        "val_precision": val_metrics.precision,
        "val_recall": val_metrics.recall,
        "val_f1": val_metrics.f1,
        "val_mean_iou": val_metrics.mean_iou,
        "val_best_thresh": val_metrics.best_threshold,
        "val_best_f1": val_metrics.best_f1,
        "epoch_time_s": elapsed,
    }


def _format_metrics(metrics: Dict[str, float]) -> str:
    keys = ("map50", "map50_95", "precision", "recall", "f1", "mean_iou")
    return " ".join(f"{k}={metrics[k]:.4f}" for k in keys if k in metrics)
