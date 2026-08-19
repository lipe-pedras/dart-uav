"""Run persistence.

:class:`RunRecorder` owns *all* filesystem writes for a run; the trainer never
writes a file itself. Consequences:

* the resolved configuration (FR-3.4) and the environment record (FR-7.1) are
  written before the first epoch, so an interrupted run is still reproducible;
* every run directory is self-contained, and the paper's
  ``model_comparison.csv`` (FR-4.4) is a concatenation of per-run outputs
  rather than a separate reporting pipeline;
* the efficiency profile (FR-4.5) is part of the run record, not a side script.

Layout of a run directory::

    runs/exp/
        config.yaml       fully-resolved configuration
        environment.json  versions, hardware, config fingerprint
        metrics.csv       one row per epoch
        results.json      final summary, best epoch, efficiency
        weights/best.pt   best checkpoint by the monitored metric
        weights/last.pt   most recent checkpoint (resume point)
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch

from ..config.schema import RunConfig
from ..metrics import EfficiencyProfiler, EfficiencyResult
from ..utils import environment_record

#: Column order of ``metrics.csv`` and of the aggregated comparison CSV. The
#: first block matches the paper's ``model_comparison.csv`` schema exactly so
#: existing analysis scripts keep working; the efficiency columns are the ones
#: the paper flags as missing (FR-4.5).
COMPARISON_FIELDS = [
    "variant",
    "dataset",
    "seed",
    "imgsz",
    "best_epoch",
    "total_epochs",
    "val_mAP50",
    "val_mAP50_95",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_mean_iou",
    "val_loss",
    "val_best_thresh",
    "val_best_f1",
    "params",
    "int8_size_kb",
    "latency_mean_ms",
    "latency_p95_ms",
    "fps",
    "device",
    "fingerprint",
]

EPOCH_FIELDS = [
    "epoch",
    "lr",
    "train_loss",
    "train_conf_loss",
    "train_bbox_loss",
    "val_loss",
    "val_conf_loss",
    "val_bbox_loss",
    "val_mAP50",
    "val_mAP50_95",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_mean_iou",
    "val_best_thresh",
    "val_best_f1",
    "epoch_time_s",
]


class RunRecorder:
    """Writes the on-disk record of one run.

    Args:
        run_dir: Directory the run writes into; created if absent.
        profiler: Efficiency profiler used at the end of the run.
    """

    def __init__(self, run_dir: Path, profiler: Optional[EfficiencyProfiler] = None) -> None:
        self.run_dir = Path(run_dir)
        self.weights_dir = self.run_dir / "weights"
        self.metrics_path = self.run_dir / "metrics.csv"
        self.config_path = self.run_dir / "config.yaml"
        self.environment_path = self.run_dir / "environment.json"
        self.results_path = self.run_dir / "results.json"
        self.profiler = profiler or EfficiencyProfiler()
        self._rows: List[Dict[str, Any]] = []

    @property
    def rows(self) -> List[Dict[str, Any]]:
        """The per-epoch rows written so far, in order."""
        return list(self._rows)

    # ── Lifecycle ───────────────────────────────────────────────────────

    def on_run_start(self, cfg: RunConfig, extra_environment: Optional[Mapping] = None) -> None:
        """Create the run directory and snapshot config plus environment."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        cfg.to_yaml(self.config_path)

        record = dict(environment_record())
        record["fingerprint"] = cfg.fingerprint()
        record["variant"] = cfg.variant
        record["seed"] = cfg.seed
        if extra_environment:
            record.update(dict(extra_environment))
        self._write_json(self.environment_path, record)

        with self.metrics_path.open("w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=EPOCH_FIELDS).writeheader()

    def on_epoch_end(self, epoch: int, metrics: Mapping[str, Any]) -> None:
        """Append one epoch's metrics row."""
        row = {key: metrics.get(key, "") for key in EPOCH_FIELDS}
        row["epoch"] = epoch
        self._rows.append(row)
        with self.metrics_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=EPOCH_FIELDS).writerow(row)

    def on_run_end(self, result: Any, model: Optional[torch.nn.Module] = None) -> Path:
        """Write the final summary, profiling the model's efficiency first."""
        payload = _to_plain(result)
        if model is not None:
            efficiency = self.profile(model)
            payload["efficiency"] = efficiency.to_dict()
        self._write_json(self.results_path, payload)
        return self.results_path

    # ── Checkpoints (FR-3.3) ────────────────────────────────────────────

    def save_checkpoint(
        self,
        name: str,
        model: torch.nn.Module,
        cfg: RunConfig,
        epoch: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        stopper_state: Optional[Mapping] = None,
        metrics: Optional[Mapping] = None,
    ) -> Path:
        """Write a checkpoint sufficient to resume or to reload for inference.

        The resolved config travels inside the checkpoint, so a ``.pt`` file
        alone is enough to rebuild the exact architecture that produced it.
        """
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
            "epoch": int(epoch),
            "metrics": dict(metrics or {}),
            "fingerprint": cfg.fingerprint(),
        }
        if optimizer is not None:
            payload["optimizer_state"] = optimizer.state_dict()
        if scheduler is not None:
            payload["scheduler_state"] = scheduler.state_dict()
        if stopper_state is not None:
            payload["stopper_state"] = dict(stopper_state)

        path = self.weights_dir / f"{name}.pt"
        torch.save(payload, path)
        return path

    def profile(self, model: torch.nn.Module) -> EfficiencyResult:
        """Profile the model's efficiency (FR-4.5), fusing a copy first."""
        return self.profiler.profile(model)

    # ── Reporting (FR-4.4) ──────────────────────────────────────────────

    def best_row(self, monitor: str = "val_mAP50") -> Dict[str, Any]:
        """The epoch row that scored best on ``monitor``.

        Ties break toward the lower validation loss, matching how the research
        codebase's ``compare_models.py`` picked a run's representative epoch.
        """
        rows = [r for r in self._rows if r.get(monitor) not in ("", None)]
        if not rows:
            return {}
        return max(rows, key=lambda r: (float(r[monitor]), -float(r.get("val_loss") or 0.0)))

    def write_comparison_row(
        self, cfg: RunConfig, extra: Optional[Mapping[str, Any]] = None
    ) -> Path:
        """Write this run's single row in the paper's comparison schema."""
        best = self.best_row()
        row = {field: "" for field in COMPARISON_FIELDS}
        row.update(
            {
                "variant": cfg.variant,
                "dataset": cfg.data.path or "",
                "seed": cfg.seed,
                "imgsz": cfg.model.imgsz,
                "best_epoch": best.get("epoch", ""),
                "total_epochs": len(self._rows),
                "fingerprint": cfg.fingerprint(),
            }
        )
        for key in (
            "val_mAP50",
            "val_mAP50_95",
            "val_precision",
            "val_recall",
            "val_f1",
            "val_mean_iou",
            "val_loss",
            "val_best_thresh",
            "val_best_f1",
        ):
            row[key] = best.get(key, "")
        if extra:
            row.update({k: v for k, v in extra.items() if k in row})

        path = self.run_dir / "comparison.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
            writer.writeheader()
            writer.writerow(row)
        return path

    # ── Internals ───────────────────────────────────────────────────────

    @staticmethod
    def _write_json(path: Path, payload: Mapping) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(_to_plain(payload), handle, indent=2, default=str)


def aggregate_runs(root: Path, output: Optional[Path] = None) -> Path:
    """Concatenate every run's ``comparison.csv`` under ``root`` into one file.

    This is how the paper's 135-run comparison table is regenerated from a
    fresh set of runs (FR-4.4).

    Args:
        root: Directory tree containing run directories.
        output: Destination CSV; defaults to ``root/model_comparison.csv``.
    """
    root = Path(root)
    output = Path(output) if output else root / "model_comparison.csv"
    rows: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("comparison.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _to_plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _to_plain(asdict(value))
    if isinstance(value, Mapping):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return value
