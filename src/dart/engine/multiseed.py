"""Multi-seed runs with mean ± standard deviation reporting (FR-4.3)."""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence

from ..config.schema import RunConfig
from .trainer import RunResult


@dataclass
class AggregateResult:
    """Per-metric mean and standard deviation across seeds.

    Attributes:
        seeds: The seeds that were run.
        mean: Metric name -> mean across seeds.
        std: Metric name -> sample standard deviation (0.0 for a single seed).
        runs: The individual :class:`RunResult` objects.
    """

    seeds: List[int] = field(default_factory=list)
    mean: Dict[str, float] = field(default_factory=dict)
    std: Dict[str, float] = field(default_factory=dict)
    runs: List[RunResult] = field(default_factory=list)

    def format(
        self, keys: Sequence[str] = ("map50", "precision", "recall", "f1", "mean_iou")
    ) -> str:
        """Render as ``metric: mean ± std`` lines."""
        return "\n".join(
            f"  {key:<10} {self.mean[key]:.4f} ± {self.std.get(key, 0.0):.4f}"
            for key in keys
            if key in self.mean
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seeds": self.seeds,
            "mean": self.mean,
            "std": self.std,
            "run_dirs": [str(r.run_dir) for r in self.runs],
        }


class MultiSeedRunner:
    """Runs one configuration across several seeds and aggregates the results.

    Args:
        train_fn: Callable taking ``(RunConfig, run_dir)`` and returning a
            :class:`RunResult`. Injected rather than hardcoded so the aggregation
            logic is testable without training anything.
    """

    def __init__(self, train_fn: Callable[[RunConfig, Path], RunResult]) -> None:
        self.train_fn = train_fn

    def run(self, cfg: RunConfig, seeds: Sequence[int], verbose: bool = True) -> AggregateResult:
        """Train once per seed and aggregate the best-epoch metrics.

        Each seed gets its own run directory (``<name>_seed<k>``), so no run
        overwrites another and every individual result stays inspectable.
        """
        results: List[RunResult] = []
        for seed in seeds:
            seed_cfg = RunConfig.from_dict(cfg.to_dict())
            seed_cfg.seed = int(seed)
            run_dir = Path(cfg.train.project) / f"{cfg.train.name}_seed{seed}"
            if verbose:
                print(f"\n=== seed {seed} -> {run_dir} ===")
            results.append(self.train_fn(seed_cfg, run_dir))

        aggregate = aggregate_results(results, list(seeds))
        if verbose:
            print(f"\n=== {len(results)} seeds, mean ± std ===")
            print(aggregate.format())
        return aggregate


def aggregate_results(results: Sequence[RunResult], seeds: Sequence[int]) -> AggregateResult:
    """Aggregate per-seed best metrics into means and standard deviations."""
    keys = sorted(
        {k for r in results for k, v in r.best_metrics.items() if isinstance(v, (int, float))}
    )
    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    for key in keys:
        values = [float(r.best_metrics[key]) for r in results if key in r.best_metrics]
        if not values:
            continue
        mean[key] = statistics.fmean(values)
        std[key] = statistics.stdev(values) if len(values) > 1 else 0.0
    return AggregateResult(seeds=list(seeds), mean=mean, std=std, runs=list(results))
