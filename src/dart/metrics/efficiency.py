"""Efficiency metrics: parameters, size, latency (FR-4.5).

These are first-class members of the metrics schema, not an afterthought — the
paper flags them as the columns missing from ``model_comparison.csv``.

The profiler always runs on the **fused** model. Parameter count and latency
measured before RepConv reparameterization overstate both, because the
multi-branch training graph is not what deploys.
"""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class LatencyStats:
    """Single-image inference latency, in milliseconds."""

    mean_ms: float = 0.0
    std_ms: float = 0.0
    min_ms: float = 0.0
    p50_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    max_ms: float = 0.0
    fps: float = 0.0
    device: str = "cpu"
    batch_size: int = 1


@dataclass
class EfficiencyResult:
    """Everything the efficiency columns of the metrics schema need."""

    params: int = 0
    params_k: float = 0.0
    fp32_size_kb: float = 0.0
    int8_size_kb: float = 0.0
    imgsz: int = 0
    latency: Optional[LatencyStats] = None
    extras: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        data = asdict(self)
        latency = data.pop("latency") or {}
        extras = data.pop("extras")
        data.update({f"latency_{k}": v for k, v in latency.items()})
        data.update(extras)
        return data


class EfficiencyProfiler:
    """Measures parameter count, quantized size, and latency.

    Args:
        warmup: Untimed forward passes before measurement.
        runs: Timed forward passes.
        batch_size: Batch size used for the latency measurement.
    """

    def __init__(self, warmup: int = 20, runs: int = 100, batch_size: int = 1) -> None:
        self.warmup = int(warmup)
        self.runs = int(runs)
        self.batch_size = int(batch_size)

    def profile(
        self,
        model: nn.Module,
        imgsz: Optional[int] = None,
        in_channels: int = 3,
        device: Optional[torch.device] = None,
        measure_latency: bool = True,
    ) -> EfficiencyResult:
        """Profile a model, fusing a copy of it first.

        The model handed in is never mutated: a deep copy is fused, so calling
        the profiler mid-training cannot destroy the training graph.
        """
        device = device or next(model.parameters()).device
        imgsz = imgsz or getattr(model, "imgsz", 96)

        deployed = copy.deepcopy(model).to(device).eval()
        if hasattr(deployed, "fuse"):
            deployed.fuse()

        params = sum(p.numel() for p in deployed.parameters())
        result = EfficiencyResult(
            params=params,
            params_k=round(params / 1e3, 2),
            fp32_size_kb=round(params * 4 / 1024, 2),
            # INT8 post-training quantization stores one byte per weight; the
            # same estimate the paper reports.
            int8_size_kb=round(params / 1024, 2),
            imgsz=imgsz,
        )
        if measure_latency:
            result.latency = self.measure_latency(deployed, imgsz, in_channels, device)
        return result

    @torch.no_grad()
    def measure_latency(
        self,
        model: nn.Module,
        imgsz: int,
        in_channels: int = 3,
        device: Optional[torch.device] = None,
    ) -> LatencyStats:
        """Time repeated forward passes on synthetic input."""
        device = device or next(model.parameters()).device
        model.eval()
        dummy = torch.randn(self.batch_size, in_channels, imgsz, imgsz, device=device)

        for _ in range(self.warmup):
            model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        timings = []
        for _ in range(self.runs):
            start = time.perf_counter()
            model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)

        samples = np.asarray(timings, dtype=np.float64)
        mean = float(samples.mean())
        return LatencyStats(
            mean_ms=mean,
            std_ms=float(samples.std()),
            min_ms=float(samples.min()),
            p50_ms=float(np.percentile(samples, 50)),
            p90_ms=float(np.percentile(samples, 90)),
            p95_ms=float(np.percentile(samples, 95)),
            p99_ms=float(np.percentile(samples, 99)),
            max_ms=float(samples.max()),
            fps=1000.0 / mean if mean > 0 else 0.0,
            device=str(device),
            batch_size=self.batch_size,
        )
