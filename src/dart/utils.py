"""Cross-cutting helpers: seeding, device selection, environment capture.

All filesystem work in DART goes through :mod:`pathlib` (NFR-3); there is no
string path concatenation and no hardcoded separator anywhere in the package.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG DART touches (FR-3.5).

    **Controlled:** Python's ``random``, NumPy, PyTorch CPU and CUDA,
    the DataLoader shuffling generator, and cuDNN algorithm selection when
    ``deterministic`` is set.

    **Not controlled:** non-deterministic CUDA kernels that PyTorch has no
    deterministic implementation for (these raise rather than silently vary
    when ``torch.use_deterministic_algorithms`` is enabled — DART does not
    enable it, because it would make some models untrainable); atomics in
    third-party ops; and the order in which multiple DataLoader workers finish,
    which does not affect results because batches are assembled by index.

    Args:
        seed: The seed.
        deterministic: Ask cuDNN for deterministic algorithms, trading some
            throughput for run-to-run reproducibility (NFR-2).
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not deterministic


def select_device(spec: str = "auto") -> torch.device:
    """Resolve a device string.

    Args:
        spec: ``"auto"`` (CUDA when available, else MPS, else CPU), ``"cpu"``,
            ``"cuda"``, ``"cuda:1"``, or ``"mps"``.
    """
    spec = str(spec).lower()
    if spec in ("auto", ""):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def environment_record() -> Dict[str, Any]:
    """Capture what is needed to explain a metric discrepancy (FR-7.1)."""
    record: Dict[str, Any] = {
        "dart_version": _package_version(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
        ),
    }
    for name in ("torchvision", "torchmetrics", "supervision", "cv2", "onnx"):
        record[name] = _module_version(name)
    if torch.cuda.is_available():
        record["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return record


def unique_dir(base: Path, name: str) -> Path:
    """Return ``base/name``, suffixed with ``2``, ``3``, ... if it is taken.

    Runs are never silently overwritten: a re-run of the same command produces
    ``exp2`` beside ``exp``, so the earlier run's metrics survive.
    """
    base = Path(base)
    candidate = base / name
    index = 2
    while candidate.exists():
        candidate = base / f"{name}{index}"
        index += 1
    return candidate


def _package_version() -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("dart-uav")
        except PackageNotFoundError:
            return None
    except ImportError:  # pragma: no cover - Python < 3.8 only
        return None


def _module_version(name: str) -> Optional[str]:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)
