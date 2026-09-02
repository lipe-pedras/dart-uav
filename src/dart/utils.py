"""
Cross-cutting helpers: seeding, device selection, environment capture.

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
    """
    Seed every RNG DART touches (FR-3.5).

    **Controlled:** Python's ``random``, NumPy, PyTorch CPU and CUDA,
    the DataLoader shuffling generator, and cuDNN algorithm selection when
    ``deterministic`` is set.

    **Not controlled:** non-deterministic CUDA kernels that PyTorch has no
    deterministic implementation for (these raise rather than silently vary
    when ``torch.use_deterministic_algorithms`` is enabled — DART does not
    enable it, because it would make some models untrainable); atomics in
    third-party ops; and the order in which multiple DataLoader workers finish,
    which does not affect results because batches are assembled by index.

    Parameters
    ----------
    seed : int
        The seed. Coerced with :class:`int`, and also exported as
        ``PYTHONHASHSEED`` so hash ordering is stable in child processes.
    deterministic : bool, default: True
        Ask cuDNN for deterministic algorithms, trading some throughput for
        run-to-run reproducibility (NFR-2). Setting this to ``False`` enables
        cuDNN benchmarking instead, which picks the fastest kernel per shape.

    See Also
    --------
    :class:`~dart.engine.multiseed.MultiSeedRunner` : Repeats a configuration
        across several seeds and aggregates the spread.
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
    """
    Resolve a device string to a concrete :class:`torch.device`.

    Parameters
    ----------
    spec : str, default: "auto"
        ``"auto"`` (CUDA when available, else MPS, else CPU), ``"cpu"``,
        ``"cuda"``, ``"cuda:1"``, or ``"mps"``. Case-insensitive; the empty
        string is treated as ``"auto"``.

    Returns
    -------
    torch.device
        The resolved device. Anything other than ``"auto"`` is passed straight
        to :class:`torch.device`, so an unavailable but well-formed device is
        returned here and fails later at the point of use.
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
    """
    Capture what is needed to explain a metric discrepancy (FR-7.1).

    Returns
    -------
    dict of str to Any
        Interpreter, platform and hardware identification, the versions of
        ``dart`` itself and of every dependency whose behaviour can move a
        metric, and the CUDA and cuDNN versions. Any package that is not
        installed maps to ``None`` rather than being omitted, so two records
        always have the same keys and can be compared field by field. A
        ``gpus`` key listing device names is present only when CUDA is
        available.

    See Also
    --------
    :class:`~dart.engine.recorder.RunRecorder` : Writes this record into each
        run directory.
    """
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
    """
    Return a directory under ``base`` that does not exist yet.

    Runs are never silently overwritten: a re-run of the same command produces
    ``exp2`` beside ``exp``, so the earlier run's metrics survive.

    Parameters
    ----------
    base : ~pathlib.Path
        Parent directory. Not created; only its children are tested for
        existence.
    name : str
        Preferred directory name. Used unchanged when available, otherwise
        suffixed with ``2``, ``3``, and so on until a free name is found.

    Returns
    -------
    ~pathlib.Path
        The first free path, composed with :class:`~pathlib.Path` rather than
        by string concatenation.

    Notes
    -----
    The check is not atomic: two processes racing on the same ``base`` can
    settle on the same name. DART runs one experiment per process, so this is
    not guarded against.
    """
    base = Path(base)
    candidate = base / name
    index = 2
    while candidate.exists():
        candidate = base / f"{name}{index}"
        index += 1
    return candidate


def _package_version() -> Optional[str]:
    """
    Return the installed ``dart-uav`` version, or ``None``.

    Returns
    -------
    str or None
        ``None`` when the package is not installed — which is the normal case
        for a source checkout run without ``pip install`` — so an environment
        record can still be written.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("dart-uav")
        except PackageNotFoundError:
            return None
    except ImportError:  # pragma: no cover - Python < 3.8 only
        return None


def _module_version(name: str) -> Optional[str]:
    """
    Return an importable module's ``__version__``, or ``None``.

    Parameters
    ----------
    name : str
        Import name of the module, not its distribution name — ``"cv2"``, not
        ``"opencv-python"``.

    Returns
    -------
    str or None
        ``None`` if the module is not installed or does not define
        ``__version__``. The two cases are deliberately not distinguished:
        either way the version is unknown.
    """
    try:
        module = __import__(name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)
