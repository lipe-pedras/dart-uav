"""Export orchestration (FR-5.1 – FR-5.3)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Optional

import torch.nn as nn

from ..errors import ExportError, ExportVerificationError
from .backends import BACKENDS, ExportBackend, VerifyReport, available_formats


class ExportManager:
    """Fuses, exports, and verifies — in that order, every time.

    Fusion happens automatically before the backend is invoked (FR-5.3), so a
    user can never accidentally export the multi-branch training graph. The
    fusion runs on a deep copy, leaving the caller's model trainable.
    """

    def __init__(self, backend: Optional[ExportBackend] = None) -> None:
        self._override = backend

    def backend_for(self, fmt: str) -> ExportBackend:
        """Resolve a format name to its backend."""
        if self._override is not None:
            return self._override
        key = str(fmt).lower()
        if key not in BACKENDS:
            raise ExportError(
                f"unknown export format '{fmt}'. Available: {available_formats()}. "
                "Additional backends are registered with "
                "dart.export.register_backend()."
            )
        return BACKENDS[key]

    def export(
        self,
        model: nn.Module,
        fmt: str = "onnx",
        path: Optional[Path] = None,
        verify: bool = True,
        tolerance: float = 1e-4,
        **opts: Any,
    ) -> Path:
        """Export ``model`` and return the artifact path.

        Args:
            model: A trained detector.
            fmt: Registered format name.
            path: Destination; defaults to ``dart-<format>`` beside the CWD
                with the backend's suffix.
            verify: Compare the artifact's outputs against PyTorch's.
            tolerance: Maximum acceptable absolute difference.
            **opts: Backend-specific options (``imgsz``, ``opset``, ...).

        Raises:
            ExportVerificationError: If verification runs and fails. A silently
                wrong artifact is worse than no artifact.
        """
        backend = self.backend_for(fmt)

        deployed = copy.deepcopy(model).eval()
        if hasattr(deployed, "fuse"):
            deployed.fuse()

        target = Path(path) if path else Path(f"dart-{backend.name}{backend.suffix}")
        if target.suffix != backend.suffix and backend.suffix:
            target = target.with_suffix(backend.suffix)

        opts.setdefault("imgsz", getattr(deployed, "imgsz", None))
        opts.setdefault("in_channels", _input_channels(deployed))
        artifact = backend.export(deployed, target, **opts)

        if verify:
            report = backend.verify(deployed, artifact, tolerance=tolerance, **opts)
            self._raise_if_diverged(report, artifact)
            print(f"  Export verified: {report.message}")
        return artifact

    def verify(
        self, model: nn.Module, artifact: Path, fmt: str = "onnx", **opts: Any
    ) -> VerifyReport:
        """Verify an existing artifact against a model."""
        backend = self.backend_for(fmt)
        deployed = copy.deepcopy(model).eval()
        if hasattr(deployed, "fuse"):
            deployed.fuse()
        opts.setdefault("imgsz", getattr(deployed, "imgsz", None))
        opts.setdefault("in_channels", _input_channels(deployed))
        return backend.verify(deployed, Path(artifact), **opts)

    @staticmethod
    def _raise_if_diverged(report: VerifyReport, artifact: Path) -> None:
        if report.skipped:
            print(f"  {report.message}")
            return
        if not report.passed:
            raise ExportVerificationError(
                f"{artifact}: {report.message}. The exported model is not "
                "numerically equivalent to the PyTorch model; do not deploy it."
            )


def _input_channels(model: nn.Module) -> int:
    cfg = getattr(model, "cfg", None)
    backbone = getattr(cfg, "backbone", None)
    return getattr(backbone, "in_channels", 3) if backbone else 3
