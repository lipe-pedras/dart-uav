"""Export backends (FR-5.1, FR-5.2).

Adding a target — TFLite, NCNN, direct TensorRT — means implementing
:class:`ExportBackend` and registering it. It never means changing
``ExportManager``'s signature, ``DART.export()``, or any model code.

``verify()`` is part of the interface, not an optional extra: an export that
silently produces a numerically different model is worse than a failed export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ..errors import ExportError


@dataclass
class VerifyReport:
    """Numerical comparison of an exported artifact against the PyTorch model.

    Attributes:
        passed: Whether every output stayed within tolerance.
        max_abs_diff: Largest absolute difference over all outputs.
        tolerance: The tolerance that was applied.
        details: Per-output differences.
        skipped: Set when verification could not run (e.g. no runtime
            installed); ``passed`` is then ``True`` but the check was not made.
        message: Human-readable explanation.
    """

    passed: bool = True
    max_abs_diff: float = 0.0
    tolerance: float = 1e-4
    details: Dict[str, float] = field(default_factory=dict)
    skipped: bool = False
    message: str = ""


class ExportBackend:
    """Interface every export target implements."""

    #: Registered format name, e.g. ``"onnx"``.
    name = "base"
    #: File suffix of the produced artifact.
    suffix = ""

    def export(self, model: nn.Module, path: Path, **opts: Any) -> Path:
        """Write the artifact and return its path."""
        raise NotImplementedError

    def verify(self, model: nn.Module, artifact: Path, **opts: Any) -> VerifyReport:
        """Compare the artifact's outputs against the PyTorch model's."""
        raise NotImplementedError


class ONNXBackend(ExportBackend):
    """Export to ONNX — the interchange format all three paper-validated
    embedded targets consume (ONNX Runtime on the Raspberry Pis, TensorRT via
    ONNX on the Jetson).
    """

    name = "onnx"
    suffix = ".onnx"

    def export(
        self,
        model: nn.Module,
        path: Path,
        imgsz: Optional[int] = None,
        in_channels: int = 3,
        opset: Optional[int] = None,
        dynamic_batch: bool = True,
        simplify: bool = False,
        **_: Any,
    ) -> Path:
        """Trace the model and write an ONNX graph.

        Args:
            model: The (already fused) detector.
            path: Destination ``.onnx`` path.
            imgsz: Input resolution; defaults to the model's own.
            in_channels: Input channel count.
            opset: ONNX opset version. ``None`` uses the exporter's own
                default, which is what the Dynamo exporter emits natively;
                pinning an older opset forces a graph down-conversion that can
                fail on the pooling ops DART's heads use.
            dynamic_batch: Mark the batch dimension dynamic, so one artifact
                serves both single-image and batched inference.
            simplify: Run ``onnxsim`` if it is installed.
        """
        imgsz = imgsz or getattr(model, "imgsz", 96)
        model = model.eval()
        # Trace with a batch of 2 when the batch axis is meant to stay dynamic:
        # exporters specialise dimensions of size 0 or 1 to a constant, which
        # would silently bake batch=1 into every reshape in the graph.
        batch = 2 if dynamic_batch else 1
        dummy = torch.zeros(batch, in_channels, imgsz, imgsz, device=_device_of(model))

        path.parent.mkdir(parents=True, exist_ok=True)
        last_exc: Optional[Exception] = None
        for kwargs in _exporter_kwargs_attempts(dynamic_batch):
            try:
                torch.onnx.export(
                    model,
                    dummy,
                    str(path),
                    input_names=["images"],
                    output_names=["conf", "boxes"],
                    do_constant_folding=True,
                    **({"opset_version": opset} if opset else {}),
                    **kwargs,
                )
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001 - retried, then surfaced with context
                last_exc = exc
        if last_exc is not None:
            raise ExportError(f"ONNX export failed: {last_exc}") from last_exc

        if simplify:
            _try_simplify(path)
        return path

    def verify(
        self,
        model: nn.Module,
        artifact: Path,
        imgsz: Optional[int] = None,
        in_channels: int = 3,
        tolerance: float = 1e-4,
        **_: Any,
    ) -> VerifyReport:
        """Run both models on a fixed input and compare outputs elementwise."""
        try:
            import onnxruntime as ort
        except ImportError:
            return VerifyReport(
                passed=True,
                skipped=True,
                tolerance=tolerance,
                message=(
                    "onnxruntime is not installed, so the exported model was not "
                    "verified. Install dart-uav[onnx] to enable verification."
                ),
            )

        imgsz = imgsz or getattr(model, "imgsz", 96)
        generator = torch.Generator().manual_seed(0)
        sample = torch.randn(1, in_channels, imgsz, imgsz, generator=generator)

        model = model.eval()
        with torch.no_grad():
            reference = model(sample.to(_device_of(model)))
        reference_np = [t.detach().cpu().numpy() for t in reference]

        session = ort.InferenceSession(str(artifact), providers=["CPUExecutionProvider"])
        outputs = session.run(None, {session.get_inputs()[0].name: sample.numpy()})

        details: Dict[str, float] = {}
        worst = 0.0
        names = ["conf", "boxes"]
        for i, (expected, actual) in enumerate(zip(reference_np, outputs)):
            diff = float(np.abs(expected - np.asarray(actual)).max())
            details[names[i] if i < len(names) else f"output{i}"] = diff
            worst = max(worst, diff)

        passed = worst <= tolerance
        return VerifyReport(
            passed=passed,
            max_abs_diff=worst,
            tolerance=tolerance,
            details=details,
            message=(
                f"max |onnx - torch| = {worst:.2e} (tolerance {tolerance:.0e})"
                if passed
                else f"outputs diverge: max |onnx - torch| = {worst:.2e} > {tolerance:.0e}"
            ),
        )


#: Registered export backends, by format name.
BACKENDS: Dict[str, ExportBackend] = {"onnx": ONNXBackend()}


def register_backend(backend: ExportBackend) -> ExportBackend:
    """Register a new export target."""
    BACKENDS[backend.name.lower()] = backend
    return backend


def available_formats() -> List[str]:
    """Names of the registered export formats."""
    return sorted(BACKENDS)


#: How a dynamic batch dimension is declared to the TorchScript exporter.
_LEGACY_DYNAMIC_AXES = {
    "images": {0: "batch"},
    "conf": {0: "batch"},
    "boxes": {0: "batch"},
}


def _exporter_kwargs_attempts(dynamic_batch: bool) -> List[Dict[str, Any]]:
    """Pick the ONNX exporter dialect(s) to try, in order.

    PyTorch 2.6+ defaults ``torch.onnx.export`` to the Dynamo exporter, which
    needs the optional ``onnxscript`` package and declares dynamism through
    ``dynamic_shapes``. Without ``onnxscript``, fall back to the TorchScript
    exporter and its ``dynamic_axes``. Older PyTorch releases have no
    ``dynamo`` parameter at all.

    The two are not interchangeable for DART: the TorchScript exporter cannot
    trace ``AdaptiveAvgPool2d`` over a spatially-dynamic input, which the
    FPN-Lite neck produces, so ``onnxscript`` is what the ``[onnx]`` extra
    installs. Even so, the Dynamo exporter has shown platform-specific
    breakage (e.g. onnxscript/torch mismatches on Windows) where it resolves
    to ``dynamo=False`` internally despite being asked for ``dynamic_shapes``,
    which torch itself then rejects. Rather than trust that resolution, try
    the Dynamo dialect first and fall back to the explicit legacy dialect on
    any failure.
    """
    import inspect

    legacy = {"dynamic_axes": _LEGACY_DYNAMIC_AXES} if dynamic_batch else {}

    if "dynamo" not in inspect.signature(torch.onnx.export).parameters:
        return [legacy]

    try:
        import onnxscript  # noqa: F401
    except ImportError:
        return [{"dynamo": False, **legacy}]

    if not dynamic_batch:
        return [{}]
    # Dim.AUTO lets the exporter infer the batch dimension's bounds. Naming a
    # plain Dim instead leaves it unbounded, and the shape solver then rejects
    # the graph over guards the pooling ops generate.
    batch_dim = getattr(torch.export.Dim, "AUTO", None)
    if batch_dim is None:  # pragma: no cover - PyTorch < 2.6
        batch_dim = torch.export.Dim("batch", min=1, max=65535)
    return [{"dynamo": True, "dynamic_shapes": ({0: batch_dim},)}, {"dynamo": False, **legacy}]


def _device_of(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:  # pragma: no cover - a model with no parameters
        return torch.device("cpu")


def _try_simplify(path: Path) -> None:
    try:
        import onnx
        import onnxsim
    except ImportError:
        return
    model = onnx.load(str(path))
    simplified, ok = onnxsim.simplify(model)
    if ok:
        onnx.save(simplified, str(path))
