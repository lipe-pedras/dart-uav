"""Command-line interface (FR-6.2).

Mirrors the Python API one-to-one::

    dart train  data=my_dataset model=dart-xd epochs=300 imgsz=320
    dart val    data=my_dataset model=runs/exp/weights/best.pt
    dart predict source=image.jpg model=runs/exp/weights/best.pt
    dart export model=runs/exp/weights/best.pt format=onnx

Arguments are ``key=value`` pairs, so anything the API accepts as a keyword the
CLI accepts too, including dotted config paths such as
``train.early_stop.patience=25``.

Every entry point is guarded with ``if __name__ == "__main__"`` (NFR-3), and
the console script installed by the package calls :func:`main` directly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .api import DART
from .config import ConfigResolver
from .engine import aggregate_runs
from .errors import DARTError

USAGE = """dart — lightweight detection for resource-constrained UAV navigation

Usage:
  dart train    data=<path> [model=dart-xd] [epochs=N] [imgsz=N] [key=value ...]
  dart val      data=<path> model=<checkpoint> [key=value ...]
  dart predict  source=<image|dir> model=<checkpoint> [conf=0.5]
  dart export   model=<checkpoint> [format=onnx] [path=out.onnx]
  dart profile  model=<checkpoint|variant> [imgsz=N]
  dart report   runs=<dir> [out=model_comparison.csv]
  dart variants
  dart --help

Common keys:
  model            variant name, config YAML, or checkpoint path
  data             dataset root or dataset YAML
  target_policy    largest_only | keep_all      (required for multi-annotation data)
  target_class     class id to detect           (required for multi-class data)
  epochs, batch, lr, imgsz, seed, width, workers, device, project, name

Any config path also works verbatim, e.g. train.early_stop.patience=25.
"""

TYPED_TRUE = {"true", "yes", "1", "on"}
TYPED_FALSE = {"false", "no", "0", "off"}


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Returns a process exit code."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(USAGE)
        return 0

    command, *rest = argv
    try:
        kwargs = parse_args(rest)
    except ValueError as exc:
        print(f"error: {exc}\n\n{USAGE}", file=sys.stderr)
        return 2

    handlers = {
        "train": _train,
        "val": _val,
        "predict": _predict,
        "export": _export,
        "profile": _profile,
        "report": _report,
        "variants": _variants,
    }
    if command not in handlers:
        print(f"error: unknown command '{command}'\n\n{USAGE}", file=sys.stderr)
        return 2

    try:
        return handlers[command](kwargs)
    except DARTError as exc:
        # DART's own errors are actionable by design; print them plainly rather
        # than burying the message in a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 1


def parse_args(tokens: List[str]) -> Dict[str, Any]:
    """Parse ``key=value`` tokens, coercing obvious literals."""
    kwargs: Dict[str, Any] = {}
    for token in tokens:
        if "=" not in token:
            raise ValueError(f"expected key=value, got '{token}'")
        key, _, value = token.partition("=")
        kwargs[key.strip()] = _coerce(value.strip())
    return kwargs


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in TYPED_TRUE:
        return True
    if lowered in TYPED_FALSE:
        return False
    if lowered in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [_coerce(v.strip()) for v in inner.split(",")] if inner else []
    return value


# ── Commands ────────────────────────────────────────────────────────────


def _train(kwargs: Dict[str, Any]) -> int:
    model_spec = kwargs.pop("model", "dart-xd")
    seeds = kwargs.pop("seeds", None)
    model = DART(model_spec)
    if seeds is not None:
        seed_list = seeds if isinstance(seeds, list) else [seeds]
        model.train_seeds([int(s) for s in seed_list], **kwargs)
        return 0
    model.train(**kwargs)
    return 0


def _val(kwargs: Dict[str, Any]) -> int:
    model = DART(kwargs.pop("model", "dart-xd"))
    split = kwargs.pop("split", "val")
    result = model.val(split=split, **kwargs)
    for key, value in result.to_dict().items():
        if isinstance(value, float):
            print(f"  {key:<16} {value:.4f}")
    return 0


def _predict(kwargs: Dict[str, Any]) -> int:
    source = kwargs.pop("source", None)
    if source is None:
        raise ValueError("predict needs source=<image or directory>")
    model = DART(kwargs.pop("model", "dart-xd"))
    for result in model.predict(source, conf=kwargs.pop("conf", None)):
        if not result.detections:
            print(f"  {result.source}: no detection")
            continue
        for det in result.detections:
            box = ", ".join(f"{v:.1f}" for v in det.box_xyxy)
            print(f"  {result.source}: conf={det.confidence:.3f} xyxy=[{box}]")
    return 0


def _export(kwargs: Dict[str, Any]) -> int:
    model = DART(kwargs.pop("model", "dart-xd"))
    fmt = kwargs.pop("format", "onnx")
    path = kwargs.pop("path", None)
    artifact = model.export(format=fmt, path=Path(path) if path else None, **kwargs)
    print(f"  Exported to {artifact}")
    return 0


def _profile(kwargs: Dict[str, Any]) -> int:
    model = DART(kwargs.pop("model", "dart-xd"), **kwargs)
    result = model.profile()
    print(f"  params        {result.params:,} ({result.params_k}K)")
    print(f"  fp32 size     {result.fp32_size_kb} KB")
    print(f"  int8 size     {result.int8_size_kb} KB")
    if result.latency:
        print(f"  latency       {result.latency.mean_ms:.3f} ms (p95 {result.latency.p95_ms:.3f})")
        print(f"  throughput    {result.latency.fps:.1f} FPS on {result.latency.device}")
    return 0


def _report(kwargs: Dict[str, Any]) -> int:
    runs = Path(kwargs.pop("runs", "runs"))
    out = kwargs.pop("out", None)
    path = aggregate_runs(runs, Path(out) if out else None)
    print(f"  Wrote {path}")
    return 0


def _variants(_: Dict[str, Any]) -> int:
    for name in ConfigResolver().available_presets():
        print(f"  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
