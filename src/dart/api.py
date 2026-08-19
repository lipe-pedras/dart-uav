"""The public facade (FR-6.1).

Modelled on the Ultralytics ergonomic pattern, deliberately::

    from dart import DART

    model = DART("dart-xd")
    model.train(data="my_dataset", epochs=300, imgsz=320, target_policy="largest_only")
    metrics = model.val(data="my_dataset")
    results = model.predict("image.jpg")
    model.export(format="onnx")

Everything the facade does is available piecewise underneath
(:class:`~dart.engine.trainer.Trainer`,
:class:`~dart.engine.evaluator.Evaluator`,
:class:`~dart.export.manager.ExportManager`); the facade exists for brevity,
not to hide the machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from .config import ConfigResolver, RunConfig
from .data import DatasetLoader, build_dataloader
from .data.dataset import GRAY_MEAN, GRAY_STD, IMAGENET_MEAN, IMAGENET_STD
from .engine import Evaluator, MultiSeedRunner, RunResult, Trainer
from .engine.multiseed import AggregateResult
from .errors import CheckpointError, ConfigurationError
from .export import ExportManager
from .losses import CompositeLoss
from .metrics import EfficiencyResult, MetricResult
from .models import DARTDetector
from .utils import select_device, set_seed

CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt")


@dataclass
class Detection:
    """One predicted box.

    Attributes:
        confidence: Objectness in ``[0, 1]``.
        box: ``[cx, cy, w, h]`` normalised to the input image.
        box_xyxy: ``[x1, y1, x2, y2]`` in the source image's pixel coordinates.
    """

    confidence: float
    box: List[float]
    box_xyxy: List[float]


@dataclass
class PredictResult:
    """Predictions for one image.

    Attributes:
        source: The image path, or ``"<array>"`` for in-memory input.
        detections: Boxes above the confidence threshold.
        raw_confidence: Every predicted confidence, thresholded or not.
    """

    source: str
    detections: List[Detection]
    raw_confidence: List[float]

    def __len__(self) -> int:
        return len(self.detections)

    @property
    def top(self) -> Optional[Detection]:
        """The highest-confidence detection, or ``None`` if there is none."""
        return max(self.detections, key=lambda d: d.confidence, default=None)


class DART:
    """A DART model: architecture, weights, and the operations on them.

    Args:
        model: A variant name (``"dart-xd"``), a path to a checkpoint
            (``"runs/exp/weights/best.pt"``), a path to a YAML config, or a
            ``"repo_id:file.pt"`` Hugging Face Hub reference. Checkpoints not
            present locally are resolved through the Hub (FR-2.3).
        **overrides: Any configuration override, e.g. ``imgsz=320``,
            ``width=0.5``.
    """

    def __init__(self, model: str = "dart-xd", **overrides: Any) -> None:
        self._resolver = ConfigResolver()
        self.checkpoint: Optional[Path] = None
        self._state_dict: Optional[Dict[str, torch.Tensor]] = None

        spec: Any = model
        if _looks_like_checkpoint(model):
            self.checkpoint = _resolve_checkpoint(model)
            payload = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            if "config" not in payload or "model_state" not in payload:
                raise CheckpointError(
                    f"{self.checkpoint} is not a DART checkpoint (expected "
                    "'config' and 'model_state' keys)."
                )
            spec = payload["config"]
            self._state_dict = payload["model_state"]

        self.cfg: RunConfig = self._resolver.resolve(spec, **overrides)
        self.device = select_device(self.cfg.train.device)
        self._built_seed: Optional[int] = None
        self._build_model()

    # ── Training (FR-3.1) ───────────────────────────────────────────────

    def train(self, data: Optional[str] = None, **overrides: Any) -> RunResult:
        """Train on ``data``.

        Args:
            data: Dataset root or dataset YAML. May also be given as
                ``data=`` inside ``overrides``.
            **overrides: Any configuration override — ``epochs``, ``imgsz``,
                ``batch``, ``lr``, ``seed``, ``target_policy``,
                ``target_class``, ``patience``, ``project``, ``name``, ...

        Returns:
            The :class:`~dart.engine.trainer.RunResult`, whose ``run_dir``
            holds the resolved config, metrics, and checkpoints.
        """
        cfg = self._reconfigure(data, overrides)
        loader = DatasetLoader()
        train_data = loader.load(cfg.data, cfg.model, split="train")
        val_data = _maybe_val_split(loader, cfg)

        trainer = Trainer(cfg)
        result = trainer.fit(self.model, train_data, val_data)
        return result

    def train_seeds(
        self, seeds: Sequence[int], data: Optional[str] = None, **overrides: Any
    ) -> AggregateResult:
        """Train the same configuration across several seeds (FR-4.3).

        Returns:
            Per-metric mean ± standard deviation, plus the individual runs.
        """
        cfg = self._reconfigure(data, overrides)
        runner = MultiSeedRunner(_SeedTrainer())
        return runner.run(cfg, seeds)

    # ── Evaluation (FR-4.2) ─────────────────────────────────────────────

    def val(self, data: Optional[str] = None, split: str = "val", **overrides: Any) -> MetricResult:
        """Evaluate the current weights without training.

        Args:
            data: Dataset root or dataset YAML.
            split: Which split to evaluate (``"val"`` or ``"train"``).
            **overrides: Configuration overrides, e.g. ``batch``, ``conf``.
        """
        cfg = self._reconfigure(data, overrides)
        dataset = DatasetLoader().load(cfg.data, cfg.model, split=split, augment=False)
        loader = build_dataloader(
            dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            workers=cfg.data.workers,
            pin_memory=self.device.type == "cuda",
        )
        evaluator = Evaluator(
            conf_threshold=cfg.train.conf_threshold,
            loss=CompositeLoss(cfg.train.loss),
            device=self.device,
            amp=cfg.train.amp,
        )
        return evaluator.evaluate(self.model, loader, imgsz=cfg.model.imgsz, verbose=True)

    # ── Inference ───────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        source: Union[str, Path, np.ndarray, Sequence[Union[str, Path]]],
        conf: Optional[float] = None,
    ) -> List[PredictResult]:
        """Run inference on an image, a list of images, or a directory.

        Args:
            source: Image path, directory of images, ``HxWxC`` BGR array, or a
                sequence of paths.
            conf: Confidence threshold; defaults to ``train.conf_threshold``.

        Returns:
            One :class:`PredictResult` per input image.
        """
        import cv2

        threshold = self.cfg.train.conf_threshold if conf is None else float(conf)
        self.model.eval()

        results: List[PredictResult] = []
        for name, image in _iter_sources(source):
            original_h, original_w = image.shape[:2]
            tensor = self._preprocess(image, cv2).to(self.device)
            pred_conf, pred_boxes = self.model(tensor)

            confidences = pred_conf.reshape(-1).float().cpu().tolist()
            boxes = pred_boxes.reshape(-1, 4).float().cpu().tolist()
            detections = [
                Detection(
                    confidence=score,
                    box=list(box),
                    box_xyxy=_to_pixel_xyxy(box, original_w, original_h),
                )
                for score, box in zip(confidences, boxes)
                if score >= threshold
            ]
            results.append(
                PredictResult(source=name, detections=detections, raw_confidence=confidences)
            )
        return results

    # ── Export (FR-5.1) ─────────────────────────────────────────────────

    def export(
        self,
        format: str = "onnx",
        path: Optional[Union[str, Path]] = None,
        verify: bool = True,
        **opts: Any,
    ) -> Path:
        """Export the model; RepConv fusion happens automatically (FR-5.3)."""
        manager = ExportManager()
        target = Path(path) if path else _default_export_path(self, format)
        return manager.export(
            self.model, fmt=format, path=target, verify=verify, imgsz=self.cfg.model.imgsz, **opts
        )

    # ── Introspection ───────────────────────────────────────────────────

    def profile(self, **kwargs: Any) -> EfficiencyResult:
        """Measure parameters, quantized size, and latency (FR-4.5)."""
        from .metrics import EfficiencyProfiler

        return EfficiencyProfiler(**kwargs).profile(
            self.model,
            imgsz=self.cfg.model.imgsz,
            in_channels=self.cfg.model.backbone.in_channels,
            device=self.device,
        )

    def info(self) -> Dict[str, Any]:
        """Parameter count, size estimate, and the architectural axes."""
        return self.model.model_info()

    def save(self, path: Union[str, Path]) -> Path:
        """Write a checkpoint that carries its own resolved config."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "config": self.cfg.to_dict(),
                "fingerprint": self.cfg.fingerprint(),
                "epoch": 0,
            },
            path,
        )
        return path

    @staticmethod
    def variants() -> List[str]:
        """The paper-validated variant names shipped as presets."""
        return ConfigResolver().available_presets()

    def __repr__(self) -> str:
        return f"DART(variant='{self.cfg.variant}', {self.model!r})"

    # ── Internals ───────────────────────────────────────────────────────

    def _reconfigure(self, data: Optional[str], overrides: Dict[str, Any]) -> RunConfig:
        """Apply call-site overrides on top of the model's current config.

        Rebuilding the model is only necessary when the architecture itself
        changed; weights are preserved otherwise, which is what lets ``val()``
        follow ``train()`` without reloading anything.
        """
        if data is not None:
            overrides.setdefault("data", data)
        if not overrides:
            self._require_data(self.cfg)
            return self.cfg

        cfg = self._resolver.resolve(self.cfg, **overrides)
        self._require_data(cfg)

        architecture_changed = cfg.model != self.cfg.model
        seed_changed = cfg.seed != self._built_seed

        self.cfg = cfg
        self.device = select_device(cfg.train.device)

        if architecture_changed or seed_changed:
            # Random initialisation is part of what a seed controls, so a
            # changed seed means a freshly-initialised model — otherwise "same
            # seed and config" would not imply the same result (NFR-2). A
            # changed architecture cannot keep its old weights either.
            self._build_model()
        else:
            self.model.to(self.device)
        return cfg

    def _build_model(self) -> None:
        """(Re)build the detector under the configured seed."""
        set_seed(self.cfg.seed)
        self._built_seed = self.cfg.seed
        self.model = DARTDetector(self.cfg.model)
        if self._state_dict is not None:
            self.model.load_state_dict(self._state_dict)
        self.model.to(self.device)

    @staticmethod
    def _require_data(cfg: RunConfig) -> None:
        if not cfg.data.path:
            raise ConfigurationError(
                "no dataset configured — pass data='<path to dataset or data.yaml>'"
            )

    def _preprocess(self, image: np.ndarray, cv2) -> torch.Tensor:
        """Resize, convert, and normalise one image exactly as training does."""
        grayscale = self.cfg.data.grayscale
        if grayscale:
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image[:, :, None]
        else:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        imgsz = self.cfg.model.imgsz
        resized = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = resized[:, :, None]

        pixels = resized.astype(np.float32) / 255.0
        mean = GRAY_MEAN if grayscale else IMAGENET_MEAN
        std = GRAY_STD if grayscale else IMAGENET_STD
        pixels = (pixels - mean) / std
        return torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1).unsqueeze(0).float()


class _SeedTrainer:
    """Callable that trains a freshly-initialised model for one seed."""

    def __call__(self, cfg: RunConfig, run_dir: Path) -> RunResult:
        loader = DatasetLoader()
        train_data = loader.load(cfg.data, cfg.model, split="train", verbose=False)
        val_data = _maybe_val_split(loader, cfg, verbose=False)
        set_seed(cfg.seed)
        model = DARTDetector(cfg.model)
        return Trainer(cfg, run_dir=run_dir).fit(model, train_data, val_data, verbose=False)


def _maybe_val_split(loader: DatasetLoader, cfg: RunConfig, verbose: bool = True):
    """Load the validation split, tolerating datasets that have none."""
    try:
        return loader.load(cfg.data, cfg.model, split="val", augment=False, verbose=verbose)
    except Exception as exc:  # noqa: BLE001 - degraded but explicit
        print(
            f"  Warning: no usable validation split ({exc}). "
            "Falling back to the training split — reported metrics are NOT held out."
        )
        return None


def _looks_like_checkpoint(spec: str) -> bool:
    text = str(spec)
    if any(text.endswith(suffix) for suffix in CHECKPOINT_SUFFIXES):
        return True
    return ":" in text and Path(text.split(":")[-1]).suffix in CHECKPOINT_SUFFIXES


def _resolve_checkpoint(spec: str) -> Path:
    """Resolve a checkpoint locally, falling back to the Hugging Face Hub."""
    path = Path(spec).expanduser()
    if path.exists():
        return path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise CheckpointError(
            f"checkpoint '{spec}' not found locally and huggingface-hub is not installed"
        ) from exc

    repo_id, _, filename = str(spec).partition(":")
    if not filename:
        raise CheckpointError(
            f"checkpoint '{spec}' not found locally. For a Hub checkpoint use "
            "'repo_id:filename.pt'."
        )
    try:
        return Path(hf_hub_download(repo_id=repo_id, filename=filename))
    except Exception as exc:  # noqa: BLE001 - network/permission errors vary
        raise CheckpointError(f"could not download '{spec}' from the Hub: {exc}") from exc


def _default_export_path(model: DART, fmt: str) -> Path:
    stem = model.checkpoint.stem if model.checkpoint else model.cfg.variant
    parent = model.checkpoint.parent if model.checkpoint else Path.cwd()
    return parent / f"{stem}.{fmt}"


def _iter_sources(source):
    """Yield ``(name, BGR array)`` pairs from whatever the caller passed."""
    import cv2

    from .data.adapters import IMAGE_SUFFIXES

    if isinstance(source, np.ndarray):
        yield "<array>", source
        return

    paths: List[Path]
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            paths = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        else:
            paths = [path]
    else:
        paths = [Path(p) for p in source]

    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"cannot read image: {path}")
        yield str(path), image


def _to_pixel_xyxy(box: Sequence[float], width: int, height: int) -> List[float]:
    cx, cy, w, h = box
    return [
        (cx - w / 2) * width,
        (cy - h / 2) * height,
        (cx + w / 2) * width,
        (cy + h / 2) * height,
    ]
