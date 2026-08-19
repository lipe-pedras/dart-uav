"""Typed configuration objects.

Plain dataclasses rather than a validation library: the config surface is
small, the framework already depends on enough, and dataclasses serialize to
YAML with no adapter. Every field has a default, so a
:class:`RunConfig` produced by :class:`~dart.config.resolver.ConfigResolver` is
always fully populated — which is what makes the resolved-config snapshot
(FR-3.4) mechanically complete rather than aspirational.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from functools import cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Type, TypeVar, get_type_hints

import yaml

from ..errors import ConfigurationError

T = TypeVar("T")

#: Input resolutions validated by the paper. Other values are allowed but warn.
PAPER_RESOLUTIONS = (96, 160, 224, 320)

#: ``TrainConfig`` fields that say where a run is written, not what it computes.
#: Excluded from :meth:`RunConfig.fingerprint`.
BOOKKEEPING_FIELDS = ("project", "name", "resume", "save_period")


@dataclass
class BlockSpec:
    """One backbone block, as it appears in a config file.

    Attributes:
        type: Registered block name (``repconv``, ``inverted_res``, ``ghost``,
            ``dw_sep``).
        out_ch: Output channels *before* the width multiplier is applied.
        stride: Spatial stride.
        attention: Registered attention name; ``none`` for no attention.
        act: Registered activation name.
        extra: Block-specific keyword arguments (e.g. ``expand_ratio``).
    """

    type: str = "inverted_res"
    out_ch: int = 32
    stride: int = 1
    attention: str = "none"
    act: str = "relu6"
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> BlockSpec:
        known = {f.name for f in fields(cls)} - {"extra"}
        extra = dict(data.get("extra", {}))
        extra.update({k: v for k, v in data.items() if k not in known and k != "extra"})
        return cls(**{k: v for k, v in data.items() if k in known}, extra=extra)


@dataclass
class BackboneConfig:
    """Backbone geometry.

    Attributes:
        width_mult: Channel multiplier applied to every block (FR-2.5).
        blocks: Ordered block specifications.
        tap_indices: Block indices whose outputs feed the neck. The last block
            is always tapped regardless.
        in_channels: 3 for RGB input, 1 for grayscale.
        pretrained: Optional ``state_dict`` path or ``repo_id:file`` Hub
            reference used to initialise the backbone (FR-2.6).
        pretrained_lr_mult: Learning-rate multiplier applied to backbone
            parameters relative to head parameters.
    """

    width_mult: float = 1.0
    blocks: List[BlockSpec] = field(default_factory=list)
    tap_indices: List[int] = field(default_factory=list)
    in_channels: int = 3
    pretrained: Optional[str] = None
    pretrained_lr_mult: float = 1.0


@dataclass
class NeckConfig:
    """Neck selection and its keyword arguments."""

    name: str = "identity"
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HeadConfig:
    """Head selection, target count, and keyword arguments.

    ``num_targets`` is 1 for the single-target task the paper defines; the
    field exists so multi-target becomes a value change plus a policy, not a
    signature change (NFR-8).
    """

    name: str = "gap"
    num_targets: int = 1
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Everything needed to build a :class:`~dart.models.detector.DARTDetector`.

    Attributes:
        imgsz: Square input resolution (FR-2.4).
        act: Default activation for blocks that do not override it.
        num_classes: Detected classes. v1 supports 1; the field exists so
            multi-class is an added head, not a rewrite.
    """

    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    neck: NeckConfig = field(default_factory=NeckConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    imgsz: int = 96
    act: str = "relu6"
    num_classes: int = 1


@dataclass
class AugmentConfig:
    """Augmentation probabilities and magnitudes (FR-1.6).

    Defaults reproduce the research codebase's pipeline exactly. Set
    ``enabled=False`` (the default for validation splits) to disable all of it.
    """

    enabled: bool = True
    hflip: float = 0.5
    vflip: float = 0.5
    brightness: float = 0.7
    brightness_range: Sequence[float] = (0.6, 1.4)
    contrast: float = 0.7
    contrast_range: Sequence[float] = (0.7, 1.3)
    noise: float = 0.6
    noise_sigma: Sequence[float] = (2.0, 12.0)
    crop_resize: float = 0.6
    crop_scale: Sequence[float] = (0.75, 0.98)
    rotate: float = 0.5
    rotate_degrees: float = 15.0
    blur: float = 0.4
    blur_kernels: Sequence[int] = (3, 5)


@dataclass
class DataConfig:
    """Dataset location, interpretation, and preprocessing.

    Attributes:
        path: Dataset root, or a dataset YAML describing the splits.
        format: ``yolo``, ``coco``, ``voc``, ``createml``, or ``auto``.
        train / val: Split sub-paths relative to ``path``.
        target_policy: ``largest_only`` or ``keep_all``. **No default** — an
            unset policy on a multi-annotation dataset is an error (FR-1.3).
        target_class: Class id kept as the detection target. **No default** —
            an unset selector on a multi-class dataset is an error (FR-1.4).
        class_map: Explicit ``old_id: new_id`` remapping applied at ingestion
            (FR-1.4). ``new_id: null`` drops the class to background. Never
            inferred — an unset map means the ids are used as they are.
        class_names: Optional id -> name mapping used in error messages.
        grayscale: Convert images to a single channel.
        batch_size / workers: DataLoader settings.
    """

    path: Optional[str] = None
    format: str = "auto"
    train: str = "train"
    val: str = "val"
    target_policy: Optional[str] = None
    target_class: Optional[int] = None
    class_map: Dict[int, Optional[int]] = field(default_factory=dict)
    class_names: Dict[int, str] = field(default_factory=dict)
    grayscale: bool = False
    batch_size: int = 32
    workers: int = 4
    augment: AugmentConfig = field(default_factory=AugmentConfig)


@dataclass
class LossConfig:
    """Loss preset (FR-3.6).

    ``conf`` is ``bce`` or ``focal``; ``bbox`` is ``smooth_l1`` or ``ciou``.
    ``total = alpha * conf_loss + beta * bbox_loss``.
    """

    conf: str = "bce"
    bbox: str = "smooth_l1"
    alpha: float = 1.0
    beta: float = 2.0
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0


@dataclass
class EarlyStopConfig:
    """Compound early-stopping criterion (FR-3.2).

    The primary metric drives patience. The optional secondary metric can veto
    a stop while it is still improving, but only once the primary has reached
    ``primary_saturated`` — otherwise a still-wandering secondary metric would
    keep a dead run alive. Setting ``secondary_metric=None`` degenerates the
    whole mechanism to standard patience-based stopping.

    The paper's criterion is the default: primary mAP@0.5, secondary mean IoU
    at 3% relative gain over a 3-epoch window, patience 40.
    """

    enabled: bool = True
    primary_metric: str = "map50"
    direction: str = "max"
    patience: int = 40
    min_delta: float = 1e-3
    secondary_metric: Optional[str] = "mean_iou"
    secondary_rel_gain: float = 0.03
    secondary_window: int = 3
    primary_saturated: float = 0.999


@dataclass
class TrainConfig:
    """Optimisation schedule and run bookkeeping."""

    epochs: int = 300
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    min_lr: float = 1e-6
    grad_clip: float = 10.0
    amp: bool = True
    conf_threshold: float = 0.5
    device: str = "auto"
    workers: int = 4
    project: str = "runs"
    name: str = "exp"
    resume: Optional[str] = None
    save_period: int = 0
    loss: LossConfig = field(default_factory=LossConfig)
    early_stop: EarlyStopConfig = field(default_factory=EarlyStopConfig)


@dataclass
class RunConfig:
    """The fully-resolved configuration of one run.

    Produced only by :class:`~dart.config.resolver.ConfigResolver`, never
    constructed piecemeal by callers, so that
    :meth:`to_yaml` always writes something re-runnable on its own (FR-3.4).
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    seed: int = 0
    variant: str = "custom"

    # ── Serialization ───────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Plain nested dicts, suitable for YAML/JSON."""
        return _to_plain(asdict(self))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RunConfig:
        return _from_dict(cls, data)

    def to_yaml(self, path: str | Path) -> Path:
        """Write the resolved config; returns the path written."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, sort_keys=False, default_flow_style=False)
        return path

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        with Path(path).open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)

    def fingerprint(self) -> str:
        """Stable 16-hex-character digest of the resolved configuration (FR-7.2).

        Two runs sharing a fingerprint were launched from the same scientific
        configuration, which is what makes a reported number traceable to what
        produced it. Pure bookkeeping fields — the output directory, the run
        name, the resume path, the checkpoint period — are excluded, so writing
        the same run to a different folder does not change its identity.
        """
        payload = self.to_dict()
        for key in BOOKKEEPING_FIELDS:
            payload["train"].pop(key, None)
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:16]


# ── dict <-> dataclass plumbing ─────────────────────────────────────────


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


@cache
def _resolved_hints(cls: type) -> Dict[str, Any]:
    """Field annotations as real objects.

    ``dataclasses.fields()`` hands back annotation *strings* under
    ``from __future__ import annotations``, so nested dataclasses have to be
    recovered with ``get_type_hints``.
    """
    return get_type_hints(cls)


def _from_dict(cls: Type[T], data: Any) -> T:
    """Recursively build a dataclass from nested dicts, rejecting stray keys."""
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"expected a mapping for {cls.__name__}, got {type(data).__name__}"
        )

    kwargs: Dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    hints = _resolved_hints(cls)
    unknown = set(data) - set(known)
    if unknown:
        raise ConfigurationError(
            f"unknown config key(s) for {cls.__name__}: {sorted(unknown)}. "
            f"Valid keys: {sorted(known)}"
        )

    for name in known:
        if name not in data:
            continue
        value = data[name]
        target = hints.get(name)
        if is_dataclass(target) and isinstance(target, type):
            kwargs[name] = _from_dict(target, value)
        elif name == "blocks":
            kwargs[name] = [
                b if isinstance(b, BlockSpec) else BlockSpec.from_dict(b) for b in value
            ]
        elif name == "class_names" and isinstance(value, dict):
            kwargs[name] = {int(k): str(v) for k, v in value.items()}
        elif name == "class_map" and isinstance(value, dict):
            kwargs[name] = {int(k): (None if v is None else int(v)) for k, v in value.items()}
        else:
            kwargs[name] = value
    return cls(**kwargs)
