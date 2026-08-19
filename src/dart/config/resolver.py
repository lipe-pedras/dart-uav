"""The single place configuration defaults are applied.

``ConfigResolver.resolve()`` takes a preset name, a YAML path, or a dict, plus
arbitrary overrides, and returns a fully-populated :class:`RunConfig`. Nothing
else in the framework may apply a default, which is what makes the snapshot
written by :class:`~dart.engine.recorder.RunRecorder` self-contained (FR-3.4):
re-running a run directory needs only its ``config.yaml``, with no knowledge of
which package version produced it.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

import yaml

from ..errors import ConfigurationError
from .schema import PAPER_RESOLUTIONS, RunConfig

PRESET_DIR = Path(__file__).parent / "presets"

#: Aliases accepted alongside the canonical preset names.
PRESET_ALIASES = {
    "dart-base": "dart-b",
    "dart-b-focal": "dart-b-ciou",
}

#: Dotted-path shorthands for the keyword arguments the public API accepts, so
#: ``model.train(epochs=300, imgsz=320)`` and a YAML file mean the same thing.
KWARG_ALIASES = {
    "data": "data.path",
    "format": "data.format",
    "target_policy": "data.target_policy",
    "target_class": "data.target_class",
    "grayscale": "data.grayscale",
    "workers": "data.workers",
    "epochs": "train.epochs",
    "batch": "train.batch_size",
    "batch_size": "train.batch_size",
    "lr": "train.lr",
    "weight_decay": "train.weight_decay",
    "device": "train.device",
    "amp": "train.amp",
    "patience": "train.early_stop.patience",
    "resume": "train.resume",
    "project": "train.project",
    "name": "train.name",
    "conf": "train.conf_threshold",
    "imgsz": "model.imgsz",
    "width": "model.backbone.width_mult",
    "width_mult": "model.backbone.width_mult",
    "pretrained": "model.backbone.pretrained",
    "seed": "seed",
}

Spec = Union[str, Path, Mapping[str, Any], RunConfig, None]


class ConfigResolver:
    """Merges preset, file, and keyword configuration into one `RunConfig`."""

    def __init__(self, preset_dir: Path | None = None) -> None:
        self.preset_dir = Path(preset_dir) if preset_dir else PRESET_DIR

    # ── Public API ──────────────────────────────────────────────────────

    def resolve(self, spec: Spec = None, **overrides: Any) -> RunConfig:
        """Produce a fully-resolved configuration.

        Args:
            spec: A preset name (``"dart-xd"``), a path to a YAML config, a
                mapping, or an existing :class:`RunConfig` to override.
            **overrides: Keyword overrides. Both dotted paths
                (``train__epochs`` or ``{"train.epochs": 300}``) and the
                shorthands in :data:`KWARG_ALIASES` are accepted.

        Returns:
            A :class:`RunConfig` with every field populated.

        Raises:
            ConfigurationError: If a key is unknown or a value is invalid.
        """
        base: Dict[str, Any] = {}

        if isinstance(spec, RunConfig):
            base = spec.to_dict()
        elif isinstance(spec, Mapping):
            base = copy.deepcopy(dict(spec))
        elif spec is not None:
            base = self._load_spec(str(spec))

        # A YAML config may itself name a preset to build on.
        preset_name = base.pop("preset", None) or base.pop("extends", None)
        if preset_name:
            base = self.merge(self.load_preset(str(preset_name)), base)

        merged = self.merge(base, self._expand_overrides(overrides))
        _derive_input_channels(merged)
        cfg = RunConfig.from_dict(merged)
        self.validate(cfg)
        return cfg

    def load_preset(self, name: str) -> Dict[str, Any]:
        """Read a bundled preset by name, e.g. ``"dart-xd"``."""
        key = PRESET_ALIASES.get(name.lower(), name.lower())
        path = self.preset_dir / f"{key}.yaml"
        if not path.exists():
            raise ConfigurationError(
                f"unknown DART variant '{name}'. Available presets: "
                f"{', '.join(self.available_presets())}. "
                f"To use a custom variant, pass the path to a YAML config instead."
            )
        return self.load_yaml(path)

    def available_presets(self) -> List[str]:
        """Names of the bundled variant presets, sorted."""
        return sorted(p.stem for p in self.preset_dir.glob("*.yaml"))

    @staticmethod
    def load_yaml(path: str | Path) -> Dict[str, Any]:
        """Read a YAML config file into a plain dict."""
        path = Path(path)
        if not path.exists():
            raise ConfigurationError(f"config file not found: {path}")
        with path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigurationError(f"{path} must contain a YAML mapping at the top level")
        return data

    @staticmethod
    def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
        """Recursively merge ``override`` into ``base``.

        Mappings merge key-by-key; every other value (lists included, so a
        replacement block list fully replaces the original) is overwritten.
        """
        result = copy.deepcopy(dict(base))
        for key, value in override.items():
            if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
                result[key] = ConfigResolver.merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    # ── Validation ──────────────────────────────────────────────────────

    @staticmethod
    def validate(cfg: RunConfig) -> None:
        """Check cross-field consistency; raise `ConfigurationError` on failure.

        Only checks that need no filesystem access happen here. Dataset-versus-
        model consistency (FR-1.7) is the job of
        :class:`~dart.data.validator.DatasetValidator`, which needs the data.
        """
        from ..models.registry import BLOCKS, COMPONENTS

        model = cfg.model
        if not model.backbone.blocks:
            raise ConfigurationError(
                "model.backbone.blocks is empty — a variant must list at least "
                "one block. Start from a preset (e.g. model='dart-xd')."
            )
        for i, block in enumerate(model.backbone.blocks):
            if block.type not in BLOCKS:
                raise ConfigurationError(
                    f"model.backbone.blocks[{i}].type = '{block.type}' is not a "
                    f"registered block. Available: {sorted(BLOCKS.names())}"
                )
            _require(
                COMPONENTS, "attention", block.attention, f"model.backbone.blocks[{i}].attention"
            )
            _require(COMPONENTS, "activation", block.act, f"model.backbone.blocks[{i}].act")

        n_blocks = len(model.backbone.blocks)
        for tap in model.backbone.tap_indices:
            if not -n_blocks <= tap < n_blocks:
                raise ConfigurationError(
                    f"model.backbone.tap_indices contains {tap}, which is out of "
                    f"range for a {n_blocks}-block backbone. Taps are zero-based "
                    "block indices (negatives count from the end); the last block "
                    "is always tapped and need not be listed."
                )

        _require(COMPONENTS, "neck", model.neck.name, "model.neck.name")
        _require(COMPONENTS, "head", model.head.name, "model.head.name")
        _require(COMPONENTS, "activation", model.act, "model.act")

        if model.imgsz <= 0 or (model.imgsz % 32 != 0 and model.imgsz not in PAPER_RESOLUTIONS):
            raise ConfigurationError(
                f"model.imgsz = {model.imgsz} is invalid; use a positive multiple "
                f"of 32. The paper validates {PAPER_RESOLUTIONS}."
            )
        if model.backbone.width_mult <= 0:
            raise ConfigurationError(
                f"model.backbone.width_mult = {model.backbone.width_mult} must be > 0"
            )
        if model.num_classes != 1:
            raise ConfigurationError(
                f"model.num_classes = {model.num_classes}: DART v1 is a "
                "single-class detector. Multi-class support is deliberately "
                "out of scope (requirements §3.2)."
            )

        train = cfg.train
        if train.epochs < 1:
            raise ConfigurationError(f"train.epochs = {train.epochs} must be >= 1")
        if train.batch_size < 1:
            raise ConfigurationError(f"train.batch_size = {train.batch_size} must be >= 1")
        if train.loss.conf not in ("bce", "focal"):
            raise ConfigurationError(
                f"train.loss.conf = '{train.loss.conf}' is not a known confidence "
                "loss; use 'bce' or 'focal'."
            )
        if train.loss.bbox not in ("smooth_l1", "ciou"):
            raise ConfigurationError(
                f"train.loss.bbox = '{train.loss.bbox}' is not a known box loss; "
                "use 'smooth_l1' or 'ciou'."
            )

        stop = train.early_stop
        if stop.direction not in ("max", "min"):
            raise ConfigurationError(
                f"train.early_stop.direction = '{stop.direction}' must be 'max' or 'min'"
            )
        if stop.patience < 1:
            raise ConfigurationError("train.early_stop.patience must be >= 1")
        if stop.secondary_metric and stop.secondary_window < 1:
            raise ConfigurationError("train.early_stop.secondary_window must be >= 1")

        expected_ch = 1 if cfg.data.grayscale else 3
        if model.backbone.in_channels != expected_ch:
            raise ConfigurationError(
                f"model.backbone.in_channels = {model.backbone.in_channels} but "
                f"data.grayscale = {cfg.data.grayscale} implies {expected_ch}. "
                "Set both consistently."
            )

        if cfg.data.target_policy not in (None, "largest_only", "keep_all"):
            raise ConfigurationError(
                f"data.target_policy = '{cfg.data.target_policy}' is unknown; "
                "use 'largest_only' or 'keep_all'."
            )

    # ── Internals ───────────────────────────────────────────────────────

    def _load_spec(self, spec: str) -> Dict[str, Any]:
        """Interpret a string spec as a preset name or a path to a YAML file."""
        candidate = Path(spec)
        if candidate.suffix.lower() in (".yaml", ".yml"):
            return self.load_yaml(candidate)
        return self.load_preset(spec)

    @staticmethod
    def _expand_overrides(overrides: Mapping[str, Any]) -> Dict[str, Any]:
        """Turn flat keyword overrides into a nested dict."""
        nested: Dict[str, Any] = {}
        for raw_key, value in overrides.items():
            if value is None and raw_key not in ("target_class", "pretrained", "resume"):
                continue  # an unset keyword must not clobber a configured value
            key = KWARG_ALIASES.get(raw_key, raw_key.replace("__", "."))
            _assign(nested, key.split("."), value)
        return nested


def _derive_input_channels(merged: Dict[str, Any]) -> None:
    """Keep ``backbone.in_channels`` in step with ``data.grayscale``.

    Deriving one from the other is bookkeeping, not the kind of silent
    adaptation FR-1.7 forbids: nothing about the dataset's *content* is being
    guessed. An explicitly-set ``in_channels`` is always left alone, so a
    genuine mismatch still reaches the validator and errors.
    """
    backbone = merged.setdefault("model", {}).setdefault("backbone", {})
    if "in_channels" in backbone:
        return
    if merged.get("data", {}).get("grayscale"):
        backbone["in_channels"] = 1


def _assign(target: Dict[str, Any], path: Iterable[str], value: Any) -> None:
    keys = list(path)
    node = target
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ConfigurationError(f"conflicting override for '{'.'.join(keys)}'")
    node[keys[-1]] = value


def _require(components, kind: str, name: Optional[str], key: str) -> None:
    if name is None:
        raise ConfigurationError(f"{key} must be set")
    registry = components.kind(kind)
    if name not in registry:
        raise ConfigurationError(
            f"{key} = '{name}' is not a registered {kind}. Available: {sorted(registry.names())}"
        )


def resolve_config(spec: Spec = None, **overrides: Any) -> RunConfig:
    """Convenience wrapper around :meth:`ConfigResolver.resolve`."""
    return ConfigResolver().resolve(spec, **overrides)
