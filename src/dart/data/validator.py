"""Dataset-versus-model validation (FR-1.7).

Runs on the **raw** annotations, before any policy or class selector is
applied. If it ran afterwards, a ``largest_only`` policy would already have
discarded the extra annotations and the multi-target violation would be
invisible — the system would have silently adapted, which the requirements
forbid.

Every message names the offending dataset, the offending class or image, and
the configuration key that resolves it.
"""

from __future__ import annotations

from typing import List, Optional

from ..config.schema import DataConfig, ModelConfig
from ..errors import ConfigurationError
from .adapters import RawAnnotations


class DatasetValidator:
    """Checks a split's annotations against the model configuration."""

    def validate(self, raw: RawAnnotations, model_cfg: ModelConfig, data_cfg: DataConfig) -> None:
        """Raise :class:`ConfigurationError` on the first violation found."""
        self._check_non_empty(raw)
        self._check_class_cardinality(raw, model_cfg, data_cfg)
        self._check_target_cardinality(raw, model_cfg, data_cfg)

    # ── Individual checks ───────────────────────────────────────────────

    @staticmethod
    def _check_non_empty(raw: RawAnnotations) -> None:
        if len(raw) == 0:
            raise ConfigurationError(
                f"no images found for dataset at {raw.root} (format '{raw.format}'). "
                "Check data.path and the split directory names (data.train / data.val)."
            )

    @staticmethod
    def _check_class_cardinality(
        raw: RawAnnotations, model_cfg: ModelConfig, data_cfg: DataConfig
    ) -> None:
        # An explicit class_map is a written decision about what the ids mean,
        # so cardinality is judged in the remapped space. The *target*
        # cardinality check below still runs on raw annotations, because a
        # policy would hide the violation it exists to catch.
        present = _effective_classes(raw, data_cfg)
        if model_cfg.num_classes > 1 or len(present) <= 1:
            return
        if data_cfg.target_class is not None:
            if data_cfg.target_class not in present:
                raise ConfigurationError(
                    f"dataset at {raw.root}: data.target_class = "
                    f"{data_cfg.target_class} is not annotated anywhere in this "
                    f"split. Classes present: {_describe(present, raw, data_cfg)}."
                )
            return
        raise ConfigurationError(
            f"dataset at {raw.root} annotates {len(present)} classes "
            f"({_describe(present, raw, data_cfg)}) but the model is configured "
            "single-class. DART will not guess which class is the target: set "
            "'data.target_class' to the class id you want to detect. Every other "
            "class then becomes background."
        )

    @staticmethod
    def _check_target_cardinality(
        raw: RawAnnotations, model_cfg: ModelConfig, data_cfg: DataConfig
    ) -> None:
        if model_cfg.head.num_targets > 1 or data_cfg.target_policy is not None:
            return

        target_class = data_cfg.target_class
        for item in raw.items:
            count = (
                len(item) if target_class is None else int((item.class_ids == target_class).sum())
            )
            if count > 1:
                raise ConfigurationError(
                    f"dataset at {raw.root}: image '{item.path}' has {count} "
                    "annotations but the model is configured single-target. "
                    "DART will not choose which one to keep: set "
                    "'data.target_policy' to 'largest_only' (the paper's "
                    "convention) or 'keep_all'."
                )


def _effective_classes(raw: RawAnnotations, data_cfg: DataConfig) -> List[int]:
    """Class ids as they will be seen after any configured remapping."""
    present = raw.class_ids_present()
    if not data_cfg.class_map:
        return present
    mapped = set()
    for cid in present:
        target = data_cfg.class_map.get(cid, cid)
        if target is not None:
            mapped.add(int(target))
    return sorted(mapped)


def _describe(class_ids: List[int], raw: RawAnnotations, data_cfg: DataConfig) -> str:
    """Render class ids with their names, when names are known."""
    names = {**raw.class_names, **(data_cfg.class_names or {})}
    return ", ".join(f"{cid}" + (f" ({names[cid]})" if cid in names else "") for cid in class_ids)


def summarize(raw: RawAnnotations, target_class: Optional[int] = None) -> dict:
    """Basic split statistics for the load-time report (FR-1.5)."""
    positives = 0
    total_boxes = 0
    for item in raw.items:
        count = len(item) if target_class is None else int((item.class_ids == target_class).sum())
        total_boxes += count
        positives += int(count > 0)
    images = len(raw)
    return {
        "images": images,
        "positives": positives,
        "negatives": images - positives,
        "positive_rate": positives / images if images else 0.0,
        "annotations": total_boxes,
        "classes": raw.class_ids_present(),
        "max_annotations_per_image": raw.max_annotations_per_image(),
    }
