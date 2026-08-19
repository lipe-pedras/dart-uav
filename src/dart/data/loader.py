"""Turning a path on disk into validated, batched training data.

``DatasetLoader.load()`` is the only entry point: it locates the split,
selects a format adapter, runs :class:`~dart.data.validator.DatasetValidator`
against the model configuration *before* anything is filtered, and only then
builds a :class:`~dart.data.dataset.DARTDataset`.
"""

from __future__ import annotations

import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import yaml
from torch.utils.data import DataLoader

from ..config.schema import DataConfig, ModelConfig
from ..errors import ConfigurationError, DatasetError, SpawnSafetyError
from .adapters import adapter_for, detect_format
from .augment import AugmentationPipeline
from .dataset import DARTDataset, stats_from_raw
from .policies import ClassRemapper, ClassSelector, build_policy
from .validator import DatasetValidator

#: Directory names searched for a split's images and annotations.
IMAGE_DIR_NAMES = ("images", "image", "imgs")
LABEL_DIR_NAMES = ("labels", "annotations", "labels_json", "anns")


@dataclass
class SplitPaths:
    """Where a split's images and annotations live."""

    images: Path
    annotations: Path


class DatasetLoader:
    """Builds :class:`DARTDataset` objects from a :class:`DataConfig`."""

    def __init__(self, validator: Optional[DatasetValidator] = None) -> None:
        self.validator = validator or DatasetValidator()

    def load(
        self,
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        split: str = "train",
        augment: Optional[bool] = None,
        verbose: bool = True,
    ) -> DARTDataset:
        """Load one split.

        Args:
            data_cfg: Resolved data configuration.
            model_cfg: Resolved model configuration, used for validation.
            split: ``"train"`` or ``"val"``.
            augment: Override the config's augmentation switch. Defaults to
                augmenting the training split only.
            verbose: Print the split statistics (FR-1.5).

        Returns:
            A validated, ready-to-batch dataset.

        Raises:
            ConfigurationError: If the dataset conflicts with the model
                configuration and no explicit policy resolves it (FR-1.7).
        """
        paths = self.locate_split(data_cfg, split)
        fmt = data_cfg.format
        if not fmt or fmt == "auto":
            fmt = detect_format(paths.images, paths.annotations)

        raw = adapter_for(fmt).read(paths.images, paths.annotations)
        raw.root = paths.images
        raw.format = fmt

        # Validation runs on the *raw* annotations, before any filtering.
        self.validator.validate(raw, model_cfg, data_cfg)

        stats = stats_from_raw(raw, data_cfg.target_class)
        if verbose:
            print(f"  [{split}] {stats}")

        do_augment = (split == "train") if augment is None else augment
        pipeline = (
            AugmentationPipeline.from_config(data_cfg.augment)
            if do_augment
            else AugmentationPipeline([])
        )

        # A policy is only built when one is needed or explicitly configured;
        # a class selector is only built when explicitly configured (FR-1.4).
        policy = None
        if data_cfg.target_policy is not None:
            policy = build_policy(data_cfg.target_policy)
        elif raw.max_annotations_per_image() > model_cfg.head.num_targets:
            # Unreachable while the validator is in place; kept as a guard so a
            # future caller cannot bypass the explicit-policy requirement.
            raise ConfigurationError(
                "data.target_policy must be set for a dataset with more "
                "annotations per image than the model has targets"
            )

        # Both are built only from explicit configuration; there is no code
        # path that infers either one (FR-1.4).
        remapper = ClassRemapper(data_cfg.class_map) if data_cfg.class_map else None
        selector = (
            ClassSelector(data_cfg.target_class) if data_cfg.target_class is not None else None
        )

        return DARTDataset(
            items=raw.items,
            imgsz=model_cfg.imgsz,
            num_targets=model_cfg.head.num_targets,
            policy=policy,
            class_remapper=remapper,
            class_selector=selector,
            augment=pipeline,
            grayscale=data_cfg.grayscale,
            stats=stats,
            class_names={**raw.class_names, **(data_cfg.class_names or {})},
        )

    # ── Split discovery ─────────────────────────────────────────────────

    def locate_split(self, data_cfg: DataConfig, split: str) -> SplitPaths:
        """Find a split's image and annotation paths.

        Understands three layouts, in this order:

        1. an Ultralytics-style dataset YAML naming ``path``/``train``/``val``;
        2. ``<root>/images/<split>`` with ``<root>/labels/<split>``;
        3. ``<root>/<split>/images`` with ``<root>/<split>/labels``, or a
           single ``<root>/<split>`` holding images and annotations together.
        """
        if not data_cfg.path:
            raise ConfigurationError(
                "data.path is not set — pass data='<path to dataset or data.yaml>'"
            )
        root = Path(data_cfg.path).expanduser()
        split_name = getattr(data_cfg, split, split)

        if root.suffix.lower() in (".yaml", ".yml"):
            root, split_name = _read_dataset_yaml(root, split, split_name)

        if not root.exists():
            raise DatasetError(f"dataset path does not exist: {root}")

        candidate = Path(split_name)
        if candidate.is_absolute() and candidate.exists():
            return _pair_for(candidate)

        for images_name in IMAGE_DIR_NAMES:
            images = root / images_name / split_name
            if images.is_dir():
                for labels_name in LABEL_DIR_NAMES:
                    labels = root / labels_name / split_name
                    if labels.exists():
                        return SplitPaths(images, labels)
                return _pair_for(images)

        split_dir = root / split_name
        if split_dir.is_dir():
            return _pair_for(split_dir)

        raise DatasetError(
            f"could not find the '{split}' split under {root}. Expected one of: "
            f"{root}/images/{split_name}, {root}/{split_name}/images, or "
            f"{root}/{split_name}. Set data.{split} to the split's directory name."
        )


def _pair_for(directory: Path) -> SplitPaths:
    """Resolve a split directory into an (images, annotations) pair."""
    for images_name in IMAGE_DIR_NAMES:
        images = directory / images_name
        if images.is_dir():
            for labels_name in LABEL_DIR_NAMES:
                labels = directory / labels_name
                if labels.exists():
                    return SplitPaths(images, labels)
            return SplitPaths(images, directory)
    # Images and annotations share one directory (common for VOC and COCO).
    for labels_name in LABEL_DIR_NAMES:
        labels = directory / labels_name
        if labels.exists():
            return SplitPaths(directory, labels)
    return SplitPaths(directory, directory)


def _read_dataset_yaml(path: Path, split: str, fallback: str) -> Tuple[Path, str]:
    """Read an Ultralytics-style dataset YAML into ``(root, split_path)``."""
    if not path.exists():
        raise DatasetError(f"dataset YAML not found: {path}")
    with path.open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle) or {}
    if not isinstance(spec, dict):
        raise DatasetError(f"{path} must contain a YAML mapping")

    root = Path(spec.get("path", path.parent)).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    entry = spec.get(split, fallback)
    if isinstance(entry, (list, tuple)):
        entry = entry[0]
    return root, str(entry)


# ── DataLoader construction (NFR-3) ─────────────────────────────────────


def build_dataloader(
    dataset: DARTDataset,
    batch_size: int,
    shuffle: bool,
    workers: int = 0,
    seed: Optional[int] = None,
    drop_last: bool = False,
    pin_memory: bool = False,
) -> DataLoader:
    """Wrap a dataset in a ``DataLoader``, checking spawn safety first.

    Args:
        dataset: The dataset to batch.
        batch_size: Samples per batch.
        shuffle: Shuffle between epochs (training only).
        workers: Worker processes; ``0`` is always a correct fallback.
        seed: Seed for the shuffling generator, so a seeded run is reproducible.
        drop_last: Drop a trailing partial batch (training only).
        pin_memory: Pin host memory for faster host-to-device copies.

    Raises:
        SpawnSafetyError: If workers were requested from a script that would
            re-execute itself under the ``spawn`` start method.
    """
    workers = max(0, int(workers))
    if workers > 0:
        check_spawn_safety(workers)

    generator = None
    if seed is not None:
        import torch

        generator = torch.Generator()
        generator.manual_seed(int(seed))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
        persistent_workers=workers > 0,
    )


def check_spawn_safety(workers: int) -> None:
    """Refuse to spawn workers from a script that would re-execute itself.

    Under the ``spawn`` start method (mandatory on Windows, default on macOS)
    each worker re-imports the ``__main__`` module. A user script without an
    ``if __name__ == "__main__":`` guard therefore re-runs its own training
    call in every worker, spawning processes without bound. Detecting the
    condition and naming the fix is required by NFR-3.
    """
    if multiprocessing.get_start_method(allow_none=True) not in ("spawn", None):
        return
    if sys.platform not in ("win32", "darwin"):
        return  # fork is the default and is safe here

    main = sys.modules.get("__main__")
    script = getattr(main, "__file__", None)
    if not script:
        return  # REPL / notebook: there is no module for a worker to re-import

    try:
        source = Path(script).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return
    if "__main__" in source:
        return

    raise SpawnSafetyError(
        f"{script} calls DART with workers={workers}, but this platform starts "
        "DataLoader workers with 'spawn', which re-imports your script in every "
        "worker process. Wrap the entry point:\n\n"
        "    if __name__ == '__main__':\n"
        "        main()\n\n"
        "or pass workers=0, which is always correct and never spawns."
    )
