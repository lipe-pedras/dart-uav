"""The internal dataset representation and its batch contract.

``DARTDataset`` holds nothing but picklable data — paths, numpy arrays, and
transform objects — so it survives the ``spawn`` start method Windows requires
(NFR-3). In particular it never holds an open file handle, a cv2 capture, or a
lambda.

Batch contract, matching the head output contract in
:mod:`dart.models.heads`::

    images  : (B, C, H, W)  float32, normalised
    conf    : (B, N)        1.0 where target ``n`` is present, else 0.0
    boxes   : (B, N, 4)     [cx, cy, w, h] in [0, 1]; zeros where absent
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..errors import DatasetError
from .adapters import ImageAnnotations, RawAnnotations
from .augment import AugmentationPipeline
from .policies import ClassRemapper, ClassSelector, TargetPolicy

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
GRAY_MEAN = np.array([0.5], dtype=np.float32)
GRAY_STD = np.array([0.5], dtype=np.float32)


@dataclass
class Stats:
    """Split statistics reported on load (FR-1.5)."""

    images: int
    positives: int
    negatives: int
    positive_rate: float
    annotations: int
    classes: List[int]
    max_annotations_per_image: int

    def __str__(self) -> str:
        return (
            f"{self.images} images ({self.positives} positive, "
            f"{self.negatives} negative, {self.positive_rate:.1%} positive rate), "
            f"{self.annotations} annotations, classes={self.classes}"
        )


class DARTDataset(Dataset):
    """Images plus normalised targets, ready for the training loop.

    Args:
        items: Per-image raw annotations, in absolute-pixel xyxy.
        imgsz: Square resolution every image is resized to.
        num_targets: Boxes emitted per sample (``N`` in the batch contract).
        policy: Target-selection policy; ``None`` only when every image already
            has at most ``num_targets`` annotations.
        class_remapper: Class id rewriter; ``None`` unless explicitly
            configured. Applied before ``class_selector``, so ``target_class``
            names an id in the *remapped* space.
        class_selector: Class filter; ``None`` unless explicitly configured.
        augment: Augmentation pipeline; empty for validation splits.
        grayscale: Load as a single channel.
        stats: Precomputed split statistics.
        class_names: Class id -> name, for reporting.
    """

    def __init__(
        self,
        items: List[ImageAnnotations],
        imgsz: int = 96,
        num_targets: int = 1,
        policy: Optional[TargetPolicy] = None,
        class_remapper: Optional[ClassRemapper] = None,
        class_selector: Optional[ClassSelector] = None,
        augment: Optional[AugmentationPipeline] = None,
        grayscale: bool = False,
        stats: Optional[Stats] = None,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        self.items = list(items)
        self.imgsz = int(imgsz)
        self.num_targets = int(num_targets)
        self.policy = policy
        self.class_remapper = class_remapper
        self.class_selector = class_selector
        self.augment = augment or AugmentationPipeline([])
        self.grayscale = bool(grayscale)
        self.stats = stats
        self.class_names = dict(class_names or {})

    # ── Dataset protocol ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        item = self.items[index]
        image = self._read_image(item.path)
        height, width = image.shape[:2]

        boxes = _to_normalized_cxcywh(item.boxes, width, height)
        class_ids = item.class_ids
        if self.class_remapper is not None:
            boxes, class_ids = self.class_remapper.apply(boxes, class_ids)
        if self.class_selector is not None:
            boxes, class_ids = self.class_selector.apply(boxes, class_ids)
        if self.policy is not None:
            boxes, class_ids = self.policy.apply(boxes, class_ids)

        image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        if image.ndim == 2:
            image = image[:, :, None]

        image, boxes = self.augment.apply(image, boxes)
        return self._to_tensors(image, boxes)

    # ── Internals ───────────────────────────────────────────────────────

    def _read_image(self, path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise DatasetError(f"cannot read image: {path}")
        if self.grayscale:
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return image[:, :, None]
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _to_tensors(
        self, image: np.ndarray, boxes: np.ndarray
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pixels = image.astype(np.float32) / 255.0
        mean = GRAY_MEAN if self.grayscale else IMAGENET_MEAN
        std = GRAY_STD if self.grayscale else IMAGENET_STD
        pixels = (pixels - mean) / std

        conf = np.zeros(self.num_targets, dtype=np.float32)
        padded = np.zeros((self.num_targets, 4), dtype=np.float32)
        keep = min(len(boxes), self.num_targets)
        if keep:
            padded[:keep] = boxes[:keep]
            conf[:keep] = 1.0

        return (
            torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1).float(),
            torch.from_numpy(conf),
            torch.from_numpy(padded),
        )


def _to_normalized_cxcywh(boxes_xyxy: np.ndarray, width: int, height: int) -> np.ndarray:
    """Absolute-pixel ``xyxy`` -> normalised ``cxcywh``, clipped to the image."""
    if len(boxes_xyxy) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.asarray(boxes_xyxy, dtype=np.float32).copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    cx = (boxes[:, 0] + boxes[:, 2]) / 2.0 / width
    cy = (boxes[:, 1] + boxes[:, 3]) / 2.0 / height
    bw = (boxes[:, 2] - boxes[:, 0]) / width
    bh = (boxes[:, 3] - boxes[:, 1]) / height
    return np.stack([cx, cy, bw, bh], axis=1).astype(np.float32)


def stats_from_raw(raw: RawAnnotations, target_class: Optional[int] = None) -> Stats:
    """Build :class:`Stats` from raw annotations."""
    from .validator import summarize

    return Stats(**summarize(raw, target_class))
