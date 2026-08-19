"""The augmentation pipeline (FR-1.6).

Every transform is a small picklable class — no lambdas, no closures, no open
handles — because on Windows the ``spawn`` start method pickles the dataset,
and its transforms with it (NFR-3).

Transforms operate on ``(image, boxes)`` where ``image`` is ``HxWxC`` uint8 and
``boxes`` is an ``(n, 4)`` float array of ``[cx, cy, w, h]`` normalised to
``[0, 1]``. Working in normalised coordinates keeps flip and crop arithmetic
resolution-independent.

Probabilities and magnitudes default to the values used in the paper, so
``AugmentationPipeline.from_config(AugmentConfig())`` reproduces the research
pipeline exactly.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import cv2
import numpy as np

from ..config.schema import AugmentConfig

Sample = Tuple[np.ndarray, np.ndarray]


class Transform:
    """Base class: applies itself with probability ``p``.

    Args:
        p: Probability of applying the transform to a given sample.
    """

    #: Transforms that adjust box geometry are skipped on negative samples,
    #: matching the research pipeline (there is no box to keep consistent).
    requires_boxes = False

    def __init__(self, p: float = 0.5) -> None:
        self.p = float(p)

    def apply(self, image: np.ndarray, boxes: np.ndarray) -> Sample:
        raise NotImplementedError

    def __call__(self, image: np.ndarray, boxes: np.ndarray) -> Sample:
        if self.p <= 0.0 or random.random() >= self.p:
            return image, boxes
        if self.requires_boxes and len(boxes) == 0:
            return image, boxes
        return self.apply(image, boxes)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(p={self.p})"


class HorizontalFlip(Transform):
    """Mirror left-right; ``cx`` mirrors with it."""

    def apply(self, image: np.ndarray, boxes: np.ndarray) -> Sample:
        image = np.fliplr(image).copy()
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, 0] = 1.0 - boxes[:, 0]
        return image, boxes


class VerticalFlip(Transform):
    """Mirror top-bottom; ``cy`` mirrors with it."""

    def apply(self, image: np.ndarray, boxes: np.ndarray) -> Sample:
        image = np.flipud(image).copy()
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, 1] = 1.0 - boxes[:, 1]
        return image, boxes


class RandomBrightness(Transform):
    """Scale pixel intensity by a random factor."""

    def __init__(self, p: float = 0.7, factor_range: Sequence[float] = (0.6, 1.4)) -> None:
        super().__init__(p)
        self.factor_range = tuple(factor_range)

    def apply(self, image: np.ndarray, boxes: np.ndarray) -> Sample:
        factor = random.uniform(*self.factor_range)
        return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8), boxes


class RandomContrast(Transform):
    """Scale deviation from the image mean by a random factor."""

    def __init__(self, p: float = 0.7, factor_range: Sequence[float] = (0.7, 1.3)) -> None:
        super().__init__(p)
        self.factor_range = tuple(factor_range)

    def apply(self, image: np.ndarray, boxes: np.ndarray) -> Sample:
        factor = random.uniform(*self.factor_range)
        mean = float(image.mean())
        adjusted = (image.astype(np.float32) - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8), boxes


class GaussianNoise(Transform):
    """Additive Gaussian noise — stands in for low-light sensor noise."""

    def __init__(self, p: float = 0.6, sigma_range: Sequence[float] = (2.0, 12.0)) -> None:
        super().__init__(p)
        self.sigma_range = tuple(sigma_range)

    def apply(self, image: np.ndarray, boxes: np.ndarray) -> Sample:
        sigma = random.uniform(*self.sigma_range)
        noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8), boxes


class RandomCropResize(Transform):
    """Crop a random sub-window and resize back — simulates altitude change."""

    requires_boxes = True

    def __init__(self, p: float = 0.6, scale_range: Sequence[float] = (0.75, 0.98)) -> None:
        super().__init__(p)
        self.scale_range = tuple(scale_range)

    def apply(self, image: np.ndarray, boxes: np.ndarray) -> Sample:
        height, width = image.shape[:2]
        scale = random.uniform(*self.scale_range)
        new_h, new_w = max(1, int(height * scale)), max(1, int(width * scale))
        top = random.randint(0, height - new_h)
        left = random.randint(0, width - new_w)

        cropped = image[top : top + new_h, left : left + new_w]
        resized = cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)
        if resized.ndim == 2:
            resized = resized[:, :, None]

        boxes = boxes.copy()
        boxes[:, 0] = (boxes[:, 0] * width - left) / new_w
        boxes[:, 1] = (boxes[:, 1] * height - top) / new_h
        boxes[:, 2] = boxes[:, 2] * width / new_w
        boxes[:, 3] = boxes[:, 3] * height / new_h
        return resized, np.clip(boxes, 0.0, 1.0)


class RandomRotate(Transform):
    """Rotate about the image centre — simulates camera roll / yaw drift.

    Only the box centre is rotated; width and height are left alone, which is a
    good approximation for the small angles used here and matches the research
    pipeline the paper's numbers came from.
    """

    requires_boxes = True

    def __init__(self, p: float = 0.5, degrees: float = 15.0) -> None:
        super().__init__(p)
        self.degrees = float(degrees)

    def apply(self, image: np.ndarray, boxes: np.ndarray) -> Sample:
        height, width = image.shape[:2]
        angle = random.uniform(-self.degrees, self.degrees)
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        if rotated.ndim == 2:
            rotated = rotated[:, :, None]

        boxes = boxes.copy()
        points = np.stack([boxes[:, 0] * width, boxes[:, 1] * height, np.ones(len(boxes))], axis=1)
        moved = points @ matrix.T
        boxes[:, 0] = np.clip(moved[:, 0] / width, 0.0, 1.0)
        boxes[:, 1] = np.clip(moved[:, 1] / height, 0.0, 1.0)
        return rotated, boxes


class GaussianBlur(Transform):
    """Blur with a random odd kernel — simulates motion blur or defocus."""

    def __init__(self, p: float = 0.4, kernels: Sequence[int] = (3, 5)) -> None:
        super().__init__(p)
        self.kernels = tuple(int(k) for k in kernels)

    def apply(self, image: np.ndarray, boxes: np.ndarray) -> Sample:
        k = random.choice(self.kernels)
        blurred = cv2.GaussianBlur(image, (k, k), 0)
        if blurred.ndim == 2:
            blurred = blurred[:, :, None]
        return blurred, boxes


class AugmentationPipeline:
    """An ordered, swappable list of transforms.

    Replace the whole pipeline by constructing one directly::

        pipeline = AugmentationPipeline([HorizontalFlip(0.5), GaussianBlur(0.2)])

    Args:
        transforms: Transforms applied in order.
    """

    def __init__(self, transforms: Sequence[Transform] | None = None) -> None:
        self.transforms: List[Transform] = list(transforms or [])

    @classmethod
    def from_config(cls, cfg: AugmentConfig) -> AugmentationPipeline:
        """Build the paper's pipeline from an :class:`AugmentConfig`."""
        if not cfg.enabled:
            return cls([])
        return cls(
            [
                HorizontalFlip(cfg.hflip),
                VerticalFlip(cfg.vflip),
                RandomBrightness(cfg.brightness, cfg.brightness_range),
                RandomContrast(cfg.contrast, cfg.contrast_range),
                GaussianNoise(cfg.noise, cfg.noise_sigma),
                RandomCropResize(cfg.crop_resize, cfg.crop_scale),
                RandomRotate(cfg.rotate, cfg.rotate_degrees),
                GaussianBlur(cfg.blur, cfg.blur_kernels),
            ]
        )

    def apply(self, image: np.ndarray, boxes: np.ndarray) -> Sample:
        for transform in self.transforms:
            image, boxes = transform(image, boxes)
        return image, boxes

    def __len__(self) -> int:
        return len(self.transforms)

    def __repr__(self) -> str:
        return f"AugmentationPipeline({self.transforms})"
