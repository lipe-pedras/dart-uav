"""Shared fixtures: tiny synthetic datasets in every supported format.

Everything here is generated on the fly into ``tmp_path``, so the test suite
has no binary fixtures to keep in the repository and CI needs no downloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import pytest

IMG_SIZE = 64


def _draw(path: Path, boxes: Sequence[Tuple[float, float, float, float]]) -> None:
    """Write an image with a filled white rectangle per normalised box."""
    image = np.full((IMG_SIZE, IMG_SIZE, 3), 32, dtype=np.uint8)
    for cx, cy, w, h in boxes:
        x1 = int((cx - w / 2) * IMG_SIZE)
        y1 = int((cy - h / 2) * IMG_SIZE)
        x2 = int((cx + w / 2) * IMG_SIZE)
        y2 = int((cy + h / 2) * IMG_SIZE)
        cv2.rectangle(image, (x1, y1), (x2, y2), (240, 240, 240), -1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def _sample_boxes(index: int) -> List[Tuple[int, Tuple[float, float, float, float]]]:
    """Deterministic ``(class_id, box)`` list for image ``index``.

    Every third image is a negative (no annotation), which gives the metrics
    something to be wrong about.
    """
    if index % 3 == 2:
        return []
    cx = 0.3 + 0.1 * (index % 4)
    cy = 0.3 + 0.1 * ((index // 2) % 4)
    return [(0, (cx, cy, 0.25, 0.25))]


def make_yolo_dataset(
    root: Path,
    splits: Sequence[str] = ("train", "val"),
    n_per_split: int = 6,
    multi_class: bool = False,
    multi_target: bool = False,
) -> Path:
    """Build a YOLO-layout dataset under ``root`` and return the root."""
    for split in splits:
        for i in range(n_per_split):
            annotations = _sample_boxes(i)
            if multi_class and i == 0:
                annotations = [(0, (0.3, 0.3, 0.2, 0.2)), (1, (0.7, 0.7, 0.2, 0.2))]
            if multi_target and i == 1:
                annotations = [(0, (0.3, 0.3, 0.2, 0.2)), (0, (0.7, 0.7, 0.3, 0.3))]

            stem = f"img{i:03d}"
            _draw(root / "images" / split / f"{stem}.jpg", [b for _, b in annotations])
            label_path = root / "labels" / split / f"{stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text(
                "".join(
                    f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n"
                    for cid, (cx, cy, w, h) in annotations
                ),
                encoding="utf-8",
            )

    names = ["target", "other"] if multi_class else ["target"]
    (root / "data.yaml").write_text(
        f"path: .\ntrain: images/train\nval: images/val\nnc: {len(names)}\nnames: {names}\n",
        encoding="utf-8",
    )
    return root


def make_coco_dataset(root: Path, split: str = "train", n: int = 4) -> Path:
    images, annotations = [], []
    ann_id = 1
    for i in range(n):
        stem = f"img{i:03d}.jpg"
        boxes = _sample_boxes(i)
        _draw(root / split / stem, [b for _, b in boxes])
        images.append({"id": i + 1, "file_name": stem, "width": IMG_SIZE, "height": IMG_SIZE})
        for cid, (cx, cy, w, h) in boxes:
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": i + 1,
                    "category_id": cid + 1,
                    "bbox": [
                        (cx - w / 2) * IMG_SIZE,
                        (cy - h / 2) * IMG_SIZE,
                        w * IMG_SIZE,
                        h * IMG_SIZE,
                    ],
                    "area": w * h * IMG_SIZE * IMG_SIZE,
                    "iscrowd": 0,
                }
            )
            ann_id += 1
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "target", "supercategory": "none"}],
    }
    (root / split / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def make_voc_dataset(root: Path, split: str = "train", n: int = 4) -> Path:
    for i in range(n):
        stem = f"img{i:03d}"
        boxes = _sample_boxes(i)
        _draw(root / split / f"{stem}.jpg", [b for _, b in boxes])
        objects = "".join(
            "<object><name>target</name><bndbox>"
            f"<xmin>{int((cx - w / 2) * IMG_SIZE)}</xmin>"
            f"<ymin>{int((cy - h / 2) * IMG_SIZE)}</ymin>"
            f"<xmax>{int((cx + w / 2) * IMG_SIZE)}</xmax>"
            f"<ymax>{int((cy + h / 2) * IMG_SIZE)}</ymax>"
            "</bndbox></object>"
            for _, (cx, cy, w, h) in boxes
        )
        (root / split / f"{stem}.xml").write_text(
            f"<annotation><filename>{stem}.jpg</filename>"
            f"<size><width>{IMG_SIZE}</width><height>{IMG_SIZE}</height><depth>3</depth></size>"
            f"{objects}</annotation>",
            encoding="utf-8",
        )
    return root


def make_createml_dataset(root: Path, split: str = "train", n: int = 4) -> Path:
    records = []
    for i in range(n):
        stem = f"img{i:03d}.jpg"
        boxes = _sample_boxes(i)
        _draw(root / split / stem, [b for _, b in boxes])
        records.append(
            {
                "image": stem,
                "annotations": [
                    {
                        "label": "target",
                        "coordinates": {
                            "x": cx * IMG_SIZE,
                            "y": cy * IMG_SIZE,
                            "width": w * IMG_SIZE,
                            "height": h * IMG_SIZE,
                        },
                    }
                    for _, (cx, cy, w, h) in boxes
                ],
            }
        )
    (root / split / "_annotations.createml.json").write_text(json.dumps(records), encoding="utf-8")
    return root


@pytest.fixture
def yolo_dataset(tmp_path: Path) -> Path:
    """A clean single-class, single-target YOLO dataset with train and val."""
    return make_yolo_dataset(tmp_path / "yolo")


@pytest.fixture
def multiclass_dataset(tmp_path: Path) -> Path:
    """A YOLO dataset annotating two classes."""
    return make_yolo_dataset(tmp_path / "multiclass", multi_class=True)


@pytest.fixture
def multitarget_dataset(tmp_path: Path) -> Path:
    """A YOLO dataset with an image carrying two annotations of one class."""
    return make_yolo_dataset(tmp_path / "multitarget", multi_target=True)
