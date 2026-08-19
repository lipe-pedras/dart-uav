"""Format adapters: any supported annotation layout -> one internal form.

Parsing is delegated to ``supervision`` (design principle 3, NFR-5) rather than
hand-written per-format parsers. The one exception is CreateML: ``supervision``
ships no reader for it, so a small JSON reader lives here behind the same
:class:`FormatAdapter` interface, and swapping it for an upstream adapter later
touches nothing else.

Everything downstream sees :class:`RawAnnotations`, so the training engine is
format-agnostic (FR-1.2). Boxes stay in **absolute pixel xyxy** at this stage;
normalisation happens in :class:`~dart.data.dataset.DARTDataset`, where the
image is being read anyway and its true size is known for free.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..errors import DatasetError

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass
class ImageAnnotations:
    """Annotations of a single image.

    Attributes:
        path: Image file path.
        boxes: ``(n, 4)`` float array of absolute-pixel ``xyxy`` boxes.
        class_ids: ``(n,)`` int array of class ids, aligned with ``boxes``.
    """

    path: Path
    boxes: np.ndarray
    class_ids: np.ndarray

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.class_ids = np.asarray(self.class_ids, dtype=np.int64).reshape(-1)
        if len(self.boxes) != len(self.class_ids):
            raise DatasetError(
                f"{self.path}: {len(self.boxes)} boxes but {len(self.class_ids)} class ids"
            )

    def __len__(self) -> int:
        return len(self.boxes)


@dataclass
class RawAnnotations:
    """A whole split, before any policy or class selection is applied.

    The validator (FR-1.7) deliberately runs on *this*, not on the filtered
    result: after a target policy has dropped the extra boxes, the violation it
    is meant to catch would already be invisible.
    """

    items: List[ImageAnnotations] = field(default_factory=list)
    class_names: Dict[int, str] = field(default_factory=dict)
    root: Optional[Path] = None
    format: str = "unknown"

    def __len__(self) -> int:
        return len(self.items)

    def class_ids_present(self) -> List[int]:
        found = set()
        for item in self.items:
            found.update(int(c) for c in item.class_ids)
        return sorted(found)

    def max_annotations_per_image(self) -> int:
        return max((len(item) for item in self.items), default=0)


class FormatAdapter:
    """Interface every annotation-format reader implements."""

    name = "base"

    def read(self, images_dir: Path, annotations: Path) -> RawAnnotations:
        """Read one split.

        Args:
            images_dir: Directory holding the split's images.
            annotations: Directory or file holding the split's annotations.
        """
        raise NotImplementedError

    # ── Shared helpers ──────────────────────────────────────────────────

    @staticmethod
    def _from_supervision(dataset, root: Path, fmt: str) -> RawAnnotations:
        """Convert a ``supervision.DetectionDataset`` to `RawAnnotations`."""
        items: List[ImageAnnotations] = []
        for image_path, detections in dataset.annotations.items():
            boxes = (
                np.zeros((0, 4), dtype=np.float32)
                if detections.xyxy is None
                else np.asarray(detections.xyxy, dtype=np.float32)
            )
            class_ids = (
                np.zeros((0,), dtype=np.int64)
                if detections.class_id is None
                else np.asarray(detections.class_id, dtype=np.int64)
            )
            items.append(ImageAnnotations(Path(image_path), boxes, class_ids))
        items.sort(key=lambda item: str(item.path))
        names = {i: str(n) for i, n in enumerate(dataset.classes or [])}
        return RawAnnotations(items=items, class_names=names, root=root, format=fmt)


class YoloAdapter(FormatAdapter):
    """YOLO layout: ``images/<split>`` plus ``labels/<split>`` and a data YAML."""

    name = "yolo"

    def read(self, images_dir: Path, annotations: Path) -> RawAnnotations:
        import supervision as sv

        data_yaml = _find_data_yaml(images_dir, annotations)
        dataset = sv.DetectionDataset.from_yolo(
            images_directory_path=str(images_dir),
            annotations_directory_path=str(annotations),
            data_yaml_path=str(data_yaml) if data_yaml else "",
        )
        return self._from_supervision(dataset, images_dir.parent, self.name)


class CocoAdapter(FormatAdapter):
    """COCO layout: an images directory plus one ``_annotations.coco.json``."""

    name = "coco"

    def read(self, images_dir: Path, annotations: Path) -> RawAnnotations:
        import supervision as sv

        path = annotations if annotations.is_file() else _find_json(annotations, "coco")
        dataset = sv.DetectionDataset.from_coco(
            images_directory_path=str(images_dir), annotations_path=str(path)
        )
        return self._from_supervision(dataset, images_dir.parent, self.name)


class VocAdapter(FormatAdapter):
    """Pascal VOC layout: an images directory plus per-image XML annotations."""

    name = "voc"

    def read(self, images_dir: Path, annotations: Path) -> RawAnnotations:
        import supervision as sv

        dataset = sv.DetectionDataset.from_pascal_voc(
            images_directory_path=str(images_dir),
            annotations_directory_path=str(annotations),
        )
        return self._from_supervision(dataset, images_dir.parent, self.name)


class CreateMLAdapter(FormatAdapter):
    """CreateML layout: an images directory plus one JSON array of records.

    Each record is ``{"image": name, "annotations": [{"label": str,
    "coordinates": {"x": cx, "y": cy, "width": w, "height": h}}]}`` with
    absolute-pixel, centre-based coordinates.
    """

    name = "createml"

    def read(self, images_dir: Path, annotations: Path) -> RawAnnotations:
        path = annotations if annotations.is_file() else _find_json(annotations, "createml")
        try:
            records = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DatasetError(f"cannot read CreateML annotations at {path}: {exc}") from exc
        if not isinstance(records, list):
            raise DatasetError(f"{path}: CreateML annotations must be a JSON array")

        labels: List[str] = []
        items: List[ImageAnnotations] = []
        for record in records:
            image_path = images_dir / str(record.get("image", ""))
            boxes, class_ids = [], []
            for ann in record.get("annotations", []) or []:
                label = str(ann.get("label", "object"))
                if label not in labels:
                    labels.append(label)
                coords = ann.get("coordinates", {})
                cx, cy = float(coords["x"]), float(coords["y"])
                width, height = float(coords["width"]), float(coords["height"])
                boxes.append([cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2])
                class_ids.append(labels.index(label))
            items.append(
                ImageAnnotations(
                    image_path,
                    np.array(boxes, dtype=np.float32).reshape(-1, 4),
                    np.array(class_ids, dtype=np.int64),
                )
            )
        items.sort(key=lambda item: str(item.path))
        return RawAnnotations(
            items=items,
            class_names=dict(enumerate(labels)),
            root=images_dir.parent,
            format=self.name,
        )


ADAPTERS: Dict[str, FormatAdapter] = {
    "yolo": YoloAdapter(),
    "coco": CocoAdapter(),
    "voc": VocAdapter(),
    "pascal_voc": VocAdapter(),
    "createml": CreateMLAdapter(),
}


def adapter_for(fmt: str) -> FormatAdapter:
    """Look up an adapter by format name."""
    key = str(fmt).lower()
    if key not in ADAPTERS:
        raise DatasetError(
            f"unsupported dataset format '{fmt}'. Supported: {sorted(set(ADAPTERS))}"
        )
    return ADAPTERS[key]


def detect_format(images_dir: Path, annotations: Path) -> str:
    """Infer the annotation format from what is actually on disk.

    Inferring the *format* is safe — it is a fact about the files, verifiable
    by looking. It is inferring *semantics* (which class matters, which
    annotation to keep) that FR-1.4 and FR-1.3 forbid.
    """
    if annotations.is_file():
        if annotations.suffix.lower() == ".json":
            return "coco" if _looks_like_coco(annotations) else "createml"
        raise DatasetError(f"cannot infer a format from the annotation file {annotations}")

    if not annotations.exists():
        raise DatasetError(f"annotation path does not exist: {annotations}")

    if any(annotations.glob("*.txt")):
        return "yolo"
    if any(annotations.glob("*.xml")):
        return "voc"
    for candidate in sorted(annotations.glob("*.json")):
        return "coco" if _looks_like_coco(candidate) else "createml"
    raise DatasetError(
        f"no .txt, .xml, or .json annotations found under {annotations}; "
        "set data.format explicitly if the layout is non-standard"
    )


def list_images(images_dir: Path) -> List[Path]:
    """Every image file directly under ``images_dir``, sorted."""
    return sorted(
        p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )


def _looks_like_coco(path: Path) -> bool:
    try:
        head = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(head, dict) and "images" in head and "annotations" in head


def _find_json(directory: Path, kind: str) -> Path:
    candidates = sorted(directory.glob("*.json"))
    if not candidates:
        raise DatasetError(f"no {kind} .json annotation file found in {directory}")
    return candidates[0]


def _find_data_yaml(images_dir: Path, annotations: Path) -> Optional[Path]:
    """Locate the YOLO ``data.yaml`` that names the classes, if there is one."""
    seen: Sequence[Path] = (
        images_dir.parent,
        images_dir.parent.parent,
        annotations.parent,
        annotations.parent.parent,
    )
    for directory in seen:
        for name in ("data.yaml", "data.yml", "dataset.yaml"):
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None
