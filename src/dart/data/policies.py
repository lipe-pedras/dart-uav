"""Target-selection and class-selection policies.

Both are explicit, configured choices. There is no code path that constructs a
:class:`ClassSelector` from inference (FR-1.4), and no target policy is applied
unless the user named one (FR-1.3) — the validator raises first.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..errors import ConfigurationError

#: Maximum boxes a ``keep_all`` sample may carry into a batch. Fixed-width
#: padding keeps the collate function trivial and the batch shape static, which
#: is what ONNX export and the ``(B, N, ...)`` head contract both want.
MAX_TARGETS = 32


class TargetPolicy:
    """Decides which of an image's annotations survive into training."""

    name = "base"

    def apply(self, boxes: np.ndarray, class_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Filter ``(n, 4)`` boxes and ``(n,)`` class ids down to the kept set."""
        raise NotImplementedError


class LargestOnlyPolicy(TargetPolicy):
    """Keep the single largest-area annotation — the paper's convention.

    Ties are broken by original order, so the policy is deterministic.
    """

    name = "largest_only"

    def apply(self, boxes: np.ndarray, class_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(boxes) <= 1:
            return boxes, class_ids
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = int(np.argmax(areas))
        return boxes[keep : keep + 1], class_ids[keep : keep + 1]


class KeepAllPolicy(TargetPolicy):
    """Keep every annotation, up to :data:`MAX_TARGETS`.

    Only meaningful once a head emits ``N > 1``; it exists now so multi-target
    support is an added head rather than a change to the ingestion path
    (NFR-8).
    """

    name = "keep_all"

    def __init__(self, max_targets: int = MAX_TARGETS) -> None:
        self.max_targets = max_targets

    def apply(self, boxes: np.ndarray, class_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return boxes[: self.max_targets], class_ids[: self.max_targets]


POLICIES: Dict[str, type] = {
    "largest_only": LargestOnlyPolicy,
    "keep_all": KeepAllPolicy,
}


def build_policy(name: str) -> TargetPolicy:
    """Instantiate a target policy by name.

    Raises:
        ConfigurationError: If ``name`` is ``None`` or unknown. A missing
            policy is an error, never a silently-chosen default (FR-1.3).
    """
    if name is None:
        raise ConfigurationError(
            "data.target_policy is not set. DART will not choose one for you: "
            "set it to 'largest_only' (the paper's convention) or 'keep_all'."
        )
    key = str(name).lower()
    if key not in POLICIES:
        raise ConfigurationError(
            f"data.target_policy = '{name}' is unknown; use one of {sorted(POLICIES)}"
        )
    return POLICIES[key]()


class ClassSelector:
    """Keeps one class as the detection target and drops the rest.

    Constructed *only* when ``data.target_class`` is explicitly configured.
    Everything not selected becomes background — which is exactly the
    ``medicalkit-drone-dataset`` case from the paper (class 0 = landing base,
    class 1 = package): keeping the package and treating the base as
    background must be a written choice, never a default.

    Args:
        target_class: The class id to keep.
    """

    def __init__(self, target_class: int) -> None:
        if target_class is None:
            raise ConfigurationError("ClassSelector requires an explicit target_class")
        self.target_class = int(target_class)

    def apply(self, boxes: np.ndarray, class_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        keep = class_ids == self.target_class
        return boxes[keep], class_ids[keep]

    def __repr__(self) -> str:
        return f"ClassSelector(target_class={self.target_class})"


class ClassRemapper:
    """Rewrites class ids at ingestion time, and drops those mapped to nothing.

    Constructed *only* when ``data.class_map`` is explicitly configured, and
    applied before :class:`ClassSelector`, so a dataset whose ids do not match
    the user's intended labelling can be brought into line without editing the
    annotations on disk. Ids absent from the map are left untouched; an id
    mapped to ``None`` becomes background.

    Args:
        class_map: ``old_id -> new_id`` (or ``old_id -> None`` to drop).
    """

    def __init__(self, class_map: Dict[int, Optional[int]]) -> None:
        if not class_map:
            raise ConfigurationError("ClassRemapper requires a non-empty class_map")
        self.class_map = {int(k): (None if v is None else int(v)) for k, v in class_map.items()}

    def apply(self, boxes: np.ndarray, class_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(class_ids) == 0:
            return boxes, class_ids
        remapped = class_ids.copy()
        keep = np.ones(len(class_ids), dtype=bool)
        for index, cid in enumerate(class_ids):
            if int(cid) not in self.class_map:
                continue
            target = self.class_map[int(cid)]
            if target is None:
                keep[index] = False
            else:
                remapped[index] = target
        return boxes[keep], remapped[keep]

    def __repr__(self) -> str:
        return f"ClassRemapper(class_map={self.class_map})"
