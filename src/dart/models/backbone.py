"""The DART backbone: a configurable sequence of registered blocks.

The backbone is deliberately constructible on its own, with no neck and no
head (FR-2.6), so pretrained weights can be produced and loaded independently
of any detection head.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from ..errors import CheckpointError
from .registry import build_block, scale_channels


@dataclass
class LoadReport:
    """Outcome of loading an external ``state_dict`` into a backbone.

    Attributes:
        loaded: Parameter names that were copied in.
        skipped_shape: ``name -> (checkpoint_shape, model_shape)`` for entries
            that matched by name but not by shape.
        missing: Model parameters no checkpoint entry supplied.
        unexpected: Checkpoint entries with no counterpart in the model.
    """

    loaded: List[str] = field(default_factory=list)
    skipped_shape: Dict[str, Tuple[tuple, tuple]] = field(default_factory=dict)
    missing: List[str] = field(default_factory=list)
    unexpected: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"loaded {len(self.loaded)} tensors, "
            f"skipped {len(self.skipped_shape)} on shape mismatch, "
            f"{len(self.missing)} missing, {len(self.unexpected)} unexpected"
        )


class Backbone(nn.Module):
    """Feature extractor built from a list of block specifications.

    The block *sequence itself* is configuration, not code: a backbone with six
    blocks, or with a different block type in position 4, is a YAML edit
    (design principle 1, NFR-8).

    Args:
        blocks: Ordered block specifications
            (:class:`~dart.config.schema.BlockSpec`).
        in_channels: Channels of the input image (3 for RGB, 1 for grayscale).
        width_mult: Channel multiplier applied to every block's ``out_ch``.
        tap_indices: Zero-based indices of the block outputs retained as
            feature maps for a neck. The final block is always retained.
    """

    def __init__(
        self,
        blocks: Sequence,
        in_channels: int = 3,
        width_mult: float = 1.0,
        tap_indices: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if not blocks:
            raise ValueError("a backbone needs at least one block")

        self.width_mult = width_mult
        self.in_channels = in_channels

        modules: List[nn.Module] = []
        channels: List[int] = []
        in_ch = in_channels
        for spec in blocks:
            block = build_block(spec, in_ch=in_ch, width_mult=width_mult)
            modules.append(block)
            in_ch = block.out_ch
            channels.append(in_ch)

        self.blocks = nn.ModuleList(modules)
        self.out_channels_per_block = channels
        self.out_channels = channels[-1]

        last = len(modules) - 1
        taps = list(tap_indices) if tap_indices else []
        taps = [t if t >= 0 else len(modules) + t for t in taps]
        if last not in taps:
            taps.append(last)
        for t in taps:
            if not 0 <= t < len(modules):
                raise ValueError(
                    f"tap index {t} is out of range for a {len(modules)}-block backbone"
                )
        self._tap_indices = sorted(set(taps))

    # ── Introspection ───────────────────────────────────────────────────

    def tap_indices(self) -> List[int]:
        """Block indices whose outputs :meth:`forward` returns."""
        return list(self._tap_indices)

    def tap_channels(self) -> List[int]:
        """Channel count of each tapped feature map, in tap order."""
        return [self.out_channels_per_block[i] for i in self._tap_indices]

    # ── Forward / fuse ──────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Return the tapped feature maps, ordered shallowest to deepest."""
        taps: List[torch.Tensor] = []
        tap_set = self._tap_indices
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in tap_set:
                taps.append(x)
        return taps

    def fuse(self) -> None:
        """Fuse every fusible block. Unconditional, so new fusible block types
        need no change here (FR-5.3)."""
        for block in self.blocks:
            block.fuse()

    # ── Pretrained weights (FR-2.6) ─────────────────────────────────────

    def load_backbone_weights(
        self, source: str | Path | Dict[str, torch.Tensor], strict: bool = False
    ) -> LoadReport:
        """Load externally-supplied backbone weights.

        Only shape-compatible tensors are copied; the rest are reported rather
        than raising, so a backbone pretrained at a different width or with a
        different stem still contributes what it can.

        Args:
            source: A ``state_dict``, a local ``.pt``/``.pth`` path, or a
                ``"repo_id:filename"`` Hugging Face Hub reference.
            strict: When ``True``, raise :class:`~dart.errors.CheckpointError`
                if anything was skipped or missing.

        Returns:
            A :class:`LoadReport` naming exactly what happened to every tensor.
        """
        state = _resolve_state_dict(source)
        state = _strip_prefixes(state, ("backbone.", "module."))

        own = self.state_dict()
        report = LoadReport()
        to_load: Dict[str, torch.Tensor] = {}
        for name, tensor in state.items():
            if name not in own:
                report.unexpected.append(name)
            elif tuple(tensor.shape) != tuple(own[name].shape):
                report.skipped_shape[name] = (tuple(tensor.shape), tuple(own[name].shape))
            else:
                to_load[name] = tensor
                report.loaded.append(name)
        report.missing = [k for k in own if k not in to_load]

        self.load_state_dict(to_load, strict=False)

        if strict and (report.skipped_shape or report.missing):
            raise CheckpointError(
                f"strict backbone load failed: {report.summary()}; "
                f"skipped={sorted(report.skipped_shape)} missing={report.missing}"
            )
        return report


def _resolve_state_dict(source) -> Dict[str, torch.Tensor]:
    """Turn a state_dict / path / Hub reference into a flat tensor mapping."""
    if isinstance(source, dict):
        obj = source
    else:
        path = Path(str(source))
        if not path.exists():
            path = Path(_download_from_hub(str(source)))
        obj = torch.load(path, map_location="cpu", weights_only=False)

    for key in ("backbone_state", "model_state", "state_dict", "model"):
        if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict):
        raise CheckpointError(f"cannot interpret {source!r} as a state_dict")
    return {k: v for k, v in obj.items() if isinstance(v, torch.Tensor)}


def _download_from_hub(reference: str) -> str:
    """Resolve ``"repo_id:filename"`` (or ``"repo_id"``) to a local cached file."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise CheckpointError(
            f"'{reference}' is not a local file and huggingface-hub is not installed"
        ) from exc

    repo_id, _, filename = reference.partition(":")
    return hf_hub_download(repo_id=repo_id, filename=filename or "backbone.pt")


def _strip_prefixes(
    state: Dict[str, torch.Tensor], prefixes: Sequence[str]
) -> Dict[str, torch.Tensor]:
    """Drop a leading wrapper prefix when *every* key carries it."""
    for prefix in prefixes:
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix) :]: v for k, v in state.items()}
    return state


def infer_backbone_channels(blocks: Sequence, width_mult: float) -> List[int]:
    """Channel counts a backbone built from ``blocks`` would produce.

    Lets a neck or head be sized without instantiating the backbone.
    """
    return [scale_channels(spec.out_ch, width_mult) for spec in blocks]
