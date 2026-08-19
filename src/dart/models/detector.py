"""The assembled detector: backbone + neck + head.

``dart.models`` deliberately knows nothing about ``dart.data`` or
``dart.engine`` (UML design §3), so a detector can be built, run, and exported
with no training machinery present.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn

from ..config.schema import ModelConfig
from .backbone import Backbone, LoadReport
from .heads import build_head
from .necks import build_neck


class DARTDetector(nn.Module):
    """A DART variant, assembled from a resolved :class:`ModelConfig`.

    Forward returns ``(conf, boxes)`` with shapes ``(B, N, 1)`` and
    ``(B, N, 4)``; boxes are ``[cx, cy, w, h]`` normalised to ``[0, 1]``.

    Args:
        cfg: Fully-resolved model configuration.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.imgsz = cfg.imgsz

        self.backbone = Backbone(
            blocks=cfg.backbone.blocks,
            in_channels=cfg.backbone.in_channels,
            width_mult=cfg.backbone.width_mult,
            tap_indices=cfg.backbone.tap_indices,
        )
        self.neck = build_neck(cfg.neck.name, self.backbone.tap_channels(), **cfg.neck.kwargs)
        self.head = build_head(
            cfg.head.name,
            in_ch=self.neck.out_channels,
            num_targets=cfg.head.num_targets,
            act=cfg.act,
            **cfg.head.kwargs,
        )
        self._init_weights()

        if cfg.backbone.pretrained:
            self.load_backbone_weights(cfg.backbone.pretrained)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    # ── Inference ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.neck(self.backbone(x)))

    # ── Deployment ──────────────────────────────────────────────────────

    def fuse(self) -> None:
        """Reparameterize every fusible block for inference (FR-5.3).

        Idempotent, and invoked automatically by
        :class:`~dart.export.manager.ExportManager` and the efficiency
        profiler, so the user never has to remember it.
        """
        self.backbone.fuse()

    @property
    def is_fused(self) -> bool:
        return all(getattr(block, "deployed", True) for block in self.backbone.blocks)

    # ── Pretrained weights and optimisation groups (FR-2.6) ─────────────

    def load_backbone_weights(self, source, strict: bool = False) -> LoadReport:
        """Load backbone-only weights; see :meth:`Backbone.load_backbone_weights`."""
        return self.backbone.load_backbone_weights(source, strict=strict)

    def param_groups(self, lr: float, backbone_lr_mult: float | None = None) -> List[dict]:
        """Optimizer parameter groups with a separate backbone learning rate.

        Randomly-initialised heads want a larger step than a pretrained
        backbone; keeping the two groups separate is what lets the trainer
        honour ``backbone.pretrained_lr_mult``.
        """
        mult = (
            self.cfg.backbone.pretrained_lr_mult if backbone_lr_mult is None else backbone_lr_mult
        )
        backbone_params = list(self.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        other_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        groups = [{"params": backbone_params, "lr": lr * mult, "name": "backbone"}]
        if other_params:
            groups.append({"params": other_params, "lr": lr, "name": "head"})
        return groups

    # ── Introspection ───────────────────────────────────────────────────

    def count_params(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def model_info(self) -> Dict[str, Any]:
        """Parameter count and INT8 size estimate, as reported in the paper."""
        n = self.count_params()
        return {
            "params": n,
            "params_k": round(n / 1e3, 1),
            "est_int8_kb": round(n / 1024, 1),
            "imgsz": self.imgsz,
            "width_mult": self.cfg.backbone.width_mult,
            "neck": self.cfg.neck.name,
            "head": self.cfg.head.name,
        }

    def __repr__(self) -> str:
        info = self.model_info()
        return (
            f"DARTDetector(params={info['params_k']}K, INT8~{info['est_int8_kb']}KB, "
            f"imgsz={info['imgsz']}, width={info['width_mult']}, "
            f"neck={info['neck']}, head={info['head']})"
        )
