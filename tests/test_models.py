"""Model assembly, the output contract, and RepConv fusion equivalence."""

from __future__ import annotations

import pytest
import torch

from dart.config import ConfigResolver
from dart.errors import RegistryError
from dart.models import BLOCKS, COMPONENTS, Backbone, BackboneBlock, DARTDetector, register_block

VARIANTS = ConfigResolver().available_presets()
RESOLUTIONS = [96, 160, 224, 320]


def build(variant: str, **overrides) -> DARTDetector:
    return DARTDetector(ConfigResolver().resolve(variant, **overrides).model)


@pytest.mark.parametrize("variant", VARIANTS)
def test_every_variant_honours_the_output_contract(variant):
    model = build(variant).eval()
    conf, boxes = model(torch.randn(2, 3, 96, 96))
    assert conf.shape == (2, 1, 1)
    assert boxes.shape == (2, 1, 4)
    assert torch.all((conf >= 0) & (conf <= 1))
    assert torch.all((boxes >= 0) & (boxes <= 1))


@pytest.mark.parametrize("imgsz", RESOLUTIONS)
def test_all_four_paper_resolutions_work(imgsz):
    model = build("dart-xd", imgsz=imgsz).eval()
    conf, boxes = model(torch.randn(1, 3, imgsz, imgsz))
    assert conf.shape == (1, 1, 1) and boxes.shape == (1, 1, 4)


@pytest.mark.parametrize("width", [0.5, 0.75, 1.0])
def test_width_multiplier_scales_capacity_monotonically(width):
    model = build("dart-xd", width=width)
    assert model.count_params() > 0
    if width == 1.0:
        # The paper's widths: 16 -> 32 -> 48 -> 64.
        assert model.backbone.out_channels_per_block == [16, 32, 48, 64]


def test_width_multiplier_shrinks_the_model():
    assert build("dart-xd", width=0.5).count_params() < build("dart-xd").count_params()


# ── RepConv fusion (FR-5.3) ─────────────────────────────────────────────
# The one place where a regression silently changes every exported model's
# behaviour while leaving training metrics untouched.


@pytest.mark.parametrize("variant", VARIANTS)
def test_repconv_fusion_preserves_outputs(variant):
    torch.manual_seed(0)
    model = build(variant)

    # Give the BatchNorms non-trivial running statistics: with the identity
    # init (mean 0, var 1) the fusion is trivially exact and proves nothing.
    model.train()
    for _ in range(3):
        model(torch.randn(4, 3, 96, 96))
    model.eval()

    sample = torch.randn(2, 3, 96, 96)
    with torch.no_grad():
        before = model(sample)
        model.fuse()
        after = model(sample)

    assert model.is_fused
    for expected, actual in zip(before, after):
        assert torch.allclose(expected, actual, atol=1e-5), (
            f"{variant}: fused output diverged by {(expected - actual).abs().max():.2e}"
        )


def test_fuse_is_idempotent():
    model = build("dart-b").eval()
    model.fuse()
    model.fuse()
    conf, _ = model(torch.randn(1, 3, 96, 96))
    assert conf.shape == (1, 1, 1)


# ── Registry and extensibility (NFR-8) ──────────────────────────────────


def test_every_architectural_axis_is_registry_resolved():
    assert {"repconv", "inverted_res", "ghost", "dw_sep"} <= set(BLOCKS.names())
    assert {"none", "se", "coord"} <= set(COMPONENTS.names("attention"))
    assert {"identity", "fpn_lite"} <= set(COMPONENTS.names("neck"))
    assert {"gap", "spp", "decoupled_spp"} <= set(COMPONENTS.names("head"))


def test_unknown_component_names_are_rejected():
    with pytest.raises(RegistryError):
        BLOCKS.get("does_not_exist")


def test_a_new_block_type_needs_no_change_to_existing_code():
    @register_block("test_passthrough")
    class PassThroughBlock(BackboneBlock):
        def __init__(self, in_ch, out_ch, stride=1, attention="none", act="relu6"):
            super().__init__(in_ch, out_ch, stride, attention, act)
            self.proj = torch.nn.Conv2d(in_ch, out_ch, 1, stride=stride)

        def _forward_impl(self, x):
            return self.proj(x)

    cfg = ConfigResolver().resolve(
        "dart-b",
        **{
            "model.backbone.blocks": [
                {"type": "test_passthrough", "out_ch": 16, "stride": 2, "attention": "coord"},
                {"type": "ghost", "out_ch": 32, "stride": 2},
            ]
        },
    )
    model = DARTDetector(cfg.model).eval()
    conf, boxes = model(torch.randn(1, 3, 96, 96))
    assert conf.shape == (1, 1, 1) and boxes.shape == (1, 1, 4)


def test_attention_attaches_to_any_block_type():
    for block_type in ("repconv", "inverted_res", "ghost", "dw_sep"):
        cfg = ConfigResolver().resolve(
            "dart-b",
            **{
                "model.backbone.blocks": [
                    {"type": block_type, "out_ch": 16, "stride": 2, "attention": "coord"}
                ],
                "model.backbone.tap_indices": [],
                "model.neck.name": "identity",
            },
        )
        model = DARTDetector(cfg.model).eval()
        assert model(torch.randn(1, 3, 96, 96))[0].shape == (1, 1, 1)


def test_fpn_neck_reads_taps_from_the_backbone_not_from_hardcoded_indices():
    cfg = ConfigResolver().resolve(
        "dart-xd",
        **{
            "model.backbone.tap_indices": [2],
            "model.backbone.blocks": [
                {"type": "repconv", "out_ch": 16, "stride": 2},
                {"type": "inverted_res", "out_ch": 24, "stride": 2},
                {"type": "inverted_res", "out_ch": 32, "stride": 1},
                {"type": "inverted_res", "out_ch": 48, "stride": 2},
                {"type": "ghost", "out_ch": 64, "stride": 2},
            ],
        },
    )
    model = DARTDetector(cfg.model).eval()
    assert model.backbone.tap_indices() == [2, 4]
    assert model(torch.randn(1, 3, 96, 96))[0].shape == (1, 1, 1)


# ── Pretrained backbone infrastructure (FR-2.6) ─────────────────────────


def test_backbone_is_constructible_without_neck_or_head():
    cfg = ConfigResolver().resolve("dart-xd")
    backbone = Backbone(cfg.model.backbone.blocks, tap_indices=cfg.model.backbone.tap_indices)
    features = backbone(torch.randn(1, 3, 96, 96))
    assert len(features) == 2


def test_loading_a_backbone_reports_what_loaded_and_what_did_not():
    source = build("dart-xd")
    target = build("dart-xd")
    report = target.load_backbone_weights(source.backbone.state_dict())
    assert report.loaded and not report.skipped_shape and not report.missing

    for name, tensor in target.backbone.state_dict().items():
        assert torch.equal(tensor, source.backbone.state_dict()[name]), name


def test_shape_mismatched_layers_are_skipped_not_fatal():
    wide = build("dart-xd", width=1.0)
    narrow = build("dart-xd", width=0.5)
    report = narrow.load_backbone_weights(wide.backbone.state_dict())
    assert report.skipped_shape, "differing widths must report skipped tensors"
    assert narrow(torch.randn(1, 3, 96, 96))[0].shape == (1, 1, 1)


def test_param_groups_separate_backbone_from_head():
    model = build("dart-xd")
    groups = model.param_groups(lr=1e-3, backbone_lr_mult=0.1)
    assert [g["name"] for g in groups] == ["backbone", "head"]
    assert groups[0]["lr"] == pytest.approx(1e-4)
    assert groups[1]["lr"] == pytest.approx(1e-3)
    total = sum(len(g["params"]) for g in groups)
    assert total == len(list(model.parameters()))


def test_grayscale_models_accept_single_channel_input():
    cfg = ConfigResolver().resolve("dart-b", grayscale=True)
    model = DARTDetector(cfg.model).eval()
    assert model(torch.randn(1, 1, 96, 96))[0].shape == (1, 1, 1)
