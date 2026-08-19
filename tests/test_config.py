"""Config resolution: presets, merging, validation, snapshotting (FR-2.1–2.5, FR-3.4)."""

from __future__ import annotations

import pytest

from dart.config import ConfigResolver, RunConfig
from dart.errors import ConfigurationError

PAPER_VARIANTS = [
    ("dart-b", "identity", "gap", "se", "relu6", "bce", "smooth_l1"),
    ("dart-b-ciou", "identity", "gap", "se", "relu6", "focal", "ciou"),
    ("dart-s", "identity", "spp", "se", "relu6", "focal", "ciou"),
    ("dart-d", "identity", "decoupled_spp", "se", "relu6", "focal", "ciou"),
    ("dart-f", "fpn_lite", "gap", "se", "relu6", "focal", "ciou"),
    ("dart-x", "fpn_lite", "spp", "coord", "silu", "focal", "ciou"),
    ("dart-xd", "fpn_lite", "decoupled_spp", "coord", "silu", "focal", "ciou"),
]


def test_all_seven_paper_variants_ship_as_presets():
    assert ConfigResolver().available_presets() == sorted(v[0] for v in PAPER_VARIANTS)


@pytest.mark.parametrize("name,neck,head,attention,act,conf,bbox", PAPER_VARIANTS)
def test_preset_matches_the_papers_variant_table(name, neck, head, attention, act, conf, bbox):
    cfg = ConfigResolver().resolve(name)
    assert cfg.model.neck.name == neck
    assert cfg.model.head.name == head
    assert cfg.model.act == act
    assert cfg.train.loss.conf == conf
    assert cfg.train.loss.bbox == bbox
    # Attention lives on blocks 2 and 3 only, per the paper's ablation table.
    assert [b.attention for b in cfg.model.backbone.blocks] == [
        "none",
        attention,
        attention,
        "none",
    ]


def test_every_preset_shares_the_same_block_sequence():
    sequences = {
        tuple(b.type for b in ConfigResolver().resolve(name).model.backbone.blocks)
        for name, *_ in PAPER_VARIANTS
    }
    assert sequences == {("repconv", "inverted_res", "inverted_res", "ghost")}


def test_dart_b_and_b_ciou_differ_only_in_the_loss():
    resolver = ConfigResolver()
    base = resolver.resolve("dart-b").to_dict()
    ciou = resolver.resolve("dart-b-ciou").to_dict()
    base.pop("variant"), ciou.pop("variant")
    assert base["model"] == ciou["model"]
    assert base["train"]["loss"] != ciou["train"]["loss"]


def test_keyword_shorthands_and_dotted_paths_both_resolve():
    cfg = ConfigResolver().resolve(
        "dart-xd", imgsz=320, epochs=7, width=0.5, **{"train.early_stop.patience": 11}
    )
    assert cfg.model.imgsz == 320
    assert cfg.train.epochs == 7
    assert cfg.model.backbone.width_mult == 0.5
    assert cfg.train.early_stop.patience == 11


def test_resolved_config_roundtrips_through_yaml(tmp_path):
    cfg = ConfigResolver().resolve("dart-x", imgsz=160, seed=3)
    path = cfg.to_yaml(tmp_path / "config.yaml")
    assert RunConfig.from_yaml(path).to_dict() == cfg.to_dict()


def test_snapshot_contains_defaults_the_user_never_typed():
    """FR-3.4: the snapshot is the *resolved* config, not the command line."""
    snapshot = ConfigResolver().resolve("dart-xd", epochs=5).to_dict()
    assert snapshot["train"]["early_stop"]["patience"] == 40
    assert snapshot["train"]["lr"] == 1e-3
    assert snapshot["data"]["augment"]["hflip"] == 0.5


def test_fingerprint_is_stable_and_config_sensitive():
    resolver = ConfigResolver()
    a = resolver.resolve("dart-xd", seed=0)
    b = resolver.resolve("dart-xd", seed=0)
    c = resolver.resolve("dart-xd", seed=1)
    assert a.fingerprint() == b.fingerprint()
    assert a.fingerprint() != c.fingerprint()


def test_yaml_config_may_extend_a_preset(tmp_path):
    path = tmp_path / "mine.yaml"
    path.write_text("extends: dart-xd\nmodel:\n  imgsz: 224\n", encoding="utf-8")
    cfg = ConfigResolver().resolve(path)
    assert cfg.model.imgsz == 224
    assert cfg.model.head.name == "decoupled_spp"


def test_block_list_is_fully_replaceable_from_config(tmp_path):
    path = tmp_path / "mine.yaml"
    path.write_text(
        "extends: dart-b\n"
        "model:\n"
        "  backbone:\n"
        "    blocks:\n"
        "      - {type: repconv, out_ch: 8, stride: 2}\n"
        "      - {type: dw_sep, out_ch: 16, stride: 2, attention: coord}\n",
        encoding="utf-8",
    )
    cfg = ConfigResolver().resolve(path)
    assert [b.type for b in cfg.model.backbone.blocks] == ["repconv", "dw_sep"]


def test_unknown_variant_lists_the_available_ones():
    with pytest.raises(ConfigurationError, match="dart-xd"):
        ConfigResolver().resolve("dart-zzz")


def test_unknown_config_key_is_rejected_rather_than_ignored(tmp_path):
    path = tmp_path / "typo.yaml"
    path.write_text("extends: dart-b\ntrain:\n  epocs: 10\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="epocs"):
        ConfigResolver().resolve(path)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"model.head.name": "nope"}, "not a registered head"),
        ({"model.neck.name": "nope"}, "not a registered neck"),
        ({"train.loss.conf": "hinge"}, "confidence loss"),
        ({"train.loss.bbox": "giou"}, "box loss"),
        ({"train.epochs": 0}, "epochs"),
        ({"model.backbone.width_mult": 0}, "width_mult"),
        ({"model.num_classes": 2}, "single-class"),
        ({"train.early_stop.direction": "sideways"}, "direction"),
    ],
)
def test_invalid_values_fail_with_an_actionable_message(overrides, message):
    with pytest.raises(ConfigurationError, match=message):
        ConfigResolver().resolve("dart-b", **overrides)


def test_grayscale_derives_the_input_channel_count():
    cfg = ConfigResolver().resolve("dart-b", grayscale=True)
    assert cfg.model.backbone.in_channels == 1


def test_explicit_channel_mismatch_still_errors():
    with pytest.raises(ConfigurationError, match="in_channels"):
        ConfigResolver().resolve("dart-b", grayscale=True, **{"model.backbone.in_channels": 3})


def test_unset_keywords_do_not_clobber_configured_values():
    resolver = ConfigResolver()
    cfg = resolver.resolve("dart-b", epochs=42)
    assert resolver.resolve(cfg, epochs=None).train.epochs == 42


def test_out_of_range_tap_index_is_caught_at_config_time():
    with pytest.raises(ConfigurationError, match="tap_indices"):
        ConfigResolver().resolve(
            "dart-b",
            **{
                "model.backbone.blocks": [{"type": "ghost", "out_ch": 16, "stride": 2}],
                "model.backbone.tap_indices": [3],
            },
        )
