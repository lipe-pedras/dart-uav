"""Dataset ingestion, validation, and policies (FR-1.1 – FR-1.7)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import make_coco_dataset, make_createml_dataset, make_voc_dataset

from dart.config import ConfigResolver
from dart.data import (
    AugmentationPipeline,
    ClassSelector,
    DatasetLoader,
    KeepAllPolicy,
    LargestOnlyPolicy,
    build_policy,
    detect_format,
)
from dart.data.augment import HorizontalFlip, RandomCropResize
from dart.errors import ConfigurationError


def load(root, split="train", **overrides):
    cfg = ConfigResolver().resolve("dart-b", data=str(root), **overrides)
    return DatasetLoader().load(cfg.data, cfg.model, split=split, verbose=False)


# ── Format ingestion (FR-1.1, FR-1.2) ───────────────────────────────────


def test_yolo_ingestion(yolo_dataset):
    dataset = load(yolo_dataset, target_policy="largest_only")
    assert len(dataset) == 6
    image, conf, boxes = dataset[0]
    assert image.shape == (3, 96, 96)
    assert conf.shape == (1,) and boxes.shape == (1, 4)


@pytest.mark.parametrize(
    "builder,fmt",
    [
        (make_coco_dataset, "coco"),
        (make_voc_dataset, "voc"),
        (make_createml_dataset, "createml"),
    ],
)
def test_every_supported_format_normalises_to_one_representation(tmp_path, builder, fmt):
    root = builder(tmp_path / fmt)
    dataset = load(root, target_policy="largest_only")
    assert len(dataset) == 4
    image, conf, boxes = dataset[0]
    # The training engine sees the same tensors regardless of input format.
    assert image.shape == (3, 96, 96)
    assert conf.shape == (1,) and boxes.shape == (1, 4)
    assert float(conf[0]) == 1.0
    assert torch.all((boxes >= 0) & (boxes <= 1))


def test_format_detection_identifies_each_layout(tmp_path):
    from dart.data.loader import DatasetLoader as Loader

    for builder, expected in (
        (make_coco_dataset, "coco"),
        (make_voc_dataset, "voc"),
        (make_createml_dataset, "createml"),
    ):
        root = builder(tmp_path / expected)
        cfg = ConfigResolver().resolve("dart-b", data=str(root))
        paths = Loader().locate_split(cfg.data, "train")
        assert detect_format(paths.images, paths.annotations) == expected


def test_boxes_survive_the_roundtrip_to_normalised_coordinates(yolo_dataset):
    dataset = load(yolo_dataset, target_policy="largest_only")
    dataset.augment = AugmentationPipeline([])  # isolate ingestion from augmentation
    _, conf, boxes = dataset[0]
    assert float(conf[0]) == 1.0
    # conftest draws image 0 at (0.3, 0.3) with size 0.25.
    assert boxes[0, 0] == pytest.approx(0.3, abs=0.03)
    assert boxes[0, 2] == pytest.approx(0.25, abs=0.03)


def test_negative_images_are_labelled_as_background(yolo_dataset):
    dataset = load(yolo_dataset, target_policy="largest_only")
    confidences = [float(dataset[i][1][0]) for i in range(len(dataset))]
    assert 0.0 in confidences and 1.0 in confidences


# ── Validation: explicit failure over silent adaptation (FR-1.7) ────────


def test_multiclass_dataset_against_single_class_model_errors(multiclass_dataset):
    with pytest.raises(ConfigurationError) as excinfo:
        load(multiclass_dataset, target_policy="largest_only")
    message = str(excinfo.value)
    assert "target_class" in message, "the error must name the resolving config key"
    assert "2 classes" in message


def test_multi_annotation_image_against_single_target_model_errors(multitarget_dataset):
    with pytest.raises(ConfigurationError) as excinfo:
        load(multitarget_dataset)
    message = str(excinfo.value)
    assert "target_policy" in message, "the error must name the resolving config key"
    assert "img001" in message, "the error must name the offending image"


def test_explicit_class_selection_resolves_the_multiclass_error(multiclass_dataset):
    dataset = load(multiclass_dataset, target_class=1, target_policy="largest_only")
    assert isinstance(dataset.class_selector, ClassSelector)
    assert dataset.class_selector.target_class == 1


def test_explicit_policy_resolves_the_multitarget_error(multitarget_dataset):
    dataset = load(multitarget_dataset, target_policy="largest_only")
    _, conf, boxes = dataset[1]
    assert conf.shape == (1,) and boxes.shape == (1, 4)


def test_selecting_an_absent_class_errors(multiclass_dataset):
    with pytest.raises(ConfigurationError, match="not annotated"):
        load(multiclass_dataset, target_class=7, target_policy="largest_only")


def test_validation_happens_before_the_policy_is_applied(multitarget_dataset):
    """The whole point of validating raw annotations.

    With ``largest_only`` in place the extra box is gone by the time the
    dataset exists — so if validation ran afterwards, the violation would be
    invisible and the system would have silently adapted.
    """
    with pytest.raises(ConfigurationError):
        load(multitarget_dataset)  # no policy set -> must raise
    load(multitarget_dataset, target_policy="largest_only")  # policy set -> fine


# ── Policies (FR-1.3, FR-1.4) ───────────────────────────────────────────


def test_largest_only_keeps_the_biggest_box():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 40, 40], [0, 0, 5, 5]], dtype=np.float32)
    kept, ids = LargestOnlyPolicy().apply(boxes, np.zeros(3, dtype=np.int64))
    assert kept.shape == (1, 4) and kept[0, 2] == 40 and len(ids) == 1


def test_keep_all_keeps_everything():
    boxes = np.zeros((5, 4), dtype=np.float32)
    kept, _ = KeepAllPolicy().apply(boxes, np.zeros(5, dtype=np.int64))
    assert len(kept) == 5


def test_a_missing_policy_is_an_error_not_a_default():
    with pytest.raises(ConfigurationError, match="will not choose one for you"):
        build_policy(None)


def test_class_selector_is_never_constructed_by_inference(yolo_dataset):
    dataset = load(yolo_dataset, target_policy="largest_only")
    assert dataset.class_selector is None


def test_class_selector_drops_non_target_classes():
    boxes = np.array([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=np.float32)
    kept, ids = ClassSelector(1).apply(boxes, np.array([0, 1], dtype=np.int64))
    assert len(kept) == 1 and ids[0] == 1


# ── Statistics (FR-1.5) ─────────────────────────────────────────────────


def test_stats_report_positive_and_negative_counts(yolo_dataset):
    stats = load(yolo_dataset, target_policy="largest_only").stats
    assert stats.images == 6
    assert stats.positives + stats.negatives == stats.images
    assert 0.0 < stats.positive_rate < 1.0


# ── Augmentation (FR-1.6) ───────────────────────────────────────────────


def test_augmentation_is_on_for_train_and_off_for_val(yolo_dataset):
    train = load(yolo_dataset, target_policy="largest_only", split="train")
    val = load(yolo_dataset, target_policy="largest_only", split="val")
    assert len(train.augment) > 0
    assert len(val.augment) == 0


def test_the_pipeline_is_swappable(yolo_dataset):
    dataset = load(yolo_dataset, target_policy="largest_only")
    dataset.augment = AugmentationPipeline([HorizontalFlip(p=1.0)])
    assert len(dataset.augment) == 1


def test_horizontal_flip_mirrors_the_box_centre():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    boxes = np.array([[0.25, 0.5, 0.2, 0.2]], dtype=np.float32)
    _, flipped = HorizontalFlip(p=1.0)(image, boxes)
    assert flipped[0, 0] == pytest.approx(0.75)


def test_geometry_transforms_skip_negative_samples():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    empty = np.zeros((0, 4), dtype=np.float32)
    out_image, out_boxes = RandomCropResize(p=1.0)(image, empty)
    assert out_boxes.shape == (0, 4)
    assert out_image.shape == image.shape


def test_augmentation_keeps_boxes_inside_the_image(yolo_dataset):
    dataset = load(yolo_dataset, target_policy="largest_only")
    for _ in range(20):
        _, _, boxes = dataset[0]
        assert torch.all((boxes >= 0) & (boxes <= 1))


# ── Portability (NFR-3) ─────────────────────────────────────────────────


def test_dataset_is_picklable_for_spawn_start_method(yolo_dataset):
    import pickle

    dataset = load(yolo_dataset, target_policy="largest_only")
    restored = pickle.loads(pickle.dumps(dataset))
    assert len(restored) == len(dataset)
    assert restored[0][0].shape == dataset[0][0].shape


def test_paths_are_pathlib_objects(yolo_dataset):
    from pathlib import Path

    dataset = load(yolo_dataset, target_policy="largest_only")
    assert all(isinstance(item.path, Path) for item in dataset.items)


def test_missing_split_names_the_layouts_it_looked_for(tmp_path):
    from dart.errors import DatasetError

    (tmp_path / "empty").mkdir()
    with pytest.raises(DatasetError, match="could not find"):
        load(tmp_path / "empty", target_policy="largest_only")


# ── Class remapping (FR-1.4) ────────────────────────────────────────────


def test_class_remapping_is_applied_only_when_configured(yolo_dataset):
    dataset = load(yolo_dataset, target_policy="largest_only")
    assert dataset.class_remapper is None


def test_an_explicit_class_map_resolves_a_multiclass_dataset(multiclass_dataset):
    """Merging two annotated classes into one is a written decision."""
    dataset = load(
        multiclass_dataset,
        target_policy="largest_only",
        **{"data.class_map": {1: 0}},
    )
    assert dataset.class_remapper is not None
    _, conf, _ = dataset[0]
    assert float(conf[0]) == 1.0


def test_mapping_a_class_to_null_turns_it_into_background(multiclass_dataset):
    dataset = load(
        multiclass_dataset,
        target_policy="largest_only",
        **{"data.class_map": {1: None}},
    )
    boxes = np.array([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=np.float32)
    kept, ids = dataset.class_remapper.apply(boxes, np.array([0, 1], dtype=np.int64))
    assert len(kept) == 1 and ids.tolist() == [0]


def test_remapping_runs_before_class_selection(multiclass_dataset):
    """target_class names an id in the remapped space, not the raw one."""
    dataset = load(
        multiclass_dataset,
        target_policy="largest_only",
        target_class=5,
        **{"data.class_map": {1: 5}},
    )
    boxes = np.array([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=np.float32)
    class_ids = np.array([0, 1], dtype=np.int64)
    boxes, class_ids = dataset.class_remapper.apply(boxes, class_ids)
    kept, ids = dataset.class_selector.apply(boxes, class_ids)
    assert ids.tolist() == [5]
