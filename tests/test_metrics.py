"""Metric computation against hand-checked fixtures (FR-4.1).

These are the tests that stop a silent bug from corrupting published numbers,
so every expected value here is worked out by hand in the test itself rather
than recorded from a previous run.
"""

from __future__ import annotations

import pytest
import torch

from dart.losses.functional import box_iou_pairwise, ciou_loss, focal_bce
from dart.metrics import EfficiencyProfiler, MetricSet


def as_batch(conf, boxes):
    """Shape flat fixtures into the ``(B, N, ...)`` contract."""
    conf_t = torch.tensor(conf, dtype=torch.float32).view(-1, 1, 1)
    boxes_t = torch.tensor(boxes, dtype=torch.float32).view(-1, 1, 4)
    return conf_t, boxes_t


BOX = [0.5, 0.5, 0.4, 0.4]
EMPTY = [0.0, 0.0, 0.0, 0.0]


def test_perfect_predictions_score_one_everywhere():
    metrics = MetricSet(conf_threshold=0.5, imgsz=100)
    pred = as_batch([0.9, 0.9, 0.05], [BOX, BOX, EMPTY])
    truth = as_batch([1.0, 1.0, 0.0], [BOX, BOX, EMPTY])
    metrics.update(pred[0], pred[1], truth[0].squeeze(-1), truth[1])

    result = metrics.compute()
    assert result.map50 == pytest.approx(1.0)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)
    assert result.mean_iou == pytest.approx(1.0)


def test_a_confident_but_badly_localised_box_is_a_false_positive():
    """Two GT positives; one is predicted far away (IoU 0), one is exact.

    Hand-checked: tp = 1, fp = 1 -> precision 0.5; the missed object still
    counts in recall's denominator -> recall 0.5; F1 0.5.
    """
    metrics = MetricSet(conf_threshold=0.5, imgsz=100)
    pred = as_batch([0.9, 0.9], [BOX, [0.1, 0.1, 0.05, 0.05]])
    truth = as_batch([1.0, 1.0], [BOX, BOX])
    metrics.update(pred[0], pred[1], truth[0].squeeze(-1), truth[1])

    result = metrics.compute()
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(0.5)
    assert result.f1 == pytest.approx(0.5)


def test_poor_localisation_cannot_inflate_recall():
    """Every GT object is detected confidently, all of them badly located."""
    metrics = MetricSet(conf_threshold=0.5, imgsz=100)
    pred = as_batch([0.99, 0.99], [[0.1, 0.1, 0.05, 0.05]] * 2)
    truth = as_batch([1.0, 1.0], [BOX, BOX])
    metrics.update(pred[0], pred[1], truth[0].squeeze(-1), truth[1])

    result = metrics.compute()
    assert result.recall == pytest.approx(0.0)
    assert result.precision == pytest.approx(0.0)


def test_a_detection_announced_on_a_negative_image_is_a_false_positive():
    metrics = MetricSet(conf_threshold=0.5, imgsz=100)
    pred = as_batch([0.9, 0.9], [BOX, BOX])
    truth = as_batch([1.0, 0.0], [BOX, EMPTY])
    metrics.update(pred[0], pred[1], truth[0].squeeze(-1), truth[1])

    result = metrics.compute()
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(1.0)


def test_mean_iou_averages_over_positive_images_only():
    """IoUs 1.0 and 0.5 over two positives, plus one negative that is ignored."""
    half = [0.5, 0.5, 0.2, 0.4]  # exactly half the width of BOX -> IoU 0.5
    metrics = MetricSet(conf_threshold=0.5, imgsz=100)
    pred = as_batch([0.9, 0.9, 0.9], [BOX, half, [0.9, 0.9, 0.4, 0.4]])
    truth = as_batch([1.0, 1.0, 0.0], [BOX, BOX, EMPTY])
    metrics.update(pred[0], pred[1], truth[0].squeeze(-1), truth[1])

    assert metrics.compute().mean_iou == pytest.approx((1.0 + 0.5) / 2, abs=1e-4)


def test_threshold_sweep_finds_the_operating_point_a_fixed_threshold_misses():
    """All confidences sit below 0.5, so the fixed threshold sees nothing."""
    metrics = MetricSet(conf_threshold=0.5, imgsz=100)
    pred = as_batch([0.3, 0.3], [BOX, BOX])
    truth = as_batch([1.0, 1.0], [BOX, BOX])
    metrics.update(pred[0], pred[1], truth[0].squeeze(-1), truth[1])

    result = metrics.compute()
    assert result.f1 == pytest.approx(0.0)
    assert result.best_f1 == pytest.approx(1.0)
    assert result.best_threshold <= 0.3


def test_reset_clears_accumulated_state():
    metrics = MetricSet(conf_threshold=0.5, imgsz=100)
    pred = as_batch([0.9], [BOX])
    truth = as_batch([1.0], [BOX])
    metrics.update(pred[0], pred[1], truth[0].squeeze(-1), truth[1])
    metrics.reset()
    assert metrics.compute().map50 == pytest.approx(0.0)


def test_metric_result_exposes_the_keys_early_stopping_monitors():
    keys = MetricSet().compute().to_dict()
    for name in ("map50", "map50_95", "precision", "recall", "f1", "mean_iou"):
        assert name in keys


# ── Loss primitives ─────────────────────────────────────────────────────


def test_iou_of_identical_boxes_is_one():
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    assert float(box_iou_pairwise(boxes, boxes)) == pytest.approx(1.0, abs=1e-5)


def test_iou_of_disjoint_boxes_is_zero():
    a = torch.tensor([[0.2, 0.2, 0.1, 0.1]])
    b = torch.tensor([[0.8, 0.8, 0.1, 0.1]])
    assert float(box_iou_pairwise(a, b)) == pytest.approx(0.0, abs=1e-6)


def test_ciou_loss_is_zero_for_a_perfect_match():
    boxes = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    assert float(ciou_loss(boxes, boxes)) == pytest.approx(0.0, abs=1e-5)


def test_ciou_loss_grows_with_centroid_distance():
    truth = torch.tensor([[0.5, 0.5, 0.4, 0.4]])
    near = torch.tensor([[0.55, 0.5, 0.4, 0.4]])
    far = torch.tensor([[0.9, 0.5, 0.4, 0.4]])
    assert float(ciou_loss(near, truth)) < float(ciou_loss(far, truth))


def test_focal_loss_downweights_easy_examples_relative_to_bce():
    import torch.nn.functional as F

    easy_pred = torch.tensor([0.99])
    easy_true = torch.tensor([1.0])
    bce = float(F.binary_cross_entropy(easy_pred, easy_true))
    focal = float(focal_bce(easy_pred, easy_true))
    assert focal < bce


# ── Efficiency (FR-4.5) ─────────────────────────────────────────────────


def test_profiler_measures_the_fused_model_without_mutating_the_original():
    from dart.config import ConfigResolver
    from dart.models import DARTDetector

    model = DARTDetector(ConfigResolver().resolve("dart-b").model)
    result = EfficiencyProfiler(warmup=1, runs=3).profile(model, imgsz=96)

    assert result.params > 0
    assert result.int8_size_kb == pytest.approx(result.params / 1024, abs=0.01)
    assert result.latency is not None and result.latency.mean_ms > 0
    assert result.latency.fps > 0
    assert not model.is_fused, "profiling must not fuse the caller's model"
