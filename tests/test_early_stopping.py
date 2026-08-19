"""The compound early-stopping criterion (FR-3.2).

The stopper is a pure decision function, so every case here is verified against
a synthetic metric sequence with no model and no filesystem involved.
"""

from __future__ import annotations

from typing import List, Sequence

import pytest

from dart.config import EarlyStopConfig
from dart.engine import EarlyStopper
from dart.errors import ConfigurationError


def run(cfg: EarlyStopConfig, sequence: Sequence[dict]) -> List:
    stopper = EarlyStopper(cfg)
    decisions = []
    for epoch, metrics in enumerate(sequence, start=1):
        decision = stopper.step(metrics, epoch=epoch)
        decisions.append(decision)
        if decision.stop:
            break
    return decisions


def plateau(value: float, n: int, metric: str = "map50") -> List[dict]:
    return [{metric: value, "mean_iou": 0.4} for _ in range(n)]


def test_patience_runs_down_on_a_plateau():
    cfg = EarlyStopConfig(patience=3, secondary_metric=None)
    decisions = run(cfg, [{"map50": 0.5}] + plateau(0.5, 10))
    assert decisions[-1].stop
    assert len(decisions) == 4  # first epoch improves, then three flat epochs


def test_improvement_resets_patience():
    cfg = EarlyStopConfig(patience=2, secondary_metric=None)
    sequence = [{"map50": 0.5}, {"map50": 0.5}, {"map50": 0.6}, {"map50": 0.6}]
    decisions = run(cfg, sequence)
    assert [d.improved for d in decisions] == [True, False, True, False]
    assert not any(d.stop for d in decisions)


def test_improvement_must_exceed_min_delta():
    cfg = EarlyStopConfig(patience=5, min_delta=0.01, secondary_metric=None)
    decisions = run(cfg, [{"map50": 0.5}, {"map50": 0.505}, {"map50": 0.52}])
    assert [d.improved for d in decisions] == [True, False, True]


def test_minimisation_direction_is_supported():
    cfg = EarlyStopConfig(
        primary_metric="val_loss", direction="min", patience=2, secondary_metric=None
    )
    decisions = run(cfg, [{"val_loss": 1.0}, {"val_loss": 0.5}, {"val_loss": 0.9}])
    assert [d.improved for d in decisions] == [True, True, False]


def test_no_secondary_metric_degenerates_to_plain_patience():
    """FR-3.2 requires the compound criterion to be optional."""
    cfg = EarlyStopConfig(patience=2, secondary_metric=None)
    decisions = run(cfg, [{"map50": 1.0}] + plateau(1.0, 5))
    assert decisions[-1].stop
    assert len(decisions) == 3


# ── The compound criterion ──────────────────────────────────────────────


def test_secondary_metric_keeps_a_saturated_run_alive_while_it_improves():
    """The paper's criterion: mAP has hit 1.0, mean IoU is still climbing."""
    cfg = EarlyStopConfig(
        patience=3, secondary_metric="mean_iou", secondary_rel_gain=0.03, secondary_window=2
    )
    sequence = [{"map50": 1.0, "mean_iou": 0.40}] + [
        {"map50": 1.0, "mean_iou": iou} for iou in (0.45, 0.50, 0.56, 0.63, 0.70)
    ]
    decisions = run(cfg, sequence)
    assert not any(d.stop for d in decisions), "a still-improving secondary must veto stopping"
    assert all(d.improved for d in decisions)


def test_a_saturated_run_stops_as_soon_as_the_secondary_flattens():
    cfg = EarlyStopConfig(
        patience=10, secondary_metric="mean_iou", secondary_rel_gain=0.03, secondary_window=2
    )
    sequence = [{"map50": 1.0, "mean_iou": 0.40}] + [
        {"map50": 1.0, "mean_iou": iou} for iou in (0.45, 0.50, 0.501, 0.502)
    ]
    decisions = run(cfg, sequence)
    assert decisions[-1].stop
    assert "below" in decisions[-1].reason


def test_the_secondary_cannot_veto_before_the_primary_saturates():
    """A wandering secondary must not keep a dead run alive."""
    cfg = EarlyStopConfig(
        patience=2, secondary_metric="mean_iou", secondary_rel_gain=0.03, secondary_window=2
    )
    sequence = [{"map50": 0.6, "mean_iou": 0.4}] + [
        {"map50": 0.6, "mean_iou": iou} for iou in (0.5, 0.6, 0.7, 0.8)
    ]
    decisions = run(cfg, sequence)
    assert decisions[-1].stop
    assert len(decisions) == 3, "patience must still expire while mAP is unsaturated"


def test_the_papers_exact_criterion_is_the_default():
    cfg = EarlyStopConfig()
    assert (cfg.primary_metric, cfg.direction, cfg.patience) == ("map50", "max", 40)
    assert (cfg.secondary_metric, cfg.secondary_rel_gain, cfg.secondary_window) == (
        "mean_iou",
        0.03,
        3,
    )


# ── Contracts ───────────────────────────────────────────────────────────


def test_a_missing_monitored_metric_is_an_actionable_error():
    stopper = EarlyStopper(EarlyStopConfig(primary_metric="mAP", secondary_metric=None))
    with pytest.raises(ConfigurationError, match="did not report"):
        stopper.step({"map50": 0.5})


def test_disabled_stopper_never_stops_but_still_tracks_the_best():
    cfg = EarlyStopConfig(enabled=False, patience=1, secondary_metric=None)
    decisions = run(cfg, [{"map50": 0.5}, {"map50": 0.4}, {"map50": 0.9}])
    assert not any(d.stop for d in decisions)
    assert decisions[-1].best_value == pytest.approx(0.9)


def test_state_survives_a_resume_roundtrip():
    stopper = EarlyStopper(EarlyStopConfig(patience=5, secondary_metric=None))
    stopper.step({"map50": 0.7}, epoch=1)
    stopper.step({"map50": 0.6}, epoch=2)
    state = stopper.state_dict()

    restored = EarlyStopper(EarlyStopConfig(patience=5, secondary_metric=None))
    restored.load_state_dict(state)
    assert restored.best_value == pytest.approx(0.7)
    assert restored.counter == 1
    assert restored.best_epoch == 1


def test_the_stopper_holds_no_reference_to_a_model_or_the_filesystem():
    """Structural guarantee, not a behavioural one — it is what makes this
    whole test file possible without training anything."""
    stopper = EarlyStopper(EarlyStopConfig())
    attributes = {type(v).__name__ for v in vars(stopper).values()}
    assert not attributes & {"Module", "DARTDetector", "Path", "PosixPath", "Trainer"}


def test_the_stop_reason_names_which_secondary_condition_failed():
    cfg = EarlyStopConfig(
        patience=10, secondary_metric="mean_iou", secondary_rel_gain=0.03, secondary_window=2
    )
    # The secondary regresses from its best rather than merely flattening.
    sequence = [{"map50": 1.0, "mean_iou": 0.40}] + [
        {"map50": 1.0, "mean_iou": iou} for iou in (0.50, 0.62, 0.55)
    ]
    decisions = run(cfg, sequence)
    assert decisions[-1].stop
    assert "fell back" in decisions[-1].reason
