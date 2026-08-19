"""Configurable early stopping (FR-3.2).

:class:`EarlyStopper` is a **pure decision function**. It takes a metrics dict
and returns a :class:`StopDecision`; it holds no reference to the model, the
trainer, or the filesystem. That is what makes the compound criterion testable
against a synthetic metric sequence without training anything — which is
exactly what the CI correctness gate needs.

The paper's criterion (primary mAP@0.5, secondary mean IoU at 3% relative gain
over a 3-epoch window, patience 40) is one *configuration* of this mechanism,
not hardcoded behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional

from ..config.schema import EarlyStopConfig
from ..errors import ConfigurationError


@dataclass
class StopDecision:
    """What the stopper concluded about one epoch.

    Attributes:
        stop: Training should end now.
        improved: This epoch is the new best; the caller should checkpoint it.
        reason: Human-readable explanation, written to the run log.
        best_value: Best primary-metric value seen so far.
        best_epoch: Epoch at which ``best_value`` occurred.
        epochs_without_improvement: Current patience counter.
    """

    stop: bool = False
    improved: bool = False
    reason: str = ""
    best_value: float = 0.0
    best_epoch: int = 0
    epochs_without_improvement: int = 0


class EarlyStopper:
    """Patience-based stopping with an optional secondary-metric veto.

    The primary metric drives patience. Once the primary has *saturated*
    (reached ``primary_saturated``) there is no more of it to gain, and the
    secondary metric takes over: training continues while the secondary is
    still improving by at least ``secondary_rel_gain`` over the last
    ``secondary_window`` epochs, and stops as soon as it is not.

    Gating the veto behind saturation is deliberate. A secondary metric allowed
    to veto at any time would keep a run alive that has stopped learning
    anything about its primary objective.

    Args:
        cfg: The criterion configuration. ``secondary_metric=None`` reduces
            this to standard patience-based stopping.
    """

    def __init__(self, cfg: Optional[EarlyStopConfig] = None) -> None:
        self.cfg = cfg or EarlyStopConfig()
        if self.cfg.direction not in ("max", "min"):
            raise ConfigurationError(
                f"early_stop.direction must be 'max' or 'min', got '{self.cfg.direction}'"
            )
        self.maximize = self.cfg.direction == "max"
        self.reset()

    def reset(self) -> None:
        """Clear all history — used when a run restarts from scratch."""
        self.best_value = float("-inf") if self.maximize else float("inf")
        self.best_secondary = float("-inf")
        self.best_epoch = 0
        self.counter = 0
        self.epoch = 0
        self._secondary_history: List[float] = []

    # ── Decision ────────────────────────────────────────────────────────

    def step(self, metrics: Mapping[str, float], epoch: Optional[int] = None) -> StopDecision:
        """Judge one epoch's metrics.

        Args:
            metrics: Metric name -> value for the epoch just finished. Must
                contain ``cfg.primary_metric`` (and ``cfg.secondary_metric``
                when one is configured).
            epoch: Epoch number; defaults to an internal counter.

        Returns:
            A :class:`StopDecision`.
        """
        self.epoch = self.epoch + 1 if epoch is None else int(epoch)
        primary = _require_metric(metrics, self.cfg.primary_metric, "primary")

        if not self.cfg.enabled:
            improved = self._is_improvement(primary)
            if improved:
                self._accept(primary, metrics)
            return self._decision(False, improved, "early stopping disabled")

        if self._is_improvement(primary):
            self._accept(primary, metrics)
            return self._decision(
                False, True, f"{self.cfg.primary_metric} improved to {primary:.4f}"
            )

        if self.cfg.secondary_metric and self._primary_saturated():
            return self._secondary_step(metrics)

        self.counter += 1
        stop = self.counter >= self.cfg.patience
        reason = (
            f"{self.cfg.primary_metric} has not improved for {self.counter} epoch(s)"
            f"{'; patience exhausted' if stop else ''}"
        )
        return self._decision(stop, False, reason)

    # ── Internals ───────────────────────────────────────────────────────

    def _secondary_step(self, metrics: Mapping[str, float]) -> StopDecision:
        name = self.cfg.secondary_metric
        value = _require_metric(metrics, name, "secondary")
        self._secondary_history.append(value)

        window = self.cfg.secondary_window
        if len(self._secondary_history) > window:
            past = self._secondary_history[-window - 1]
            rel_gain = (value - past) / max(abs(past), 1e-6)
        else:
            # Not enough history to judge; give the secondary the benefit of
            # the doubt rather than stopping on no evidence.
            rel_gain = float("inf")

        if value > self.best_secondary and rel_gain >= self.cfg.secondary_rel_gain:
            self.best_secondary = value
            self.best_epoch = self.epoch
            self.counter = 0
            return self._decision(
                False,
                True,
                f"{self.cfg.primary_metric} saturated; {name} still improving "
                f"({value:.4f}, +{rel_gain:.1%} over {window} epochs)",
            )

        self.counter += 1
        cause = (
            f"{name} fell back to {value:.4f} from its best {self.best_secondary:.4f}"
            if value <= self.best_secondary
            else f"{name} gained {rel_gain:.1%} over {window} epochs, below the "
            f"{self.cfg.secondary_rel_gain:.1%} threshold"
        )
        return self._decision(True, False, f"{self.cfg.primary_metric} saturated and {cause}")

    def _is_improvement(self, value: float) -> bool:
        if self.maximize:
            return value > self.best_value + self.cfg.min_delta
        return value < self.best_value - self.cfg.min_delta

    def _primary_saturated(self) -> bool:
        if self.maximize:
            return self.best_value >= self.cfg.primary_saturated
        return self.best_value <= self.cfg.primary_saturated

    def _accept(self, primary: float, metrics: Mapping[str, float]) -> None:
        self.best_value = primary
        self.best_epoch = self.epoch
        self.counter = 0
        if self.cfg.secondary_metric and self.cfg.secondary_metric in metrics:
            secondary = float(metrics[self.cfg.secondary_metric])
            self._secondary_history.append(secondary)
            self.best_secondary = max(self.best_secondary, secondary)

    def _decision(self, stop: bool, improved: bool, reason: str) -> StopDecision:
        return StopDecision(
            stop=stop,
            improved=improved,
            reason=reason,
            best_value=self.best_value,
            best_epoch=self.best_epoch,
            epochs_without_improvement=self.counter,
        )

    # ── Resume support (FR-3.3) ─────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "best_value": self.best_value,
            "best_secondary": self.best_secondary,
            "best_epoch": self.best_epoch,
            "counter": self.counter,
            "epoch": self.epoch,
            "secondary_history": list(self._secondary_history),
        }

    def load_state_dict(self, state: Mapping) -> None:
        self.best_value = state.get("best_value", self.best_value)
        self.best_secondary = state.get("best_secondary", self.best_secondary)
        self.best_epoch = state.get("best_epoch", 0)
        self.counter = state.get("counter", 0)
        self.epoch = state.get("epoch", 0)
        self._secondary_history = list(state.get("secondary_history", []))


def _require_metric(metrics: Mapping[str, float], name: str, role: str) -> float:
    if name not in metrics:
        raise ConfigurationError(
            f"early stopping is configured with {role} metric '{name}', which the "
            f"evaluator did not report. Available: {sorted(metrics)}"
        )
    return float(metrics[name])
