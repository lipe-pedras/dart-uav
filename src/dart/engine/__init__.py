"""Training engine: trainer, evaluator, early stopping, run recording."""

from .early_stopping import EarlyStopper, StopDecision
from .evaluator import Evaluator
from .multiseed import AggregateResult, MultiSeedRunner, aggregate_results
from .recorder import COMPARISON_FIELDS, EPOCH_FIELDS, RunRecorder, aggregate_runs
from .trainer import RunResult, Trainer

__all__ = [
    "COMPARISON_FIELDS",
    "EPOCH_FIELDS",
    "AggregateResult",
    "EarlyStopper",
    "Evaluator",
    "MultiSeedRunner",
    "RunRecorder",
    "RunResult",
    "StopDecision",
    "Trainer",
    "aggregate_results",
    "aggregate_runs",
]
