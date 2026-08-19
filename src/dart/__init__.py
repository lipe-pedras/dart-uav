"""DART — a family of lightweight CNNs for real-time detection in
resource-constrained UAV navigation.

Quickstart::

    from dart import DART

    model = DART("dart-xd")                       # architecture, random init
    model.train(data="my_dataset", epochs=300, imgsz=320,
                target_policy="largest_only", seed=0)
    metrics = model.val(data="my_dataset")
    model.export(format="onnx")

The seven paper-validated variants — ``dart-b``, ``dart-b-ciou``, ``dart-s``,
``dart-d``, ``dart-f``, ``dart-x``, ``dart-xd`` — ship as configuration files,
not as classes, so an eighth variant is a YAML file rather than a code change.
"""

from .api import DART, Detection, PredictResult
from .config import ConfigResolver, RunConfig
from .engine import Evaluator, MultiSeedRunner, Trainer
from .errors import (
    CheckpointError,
    ConfigurationError,
    DARTError,
    DatasetError,
    ExportError,
    ExportVerificationError,
    RegistryError,
    SpawnSafetyError,
)
from .metrics import MetricResult
from .models import DARTDetector

try:  # pragma: no cover - trivial
    from importlib.metadata import PackageNotFoundError, version

    try:
        __version__ = version("dart-uav")
    except PackageNotFoundError:
        __version__ = "0.1.0"
except ImportError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = [
    "DART",
    "CheckpointError",
    "ConfigResolver",
    "ConfigurationError",
    "DARTDetector",
    "DARTError",
    "DatasetError",
    "Detection",
    "Evaluator",
    "ExportError",
    "ExportVerificationError",
    "MetricResult",
    "MultiSeedRunner",
    "PredictResult",
    "RegistryError",
    "RunConfig",
    "SpawnSafetyError",
    "Trainer",
    "__version__",
]
