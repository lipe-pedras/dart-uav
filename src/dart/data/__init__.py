"""Dataset ingestion: adapters, validation, policies, augmentation, batching."""

from .adapters import (
    ADAPTERS,
    CocoAdapter,
    CreateMLAdapter,
    FormatAdapter,
    ImageAnnotations,
    RawAnnotations,
    VocAdapter,
    YoloAdapter,
    adapter_for,
    detect_format,
)
from .augment import AugmentationPipeline, Transform
from .dataset import DARTDataset, Stats
from .loader import DatasetLoader, SplitPaths, build_dataloader, check_spawn_safety
from .policies import (
    ClassRemapper,
    ClassSelector,
    KeepAllPolicy,
    LargestOnlyPolicy,
    TargetPolicy,
    build_policy,
)
from .validator import DatasetValidator, summarize

__all__ = [
    "ADAPTERS",
    "AugmentationPipeline",
    "ClassRemapper",
    "ClassSelector",
    "CocoAdapter",
    "CreateMLAdapter",
    "DARTDataset",
    "DatasetLoader",
    "DatasetValidator",
    "FormatAdapter",
    "ImageAnnotations",
    "KeepAllPolicy",
    "LargestOnlyPolicy",
    "RawAnnotations",
    "SplitPaths",
    "Stats",
    "TargetPolicy",
    "Transform",
    "VocAdapter",
    "YoloAdapter",
    "adapter_for",
    "build_dataloader",
    "build_policy",
    "check_spawn_safety",
    "detect_format",
    "summarize",
]
