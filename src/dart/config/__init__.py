"""Configuration system: presets, YAML, and the resolver that merges them."""

from .resolver import PRESET_DIR, ConfigResolver, resolve_config
from .schema import (
    AugmentConfig,
    BackboneConfig,
    BlockSpec,
    DataConfig,
    EarlyStopConfig,
    HeadConfig,
    LossConfig,
    ModelConfig,
    NeckConfig,
    RunConfig,
    TrainConfig,
)

__all__ = [
    "PRESET_DIR",
    "AugmentConfig",
    "BackboneConfig",
    "BlockSpec",
    "ConfigResolver",
    "DataConfig",
    "EarlyStopConfig",
    "HeadConfig",
    "LossConfig",
    "ModelConfig",
    "NeckConfig",
    "RunConfig",
    "TrainConfig",
    "resolve_config",
]
