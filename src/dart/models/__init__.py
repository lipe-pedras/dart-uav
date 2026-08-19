"""DART model architecture: registry-resolved backbone, neck, and head.

Importing this package registers every built-in component, which is what makes
``COMPONENTS.names("head")`` and friends complete.
"""

from .attention import (
    CoordAttention,
    IdentityAttention,
    SEAttention,
    make_activation,
    make_attention,
)
from .backbone import Backbone, LoadReport
from .blocks import (
    BackboneBlock,
    DepthwiseSeparableBlock,
    GhostBlock,
    InvertedResidualBlock,
    RepConvBlock,
)
from .detector import DARTDetector
from .heads import (
    DecoupledSPPHead,
    GAPHead,
    Head,
    SoftArgmaxHead,
    SPPLite,
    SPPLiteHead,
    build_head,
)
from .necks import FPNLiteNeck, IdentityNeck, build_neck
from .registry import BLOCKS, COMPONENTS, build_block, register_block, register_component

__all__ = [
    "BLOCKS",
    "COMPONENTS",
    "Backbone",
    "BackboneBlock",
    "CoordAttention",
    "DARTDetector",
    "DecoupledSPPHead",
    "DepthwiseSeparableBlock",
    "FPNLiteNeck",
    "GAPHead",
    "GhostBlock",
    "Head",
    "IdentityAttention",
    "IdentityNeck",
    "InvertedResidualBlock",
    "LoadReport",
    "RepConvBlock",
    "SEAttention",
    "SPPLite",
    "SPPLiteHead",
    "SoftArgmaxHead",
    "build_block",
    "build_head",
    "build_neck",
    "make_activation",
    "make_attention",
    "register_block",
    "register_component",
]
