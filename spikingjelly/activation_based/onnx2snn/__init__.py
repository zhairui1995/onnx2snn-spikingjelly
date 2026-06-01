"""
ONNX-to-SNN conversion helpers for classification CNN/ResNet-style models.
"""

from .converter import convert_onnx_to_snn
from .core import ConversionArtifacts, ConversionConfig, UnsupportedONNXError

__all__ = [
    "ConversionArtifacts",
    "ConversionConfig",
    "UnsupportedONNXError",
    "convert_onnx_to_snn",
]

