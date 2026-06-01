"""
ONNX-to-SNN conversion helpers for classification CNN/ResNet-style models.
"""

from .converter import convert_onnx_to_snn
from .core import ConversionArtifacts, ConversionConfig, PatternGroup, UnsupportedONNXError
from .patterns import analyze_patterns, pattern_report

__all__ = [
    "ConversionArtifacts",
    "ConversionConfig",
    "PatternGroup",
    "UnsupportedONNXError",
    "analyze_patterns",
    "convert_onnx_to_snn",
    "pattern_report",
]
