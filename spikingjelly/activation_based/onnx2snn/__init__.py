"""
ONNX-to-SNN conversion helpers for classification CNN/ResNet-style models.
"""

from .converter import convert_onnx_to_snn
from .core import ConversionArtifacts, ConversionConfig, PatternGroup, UnsupportedONNXError
from .patterns import analyze_patterns, pattern_report
from .structured import (
    BasicBlock,
    BottleneckBlock,
    StructuredOnnxGraphModule,
    build_structured_ann_model,
)

__all__ = [
    "ConversionArtifacts",
    "ConversionConfig",
    "PatternGroup",
    "BasicBlock",
    "BottleneckBlock",
    "StructuredOnnxGraphModule",
    "UnsupportedONNXError",
    "analyze_patterns",
    "build_structured_ann_model",
    "convert_onnx_to_snn",
    "pattern_report",
]
