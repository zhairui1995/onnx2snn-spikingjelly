from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch.nn as nn


SUPPORTED_OPS = {
    "Add",
    "AveragePool",
    "BatchNormalization",
    "Concat",
    "Constant",
    "Conv",
    "Flatten",
    "Gemm",
    "GlobalAveragePool",
    "Identity",
    "MatMul",
    "MaxPool",
    "Pad",
    "Relu",
    "ReduceMean",
    "Reshape",
    "Transpose",
}


@dataclass
class ConversionConfig:
    """Configuration for the first ONNX-to-SNN conversion path."""

    input_shape: tuple[int, ...] | None = None
    t: int = 50
    scale_mode: str | float = "max"
    momentum: float = 0.1
    target_relative_metric: float = 0.95
    device: str = "cpu"
    compare_onnxruntime: bool = True
    replace_maxpool_with_avgpool_in_snn: bool = True
    synthetic_calibration_batches: int = 1
    synthetic_batch_size: int = 2

    @classmethod
    def from_user_config(cls, config: "ConversionConfig | dict[str, Any] | None"):
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if isinstance(config, dict):
            values = dict(config)
            if values.get("input_shape") is not None:
                values["input_shape"] = tuple(int(v) for v in values["input_shape"])
            return cls(**values)
        raise TypeError(f"Unsupported config type: {type(config)!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConversionArtifacts:
    """Conversion outputs returned by :func:`convert_onnx_to_snn`."""

    ann_model: nn.Module
    structured_ann_model: nn.Module | None
    snn_model: nn.Module
    config: ConversionConfig
    calibration_stats: dict[str, Any]
    report: dict[str, Any]
    output_dir: Path


@dataclass
class CanonicalNode:
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)
    module_name: str | None = None


@dataclass
class CanonicalGraph:
    input_names: list[str]
    output_names: list[str]
    nodes: list[CanonicalNode]
    initializers: dict[str, Any]
    value_shapes: dict[str, tuple[int | None, ...]]
    opset_imports: dict[str, int]
    module_kinds: dict[str, str] = field(default_factory=dict)


@dataclass
class PatternGroup:
    name: str
    pattern_type: str
    node_indices: list[int]
    node_names: list[str]
    op_types: list[str]
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UnsupportedONNXError(RuntimeError):
    """Raised when the ONNX graph uses operators outside the v1 scope."""

    def __init__(self, unsupported_ops: list[str]):
        self.unsupported_ops = sorted(set(unsupported_ops))
        msg = "Unsupported ONNX operators for onnx2snn v1: "
        msg += ", ".join(self.unsupported_ops)
        super().__init__(msg)
