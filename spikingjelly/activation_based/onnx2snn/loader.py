from __future__ import annotations

import re
from typing import Any

import numpy as np
from onnx import TensorProto, checker, load, numpy_helper, shape_inference

from .core import SUPPORTED_OPS, CanonicalGraph, CanonicalNode, UnsupportedONNXError


def sanitize_name(name: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z_]", "_", name)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = fallback
    if cleaned[0].isdigit():
        cleaned = f"n_{cleaned}"
    return cleaned


def load_onnx_graph(onnx_path: str) -> CanonicalGraph:
    model = load(onnx_path)
    checker.check_model(model)
    inferred = shape_inference.infer_shapes(model)
    graph = inferred.graph

    initializers = {
        tensor.name: numpy_helper.to_array(tensor).copy()
        for tensor in graph.initializer
    }
    initializer_names = set(initializers)
    input_names = [
        value.name for value in graph.input if value.name not in initializer_names
    ]
    output_names = [value.name for value in graph.output]
    value_shapes = _collect_value_shapes(inferred)
    opset_imports = {
        opset.domain or "": int(opset.version) for opset in inferred.opset_import
    }

    unsupported = [
        node.op_type for node in graph.node if node.op_type not in SUPPORTED_OPS
    ]
    if unsupported:
        raise UnsupportedONNXError(unsupported)

    nodes: list[CanonicalNode] = []
    module_kinds: dict[str, str] = {}
    used_names: set[str] = set()
    for idx, node in enumerate(graph.node):
        base = sanitize_name(node.name or f"{node.op_type.lower()}_{idx}", f"node_{idx}")
        name = _unique_name(base, used_names)
        module_name = None
        if node.op_type in {"Conv", "BatchNormalization", "Gemm", "Relu", "Tanh"}:
            module_name = f"op_{idx}_{sanitize_name(node.op_type.lower(), 'op')}"
            module_kinds[module_name] = node.op_type
        nodes.append(
            CanonicalNode(
                name=name,
                op_type=node.op_type,
                inputs=[x for x in node.input],
                outputs=[x for x in node.output],
                attrs={attr.name: _attribute_to_python(attr) for attr in node.attribute},
                module_name=module_name,
            )
        )

    return CanonicalGraph(
        input_names=input_names,
        output_names=output_names,
        nodes=nodes,
        initializers=initializers,
        value_shapes=value_shapes,
        opset_imports=opset_imports,
        module_kinds=module_kinds,
    )


def _unique_name(base: str, used_names: set[str]) -> str:
    name = base
    idx = 1
    while name in used_names:
        name = f"{base}_{idx}"
        idx += 1
    used_names.add(name)
    return name


def _collect_value_shapes(model) -> dict[str, tuple[int | None, ...]]:
    shapes: dict[str, tuple[int | None, ...]] = {}
    values = list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    for value in values:
        tensor_type = value.type.tensor_type
        if not tensor_type.HasField("shape"):
            continue
        dims = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                dims.append(int(dim.dim_value))
            else:
                dims.append(None)
        shapes[value.name] = tuple(dims)
    return shapes


def _attribute_to_python(attr) -> Any:
    if attr.type == attr.AttributeType.FLOAT:
        return float(attr.f)
    if attr.type == attr.AttributeType.INT:
        return int(attr.i)
    if attr.type == attr.AttributeType.STRING:
        return attr.s.decode("utf-8")
    if attr.type == attr.AttributeType.FLOATS:
        return [float(v) for v in attr.floats]
    if attr.type == attr.AttributeType.INTS:
        return [int(v) for v in attr.ints]
    if attr.type == attr.AttributeType.TENSOR:
        return numpy_helper.to_array(attr.t).copy()
    if attr.type == attr.AttributeType.GRAPH:
        raise UnsupportedONNXError(["nested Graph attribute"])
    return None


def tensor_proto_dtype_to_numpy(dtype: int):
    return {
        TensorProto.FLOAT: np.float32,
        TensorProto.DOUBLE: np.float64,
        TensorProto.FLOAT16: np.float16,
        TensorProto.INT64: np.int64,
        TensorProto.INT32: np.int32,
        TensorProto.BOOL: np.bool_,
    }.get(dtype, np.float32)
