from __future__ import annotations

import copy
import hashlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.ann2snn.modules import VoltageScaler

from .core import CanonicalGraph, CanonicalNode


class OnnxGraphModule(nn.Module):
    """A readable PyTorch module that executes a canonical ONNX graph."""

    def __init__(self, graph: CanonicalGraph, modules: dict[str, nn.Module]):
        super().__init__()
        self.graph = copy.deepcopy(graph)
        self.ops = nn.ModuleDict(modules)
        self.constant_names = list(graph.initializers)
        for name, value in graph.initializers.items():
            tensor = torch.as_tensor(value)
            self.register_buffer(_buffer_name(name), tensor)

    def forward(self, *args, **kwargs):
        env: dict[str, torch.Tensor] = {}
        if kwargs:
            for name in self.graph.input_names:
                if name in kwargs:
                    env[name] = kwargs[name]
        if args:
            if len(args) == 1 and len(self.graph.input_names) == 1:
                env[self.graph.input_names[0]] = args[0]
            else:
                for name, value in zip(self.graph.input_names, args):
                    env[name] = value
        missing = [name for name in self.graph.input_names if name not in env]
        if missing:
            raise ValueError(f"Missing ONNX graph inputs: {missing}")

        for name in self.constant_names:
            env[name] = getattr(self, _buffer_name(name))

        for node in self.graph.nodes:
            outputs = self._run_node(node, env)
            if len(node.outputs) == 1:
                env[node.outputs[0]] = outputs
            else:
                for name, value in zip(node.outputs, outputs):
                    env[name] = value

        out = [env[name] for name in self.graph.output_names]
        return out[0] if len(out) == 1 else tuple(out)

    def _run_node(self, node: CanonicalNode, env: dict[str, torch.Tensor]):
        inputs = [env[name] for name in node.inputs if name]
        if node.module_name is not None:
            if node.op_type in {"Conv", "BatchNormalization", "Gemm"}:
                return self.ops[node.module_name](inputs[0])
            return self.ops[node.module_name](*inputs)
        if node.op_type == "Add":
            return inputs[0] + inputs[1]
        if node.op_type == "AveragePool":
            return _pool(inputs[0], node.attrs, mode="avg")
        if node.op_type == "Concat":
            return torch.cat(inputs, dim=int(node.attrs.get("axis", 0)))
        if node.op_type == "Flatten":
            return torch.flatten(inputs[0], start_dim=int(node.attrs.get("axis", 1)))
        if node.op_type == "GlobalAveragePool":
            dims = tuple(range(2, inputs[0].dim()))
            return inputs[0].mean(dim=dims, keepdim=True)
        if node.op_type == "Identity":
            return inputs[0]
        if node.op_type == "MatMul":
            return torch.matmul(inputs[0], inputs[1])
        if node.op_type == "MaxPool":
            return _pool(inputs[0], node.attrs, mode="max")
        if node.op_type == "ReduceMean":
            axes = node.attrs.get("axes")
            if axes is None and len(inputs) > 1:
                axes = [int(v) for v in inputs[1].detach().cpu().flatten().tolist()]
            keepdim = bool(node.attrs.get("keepdims", 1))
            return inputs[0].mean(dim=tuple(axes) if axes is not None else None, keepdim=keepdim)
        if node.op_type == "Reshape":
            allowzero = int(node.attrs.get("allowzero", 0))
            shape = [int(v) for v in inputs[1].detach().cpu().tolist()]
            if not allowzero:
                shape = [
                    inputs[0].shape[idx] if dim == 0 else dim
                    for idx, dim in enumerate(shape)
                ]
            if -1 not in shape and int(np.prod(shape)) != inputs[0].numel() and shape:
                runtime_shape = [int(inputs[0].shape[0]), *shape[1:]]
                if int(np.prod(runtime_shape)) == inputs[0].numel():
                    shape = runtime_shape
            return torch.reshape(inputs[0], tuple(shape))
        if node.op_type == "Transpose":
            perm = node.attrs.get("perm")
            return inputs[0].permute(*perm) if perm is not None else inputs[0].t()
        raise RuntimeError(f"Unsupported executable op: {node.op_type}")


def build_ann_model(graph: CanonicalGraph) -> OnnxGraphModule:
    modules: dict[str, nn.Module] = {}
    for node in graph.nodes:
        if node.module_name is None:
            continue
        if node.op_type == "Conv":
            modules[node.module_name] = _build_conv(graph, node)
        elif node.op_type == "BatchNormalization":
            modules[node.module_name] = _build_batch_norm(graph, node)
        elif node.op_type == "Gemm":
            modules[node.module_name] = _build_gemm(graph, node)
        elif node.op_type == "Relu":
            modules[node.module_name] = nn.ReLU()
    model = OnnxGraphModule(graph, modules)
    model.eval()
    return model


def build_snn_surrogate_ann_model(
    ann_model: OnnxGraphModule, replace_maxpool_with_avgpool: bool = True
) -> OnnxGraphModule:
    modules = {name: copy.deepcopy(module) for name, module in ann_model.ops.items()}
    graph = copy.deepcopy(ann_model.graph)
    if replace_maxpool_with_avgpool:
        for node in graph.nodes:
            if node.op_type == "MaxPool":
                node.op_type = "AveragePool"
                node.attrs.setdefault("count_include_pad", 0)
    model = OnnxGraphModule(graph, modules)
    model.eval()
    return model


def build_snn_model(
    ann_model: OnnxGraphModule,
    relu_scales: dict[str, float],
    replace_maxpool_with_avgpool: bool = True,
) -> OnnxGraphModule:
    modules: dict[str, nn.Module] = {}
    for name, module in ann_model.ops.items():
        if isinstance(module, nn.ReLU):
            scale = max(float(relu_scales.get(name, 1.0)), 1.0e-6)
            modules[name] = nn.Sequential(
                VoltageScaler(1.0 / scale),
                neuron.IFNode(v_threshold=1.0, v_reset=None),
                VoltageScaler(scale),
            )
        else:
            modules[name] = copy.deepcopy(module)
    surrogate_ann = build_snn_surrogate_ann_model(
        ann_model, replace_maxpool_with_avgpool=replace_maxpool_with_avgpool
    )
    snn = OnnxGraphModule(surrogate_ann.graph, modules)
    snn.eval()
    return snn


def _build_conv(graph: CanonicalGraph, node: CanonicalNode) -> nn.Module:
    weight = torch.as_tensor(graph.initializers[node.inputs[1]]).float()
    bias = None
    if len(node.inputs) > 2 and node.inputs[2] in graph.initializers:
        bias = torch.as_tensor(graph.initializers[node.inputs[2]]).float()
    dims = weight.dim() - 2
    if dims != 2:
        raise ValueError("onnx2snn v1 supports Conv2d only")
    strides = tuple(node.attrs.get("strides", [1, 1]))
    dilations = tuple(node.attrs.get("dilations", [1, 1]))
    pads = node.attrs.get("pads", [0, 0, 0, 0])
    if pads[:2] != pads[2:]:
        raise ValueError("Asymmetric Conv padding is not supported in v1")
    groups = int(node.attrs.get("group", 1))
    conv = nn.Conv2d(
        in_channels=weight.shape[1] * groups,
        out_channels=weight.shape[0],
        kernel_size=tuple(weight.shape[2:]),
        stride=strides,
        padding=tuple(pads[:2]),
        dilation=dilations,
        groups=groups,
        bias=bias is not None,
    )
    conv.weight.data.copy_(weight)
    if bias is not None:
        conv.bias.data.copy_(bias)
    return conv


def _build_batch_norm(graph: CanonicalGraph, node: CanonicalNode) -> nn.Module:
    scale = torch.as_tensor(graph.initializers[node.inputs[1]]).float()
    bias = torch.as_tensor(graph.initializers[node.inputs[2]]).float()
    mean = torch.as_tensor(graph.initializers[node.inputs[3]]).float()
    var = torch.as_tensor(graph.initializers[node.inputs[4]]).float()
    eps = float(node.attrs.get("epsilon", 1.0e-5))
    shape = graph.value_shapes.get(node.inputs[0], ())
    cls = nn.BatchNorm2d if len(shape) == 4 else nn.BatchNorm1d
    bn = cls(scale.numel(), eps=eps)
    bn.weight.data.copy_(scale)
    bn.bias.data.copy_(bias)
    bn.running_mean.data.copy_(mean)
    bn.running_var.data.copy_(var)
    bn.eval()
    return bn


def _build_gemm(graph: CanonicalGraph, node: CanonicalNode) -> nn.Module:
    b = torch.as_tensor(graph.initializers[node.inputs[1]]).float()
    c = None
    if len(node.inputs) > 2 and node.inputs[2] in graph.initializers:
        c = torch.as_tensor(graph.initializers[node.inputs[2]]).float()
    alpha = float(node.attrs.get("alpha", 1.0))
    beta = float(node.attrs.get("beta", 1.0))
    trans_b = int(node.attrs.get("transB", 0))
    if alpha != 1.0 or beta != 1.0:
        raise ValueError("Gemm with alpha/beta != 1 is not supported in v1")
    weight = b if trans_b else b.t()
    linear = nn.Linear(weight.shape[1], weight.shape[0], bias=c is not None)
    linear.weight.data.copy_(weight)
    if c is not None:
        linear.bias.data.copy_(c)
    return linear


def _pool(x: torch.Tensor, attrs: dict[str, Any], mode: str):
    kernel = tuple(attrs.get("kernel_shape", []))
    stride = tuple(attrs.get("strides", kernel))
    pads = attrs.get("pads", [0] * (2 * len(kernel)))
    if len(kernel) != 2:
        raise ValueError("onnx2snn v1 supports 2D pooling only")
    if pads[:2] != pads[2:]:
        pad = (pads[1], pads[3], pads[0], pads[2])
        x = F.pad(x, pad, value=float("-inf") if mode == "max" else 0.0)
        padding = 0
    else:
        padding = tuple(pads[:2])
    ceil_mode = bool(attrs.get("ceil_mode", 0))
    if mode == "avg":
        return F.avg_pool2d(
            x,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=bool(attrs.get("count_include_pad", 0)),
        )
    return F.max_pool2d(
        x, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode
    )


def _buffer_name(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
    return f"const_{digest}"
