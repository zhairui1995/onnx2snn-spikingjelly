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
                    if name:
                        env[name] = value

        out = [env[name] for name in self.graph.output_names]
        return out[0] if len(out) == 1 else tuple(out)

    def _run_node(self, node: CanonicalNode, env: dict[str, torch.Tensor]):
        inputs = [env[name] for name in node.inputs if name]
        if node.module_name is not None:
            if node.op_type in {"Conv", "BatchNormalization", "Gemm"}:
                return self.ops[node.module_name](inputs[0])
            return self.ops[node.module_name](*inputs)
        if node.op_type == "Abs":
            return torch.abs(inputs[0])
        if node.op_type == "Add":
            return inputs[0] + inputs[1]
        if node.op_type == "AveragePool":
            return _pool(inputs[0], node.attrs, mode="avg")
        if node.op_type == "Clip":
            return torch.clamp(
                inputs[0],
                min=_clip_bound(inputs, node.attrs, 1, "min"),
                max=_clip_bound(inputs, node.attrs, 2, "max"),
            )
        if node.op_type == "Concat":
            return torch.cat(inputs, dim=int(node.attrs.get("axis", 0)))
        if node.op_type == "Constant":
            return torch.as_tensor(node.attrs["value"])
        if node.op_type == "Div":
            return inputs[0] / inputs[1]
        if node.op_type == "Dropout":
            if len(node.outputs) > 1:
                return inputs[0], torch.ones_like(inputs[0], dtype=torch.bool)
            return inputs[0]
        if node.op_type == "Equal":
            return torch.eq(inputs[0], inputs[1])
        if node.op_type == "Expand":
            shape = _shape_from_tensor(inputs[1])
            return torch.broadcast_to(inputs[0], shape)
        if node.op_type == "Flatten":
            return torch.flatten(inputs[0], start_dim=int(node.attrs.get("axis", 1)))
        if node.op_type == "Gather":
            return _gather(inputs[0], inputs[1], int(node.attrs.get("axis", 0)))
        if node.op_type == "GlobalAveragePool":
            dims = tuple(range(2, inputs[0].dim()))
            return inputs[0].mean(dim=dims, keepdim=True)
        if node.op_type == "GlobalMaxPool":
            dims = tuple(range(2, inputs[0].dim()))
            return torch.amax(inputs[0], dim=dims, keepdim=True)
        if node.op_type == "Greater":
            return torch.gt(inputs[0], inputs[1])
        if node.op_type == "Identity":
            return inputs[0]
        if node.op_type == "Less":
            return torch.lt(inputs[0], inputs[1])
        if node.op_type == "MatMul":
            return torch.matmul(inputs[0], inputs[1])
        if node.op_type == "Max":
            if len(inputs) == 2:
                return torch.maximum(inputs[0], inputs[1])
            return torch.stack(inputs).amax(dim=0)
        if node.op_type == "MaxPool":
            return _pool(inputs[0], node.attrs, mode="max")
        if node.op_type == "Min":
            if len(inputs) == 2:
                return torch.minimum(inputs[0], inputs[1])
            return torch.stack(inputs).amin(dim=0)
        if node.op_type == "Mul":
            return inputs[0] * inputs[1]
        if node.op_type == "Neg":
            return -inputs[0]
        if node.op_type == "Pad":
            return _pad(inputs, node.attrs)
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
        if node.op_type == "Shape":
            start = int(node.attrs.get("start", 0))
            end = node.attrs.get("end")
            shape = list(inputs[0].shape)
            return torch.tensor(
                shape[start:end], dtype=torch.long, device=inputs[0].device
            )
        if node.op_type == "Squeeze":
            axes = _axes_from_inputs_or_attrs(inputs, node.attrs)
            if axes is None:
                return torch.squeeze(inputs[0])
            out = inputs[0]
            for axis in sorted(_normalize_axes(axes, out.dim()), reverse=True):
                out = torch.squeeze(out, dim=axis)
            return out
        if node.op_type == "Sub":
            return inputs[0] - inputs[1]
        if node.op_type == "Transpose":
            perm = node.attrs.get("perm")
            return inputs[0].permute(*perm) if perm is not None else inputs[0].t()
        if node.op_type == "Unsqueeze":
            axes = _axes_from_inputs_or_attrs(inputs, node.attrs)
            if axes is None:
                raise ValueError("Unsqueeze requires axes")
            out = inputs[0]
            for axis in sorted(_normalize_axes(axes, out.dim() + len(axes))):
                out = torch.unsqueeze(out, dim=axis)
            return out
        if node.op_type == "Where":
            return torch.where(inputs[0].bool(), inputs[1], inputs[2])
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
    if dims not in {1, 2, 3}:
        raise ValueError("onnx2snn v1 supports Conv1d/2d/3d only")
    strides = tuple(node.attrs.get("strides", [1] * dims))
    dilations = tuple(node.attrs.get("dilations", [1] * dims))
    pads = node.attrs.get("pads", [0] * (2 * dims))
    if pads[:dims] != pads[dims:]:
        raise ValueError("Asymmetric Conv padding is not supported in v1")
    groups = int(node.attrs.get("group", 1))
    conv_cls = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}[dims]
    conv = conv_cls(
        in_channels=weight.shape[1] * groups,
        out_channels=weight.shape[0],
        kernel_size=tuple(weight.shape[2:]),
        stride=strides,
        padding=tuple(pads[:dims]),
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
    cls = {5: nn.BatchNorm3d, 4: nn.BatchNorm2d}.get(len(shape), nn.BatchNorm1d)
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
    dims = len(kernel)
    if dims not in {1, 2, 3}:
        raise ValueError("onnx2snn v1 supports 1D/2D/3D pooling only")
    if pads[:dims] != pads[dims:]:
        pad = _onnx_pads_to_torch(pads)
        x = F.pad(x, pad, value=float("-inf") if mode == "max" else 0.0)
        padding = 0
    else:
        padding = tuple(pads[:dims])
    ceil_mode = bool(attrs.get("ceil_mode", 0))
    avg_pool = {1: F.avg_pool1d, 2: F.avg_pool2d, 3: F.avg_pool3d}[dims]
    max_pool = {1: F.max_pool1d, 2: F.max_pool2d, 3: F.max_pool3d}[dims]
    if mode == "avg":
        return avg_pool(
            x,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=bool(attrs.get("count_include_pad", 0)),
        )
    return max_pool(
        x, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode
    )


def _pad(inputs: list[torch.Tensor], attrs: dict[str, Any]):
    x = inputs[0]
    if len(inputs) > 1:
        pads = [int(v) for v in inputs[1].detach().cpu().flatten().tolist()]
    else:
        pads = [int(v) for v in attrs.get("pads", [])]
    if not pads or all(value == 0 for value in pads):
        return x
    value = 0.0
    if len(inputs) > 2:
        value = float(inputs[2].detach().cpu().flatten()[0].item())
    mode = attrs.get("mode", "constant")
    if isinstance(mode, bytes):
        mode = mode.decode("utf-8")
    mode = str(mode).lower()
    torch_pad = _onnx_pads_to_torch(pads)
    if mode == "constant":
        return F.pad(x, torch_pad, mode="constant", value=value)
    if mode == "edge":
        mode = "replicate"
    return F.pad(x, torch_pad, mode=mode)


def _axes_from_inputs_or_attrs(
    inputs: list[torch.Tensor], attrs: dict[str, Any]
) -> list[int] | None:
    if len(inputs) > 1:
        return [int(v) for v in inputs[1].detach().cpu().flatten().tolist()]
    axes = attrs.get("axes")
    if axes is None:
        return None
    return [int(v) for v in axes]


def _clip_bound(
    inputs: list[torch.Tensor], attrs: dict[str, Any], input_idx: int, attr_name: str
):
    if len(inputs) > input_idx:
        return inputs[input_idx]
    value = attrs.get(attr_name)
    return None if value is None else float(value)


def _gather(data: torch.Tensor, indices: torch.Tensor, axis: int) -> torch.Tensor:
    axis = axis if axis >= 0 else axis + data.dim()
    flat_indices = indices.to(device=data.device, dtype=torch.long).flatten()
    gathered = torch.index_select(data, axis, flat_indices)
    return gathered.reshape(
        tuple(data.shape[:axis]) + tuple(indices.shape) + tuple(data.shape[axis + 1 :])
    )


def _normalize_axes(axes: list[int], rank: int) -> list[int]:
    return [axis if axis >= 0 else axis + rank for axis in axes]


def _onnx_pads_to_torch(pads: list[int]) -> list[int]:
    dims = len(pads) // 2
    torch_pad = []
    for dim in reversed(range(dims)):
        torch_pad.extend([pads[dim], pads[dim + dims]])
    return torch_pad


def _shape_from_tensor(shape: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(v) for v in shape.detach().cpu().flatten().tolist())


def _buffer_name(name: str) -> str:
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
    return f"const_{digest}"
