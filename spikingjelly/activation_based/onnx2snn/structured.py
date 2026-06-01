from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .core import CanonicalGraph, PatternGroup
from .model import OnnxGraphModule, build_ann_model
from .patterns import analyze_patterns


class BasicBlock(OnnxGraphModule):
    """Executable ResNet BasicBlock rebuilt from an ONNX residual subgraph."""


class BottleneckBlock(OnnxGraphModule):
    """Executable ResNet Bottleneck block rebuilt from an ONNX residual subgraph."""


class StructuredOnnxGraphModule(OnnxGraphModule):
    """ONNX graph executor that calls grouped high-level modules when available."""

    def __init__(
        self,
        graph: CanonicalGraph,
        modules: dict[str, nn.Module],
        blocks: dict[str, OnnxGraphModule],
        block_specs: dict[int, dict],
    ):
        super().__init__(graph, modules)
        self.blocks = nn.ModuleDict(blocks)
        self.block_specs = block_specs
        self.block_node_indices = {
            node_idx
            for spec in block_specs.values()
            for node_idx in spec["node_indices"]
        }

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

        for idx, node in enumerate(self.graph.nodes):
            if idx in self.block_specs:
                spec = self.block_specs[idx]
                block_inputs = [env[name] for name in spec["inputs"]]
                block = self.blocks[spec["name"]]
                block_outputs = block(*block_inputs)
                if len(spec["outputs"]) == 1:
                    env[spec["outputs"][0]] = block_outputs
                else:
                    for name, value in zip(spec["outputs"], block_outputs):
                        env[name] = value
                continue
            if idx in self.block_node_indices:
                continue
            outputs = self._run_node(node, env)
            if len(node.outputs) == 1:
                env[node.outputs[0]] = outputs
            else:
                for name, value in zip(node.outputs, outputs):
                    env[name] = value

        out = [env[name] for name in self.graph.output_names]
        return out[0] if len(out) == 1 else tuple(out)


def build_structured_ann_model(graph: CanonicalGraph) -> StructuredOnnxGraphModule:
    """Build a readable ANN with real ResNet block modules when patterns are found."""

    flat = build_ann_model(graph)
    groups = _select_resnet_groups(analyze_patterns(graph))
    if not groups:
        return StructuredOnnxGraphModule(
            graph,
            {name: copy.deepcopy(module) for name, module in flat.ops.items()},
            {},
            {},
        ).eval()

    grouped_indices = {idx for group in groups for idx in group.node_indices}
    top_modules = {
        name: copy.deepcopy(module)
        for name, module in flat.ops.items()
        if _module_node_index(graph, name) not in grouped_indices
    }
    blocks: dict[str, OnnxGraphModule] = {}
    block_specs: dict[int, dict] = {}
    for group in groups:
        block_name = group.name
        block_cls = (
            BasicBlock
            if group.pattern_type == "resnet_basic_block"
            else BottleneckBlock
        )
        blocks[block_name] = _build_block_module(graph, flat, group, block_cls)
        block_specs[min(group.node_indices)] = {
            "name": block_name,
            "node_indices": set(group.node_indices),
            "inputs": group.inputs,
            "outputs": group.outputs,
            "pattern_type": group.pattern_type,
        }

    model = StructuredOnnxGraphModule(graph, top_modules, blocks, block_specs)
    model.eval()
    return model


def _select_resnet_groups(groups: list[PatternGroup]) -> list[PatternGroup]:
    selected = [
        group
        for group in groups
        if group.pattern_type in {"resnet_basic_block", "resnet_bottleneck_block"}
    ]
    selected.sort(key=lambda group: min(group.node_indices))
    return selected


def _build_block_module(
    graph: CanonicalGraph,
    flat: OnnxGraphModule,
    group: PatternGroup,
    block_cls: type[OnnxGraphModule],
) -> OnnxGraphModule:
    nodes = [copy.deepcopy(graph.nodes[idx]) for idx in group.node_indices]
    module_names = [
        node.module_name for node in nodes if node.module_name is not None
    ]
    modules = {
        name: copy.deepcopy(flat.ops[name])
        for name in module_names
        if name in flat.ops
    }
    subgraph = CanonicalGraph(
        input_names=group.inputs,
        output_names=group.outputs,
        nodes=nodes,
        initializers=_used_initializers(graph, nodes),
        value_shapes=graph.value_shapes,
        opset_imports=graph.opset_imports,
        module_kinds={
            name: graph.module_kinds[name]
            for name in module_names
            if name in graph.module_kinds
        },
    )
    return block_cls(subgraph, modules).eval()


def _used_initializers(graph: CanonicalGraph, nodes) -> dict:
    names = {
        input_name
        for node in nodes
        for input_name in node.inputs
        if input_name in graph.initializers
    }
    return {name: graph.initializers[name] for name in names}


def _module_node_index(graph: CanonicalGraph, module_name: str) -> int | None:
    for idx, node in enumerate(graph.nodes):
        if node.module_name == module_name:
            return idx
    return None


def _buffer_name(name: str) -> str:
    import hashlib

    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:16]
    return f"const_{digest}"
