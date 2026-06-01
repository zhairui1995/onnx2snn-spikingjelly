from __future__ import annotations

from collections import Counter, defaultdict

from .core import CanonicalGraph, PatternGroup


def analyze_patterns(graph: CanonicalGraph) -> list[PatternGroup]:
    """Find editable high-level patterns in a canonical ONNX graph."""

    consumers = _build_consumers(graph)
    groups: list[PatternGroup] = []
    groups.extend(_find_conv_activation_blocks(graph, consumers))
    groups.extend(_find_linear_activation_blocks(graph, consumers))
    groups.extend(_find_vgg_stages(graph))
    groups.extend(_find_residual_adds(graph, consumers))
    return groups


def pattern_report(groups: list[PatternGroup]) -> dict:
    counts = Counter(group.pattern_type for group in groups)
    return {
        "counts": dict(sorted(counts.items())),
        "groups": [group.to_dict() for group in groups],
    }


def _build_consumers(graph: CanonicalGraph) -> dict[str, list[int]]:
    consumers: dict[str, list[int]] = defaultdict(list)
    for idx, node in enumerate(graph.nodes):
        for name in node.inputs:
            if name:
                consumers[name].append(idx)
    return consumers


def _find_conv_activation_blocks(
    graph: CanonicalGraph, consumers: dict[str, list[int]]
) -> list[PatternGroup]:
    groups = []
    for idx, node in enumerate(graph.nodes):
        if node.op_type != "Conv":
            continue
        bn_idx = _single_consumer_idx(graph, consumers, idx, expected_op="BatchNormalization")
        if bn_idx is not None:
            relu_idx = _single_consumer_idx(
                graph, consumers, bn_idx, expected_op="Relu"
            )
            if relu_idx is not None:
                groups.append(
                    _make_group(
                        graph,
                        "conv_bn_relu",
                        [idx, bn_idx, relu_idx],
                        name=f"conv_bn_relu_{len(groups)}",
                    )
                )
                continue
        relu_idx = _single_consumer_idx(graph, consumers, idx, expected_op="Relu")
        if relu_idx is not None:
            groups.append(
                _make_group(
                    graph,
                    "conv_relu",
                    [idx, relu_idx],
                    name=f"conv_relu_{len(groups)}",
                )
            )
    return groups


def _find_linear_activation_blocks(
    graph: CanonicalGraph, consumers: dict[str, list[int]]
) -> list[PatternGroup]:
    groups = []
    for idx, node in enumerate(graph.nodes):
        if node.op_type not in {"Gemm", "MatMul"}:
            continue
        relu_idx = _single_consumer_idx(graph, consumers, idx, expected_op="Relu")
        if relu_idx is not None:
            groups.append(
                _make_group(
                    graph,
                    "linear_relu",
                    [idx, relu_idx],
                    name=f"linear_relu_{len(groups)}",
                )
            )
    return groups


def _find_vgg_stages(graph: CanonicalGraph) -> list[PatternGroup]:
    groups = []
    start = 0
    for idx, node in enumerate(graph.nodes):
        if node.op_type not in {"MaxPool", "AveragePool"}:
            continue
        indices = list(range(start, idx + 1))
        ops = [graph.nodes[i].op_type for i in indices]
        if "Conv" in ops and ops[-1] in {"MaxPool", "AveragePool"}:
            groups.append(
                _make_group(
                    graph,
                    "vgg_stage",
                    indices,
                    name=f"vgg_stage_{len(groups)}",
                    attrs={"pool_op": node.op_type},
                )
            )
        start = idx + 1
    return groups


def _find_residual_adds(
    graph: CanonicalGraph, consumers: dict[str, list[int]]
) -> list[PatternGroup]:
    groups = []
    for idx, node in enumerate(graph.nodes):
        if node.op_type != "Add":
            continue
        indices = [idx]
        relu_idx = _single_consumer_idx(graph, consumers, idx, expected_op="Relu")
        if relu_idx is not None:
            indices.append(relu_idx)
        groups.append(
            _make_group(
                graph,
                "residual_add",
                indices,
                name=f"residual_add_{len(groups)}",
                attrs={"has_post_relu": relu_idx is not None},
            )
        )
    return groups


def _single_consumer_idx(
    graph: CanonicalGraph,
    consumers: dict[str, list[int]],
    node_idx: int,
    expected_op: str | None = None,
) -> int | None:
    node = graph.nodes[node_idx]
    if len(node.outputs) != 1:
        return None
    consumer_indices = consumers.get(node.outputs[0], [])
    if len(consumer_indices) != 1:
        return None
    next_idx = consumer_indices[0]
    if expected_op is not None and graph.nodes[next_idx].op_type != expected_op:
        return None
    return next_idx


def _make_group(
    graph: CanonicalGraph,
    pattern_type: str,
    node_indices: list[int],
    name: str,
    attrs: dict | None = None,
) -> PatternGroup:
    nodes = [graph.nodes[idx] for idx in node_indices]
    produced = {output for node in nodes for output in node.outputs}
    consumed = {input_name for node in nodes for input_name in node.inputs if input_name}
    external_inputs = sorted(consumed - produced)
    external_outputs = [
        output
        for output in nodes[-1].outputs
        if output not in graph.initializers
    ]
    return PatternGroup(
        name=name,
        pattern_type=pattern_type,
        node_indices=node_indices,
        node_names=[node.name for node in nodes],
        op_types=[node.op_type for node in nodes],
        inputs=external_inputs,
        outputs=external_outputs,
        attrs=attrs or {},
    )
