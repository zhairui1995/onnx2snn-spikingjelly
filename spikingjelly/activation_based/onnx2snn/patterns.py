from __future__ import annotations

from collections import Counter, defaultdict

from .core import CanonicalGraph, PatternGroup


def analyze_patterns(graph: CanonicalGraph) -> list[PatternGroup]:
    """Find editable high-level patterns in a canonical ONNX graph."""

    consumers = _build_consumers(graph)
    producers = _build_producers(graph)
    groups: list[PatternGroup] = []
    groups.extend(_find_conv_activation_blocks(graph, consumers))
    groups.extend(_find_linear_activation_blocks(graph, consumers))
    groups.extend(_find_vgg_stages(graph))
    groups.extend(_find_residual_adds(graph, consumers))
    groups.extend(_find_resnet_blocks(graph, consumers, producers))
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


def _build_producers(graph: CanonicalGraph) -> dict[str, int]:
    producers: dict[str, int] = {}
    for idx, node in enumerate(graph.nodes):
        for name in node.outputs:
            producers[name] = idx
    return producers


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
        if node.op_type != "MaxPool":
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


def _find_resnet_blocks(
    graph: CanonicalGraph,
    consumers: dict[str, list[int]],
    producers: dict[str, int],
) -> list[PatternGroup]:
    groups = []
    for idx, node in enumerate(graph.nodes):
        if node.op_type != "Add":
            continue
        branches = [
            _trace_residual_branch(graph, producers, input_name)
            for input_name in node.inputs
        ]
        if len(branches) != 2:
            continue
        main_branch = max(branches, key=lambda branch: int(branch["conv_count"]))
        shortcut_branch = min(branches, key=lambda branch: int(branch["conv_count"]))
        common_conv_count = int(shortcut_branch["conv_count"])
        shortcut_conv_count = 1 if bool(shortcut_branch["starts_with_conv"]) else 0
        main_conv_count = (
            int(main_branch["conv_count"]) - common_conv_count + shortcut_conv_count
        )
        if main_conv_count == 2:
            pattern_type = "resnet_basic_block"
        elif main_conv_count == 3:
            pattern_type = "resnet_bottleneck_block"
        else:
            continue
        indices = set(
            _trim_branch_to_conv_count(
                graph, main_branch["node_indices"], main_conv_count
            )
        )
        if shortcut_conv_count:
            indices.update(
                _trim_branch_to_conv_count(
                    graph, shortcut_branch["node_indices"], shortcut_conv_count
                )
            )
        indices.add(idx)
        post_relu_idx = _single_consumer_idx(graph, consumers, idx, expected_op="Relu")
        if post_relu_idx is not None and post_relu_idx not in indices:
            indices.add(post_relu_idx)
        indices = sorted(indices)
        groups.append(
            _make_group(
                graph,
                pattern_type,
                indices,
                name=f"{pattern_type}_{len(groups)}",
                attrs={
                    "main_conv_count": main_conv_count,
                    "shortcut_conv_count": shortcut_conv_count,
                    "has_post_relu": post_relu_idx is not None,
                },
            )
        )
    return groups


def _trace_residual_branch(
    graph: CanonicalGraph,
    producers: dict[str, int],
    start_tensor: str,
) -> dict[str, int | list[int]]:
    tensor = start_tensor
    node_indices: list[int] = []
    conv_count = 0
    starts_with_conv = False
    seen: set[str] = set()
    while tensor and tensor not in seen:
        seen.add(tensor)
        node_idx = producers.get(tensor)
        if node_idx is None:
            break
        node = graph.nodes[node_idx]
        if node.op_type not in {"Conv", "BatchNormalization", "Relu", "Identity"}:
            break
        if not node_indices:
            starts_with_conv = node.op_type == "Conv"
        node_indices.append(node_idx)
        if node.op_type == "Conv":
            conv_count += 1
        if not node.inputs:
            break
        tensor = node.inputs[0]
    return {
        "node_indices": node_indices,
        "conv_count": conv_count,
        "starts_with_conv": starts_with_conv,
    }


def _trim_branch_to_conv_count(
    graph: CanonicalGraph, node_indices: list[int], conv_limit: int
) -> list[int]:
    trimmed = []
    conv_count = 0
    for node_idx in node_indices:
        trimmed.append(node_idx)
        if graph.nodes[node_idx].op_type == "Conv":
            conv_count += 1
            if conv_count >= conv_limit:
                break
    return trimmed


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
    external_inputs = sorted((consumed - produced) - set(graph.initializers))
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
