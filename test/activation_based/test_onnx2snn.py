from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from spikingjelly.activation_based.ann2snn.sample_models import cifar10_resnet
from spikingjelly.activation_based.onnx2snn import (
    ConversionConfig,
    UnsupportedONNXError,
    convert_onnx_to_snn,
)
from spikingjelly.activation_based.onnx2snn.loader import load_onnx_graph
from spikingjelly.activation_based.onnx2snn.patterns import analyze_patterns
from spikingjelly.activation_based.onnx2snn.structured import (
    BasicBlock,
    BottleneckBlock,
    build_structured_ann_model,
)

onnx = pytest.importorskip("onnx")


class _ConvBnRelu(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten(),
            nn.Linear(4 * 4 * 4, 3),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class _ResidualBlockNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 2, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(2, 2, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(2)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2, 2)

    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu2(out + identity)
        out = self.pool(out)
        return self.fc(self.flatten(out))


class _VggLikeMaxPoolNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(8 * 2 * 2, 3),
        )

    def forward(self, x):
        return self.net(x)


class _UnsupportedSigmoid(nn.Module):
    def forward(self, x):
        return torch.sigmoid(x)


def test_convert_conv_bn_relu_onnx_to_dual_artifacts(tmp_path: Path):
    torch.manual_seed(0)
    model = _ConvBnRelu().eval()
    x = torch.rand(4, 1, 8, 8)
    onnx_path = _export_onnx(model, x, tmp_path / "conv_bn_relu.onnx")
    loader = DataLoader(TensorDataset(x, torch.tensor([0, 1, 2, 1])), batch_size=2)

    artifacts = convert_onnx_to_snn(
        onnx_path,
        tmp_path / "artifacts",
        ConversionConfig(input_shape=(1, 1, 8, 8), t=4, compare_onnxruntime=True),
        calibration_loader=loader,
        eval_loader=loader,
    )

    with torch.no_grad():
        ann_out = artifacts.ann_model(x[:2])
        snn_out = artifacts.snn_model(x[:2])

    assert ann_out.shape == (2, 3)
    assert snn_out.shape == (2, 3)
    assert artifacts.report["op_counts"]["Conv"] >= 1
    assert artifacts.report["evaluation"]["num_samples"] == 4
    assert artifacts.report["calibration"]["relu_scales"]
    for name in [
        "ann_model.pt",
        "structured_ann_model.pt",
        "snn_model.pt",
        "conversion_config.json",
        "report.json",
        "run_inference.py",
        "evaluate.py",
    ]:
        assert (tmp_path / "artifacts" / name).exists()


def test_convert_residual_add_graph(tmp_path: Path):
    torch.manual_seed(1)
    model = _ResidualBlockNet().eval()
    x = torch.rand(3, 2, 8, 8)
    onnx_path = _export_onnx(model, x, tmp_path / "residual.onnx")

    artifacts = convert_onnx_to_snn(
        onnx_path,
        tmp_path / "residual_artifacts",
        {"input_shape": (1, 2, 8, 8), "t": 3, "compare_onnxruntime": False},
        calibration_loader=DataLoader(TensorDataset(x), batch_size=1),
    )

    assert artifacts.report["op_counts"]["Add"] >= 1
    assert artifacts.report["pattern_groups"]["counts"]["residual_add"] == 1
    assert len(artifacts.report["calibration"]["relu_scales"]) >= 2
    with torch.no_grad():
        assert artifacts.snn_model(x[:1]).shape == (1, 2)


def test_convert_vgg_like_maxpool_graph_replaces_snn_pooling(tmp_path: Path):
    torch.manual_seed(2)
    model = _VggLikeMaxPoolNet().eval()
    x = torch.rand(4, 3, 8, 8)
    onnx_path = _export_onnx(model, x, tmp_path / "vgg_like.onnx")

    artifacts = convert_onnx_to_snn(
        onnx_path,
        tmp_path / "vgg_artifacts",
        {"input_shape": (1, 3, 8, 8), "t": 3, "compare_onnxruntime": False},
        calibration_loader=DataLoader(TensorDataset(x), batch_size=2),
    )

    assert artifacts.report["op_counts"]["MaxPool"] == 2
    assert artifacts.report["pattern_groups"]["counts"]["vgg_stage"] == 2
    conv_block_count = artifacts.report["pattern_groups"]["counts"].get(
        "conv_bn_relu", 0
    ) + artifacts.report["pattern_groups"]["counts"].get("conv_relu", 0)
    assert conv_block_count == 2
    assert artifacts.report["snn_graph_transform"]["maxpool_to_avgpool_count"] == 2
    assert "MaxPool" not in {node.op_type for node in artifacts.snn_model.graph.nodes}
    assert artifacts.report["calibration"]["relu_scales"]
    with torch.no_grad():
        assert artifacts.ann_model(x[:1]).shape == (1, 3)
        assert artifacts.snn_model(x[:1]).shape == (1, 3)


def test_pattern_grouping_finds_editable_vgg_stages(tmp_path: Path):
    torch.manual_seed(3)
    model = _VggLikeMaxPoolNet().eval()
    onnx_path = _export_onnx(model, torch.rand(1, 3, 8, 8), tmp_path / "vgg_patterns.onnx")
    graph = load_onnx_graph(str(onnx_path))

    groups = analyze_patterns(graph)
    vgg_stages = [group for group in groups if group.pattern_type == "vgg_stage"]
    conv_blocks = [
        group
        for group in groups
        if group.pattern_type in {"conv_bn_relu", "conv_relu"}
    ]

    assert len(vgg_stages) == 2
    assert len(conv_blocks) == 2
    assert vgg_stages[0].op_types[-1] == "MaxPool"
    assert conv_blocks[0].op_types in (
        ["Conv", "BatchNormalization", "Relu"],
        ["Conv", "Relu"],
    )


@pytest.mark.parametrize(
    ("model_name", "constructor", "expected_pattern", "expected_blocks"),
    [
        ("resnet18", cifar10_resnet.ResNet18, "resnet_basic_block", 8),
        ("resnet34", cifar10_resnet.ResNet34, "resnet_basic_block", 16),
        ("resnet50", cifar10_resnet.ResNet50, "resnet_bottleneck_block", 16),
    ],
)
def test_pattern_grouping_finds_resnet_blocks(
    tmp_path: Path, model_name, constructor, expected_pattern, expected_blocks
):
    torch.manual_seed(4)
    model = constructor().eval()
    onnx_path = _export_onnx(
        model, torch.rand(1, 3, 32, 32), tmp_path / f"{model_name}.onnx"
    )
    graph = load_onnx_graph(str(onnx_path))

    groups = analyze_patterns(graph)
    counts = {}
    for group in groups:
        counts[group.pattern_type] = counts.get(group.pattern_type, 0) + 1

    assert counts["residual_add"] == expected_blocks
    assert counts[expected_pattern] == expected_blocks
    assert "vgg_stage" not in counts


@pytest.mark.parametrize(
    ("constructor", "block_cls", "expected_blocks", "expected_stage_sizes"),
    [
        (cifar10_resnet.ResNet18, BasicBlock, 8, [2, 2, 2, 2]),
        (cifar10_resnet.ResNet50, BottleneckBlock, 16, [3, 4, 6, 3]),
    ],
)
def test_structured_ann_model_uses_resnet_blocks(
    tmp_path: Path, constructor, block_cls, expected_blocks, expected_stage_sizes
):
    torch.manual_seed(5)
    model = constructor().eval()
    x = torch.rand(2, 3, 32, 32)
    onnx_path = _export_onnx(model, x, tmp_path / "resnet_structured.onnx")
    graph = load_onnx_graph(str(onnx_path))

    flat_artifacts = convert_onnx_to_snn(
        onnx_path,
        tmp_path / "structured_artifacts",
        {"input_shape": (1, 3, 32, 32), "t": 2, "compare_onnxruntime": False},
        calibration_loader=DataLoader(TensorDataset(x), batch_size=1),
    )
    structured = build_structured_ann_model(graph)

    assert sum(1 for module in structured.modules() if isinstance(module, block_cls)) == expected_blocks
    assert isinstance(structured.conv1, nn.Conv2d)
    assert isinstance(structured.relu, nn.ReLU)
    assert isinstance(structured.avgpool, nn.AvgPool2d)
    assert isinstance(structured.flatten, nn.Flatten)
    assert isinstance(structured.fc, nn.Linear)
    assert structured.readable_layer_names == ("layer1", "layer2", "layer3", "layer4")
    assert [len(getattr(structured, name)) for name in structured.readable_layer_names] == expected_stage_sizes
    first_block = next(module for module in structured.modules() if isinstance(module, block_cls))
    assert isinstance(first_block.conv1, nn.Conv2d)
    assert isinstance(first_block.relu1, nn.ReLU)
    assert "shortcut" in first_block.readable_layer_names
    if block_cls is BasicBlock:
        assert isinstance(first_block.conv2, nn.Conv2d)
        assert isinstance(first_block.relu2, nn.ReLU)
    else:
        assert isinstance(first_block.conv2, nn.Conv2d)
        assert isinstance(first_block.conv3, nn.Conv2d)
        assert isinstance(first_block.relu3, nn.ReLU)
    projection_blocks = [
        module
        for module in structured.modules()
        if isinstance(module, block_cls) and isinstance(module.shortcut, nn.Conv2d)
    ]
    assert projection_blocks
    for module in structured.modules():
        if isinstance(module, block_cls):
            object.__setattr__(
                module,
                "_graph_forward",
                lambda *args, **kwargs: (_ for _ in ()).throw(
                    AssertionError("standard ResNet forward should be used")
                ),
            )
    object.__setattr__(
        structured,
        "_graph_forward",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("standard top-level ResNet forward should be used")
        ),
    )
    with torch.no_grad():
        flat_out = flat_artifacts.ann_model(x)
        structured_out = structured(x)
    assert torch.allclose(flat_out, structured_out, atol=1.0e-6, rtol=1.0e-6)
    assert flat_artifacts.report["structured_ann"]["allclose_1e_6"]


def test_unsupported_operator_report_is_explicit(tmp_path: Path):
    x = torch.rand(1, 3)
    onnx_path = _export_onnx(_UnsupportedSigmoid().eval(), x, tmp_path / "bad.onnx")

    with pytest.raises(UnsupportedONNXError, match="Sigmoid"):
        load_onnx_graph(str(onnx_path))


def _export_onnx(model: nn.Module, x: torch.Tensor, path: Path) -> Path:
    torch.onnx.export(
        model,
        x,
        str(path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        do_constant_folding=False,
    )
    onnx.checker.check_model(str(path))
    return path
