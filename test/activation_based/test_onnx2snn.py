from pathlib import Path

import numpy as np
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


class _MnistMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


class _MnistLeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class _AlexNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=8),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class _MnistLeNet5Tanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class _DepthwiseCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1, groups=3),
            nn.ReLU(),
            nn.Conv2d(3, 8, kernel_size=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=32),
            nn.Flatten(),
            nn.Linear(8, 10),
        )

    def forward(self, x):
        return self.net(x)


class _FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_relu = nn.ReLU()
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand1x1_relu = nn.ReLU()
        self.expand3x3 = nn.Conv2d(
            squeeze_channels, expand_channels, kernel_size=3, padding=1
        )
        self.expand3x3_relu = nn.ReLU()

    def forward(self, x):
        x = self.squeeze_relu(self.squeeze(x))
        return torch.cat(
            [
                self.expand1x1_relu(self.expand1x1(x)),
                self.expand3x3_relu(self.expand3x3(x)),
            ],
            dim=1,
        )


class _SqueezeNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            _FireModule(8, 4, 8),
            _FireModule(16, 4, 8),
            nn.AvgPool2d(kernel_size=16),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


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


def test_convert_pad_constant_graph(tmp_path: Path):
    torch.manual_seed(6)
    x = torch.rand(2, 1, 8, 8)
    onnx_path = _make_pad_constant_onnx(tmp_path / "pad.onnx")

    artifacts = convert_onnx_to_snn(
        onnx_path,
        tmp_path / "pad_artifacts",
        {"input_shape": (1, 1, 8, 8), "t": 2, "compare_onnxruntime": False},
        calibration_loader=DataLoader(TensorDataset(x), batch_size=1),
    )

    assert artifacts.report["op_counts"]["Pad"] == 1
    with torch.no_grad():
        assert artifacts.ann_model(x).shape == (2, 3)
        assert torch.allclose(
            artifacts.ann_model(x),
            artifacts.structured_ann_model(x),
            atol=1.0e-6,
            rtol=1.0e-6,
        )


@pytest.mark.parametrize(
    ("model_name", "model", "input_shape", "expected_ops"),
    [
        (
            "mnist_mlp",
            _MnistMlp(),
            (1, 1, 28, 28),
            {"Reshape": 1, "Gemm": 3, "Relu": 2},
        ),
        (
            "mnist_lenet5",
            _MnistLeNet5(),
            (1, 1, 28, 28),
            {"Conv": 2, "AveragePool": 2, "Reshape": 1, "Gemm": 3, "Relu": 4},
        ),
    ],
)
def test_convert_mnist_lenet5_and_mlp_models(
    tmp_path: Path, model_name: str, model: nn.Module, input_shape, expected_ops
):
    torch.manual_seed(7)
    model.eval()
    x = torch.rand(4, *input_shape[1:])
    onnx_path = _export_onnx(model, x, tmp_path / f"{model_name}.onnx")

    artifacts = convert_onnx_to_snn(
        onnx_path,
        tmp_path / f"{model_name}_artifacts",
        {"input_shape": input_shape, "t": 3, "compare_onnxruntime": True},
        calibration_loader=DataLoader(TensorDataset(x), batch_size=2),
    )

    for op_type, count in expected_ops.items():
        assert artifacts.report["op_counts"][op_type] == count
    assert artifacts.report["onnx_vs_ann"]["allclose_1e_4"]
    assert artifacts.report["structured_ann"]["allclose_1e_6"]
    assert artifacts.report["calibration"]["relu_scales"]
    with torch.no_grad():
        assert artifacts.ann_model(x[:2]).shape == (2, 10)
        assert artifacts.snn_model(x[:2]).shape == (2, 10)


@pytest.mark.parametrize(
    ("model_name", "model", "input_shape", "required_ops"),
    [
        (
            "alexnet_small",
            _AlexNetSmall(),
            (1, 3, 32, 32),
            {"Conv": 3, "Relu": 4, "MaxPool": 2, "AveragePool": 1, "Gemm": 2},
        ),
        (
            "mnist_lenet5_tanh",
            _MnistLeNet5Tanh(),
            (1, 1, 28, 28),
            {"Conv": 2, "Tanh": 4, "AveragePool": 2, "Gemm": 3},
        ),
        (
            "depthwise_cnn",
            _DepthwiseCnn(),
            (1, 3, 32, 32),
            {"Conv": 2, "Relu": 2, "AveragePool": 1, "Gemm": 1},
        ),
        (
            "squeezenet_small",
            _SqueezeNetSmall(),
            (1, 3, 32, 32),
            {"Conv": 7, "Relu": 7, "Concat": 2, "MaxPool": 1, "Gemm": 1},
        ),
    ],
)
def test_convert_additional_classification_models(
    tmp_path: Path, model_name: str, model: nn.Module, input_shape, required_ops
):
    torch.manual_seed(12)
    model.eval()
    x = torch.rand(4, *input_shape[1:])
    onnx_path = _export_onnx(model, x, tmp_path / f"{model_name}.onnx")

    artifacts = convert_onnx_to_snn(
        onnx_path,
        tmp_path / f"{model_name}_artifacts",
        {"input_shape": input_shape, "t": 3, "compare_onnxruntime": True},
        calibration_loader=DataLoader(TensorDataset(x), batch_size=2),
    )

    for op_type, count in required_ops.items():
        assert artifacts.report["op_counts"].get(op_type, 0) >= count
    assert artifacts.report["onnx_vs_ann"]["allclose_1e_4"]
    assert artifacts.report["structured_ann"]["allclose_1e_6"]
    with torch.no_grad():
        assert artifacts.ann_model(x[:2]).shape == (2, 10)
        assert artifacts.snn_model(x[:2]).shape == (2, 10)


def test_convert_temporal_conv1d_graph(tmp_path: Path):
    torch.manual_seed(8)
    x = torch.rand(4, 2, 16)
    onnx_path = _make_temporal_conv1d_onnx(tmp_path / "temporal_conv1d.onnx")

    artifacts = convert_onnx_to_snn(
        onnx_path,
        tmp_path / "temporal_conv1d_artifacts",
        {"input_shape": (1, 2, 16), "t": 3, "compare_onnxruntime": True},
        calibration_loader=DataLoader(TensorDataset(x), batch_size=2),
    )

    assert artifacts.report["op_counts"]["Conv"] == 1
    assert artifacts.report["op_counts"]["BatchNormalization"] == 1
    assert artifacts.report["op_counts"]["MaxPool"] == 1
    assert artifacts.report["onnx_vs_ann"]["allclose_1e_4"]
    with torch.no_grad():
        assert artifacts.ann_model(x[:2]).shape == (2, 5)
        assert artifacts.snn_model(x[:2]).shape == (2, 5)


def test_convert_conv3d_global_maxpool_graph(tmp_path: Path):
    torch.manual_seed(9)
    x = torch.rand(2, 1, 4, 4, 4)
    onnx_path = _make_conv3d_global_maxpool_onnx(tmp_path / "conv3d_global_max.onnx")

    artifacts = convert_onnx_to_snn(
        onnx_path,
        tmp_path / "conv3d_global_max_artifacts",
        {"input_shape": (1, 1, 4, 4, 4), "t": 2, "compare_onnxruntime": True},
        calibration_loader=DataLoader(TensorDataset(x), batch_size=1),
    )

    assert artifacts.report["op_counts"]["Conv"] == 1
    assert artifacts.report["op_counts"]["BatchNormalization"] == 1
    assert artifacts.report["op_counts"]["GlobalMaxPool"] == 1
    assert artifacts.report["onnx_vs_ann"]["allclose_1e_4"]
    with torch.no_grad():
        assert artifacts.ann_model(x).shape == (2, 3)


def test_convert_elementwise_operator_graph(tmp_path: Path):
    torch.manual_seed(10)
    x = torch.rand(3, 4) - 0.5
    onnx_path = _make_elementwise_onnx(tmp_path / "elementwise.onnx")

    artifacts = convert_onnx_to_snn(
        onnx_path,
        tmp_path / "elementwise_artifacts",
        {"input_shape": (1, 4), "t": 1, "compare_onnxruntime": True},
        calibration_loader=DataLoader(TensorDataset(x), batch_size=3),
    )

    for op_type in [
        "Mul",
        "Sub",
        "Div",
        "Neg",
        "Abs",
        "Clip",
        "Greater",
        "Where",
        "Dropout",
    ]:
        assert artifacts.report["op_counts"][op_type] == 1
    assert artifacts.report["onnx_vs_ann"]["allclose_1e_4"]


def test_convert_shape_operator_graph(tmp_path: Path):
    torch.manual_seed(11)
    x = torch.rand(2, 1, 3, 1)
    onnx_path = _make_shape_ops_onnx(tmp_path / "shape_ops.onnx")

    artifacts = convert_onnx_to_snn(
        onnx_path,
        tmp_path / "shape_ops_artifacts",
        {"input_shape": (2, 1, 3, 1), "t": 1, "compare_onnxruntime": True},
        calibration_loader=DataLoader(TensorDataset(x), batch_size=2),
    )

    for op_type in ["Unsqueeze", "Shape", "Gather", "Expand"]:
        assert artifacts.report["op_counts"][op_type] == 1
    assert artifacts.report["op_counts"]["Squeeze"] == 2
    assert artifacts.report["onnx_vs_ann"]["allclose_1e_4"]
    with torch.no_grad():
        assert artifacts.ann_model(x).shape == (2, 3)


def test_convert_slice_split_concat_graph(tmp_path: Path):
    torch.manual_seed(13)
    x = torch.rand(2, 4, 5)
    onnx_path = _make_slice_split_concat_onnx(tmp_path / "slice_split_concat.onnx")

    artifacts = convert_onnx_to_snn(
        onnx_path,
        tmp_path / "slice_split_concat_artifacts",
        {"input_shape": (2, 4, 5), "t": 1, "compare_onnxruntime": True},
        calibration_loader=DataLoader(TensorDataset(x), batch_size=2),
    )

    assert artifacts.report["op_counts"]["Slice"] == 1
    assert artifacts.report["op_counts"]["Split"] == 1
    assert artifacts.report["op_counts"]["Concat"] == 1
    assert artifacts.report["onnx_vs_ann"]["allclose_1e_4"]
    with torch.no_grad():
        assert artifacts.ann_model(x).shape == (2, 4, 3)


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


def _make_pad_constant_onnx(path: Path) -> Path:
    helper = onnx.helper
    numpy_helper = onnx.numpy_helper
    tensor_proto = onnx.TensorProto

    weight = np.random.default_rng(6).standard_normal((2, 1, 3, 3)).astype(np.float32)
    fc_weight = np.random.default_rng(7).standard_normal((3, 128)).astype(np.float32)
    fc_bias = np.random.default_rng(8).standard_normal((3,)).astype(np.float32)
    pads = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
    pad_value = np.array([0.0], dtype=np.float32)

    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=["pads"],
                value=numpy_helper.from_array(pads, name="pads_tensor"),
            ),
            helper.make_node(
                "Constant",
                inputs=[],
                outputs=["pad_value"],
                value=numpy_helper.from_array(pad_value, name="pad_value_tensor"),
            ),
            helper.make_node(
                "Pad",
                inputs=["input", "pads", "pad_value"],
                outputs=["padded"],
                mode="constant",
            ),
            helper.make_node(
                "Conv",
                inputs=["padded", "conv_weight"],
                outputs=["conv_out"],
                pads=[0, 0, 0, 0],
            ),
            helper.make_node("Relu", inputs=["conv_out"], outputs=["relu_out"]),
            helper.make_node("Flatten", inputs=["relu_out"], outputs=["flat"], axis=1),
            helper.make_node(
                "Gemm",
                inputs=["flat", "fc_weight", "fc_bias"],
                outputs=["output"],
                transB=1,
            ),
        ],
        "pad_constant_graph",
        inputs=[
            helper.make_tensor_value_info("input", tensor_proto.FLOAT, [None, 1, 8, 8])
        ],
        outputs=[
            helper.make_tensor_value_info("output", tensor_proto.FLOAT, [None, 3])
        ],
        initializer=[
            numpy_helper.from_array(weight, name="conv_weight"),
            numpy_helper.from_array(fc_weight, name="fc_weight"),
            numpy_helper.from_array(fc_bias, name="fc_bias"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 11)],
        ir_version=7,
    )
    onnx.save(model, str(path))
    onnx.checker.check_model(str(path))
    return path


def _make_temporal_conv1d_onnx(path: Path) -> Path:
    helper = onnx.helper
    numpy_helper = onnx.numpy_helper
    tensor_proto = onnx.TensorProto
    rng = np.random.default_rng(8)

    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                inputs=["input", "conv_weight"],
                outputs=["conv_out"],
                pads=[1, 1],
            ),
            helper.make_node(
                "BatchNormalization",
                inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
                outputs=["bn_out"],
                epsilon=1.0e-5,
            ),
            helper.make_node("Relu", inputs=["bn_out"], outputs=["relu_out"]),
            helper.make_node(
                "MaxPool",
                inputs=["relu_out"],
                outputs=["pooled"],
                kernel_shape=[2],
                strides=[2],
            ),
            helper.make_node("Flatten", inputs=["pooled"], outputs=["flat"], axis=1),
            helper.make_node(
                "Gemm",
                inputs=["flat", "fc_weight", "fc_bias"],
                outputs=["output"],
                transB=1,
            ),
        ],
        "temporal_conv1d_graph",
        inputs=[
            helper.make_tensor_value_info("input", tensor_proto.FLOAT, [None, 2, 16])
        ],
        outputs=[
            helper.make_tensor_value_info("output", tensor_proto.FLOAT, [None, 5])
        ],
        initializer=[
            numpy_helper.from_array(
                rng.standard_normal((4, 2, 3)).astype(np.float32), name="conv_weight"
            ),
            numpy_helper.from_array(np.ones((4,), dtype=np.float32), name="bn_scale"),
            numpy_helper.from_array(np.zeros((4,), dtype=np.float32), name="bn_bias"),
            numpy_helper.from_array(np.zeros((4,), dtype=np.float32), name="bn_mean"),
            numpy_helper.from_array(np.ones((4,), dtype=np.float32), name="bn_var"),
            numpy_helper.from_array(
                rng.standard_normal((5, 32)).astype(np.float32), name="fc_weight"
            ),
            numpy_helper.from_array(
                rng.standard_normal((5,)).astype(np.float32), name="fc_bias"
            ),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 11)],
        ir_version=7,
    )
    onnx.save(model, str(path))
    onnx.checker.check_model(str(path))
    return path


def _make_conv3d_global_maxpool_onnx(path: Path) -> Path:
    helper = onnx.helper
    numpy_helper = onnx.numpy_helper
    tensor_proto = onnx.TensorProto
    rng = np.random.default_rng(9)

    graph = helper.make_graph(
        [
            helper.make_node(
                "Conv",
                inputs=["input", "conv_weight"],
                outputs=["conv_out"],
                pads=[1, 1, 1, 1, 1, 1],
            ),
            helper.make_node(
                "BatchNormalization",
                inputs=["conv_out", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
                outputs=["bn_out"],
                epsilon=1.0e-5,
            ),
            helper.make_node("Relu", inputs=["bn_out"], outputs=["relu_out"]),
            helper.make_node(
                "GlobalMaxPool", inputs=["relu_out"], outputs=["pooled"]
            ),
            helper.make_node("Flatten", inputs=["pooled"], outputs=["flat"], axis=1),
            helper.make_node(
                "Gemm",
                inputs=["flat", "fc_weight", "fc_bias"],
                outputs=["output"],
                transB=1,
            ),
        ],
        "conv3d_global_maxpool_graph",
        inputs=[
            helper.make_tensor_value_info(
                "input", tensor_proto.FLOAT, [None, 1, 4, 4, 4]
            )
        ],
        outputs=[
            helper.make_tensor_value_info("output", tensor_proto.FLOAT, [None, 3])
        ],
        initializer=[
            numpy_helper.from_array(
                rng.standard_normal((2, 1, 3, 3, 3)).astype(np.float32),
                name="conv_weight",
            ),
            numpy_helper.from_array(np.ones((2,), dtype=np.float32), name="bn_scale"),
            numpy_helper.from_array(np.zeros((2,), dtype=np.float32), name="bn_bias"),
            numpy_helper.from_array(np.zeros((2,), dtype=np.float32), name="bn_mean"),
            numpy_helper.from_array(np.ones((2,), dtype=np.float32), name="bn_var"),
            numpy_helper.from_array(
                rng.standard_normal((3, 2)).astype(np.float32), name="fc_weight"
            ),
            numpy_helper.from_array(
                rng.standard_normal((3,)).astype(np.float32), name="fc_bias"
            ),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 11)],
        ir_version=7,
    )
    onnx.save(model, str(path))
    onnx.checker.check_model(str(path))
    return path


def _make_elementwise_onnx(path: Path) -> Path:
    helper = onnx.helper
    numpy_helper = onnx.numpy_helper
    tensor_proto = onnx.TensorProto

    graph = helper.make_graph(
        [
            helper.make_node("Mul", inputs=["input", "scale"], outputs=["mul_out"]),
            helper.make_node("Sub", inputs=["mul_out", "bias"], outputs=["sub_out"]),
            helper.make_node("Div", inputs=["sub_out", "scale"], outputs=["div_out"]),
            helper.make_node("Neg", inputs=["div_out"], outputs=["neg_out"]),
            helper.make_node("Abs", inputs=["neg_out"], outputs=["abs_out"]),
            helper.make_node(
                "Clip", inputs=["abs_out", "clip_min", "clip_max"], outputs=["clip_out"]
            ),
            helper.make_node(
                "Greater", inputs=["clip_out", "threshold"], outputs=["mask"]
            ),
            helper.make_node("Add", inputs=["input", "bias"], outputs=["add_out"]),
            helper.make_node(
                "Where", inputs=["mask", "clip_out", "add_out"], outputs=["where_out"]
            ),
            helper.make_node(
                "Dropout",
                inputs=["where_out"],
                outputs=["output"],
                ratio=0.5,
            ),
        ],
        "elementwise_graph",
        inputs=[
            helper.make_tensor_value_info("input", tensor_proto.FLOAT, [None, 4])
        ],
        outputs=[
            helper.make_tensor_value_info("output", tensor_proto.FLOAT, [None, 4])
        ],
        initializer=[
            numpy_helper.from_array(np.array([2.0], dtype=np.float32), name="scale"),
            numpy_helper.from_array(np.array([0.25], dtype=np.float32), name="bias"),
            numpy_helper.from_array(np.array([0.0], dtype=np.float32), name="clip_min"),
            numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="clip_max"),
            numpy_helper.from_array(
                np.array([0.5], dtype=np.float32), name="threshold"
            ),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 11)],
        ir_version=7,
    )
    onnx.save(model, str(path))
    onnx.checker.check_model(str(path))
    return path


def _make_shape_ops_onnx(path: Path) -> Path:
    helper = onnx.helper
    numpy_helper = onnx.numpy_helper
    tensor_proto = onnx.TensorProto

    graph = helper.make_graph(
        [
            helper.make_node(
                "Squeeze", inputs=["input", "squeeze_axes"], outputs=["squeezed"]
            ),
            helper.make_node(
                "Unsqueeze",
                inputs=["squeezed", "unsqueeze_axes"],
                outputs=["unsqueezed"],
            ),
            helper.make_node(
                "Squeeze",
                inputs=["unsqueezed", "unsqueeze_axes"],
                outputs=["resqueezed"],
            ),
            helper.make_node("Shape", inputs=["resqueezed"], outputs=["shape"]),
            helper.make_node(
                "Gather", inputs=["shape", "gather_indices"], outputs=["expand_shape"]
            ),
            helper.make_node(
                "Expand", inputs=["one", "expand_shape"], outputs=["expanded"]
            ),
            helper.make_node(
                "Add", inputs=["resqueezed", "expanded"], outputs=["output"]
            ),
        ],
        "shape_ops_graph",
        inputs=[
            helper.make_tensor_value_info("input", tensor_proto.FLOAT, [2, 1, 3, 1])
        ],
        outputs=[
            helper.make_tensor_value_info("output", tensor_proto.FLOAT, [2, 3])
        ],
        initializer=[
            numpy_helper.from_array(
                np.array([1, 3], dtype=np.int64), name="squeeze_axes"
            ),
            numpy_helper.from_array(
                np.array([1], dtype=np.int64), name="unsqueeze_axes"
            ),
            numpy_helper.from_array(
                np.array([0, 1], dtype=np.int64), name="gather_indices"
            ),
            numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="one"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        ir_version=7,
    )
    onnx.save(model, str(path))
    onnx.checker.check_model(str(path))
    return path


def _make_slice_split_concat_onnx(path: Path) -> Path:
    helper = onnx.helper
    numpy_helper = onnx.numpy_helper
    tensor_proto = onnx.TensorProto

    graph = helper.make_graph(
        [
            helper.make_node(
                "Slice",
                inputs=["input", "starts", "ends", "axes", "steps"],
                outputs=["sliced"],
            ),
            helper.make_node(
                "Split",
                inputs=["sliced", "split_sizes"],
                outputs=["left", "right"],
                axis=1,
            ),
            helper.make_node(
                "Concat", inputs=["right", "left"], outputs=["output"], axis=1
            ),
        ],
        "slice_split_concat_graph",
        inputs=[
            helper.make_tensor_value_info("input", tensor_proto.FLOAT, [2, 4, 5])
        ],
        outputs=[
            helper.make_tensor_value_info("output", tensor_proto.FLOAT, [2, 4, 3])
        ],
        initializer=[
            numpy_helper.from_array(np.array([1], dtype=np.int64), name="starts"),
            numpy_helper.from_array(np.array([4], dtype=np.int64), name="ends"),
            numpy_helper.from_array(np.array([2], dtype=np.int64), name="axes"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps"),
            numpy_helper.from_array(
                np.array([2, 2], dtype=np.int64), name="split_sizes"
            ),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 13)],
        ir_version=7,
    )
    onnx.save(model, str(path))
    onnx.checker.check_model(str(path))
    return path
