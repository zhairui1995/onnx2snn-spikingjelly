from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .cifar10 import (
    Cifar10AlexNetSmall,
    Cifar10DepthwiseCNN,
    Cifar10NINSmall,
    Cifar10ResNet18,
    Cifar10ResNet50,
    Cifar10SqueezeNetSmall,
    Cifar10VGG16BN,
)
from .mnist import (
    MnistConvBNGAP,
    MnistLeNetReLU,
    MnistLeNetTanh,
    MnistMLP,
    MnistTinyResidual,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    dataset: str
    input_shape: tuple[int, ...]
    num_classes: int
    constructor: Callable[..., nn.Module]
    description: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "mnist_mlp": ModelSpec(
        "mnist_mlp", "mnist", (1, 28, 28), 10, MnistMLP, "MNIST fully connected MLP"
    ),
    "mnist_lenet_relu": ModelSpec(
        "mnist_lenet_relu", "mnist", (1, 28, 28), 10, MnistLeNetReLU, "LeNet5 with ReLU"
    ),
    "mnist_lenet_tanh": ModelSpec(
        "mnist_lenet_tanh", "mnist", (1, 28, 28), 10, MnistLeNetTanh, "LeNet5 with Tanh"
    ),
    "mnist_conv_bn_gap": ModelSpec(
        "mnist_conv_bn_gap",
        "mnist",
        (1, 28, 28),
        10,
        MnistConvBNGAP,
        "Conv-BN-ReLU with global average pooling",
    ),
    "mnist_tiny_residual": ModelSpec(
        "mnist_tiny_residual",
        "mnist",
        (1, 28, 28),
        10,
        MnistTinyResidual,
        "Small MNIST residual CNN",
    ),
    "cifar10_alexnet_small": ModelSpec(
        "cifar10_alexnet_small",
        "cifar10",
        (3, 32, 32),
        10,
        Cifar10AlexNetSmall,
        "Compact AlexNet-style CIFAR10 CNN",
    ),
    "cifar10_depthwise_cnn": ModelSpec(
        "cifar10_depthwise_cnn",
        "cifar10",
        (3, 32, 32),
        10,
        Cifar10DepthwiseCNN,
        "Depthwise separable CIFAR10 CNN",
    ),
    "cifar10_squeezenet_small": ModelSpec(
        "cifar10_squeezenet_small",
        "cifar10",
        (3, 32, 32),
        10,
        Cifar10SqueezeNetSmall,
        "Small SqueezeNet-style Fire-module CNN",
    ),
    "cifar10_vgg16_bn": ModelSpec(
        "cifar10_vgg16_bn",
        "cifar10",
        (3, 32, 32),
        10,
        Cifar10VGG16BN,
        "CIFAR10 VGG16 with BatchNorm",
    ),
    "cifar10_resnet18": ModelSpec(
        "cifar10_resnet18",
        "cifar10",
        (3, 32, 32),
        10,
        Cifar10ResNet18,
        "CIFAR10 ResNet18 BasicBlock",
    ),
    "cifar10_resnet50": ModelSpec(
        "cifar10_resnet50",
        "cifar10",
        (3, 32, 32),
        10,
        Cifar10ResNet50,
        "CIFAR10 ResNet50 Bottleneck",
    ),
    "cifar10_nin_small": ModelSpec(
        "cifar10_nin_small",
        "cifar10",
        (3, 32, 32),
        10,
        Cifar10NINSmall,
        "Small Network-in-Network CIFAR10 CNN",
    ),
}


def list_models(dataset: str | None = None) -> list[ModelSpec]:
    specs = list(MODEL_REGISTRY.values())
    if dataset is not None:
        specs = [spec for spec in specs if spec.dataset == dataset.lower()]
    return specs


def get_model_spec(model_name: str) -> ModelSpec:
    key = model_name.lower()
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model_name}. Choices: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]


def create_model(model_name: str, num_classes: int | None = None) -> nn.Module:
    spec = get_model_spec(model_name)
    return spec.constructor(num_classes=num_classes or spec.num_classes)


def load_model_from_checkpoint(
    model_name: str,
    checkpoint_path: str,
    map_location: str | torch.device = "cpu",
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model = create_model(model_name)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model
