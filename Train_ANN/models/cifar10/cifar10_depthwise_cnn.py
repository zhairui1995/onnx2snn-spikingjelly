from __future__ import annotations

from torch import nn


class _DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Cifar10DepthwiseCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            _DepthwiseSeparable(32, 64, stride=1),
            _DepthwiseSeparable(64, 128, stride=2),
            _DepthwiseSeparable(128, 128, stride=1),
            _DepthwiseSeparable(128, 256, stride=2),
            _DepthwiseSeparable(256, 256, stride=1),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
