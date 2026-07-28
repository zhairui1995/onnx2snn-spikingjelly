from __future__ import annotations

import torch
from torch import nn


class _Fire(nn.Module):
    def __init__(self, in_channels: int, squeeze: int, expand: int):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channels, squeeze, kernel_size=1),
            nn.ReLU(),
        )
        self.expand1x1 = nn.Sequential(
            nn.Conv2d(squeeze, expand, kernel_size=1),
            nn.ReLU(),
        )
        self.expand3x3 = nn.Sequential(
            nn.Conv2d(squeeze, expand, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.squeeze(x)
        return torch.cat([self.expand1x1(x), self.expand3x3(x)], dim=1)


class Cifar10SqueezeNetSmall(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            _Fire(64, squeeze=16, expand=64),
            _Fire(128, squeeze=16, expand=64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            _Fire(128, squeeze=32, expand=128),
            _Fire(256, squeeze=32, expand=128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
