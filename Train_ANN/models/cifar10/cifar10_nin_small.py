from __future__ import annotations

from torch import nn


class Cifar10NINSmall(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(192, 160, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(160, 96, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(p=0.5),
            nn.Conv2d(96, 192, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Dropout(p=0.5),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(192, num_classes, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        return self.flatten(x)
