from __future__ import annotations

from torch import nn


class Cifar10VGG16BN(nn.Module):
    cfg = [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ]

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = self._make_layers()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

    @classmethod
    def _make_layers(cls):
        layers = []
        in_channels = 3
        for item in cls.cfg:
            if item == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend(
                    [
                        nn.Conv2d(in_channels, item, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(item),
                        nn.ReLU(),
                    ]
                )
                in_channels = item
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*layers)
