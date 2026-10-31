
"""Multi-scale ResNet-18 for galaxy morphology classification."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

from .labels import NUM_CLASSES


class MultiScaleResNet18(nn.Module):
    """Fuse feature representations from all four ResNet-18 stages."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # Retain the pretrained ResNet-18 feature extractor.
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # Each stage is globally pooled to one feature vector.
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # 64 + 128 + 256 + 512 = 960 features.
        self.classifier = nn.Sequential(
            nn.Linear(960, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        # Pool each feature stage independently.
        p1 = self.pool(f1).flatten(1)
        p2 = self.pool(f2).flatten(1)
        p3 = self.pool(f3).flatten(1)
        p4 = self.pool(f4).flatten(1)

        # Concatenate all four representations.
        combined = torch.cat([p1, p2, p3, p4], dim=1)

        return self.classifier(combined)


def build_multiscale_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    return MultiScaleResNet18(
        num_classes=num_classes,
        pretrained=pretrained,
    )
