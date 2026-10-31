
"""Residual multi-scale feature fusion for GalaxyLens."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

from .labels import NUM_CLASSES


class ResidualMultiScaleResNet18(nn.Module):
    """
    ResNet-18 with Stage 4 as the primary representation
    and gated residual contributions from Stages 1–3.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        projection_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        stage_dims = [64, 128, 256, 512]

        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, projection_dim),
                nn.ReLU(inplace=True),
            )
            for dim in stage_dims
        ])

        # Small initial contributions from Stages 1–3.
        # Sigmoid(-2) is approximately 0.119.
        self.residual_gate_logits = nn.Parameter(
            torch.full((3,), -2.0)
        )

        self.classifier = nn.Sequential(
            nn.Linear(projection_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor, return_gates: bool = False):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        features = [f1, f2, f3, f4]

        pooled = [
            self.pool(feature).flatten(1)
            for feature in features
        ]

        projected = [
            projection(feature)
            for projection, feature in zip(self.projections, pooled)
        ]

        gates = torch.sigmoid(self.residual_gate_logits)

        # Stage 4 is the anchor. Earlier stages contribute
        # through a gated residual pathway.
        residual = sum(
            gates[i] * projected[i]
            for i in range(3)
        ) / 3.0

        fused = projected[3] + residual

        logits = self.classifier(fused)

        if return_gates:
            return logits, gates

        return logits


def build_residual_multiscale_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
) -> nn.Module:
    return ResidualMultiScaleResNet18(
        num_classes=num_classes,
        pretrained=pretrained,
    )
