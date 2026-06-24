"""CNN entraîné from scratch — premier modèle de la brique de classification supervisée.

Architecture justifiée par rapport au cours:
- Blocs répétés Conv -> BatchNorm -> ReLU -> MaxPool, comme la logique "blocs/modules
  répétitifs" introduite avec VGGNet.
- Noyaux 3x3 avec padding 1 (conservent la résolution), choix VGG.
- BatchNorm pour stabiliser l'apprentissage et régulariser implicitement.
- ReLU pour la non-linéarité.
- Global Average Pooling en fin de réseau plutôt qu'une grosse pile de couches denses :
  réduit le nombre de paramètres et le surapprentissage .
- La sortie est un vecteur de 14 LOGITS (pas de softmax) : la sigmoïde + BCE sont appliquées
  dans la perte (multi-label), conformément à l'énoncé.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Bloc Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # divise la résolution par 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNScratch(nn.Module):
    """CNN simple entraîné depuis zéro pour la classification multi-label.

    Entrée : (B, 1, H, W) — radiographies en niveaux de gris.
    Sortie : (B, num_classes) — logits (un par pathologie).
    """

    def __init__(self, num_classes: int = 14, in_channels: int = 1, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),   # H/2
            ConvBlock(32, 64),            # H/4
            ConvBlock(64, 128),           # H/8
            ConvBlock(128, 256),          # H/16
        )
        # Global Average Pooling : (B, 256, h, w) -> (B, 256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),          # régularisation (cours 4.5, slide 97)
            nn.Linear(256, num_classes),  # logits, pas d'activation finale ici
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)