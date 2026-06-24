"""Fabrique de modèles centralisée.

build_model(name, ...) construit le modèle demandé. Chaque modèle déclare aussi son nombre de
canaux d'entrée (1 = niveaux de gris pour le CNN from scratch ; 3 = RGB pour les modèles
pré-entraînés ImageNet). La boucle d'entraînement lit get_in_channels() pour préparer les images
au bon format (cf. src/data.py, paramètre rgb).
"""
from __future__ import annotations

import torch.nn as nn

from src.models.cnn_scratch import CNNScratch
from src.models.transfer import build_densenet121, build_resnet50

# Nombre de canaux attendus en entrée selon le modèle.
MODEL_CHANNELS = {
    "cnn_scratch": 1,
    "densenet121": 3,
    "resnet50": 3,
}


def get_in_channels(name: str) -> int:
    """Renvoie le nombre de canaux d'entrée attendus par le modèle."""
    name = name.lower()
    if name not in MODEL_CHANNELS:
        raise ValueError(f"Modèle inconnu : {name!r}. Choix : {list(MODEL_CHANNELS)}")
    return MODEL_CHANNELS[name]


def build_model(
    name: str,
    num_classes: int = 14,
    in_channels: int = 1,
    pretrained: bool = True,
    freeze_backbone: bool = False,
) -> nn.Module:
    """Construit le modèle demandé.

    - cnn_scratch : CNN entraîné depuis zéro (1 canal).
    - densenet121 / resnet50 : transfer learning ImageNet (3 canaux), tête multi-label.
    """
    name = name.lower()
    if name == "cnn_scratch":
        return CNNScratch(num_classes=num_classes, in_channels=in_channels)
    if name == "densenet121":
        return build_densenet121(num_classes, pretrained, freeze_backbone)
    if name == "resnet50":
        return build_resnet50(num_classes, pretrained, freeze_backbone)
    raise ValueError(f"Modèle inconnu : {name!r}. Choix : {list(MODEL_CHANNELS)}")


__all__ = ["build_model", "get_in_channels", "MODEL_CHANNELS"]