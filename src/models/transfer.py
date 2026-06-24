"""Transfer learning — modèles 2 et 3 de la brique de classification.

on réutilise un réseau déjà entraîné sur ImageNet (millions d'images naturelles). Ses premières
couches ont appris des détecteurs génériques (contours, textures) réutilisables. On remplace
seulement la tête de classification par une couche adaptée à nos 14 pathologies.

DenseNet121 est le choix principal : c'est l'architecture du modèle CheXNet, référence sur les
radiographies thoraciques, et elle figure au cours (slide 55, connexions denses / concaténation).
ResNet50 est proposé en option . Le ViT-B/16 est la 3e architecture
exigée par l'énoncé (cours 4.7, slides 131-133, traitement de l'image en séquence de patchs).

Adaptation des canaux : ces réseaux attendent 3 canaux (RGB). Les radios étant en niveaux de gris
(1 canal), on duplique le canal en 3 côté données (cf. get_transforms(rgb=True)). On garde donc
le premier étage du réseau intact, ce qui préserve les poids pré-entraînés.

Sortie : num_classes logits (pas de sigmoïde ici ; appliquée dans la perte BCE, multi-label).
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models


def _freeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def build_densenet121(num_classes: int = 14, pretrained: bool = True,
                      freeze_backbone: bool = False) -> nn.Module:
    """DenseNet121 pré-entraîné ImageNet, tête remplacée pour le multi-label."""
    weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)
    if freeze_backbone:
        _freeze(model.features)  # on ne réentraîne que la tête
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)  # 14 logits
    return model


def build_resnet50(num_classes: int = 14, pretrained: bool = True,
                  freeze_backbone: bool = False) -> nn.Module:
    """ResNet50 pré-entraîné ImageNet, tête remplacée pour le multi-label (option de comparaison)."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("fc."):
                param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_vit_b16(num_classes: int = 14, pretrained: bool = True,
                 freeze_backbone: bool = False) -> nn.Module:
    """Vision Transformer (ViT-B/16) pré-entraîné ImageNet, tête remplacée pour le multi-label.

    Le ViT découpe l'image en patchs 16x16, les traite comme une séquence de tokens et applique
    l'auto-attention pour faire interagir toutes les positions (cours 4.7, slides 131-133).
    On utilise une version PRÉ-ENTRAÎNÉE : un ViT from scratch serait mauvais sur un dataset de
    cette taille, car il manque les biais inductifs locaux d'un CNN et exige énormément de données.

    Important : torchvision charge un ViT attendant des images 224x224. Le modèle redimensionne
    automatiquement ses encodages de position si la résolution diffère, mais pour de bons
    résultats il est recommandé d'entraîner ce modèle en résolution 224 (size: 224 dans la config).
    """
    weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
    model = models.vit_b_16(weights=weights)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("heads."):
                param.requires_grad = False
    # La tête de classification du ViT torchvision est model.heads.head
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model