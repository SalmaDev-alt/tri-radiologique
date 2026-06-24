"""Chargement des données ChestMNIST.

ChestMNIST est un dataset MULTI-LABEL : chaque radiographie peut présenter 0, 1 ou plusieurs
des 14 pathologies. Les labels sont donc des vecteurs binaires de taille 14 (et non une classe
unique). C'est pourquoi on utilisera une activation sigmoïde par classe + une perte BCE
(et non softmax + cross-entropy), conformément à l'énoncé.

Les splits train / val / test sont fournis officiellement par MedMNIST : on ne mélange pas
soi-même pour éviter toute fuite de données (exigence "stratégie antifuite").
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from medmnist.dataset import ChestMNIST  # import via le submodule (contourne l'__init__ fragile)
from torch.utils.data import DataLoader
from torchvision import transforms

# Statistiques de normalisation ImageNet (pour les modèles pré-entraînés, mode RGB)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Les 14 pathologies de ChestMNIST, dans l'ordre des colonnes du vecteur de labels.
# (ordre standard NIH ChestX-ray14 dont ChestMNIST est dérivé)
PATHOLOGIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia",
]
NUM_CLASSES = len(PATHOLOGIES)


def get_transforms(train: bool, rgb: bool = False) -> transforms.Compose:
    """Renvoie les transformations d'images.

    En entraînement, on ajoute une légère augmentation (flip horizontal, petite rotation)
    — utile sur un dataset déséquilibré (cours 4.5, data augmentation). On reste prudent :
    pas de flip vertical ni de transformation agressive qui dénaturerait une radiographie.

    rgb=True : pour les modèles pré-entraînés ImageNet (DenseNet, ResNet). On convertit l'image
    en 3 canaux (le gris est dupliqué) et on applique la normalisation ImageNet, ce qui est
    indispensable pour réutiliser correctement les poids pré-entraînés.
    rgb=False : 1 canal (niveaux de gris) pour le CNN from scratch.
    """
    ops = []
    if rgb:
        ops.append(transforms.Grayscale(num_output_channels=3))  # 1 canal -> 3 canaux dupliqués
    if train:
        ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
        ]
    ops.append(transforms.ToTensor())  # -> tenseur float [0,1]
    if rgb:
        ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    else:
        ops.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    return transforms.Compose(ops)


def get_datasets(config: dict[str, Any], rgb: bool = False):
    """Charge les trois splits officiels de ChestMNIST.

    rgb : si True, images converties en 3 canaux + normalisation ImageNet (modèles pré-entraînés).
    """
    d = config["data"]
    common = dict(root=d["data_root"], size=d["size"], download=d["download"])

    train_ds = ChestMNIST(split="train", transform=get_transforms(train=True, rgb=rgb), **common)
    val_ds = ChestMNIST(split="val", transform=get_transforms(train=False, rgb=rgb), **common)
    test_ds = ChestMNIST(split="test", transform=get_transforms(train=False, rgb=rgb), **common)
    return train_ds, val_ds, test_ds


def get_dataloaders(config: dict[str, Any], smoke_test: bool = False, rgb: bool = False):
    """Construit les DataLoaders.

    smoke_test=True : ne garde qu'un petit sous-ensemble, pour vérifier rapidement que la
    chaîne tourne sur CPU avant de lancer les vrais runs sur Colab.
    rgb : transmis aux transforms (3 canaux + normalisation ImageNet pour les modèles pré-entraînés).
    """
    train_ds, val_ds, test_ds = get_datasets(config, rgb=rgb)

    if smoke_test:
        from torch.utils.data import Subset

        train_ds = Subset(train_ds, range(256))
        val_ds = Subset(val_ds, range(128))
        test_ds = Subset(test_ds, range(128))

    t = config["train"]
    loader_args = dict(num_workers=config["data"]["num_workers"], pin_memory=True)

    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, batch_size=t["batch_size"], shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader


def get_anomaly_dataloaders(config: dict[str, Any], smoke_test: bool = False):
    """DataLoaders pour la détection d'anomalies (brique 2).

    Protocole (cf. énoncé) :
    - ENTRAÎNEMENT et VALIDATION : uniquement des images NORMALES (aucune pathologie).
      L'autoencodeur apprend ainsi à reconstruire l'anatomie normale.
    - TEST : toutes les images. Une image est considérée "anomalie" si elle porte au moins
      une pathologie. L'erreur de reconstruction servira de score d'anomalie.

    Transform volontairement minimal (ToTensor seul, valeurs dans [0,1], 1 canal) : pas
    d'augmentation aléatoire, pour que l'erreur de reconstruction reste comparable d'une image
    à l'autre, et compatible avec la sortie sigmoïde du décodeur.
    """
    from torch.utils.data import Subset

    d = config["data"]
    common = dict(root=d["data_root"], size=d["size"], download=d["download"])
    tf = transforms.Compose([transforms.ToTensor()])  # [0,1], 1 canal, déterministe

    train = ChestMNIST(split="train", transform=tf, **common)
    val = ChestMNIST(split="val", transform=tf, **common)
    test = ChestMNIST(split="test", transform=tf, **common)

    # indices des images normales (somme des 14 labels == 0)
    train_normal = np.where(np.array(train.labels).sum(axis=1) == 0)[0]
    val_normal = np.where(np.array(val.labels).sum(axis=1) == 0)[0]

    train_ds = Subset(train, train_normal)
    val_ds = Subset(val, val_normal)
    test_ds = test  # toutes les images de test (normales + anormales)

    if smoke_test:
        train_ds = Subset(train, train_normal[:256])
        val_ds = Subset(val, val_normal[:128])
        test_ds = Subset(test, range(256))

    t = config["train"]
    loader_args = dict(num_workers=config["data"]["num_workers"], pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=t["batch_size"], shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, batch_size=t["batch_size"], shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, batch_size=t["batch_size"], shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader


def compute_pos_weights(config: dict[str, Any]) -> torch.Tensor:
    """Calcule un poids par classe pour compenser le déséquilibre (pos_weight de BCEWithLogitsLoss).

    Pour chaque pathologie : poids = (nb_négatifs / nb_positifs) sur le train.
    Les pathologies rares (peu de positifs) reçoivent un poids plus élevé, ce qui pénalise
    davantage le modèle quand il les rate (cours 4.4 : classes déséquilibrées).
    """
    train_ds = ChestMNIST(
        split="train",
        root=config["data"]["data_root"],
        size=config["data"]["size"],
        download=config["data"]["download"],
    )
    labels = np.array(train_ds.labels)  # shape (N, 14), valeurs 0/1
    n_pos = labels.sum(axis=0)
    n_neg = labels.shape[0] - n_pos
    # éviter la division par zéro si une classe n'a aucun positif
    pos_weight = np.where(n_pos > 0, n_neg / np.maximum(n_pos, 1), 1.0)
    return torch.tensor(pos_weight, dtype=torch.float32)