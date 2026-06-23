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


def get_transforms(train: bool) -> transforms.Compose:
    """Renvoie les transformations d'images.

    En entraînement, on ajoute une légère augmentation (flip horizontal, petite rotation)
    — utile sur un dataset déséquilibré (cours 4.5, data augmentation). On reste prudent :
    pas de flip vertical ni de transformation agressive qui dénaturerait une radiographie.
    Normalisation simple sur 1 canal (images en niveaux de gris).
    """
    base = [transforms.ToTensor()]  # -> tenseur float [0,1], shape (1, H, W)
    if train:
        aug = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
        ]
        base = aug + base
    base.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    return transforms.Compose(base)


def get_datasets(config: dict[str, Any]):
    """Charge les trois splits officiels de ChestMNIST."""
    d = config["data"]
    common = dict(root=d["data_root"], size=d["size"], download=d["download"])

    train_ds = ChestMNIST(split="train", transform=get_transforms(train=True), **common)
    val_ds = ChestMNIST(split="val", transform=get_transforms(train=False), **common)
    test_ds = ChestMNIST(split="test", transform=get_transforms(train=False), **common)
    return train_ds, val_ds, test_ds


def get_dataloaders(config: dict[str, Any], smoke_test: bool = False):
    """Construit les DataLoaders.

    smoke_test=True : ne garde qu'un petit sous-ensemble, pour vérifier rapidement que la
    chaîne tourne sur CPU avant de lancer les vrais runs sur Colab.
    """
    train_ds, val_ds, test_ds = get_datasets(config)

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