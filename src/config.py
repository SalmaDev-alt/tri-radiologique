"""Chargement de la configuration et reproductibilité.

Centralise (1) la lecture du fichier YAML de configuration et (2) le fixage des graines
aléatoires. Le seed est appliqué à random, numpy et torch pour garantir des runs comparables
(exigence de reproductibilité de l'énoncé).
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Charge la configuration YAML et renvoie un dictionnaire."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int) -> None:
    """Fixe toutes les graines aléatoires pour la reproductibilité.

    Note : on active aussi le mode déterministe de cuDNN. Cela peut légèrement ralentir
    l'entraînement mais garantit que deux runs identiques donnent les mêmes résultats,
    ce qui est attendu pour comparer équitablement des modèles (cours 4.5).
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        # torch peut ne pas être installé lors de tâches purement EDA
        pass


def ensure_dirs(config: dict[str, Any]) -> None:
    """Crée les dossiers de sortie (checkpoints, figures, data) s'ils n'existent pas."""
    for key in ("checkpoints", "figures"):
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)
    Path(config["data"]["data_root"]).mkdir(parents=True, exist_ok=True)
