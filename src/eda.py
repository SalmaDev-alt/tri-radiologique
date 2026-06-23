"""Analyse exploratoire des données (EDA).

Couvre la section "Analyse exploratoire" attendue dans le rapport :
- distribution des 14 labels et mise en évidence du déséquilibre ;
- matrice de co-occurrence entre pathologies (multi-label : plusieurs peuvent coexister) ;
- quelques exemples visuels de radiographies.

Tourne en local (pas besoin de GPU). Sauvegarde les figures dans le dossier configuré.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from medmnist.dataset import ChestMNIST

from src.config import ensure_dirs, load_config
from src.data import PATHOLOGIES


def load_labels(config: dict, split: str = "train") -> np.ndarray:
    ds = ChestMNIST(
        split=split,
        root=config["data"]["data_root"],
        size=config["data"]["size"],
        download=config["data"]["download"],
    )
    return np.array(ds.labels)


def plot_label_distribution(labels: np.ndarray, out_dir: Path) -> pd.DataFrame:
    counts = labels.sum(axis=0).astype(int)
    df = pd.DataFrame({"pathologie": PATHOLOGIES, "n_positifs": counts})
    df["taux_positif_%"] = (counts / labels.shape[0] * 100).round(2)
    df = df.sort_values("n_positifs", ascending=False)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df, x="n_positifs", y="pathologie", color="#4C72B0")
    plt.title("Distribution des pathologies (train) — déséquilibre marqué")
    plt.xlabel("Nombre d'images positives")
    plt.tight_layout()
    plt.savefig(out_dir / "label_distribution.png", dpi=150)
    plt.close()
    return df


def plot_cooccurrence(labels: np.ndarray, out_dir: Path) -> None:
    # matrice de co-occurrence : combien de fois deux pathologies apparaissent ensemble
    cooc = labels.T @ labels  # (14, 14)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cooc, xticklabels=PATHOLOGIES, yticklabels=PATHOLOGIES,
                cmap="viridis", annot=False)
    plt.title("Co-occurrences entre pathologies")
    plt.tight_layout()
    plt.savefig(out_dir / "cooccurrence.png", dpi=150)
    plt.close()


def plot_examples(config: dict, out_dir: Path, n: int = 8) -> None:
    ds = ChestMNIST(split="train", root=config["data"]["data_root"],
                    size=config["data"]["size"], download=config["data"]["download"])
    fig, axes = plt.subplots(2, n // 2, figsize=(12, 5))
    for ax, idx in zip(axes.ravel(), range(n)):
        img, label = ds[idx]
        img = np.array(img).squeeze()
        present = [PATHOLOGIES[i] for i, v in enumerate(np.array(label)) if v == 1]
        ax.imshow(img, cmap="gray")
        ax.set_title(", ".join(present) if present else "Sain", fontsize=7)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "examples.png", dpi=150)
    plt.close()


def main():
    config = load_config()
    ensure_dirs(config)
    out_dir = Path(config["paths"]["figures"])

    labels = load_labels(config, split="train")
    print(f"Train : {labels.shape[0]} images, {labels.shape[1]} pathologies.")

    # part d'images sans aucune pathologie (cas "sain")
    n_sain = int((labels.sum(axis=1) == 0).sum())
    print(f"Images sans aucune pathologie : {n_sain} "
          f"({n_sain / labels.shape[0] * 100:.1f}%)")
    print(f"Nombre moyen de pathologies par image : {labels.sum(axis=1).mean():.2f}")

    df = plot_label_distribution(labels, out_dir)
    print("\nDistribution des pathologies :")
    print(df.to_string(index=False))

    plot_cooccurrence(labels, out_dir)
    plot_examples(config, out_dir)
    print(f"\nFigures enregistrées dans : {out_dir.resolve()}")


if __name__ == "__main__":
    main()