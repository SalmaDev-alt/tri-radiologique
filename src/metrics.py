"""Métriques d'évaluation multi-label.

Sur un problème multi-label fortement déséquilibré comme ChestMNIST, l'accuracy seule est
trompeuse (cours 4.4 : "ne jamais se fier à une seule métrique"). On privilégie donc :

- AUC (aire sous la courbe ROC) par classe, puis moyennes macro et micro ;
- AP (average precision = aire sous la courbe précision-rappel) par classe : plus informative
  que l'AUC quand les positifs sont rares ;
- F1 par classe et moyennes macro / micro / weighted (cf. tableau cours slide 74) à un seuil donné.

Les fonctions prennent en entrée :
- y_true : np.ndarray (N, C), binaire 0/1
- y_prob : np.ndarray (N, C), probabilités dans [0, 1] (sortie sigmoïde)
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _safe_auc(y_true_col: np.ndarray, y_prob_col: np.ndarray) -> float:
    """AUC d'une colonne ; renvoie NaN si la classe n'a qu'une seule valeur (AUC indéfinie)."""
    if len(np.unique(y_true_col)) < 2:
        return float("nan")
    return roc_auc_score(y_true_col, y_prob_col)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    pathologies: list[str],
    threshold: float = 0.5,
) -> dict[str, float]:
    """Calcule l'ensemble des métriques globales + par classe.

    Renvoie un dictionnaire plat (clé -> valeur) directement loggable dans MLflow.
    """
    y_pred = (y_prob >= threshold).astype(int)
    metrics: dict[str, float] = {}

    # --- AUC par classe ---
    aucs = np.array([_safe_auc(y_true[:, i], y_prob[:, i]) for i in range(y_true.shape[1])])
    metrics["auc_macro"] = float(np.nanmean(aucs))
    # AUC micro : on aplatit toutes les classes ensemble
    if len(np.unique(y_true)) >= 2:
        metrics["auc_micro"] = float(roc_auc_score(y_true.ravel(), y_prob.ravel()))

    # --- Average Precision (aire sous courbe PR) ---
    aps = np.array(
        [average_precision_score(y_true[:, i], y_prob[:, i]) if y_true[:, i].sum() > 0 else np.nan
         for i in range(y_true.shape[1])]
    )
    metrics["ap_macro"] = float(np.nanmean(aps))

    # --- F1, précision, rappel (moyennes) ---
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_micro"] = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # --- Détail par classe (AUC + AP) : utile pour l'analyse critique du rapport ---
    for i, name in enumerate(pathologies):
        metrics[f"auc/{name}"] = float(aucs[i])
        metrics[f"ap/{name}"] = float(aps[i])

    return metrics


def per_class_table(metrics: dict[str, float], pathologies: list[str]) -> str:
    """Construit un tableau texte AUC/AP par pathologie (pour les logs et le rapport)."""
    lines = [f"{'Pathologie':<20} {'AUC':>8} {'AP':>8}"]
    for name in pathologies:
        auc = metrics.get(f"auc/{name}", float("nan"))
        ap = metrics.get(f"ap/{name}", float("nan"))
        lines.append(f"{name:<20} {auc:>8.3f} {ap:>8.3f}")
    return "\n".join(lines)
