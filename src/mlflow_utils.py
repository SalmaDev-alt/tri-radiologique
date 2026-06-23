"""Helpers MLflow.

MLflow est obligatoire dans l'énoncé : pour chaque run on doit tracer hyperparamètres,
métriques, artefacts (figures, courbes), meilleur modèle et configuration. Ce module
centralise la configuration pour éviter de répéter le code dans chaque script.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import mlflow


def setup_mlflow(config: dict[str, Any]) -> None:
    """Configure l'URI de tracking et l'expérience."""
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])


@contextmanager
def start_run(run_name: str, config: dict[str, Any]):
    """Ouvre un run MLflow et logge automatiquement la config et les hyperparamètres."""
    setup_mlflow(config)
    with mlflow.start_run(run_name=run_name) as run:
        # On aplatit la config pour la tracer comme paramètres
        _log_flat_params(config)
        yield run


def _log_flat_params(config: dict[str, Any], prefix: str = "") -> None:
    """Logge récursivement un dict de config en paramètres MLflow (clés aplaties)."""
    for key, value in config.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict):
            _log_flat_params(value, prefix=f"{full_key}.")
        else:
            mlflow.log_param(full_key, value)


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Logge un dictionnaire de métriques. Ignore les NaN (classes sans positifs)."""
    clean = {k: v for k, v in metrics.items() if v == v}  # v==v écarte les NaN
    mlflow.log_metrics(clean, step=step)
