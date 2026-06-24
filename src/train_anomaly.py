"""Entraînement et évaluation de la détection d'anomalies (brique 2).

Pipeline complet :
1. Entraîner un AE ou un VAE UNIQUEMENT sur des radios normales.
2. Calculer l'erreur de reconstruction sur les images normales de validation, et fixer un SEUIL
   = un percentile de ces erreurs (ex. 95e). Au-delà du seuil -> anomalie.
3. Évaluer sur le test (toutes images) : l'erreur de reconstruction sert de score d'anomalie.
   On mesure l'AUC et l'AP (anomalie = au moins une pathologie), et les scores au seuil choisi.
4. Tracer le tout dans MLflow + sauvegarder une figure de reconstructions (normales vs anormales).

Usage :
    python -m src.train_anomaly --model ae --epochs 30
    python -m src.train_anomaly --model vae --epochs 30
    python -m src.train_anomaly --model ae --epochs 1 --smoke-test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from src.config import ensure_dirs, load_config, set_seed
from src.data import get_anomaly_dataloaders
from src.mlflow_utils import log_metrics, start_run
from src.models.autoencoder import build_autoencoder, vae_loss


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, device, is_vae: bool, beta: float) -> float:
    model.train()
    losses = []
    for images, _ in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        optimizer.zero_grad()
        if is_vae:
            recon, mu, logvar = model(images)
            loss, _, _ = vae_loss(recon, images, mu, logvar, beta=beta)
        else:
            recon = model(images)
            loss = F.mse_loss(recon, images)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def reconstruction_errors(model, loader, device, is_vae: bool):
    """Erreur de reconstruction par image (MSE moyenne par pixel) + label binaire d'anomalie.

    anomalie = 1 si l'image porte au moins une pathologie, 0 sinon.
    Renvoie (errors, anomaly_labels) en numpy.
    """
    model.eval()
    errors, anomalies = [], []
    for images, labels in loader:
        images = images.to(device)
        recon = model(images)[0] if is_vae else model(images)
        # MSE moyenne par image (sur canaux + pixels)
        err = F.mse_loss(recon, images, reduction="none").mean(dim=[1, 2, 3])
        errors.append(err.cpu().numpy())
        labels = np.array(labels)
        anomalies.append((labels.sum(axis=1) > 0).astype(int))
    return np.concatenate(errors), np.concatenate(anomalies)


def save_reconstruction_figure(model, loader, device, is_vae: bool, out_path: Path, n: int = 6):
    """Sauvegarde une grille original vs reconstruction pour quelques images, avec leur erreur."""
    model.eval()
    images, labels = next(iter(loader))
    images = images[:n].to(device)
    with torch.no_grad():
        recon = model(images)[0] if is_vae else model(images)
    err = F.mse_loss(recon, images, reduction="none").mean(dim=[1, 2, 3]).cpu().numpy()
    imgs = images.cpu().numpy()
    rec = recon.cpu().numpy()
    anomaly = (np.array(labels[:n]).sum(axis=1) > 0).astype(int)

    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4.5))
    for i in range(n):
        axes[0, i].imshow(imgs[i].squeeze(), cmap="gray")
        axes[0, i].set_title(f"{'anomalie' if anomaly[i] else 'normal'}", fontsize=8)
        axes[0, i].axis("off")
        axes[1, i].imshow(rec[i].squeeze(), cmap="gray")
        axes[1, i].set_title(f"err={err[i]:.4f}", fontsize=8)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel("reconstruit")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def train(model_name: str, config: dict, epochs: int, smoke_test: bool) -> None:
    set_seed(config["seed"])
    ensure_dirs(config)
    device = get_device()
    print(f"Device : {device}")

    a = config["anomaly"]
    is_vae = model_name.lower() == "vae"
    beta = a["beta"]

    train_loader, val_loader, test_loader = get_anomaly_dataloaders(config, smoke_test=smoke_test)
    model = build_autoencoder(model_name, latent_dim=a["latent_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=a["lr"])

    best_loss = np.inf
    ckpt_path = Path(config["paths"]["checkpoints"]) / f"anomaly_{model_name}_best.pt"

    with start_run(f"anomaly_{model_name}", config):
        mlflow.log_param("anomaly_model", model_name)
        mlflow.log_param("epochs_requested", epochs)
        mlflow.log_param("smoke_test", smoke_test)

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, device, is_vae, beta)
            log_metrics({"train_loss": train_loss}, step=epoch)
            print(f"Époque {epoch:02d} | train_loss={train_loss:.4f}")
            if train_loss < best_loss:
                best_loss = train_loss
                torch.save({"model_state": model.state_dict(), "config": config,
                            "epoch": epoch, "model_name": model_name}, ckpt_path)

        # Recharger le meilleur modèle
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])

        # Seuil = percentile des erreurs sur images NORMALES de validation
        val_err, _ = reconstruction_errors(model, val_loader, device, is_vae)
        p = a["threshold_percentile"]
        threshold = float(np.percentile(val_err, p))
        mlflow.log_param("threshold_percentile", p)
        mlflow.log_metric("threshold", threshold)

        # Évaluation sur le test (toutes images)
        test_err, test_anom = reconstruction_errors(model, test_loader, device, is_vae)
        auc = float(roc_auc_score(test_anom, test_err)) if test_anom.sum() > 0 else float("nan")
        ap = float(average_precision_score(test_anom, test_err)) if test_anom.sum() > 0 else float("nan")
        preds = (test_err >= threshold).astype(int)
        metrics = {
            "test_auc": auc,
            "test_ap": ap,
            "test_precision": float(precision_score(test_anom, preds, zero_division=0)),
            "test_recall": float(recall_score(test_anom, preds, zero_division=0)),
            "test_f1": float(f1_score(test_anom, preds, zero_division=0)),
            "mean_err_normal": float(test_err[test_anom == 0].mean()) if (test_anom == 0).any() else float("nan"),
            "mean_err_anomaly": float(test_err[test_anom == 1].mean()) if (test_anom == 1).any() else float("nan"),
        }
        log_metrics(metrics)

        # Figure de reconstructions (artefact)
        fig_path = Path(config["paths"]["figures"]) / f"reconstructions_{model_name}.png"
        save_reconstruction_figure(model, test_loader, device, is_vae, fig_path)
        mlflow.log_artifact(str(fig_path))
        mlflow.log_artifact(str(ckpt_path))

        print("\n=== TEST (détection d'anomalies) ===")
        print(f"AUC={metrics['test_auc']:.4f} | AP={metrics['test_ap']:.4f} | "
              f"F1@seuil={metrics['test_f1']:.4f}")
        print(f"Erreur moyenne — normal: {metrics['mean_err_normal']:.5f} | "
              f"anomalie: {metrics['mean_err_anomaly']:.5f} "
              f"(l'anomalie doit avoir une erreur PLUS élevée)")


def parse_args():
    p = argparse.ArgumentParser(description="Détection d'anomalies AE/VAE sur ChestMNIST")
    p.add_argument("--model", choices=["ae", "vae"], default="ae")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--config", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config) if args.config else load_config()
    epochs = args.epochs if args.epochs is not None else config["anomaly"]["epochs"]
    train(args.model, config, epochs=epochs, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()