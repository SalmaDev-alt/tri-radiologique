"""Boucle d'entraînement pour la classification multi-label.

Assemble toutes les briques de la fondation :
- perte BCEWithLogitsLoss (= sigmoïde par classe + binary cross-entropy, multi-label),
  avec pos_weight pour compenser le déséquilibre des classes ;
- optimiseur AdamW + scheduler cosine ;
- early stopping sur la métrique surveillée (AUC macro par défaut) ;
- sauvegarde du MEILLEUR modèle ;
- tracking MLflow complet (hyperparamètres, métriques par époque, meilleur modèle).

Usage :
    python -m src.train --model cnn_scratch --epochs 50
    python -m src.train --model cnn_scratch --epochs 1 --smoke-test   # test rapide CPU
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.config import ensure_dirs, load_config, set_seed
from src.data import PATHOLOGIES, compute_pos_weights, get_dataloaders
from src.metrics import compute_metrics, per_class_table
from src.mlflow_utils import log_metrics, start_run
from src.models import build_model, get_in_channels


def get_device() -> torch.device:
    """GPU si disponible (Colab), sinon CPU (dev local)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_optimizer(model: nn.Module, config: dict):
    t = config["train"]
    if t["optimizer"].lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    if t["optimizer"].lower() == "adam":
        return torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    return torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=0.9,
                           weight_decay=t["weight_decay"])


def build_scheduler(optimizer, config: dict):
    s = config["train"]["scheduler"].lower()
    if s == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["train"]["epochs"])
    if s == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    return None


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, np.ndarray, np.ndarray]:
    """Passe le modèle en mode eval, renvoie (perte moyenne, y_true, y_prob)."""
    model.eval()
    losses, all_true, all_prob = [], [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        all_prob.append(torch.sigmoid(logits).cpu().numpy())
        all_true.append(labels.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)
    return float(np.mean(losses)), y_true, y_prob


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    losses = []
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.float().to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def train(model_name: str, config: dict, epochs: int, smoke_test: bool) -> None:
    set_seed(config["seed"])
    ensure_dirs(config)
    device = get_device()
    print(f"Device : {device}")

    # Le nombre de canaux dépend du modèle : 1 (CNN from scratch) ou 3 (modèles pré-entraînés).
    in_channels = get_in_channels(model_name)
    rgb = in_channels == 3
    train_loader, val_loader, test_loader = get_dataloaders(config, smoke_test=smoke_test, rgb=rgb)

    # Perte multi-label avec compensation du déséquilibre
    pos_weight = compute_pos_weights(config).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    freeze = config.get("model", {}).get("freeze_backbone", False)
    model = build_model(model_name, num_classes=len(PATHOLOGIES),
                        in_channels=in_channels, freeze_backbone=freeze).to(device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    monitor = config["train"]["monitor_metric"]
    patience = config["train"]["early_stopping_patience"]
    best_score, best_epoch, epochs_no_improve = -np.inf, -1, 0
    ckpt_path = Path(config["paths"]["checkpoints"]) / f"{model_name}_best.pt"

    run_name = f"{model_name}_{config['data']['size']}px"
    with start_run(run_name, config):
        mlflow.log_param("epochs_requested", epochs)
        mlflow.log_param("smoke_test", smoke_test)

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, y_true, y_prob = evaluate(model, val_loader, criterion, device)
            val_metrics = compute_metrics(y_true, y_prob, PATHOLOGIES)

            if scheduler is not None:
                scheduler.step()

            # Logging MLflow par époque
            log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            log_metrics({f"val_{k}": v for k, v in val_metrics.items()
                         if "/" not in k}, step=epoch)

            score = val_metrics.get(monitor, float("nan"))
            print(f"Époque {epoch:02d} | train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_{monitor}={score:.4f}")

            # Sauvegarde du meilleur modèle + early stopping
            if score > best_score:
                best_score, best_epoch, epochs_no_improve = score, epoch, 0
                torch.save({"model_state": model.state_dict(),
                            "config": config,
                            "epoch": epoch,
                            "score": score}, ckpt_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping (pas d'amélioration depuis {patience} époques).")
                    break

        # Évaluation finale sur le TEST avec le meilleur modèle
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        _, y_true, y_prob = evaluate(model, test_loader, criterion, device)
        test_metrics = compute_metrics(y_true, y_prob, PATHOLOGIES)

        log_metrics({f"test_{k}": v for k, v in test_metrics.items() if "/" not in k})
        mlflow.log_param("best_epoch", best_epoch)
        mlflow.log_artifact(str(ckpt_path))

        table = per_class_table(test_metrics, PATHOLOGIES)
        print("\n=== TEST (meilleur modèle) ===")
        print(f"AUC macro : {test_metrics['auc_macro']:.4f} | "
              f"AP macro : {test_metrics['ap_macro']:.4f}")
        print(table)
        mlflow.log_text(table, "test_per_class.txt")


def parse_args():
    p = argparse.ArgumentParser(description="Entraînement classification multi-label ChestMNIST")
    p.add_argument("--model", default="cnn_scratch", help="nom du modèle (build_model)")
    p.add_argument("--epochs", type=int, default=None, help="surcharge le nombre d'époques")
    p.add_argument("--lr", type=float, default=None,
                   help="surcharge le learning rate (utile pour le fine-tuning : ex. 0.0001)")
    p.add_argument("--smoke-test", action="store_true",
                   help="petit sous-échantillon pour tester la chaîne sur CPU")
    p.add_argument("--config", default=None, help="chemin d'un YAML de config alternatif")
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config) if args.config else load_config()
    if args.lr is not None:
        config["train"]["lr"] = args.lr
    epochs = args.epochs if args.epochs is not None else config["train"]["epochs"]
    train(args.model, config, epochs=epochs, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()