"""Entraînement multimodal image + texte sur OpenI (brique 3).

On entraîne un mode parmi : image (image seule), text (texte seul), early, late, intermediate.
En lançant les 5, on obtient la comparaison demandée par l'énoncé : les deux références
unimodales et les trois stratégies de fusion. Tâche : classification binaire normal/anormal.

Usage :
    python -m src.train_multimodal --mode early --epochs 20
    python -m src.train_multimodal --mode intermediate --epochs 20
    python -m src.train_multimodal --mode image --epochs 1 --smoke-test
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from tqdm import tqdm

from src.config import ensure_dirs, load_config, set_seed
from src.data_openi import get_openi_dataloaders
from src.mlflow_utils import log_metrics, start_run
from src.models.multimodal import build_multimodal


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    losses = []
    for image, ids, mask, label in tqdm(loader, desc="train", leave=False):
        image, ids, mask, label = image.to(device), ids.to(device), mask.to(device), label.to(device)
        optimizer.zero_grad()
        logit = model(image, ids, mask)
        loss = criterion(logit, label)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, all_true, all_prob = [], [], []
    for image, ids, mask, label in loader:
        image, ids, mask, label = image.to(device), ids.to(device), mask.to(device), label.to(device)
        logit = model(image, ids, mask)
        losses.append(criterion(logit, label).item())
        all_prob.append(torch.sigmoid(logit).cpu().numpy())
        all_true.append(label.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_prob)
    return float(np.mean(losses)), y_true, y_prob


def binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    m = {}
    if len(np.unique(y_true)) >= 2:
        m["auc"] = float(roc_auc_score(y_true, y_prob))
        m["ap"] = float(average_precision_score(y_true, y_prob))
    m["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    return m


def train(mode: str, config: dict, epochs: int, smoke_test: bool) -> None:
    set_seed(config["seed"])
    ensure_dirs(config)
    device = get_device()
    print(f"Device : {device} | mode : {mode}")

    train_loader, val_loader, test_loader, vocab = get_openi_dataloaders(config, smoke_test=smoke_test)
    print(f"Taille du vocabulaire : {len(vocab)}")

    mm = config["multimodal"]
    model = build_multimodal(len(vocab), mode=mode, dim=mm["dim"], emb_dim=mm["emb_dim"]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=mm["lr"], weight_decay=1e-4)

    best_auc, best_epoch = -np.inf, -1
    ckpt_path = Path(config["paths"]["checkpoints"]) / f"multimodal_{mode}_best.pt"

    with start_run(f"multimodal_{mode}", config):
        mlflow.log_param("fusion_mode", mode)
        mlflow.log_param("vocab_size", len(vocab))
        mlflow.log_param("epochs_requested", epochs)
        mlflow.log_param("smoke_test", smoke_test)

        for epoch in range(1, epochs + 1):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, y_true, y_prob = evaluate(model, val_loader, criterion, device)
            val_m = binary_metrics(y_true, y_prob)
            log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)
            log_metrics({f"val_{k}": v for k, v in val_m.items()}, step=epoch)
            print(f"Époque {epoch:02d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_auc={val_m.get('auc', float('nan')):.4f}")
            score = val_m.get("auc", -np.inf)
            if score > best_auc:
                best_auc, best_epoch = score, epoch
                torch.save({"model_state": model.state_dict(), "config": config,
                            "mode": mode, "epoch": epoch}, ckpt_path)

        # Évaluation finale sur le test avec le meilleur modèle
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
        _, y_true, y_prob = evaluate(model, test_loader, criterion, device)
        test_m = binary_metrics(y_true, y_prob)
        log_metrics({f"test_{k}": v for k, v in test_m.items()})
        mlflow.log_param("best_epoch", best_epoch)
        mlflow.log_artifact(str(ckpt_path))

        print(f"\n=== TEST (mode {mode}) ===")
        print(f"AUC={test_m.get('auc', float('nan')):.4f} | "
              f"AP={test_m.get('ap', float('nan')):.4f} | F1={test_m.get('f1', float('nan')):.4f}")


def parse_args():
    p = argparse.ArgumentParser(description="Entraînement multimodal OpenI (image + texte)")
    p.add_argument("--mode", choices=["image", "text", "early", "late", "intermediate"],
                   default="early")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--config", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config) if args.config else load_config()
    epochs = args.epochs if args.epochs is not None else config["multimodal"]["epochs"]
    train(args.mode, config, epochs=epochs, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()