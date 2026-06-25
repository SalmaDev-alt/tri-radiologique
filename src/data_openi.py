"""Chargement du dataset OpenI (Indiana University Chest X-ray) pour la brique multimodale.

OpenI fournit des radiographies thoraciques associées à des comptes-rendus radiologiques.
On l'utilise au format CSV (le plus simple à consommer), attendu dans un dossier contenant :
    <data_dir>/
        images/                     (fichiers .png)
        indiana_reports.csv         (uid, MeSH, Problems, findings, impression, ...)
        indiana_projections.csv     (uid, filename, projection)

Tâche (preuve de concept multimodale) : classification BINAIRE normal vs anormal.
Le label est dérivé du compte-rendu : "normal" si la colonne Problems/MeSH indique normal.
Ce choix simple permet de comparer proprement image seule / texte seul / fusion.

Anti-fuite : on découpe les splits PAR rapport (uid), pour qu'aucune image d'un même
compte-rendu ne se retrouve à la fois en train et en test. Le vocabulaire texte est construit
UNIQUEMENT sur le train.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torchvision import transforms

PAD, UNK = 0, 1  # indices réservés du vocabulaire


# --------------------------------------------------------------------------------------
# Texte : tokenisation simple et vocabulaire
# --------------------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Tokenisation simple : minuscules, mots alphanumériques."""
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-z0-9]+", text.lower())


def build_vocab(texts: list[str], min_freq: int = 2, max_size: int = 5000) -> dict[str, int]:
    """Construit le vocabulaire à partir des textes d'ENTRAÎNEMENT uniquement (anti-fuite)."""
    from collections import Counter

    counter = Counter()
    for t in texts:
        counter.update(tokenize(t))
    vocab = {"<pad>": PAD, "<unk>": UNK}
    for word, freq in counter.most_common(max_size):
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int = 200) -> list[int]:
    ids = [vocab.get(tok, UNK) for tok in tokenize(text)][:max_len]
    return ids if ids else [UNK]  # éviter une séquence vide


# --------------------------------------------------------------------------------------
# Construction du tableau (image, texte, label) à partir des CSV
# --------------------------------------------------------------------------------------
def _derive_label(row: pd.Series) -> int:
    """0 = normal, 1 = anormal, à partir des colonnes Problems / MeSH du compte-rendu."""
    for col in ("Problems", "MeSH"):
        if col in row and isinstance(row[col], str):
            val = row[col].strip().lower()
            if val == "normal" or val.startswith("normal"):
                return 0
    return 1


def _combine_text(row: pd.Series) -> str:
    parts = []
    for col in ("findings", "impression", "indication"):
        if col in row and isinstance(row[col], str):
            parts.append(row[col])
    return " ".join(parts).strip()


def load_openi_table(data_dir: str | Path) -> pd.DataFrame:
    """Assemble un DataFrame (filename, text, label) en joignant rapports et projections."""
    data_dir = Path(data_dir)
    reports = pd.read_csv(data_dir / "indiana_reports.csv")
    projections = pd.read_csv(data_dir / "indiana_projections.csv")

    reports["text"] = reports.apply(_combine_text, axis=1)
    reports["label"] = reports.apply(_derive_label, axis=1)

    df = projections.merge(reports[["uid", "text", "label"]], on="uid", how="inner")
    # privilégier les vues de face si la colonne projection existe
    if "projection" in df.columns:
        frontal = df[df["projection"].str.lower().str.contains("frontal", na=False)]
        if len(frontal) > 0:
            df = frontal
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    return df[["uid", "filename", "text", "label"]]


def split_by_uid(df: pd.DataFrame, seed: int = 42):
    """Split train/val/test PAR uid (anti-fuite), stratifié sur le label."""
    from sklearn.model_selection import train_test_split

    uids = df.drop_duplicates("uid")[["uid", "label"]]
    train_uid, temp_uid = train_test_split(
        uids, test_size=0.30, random_state=seed, stratify=uids["label"])
    val_uid, test_uid = train_test_split(
        temp_uid, test_size=0.50, random_state=seed, stratify=temp_uid["label"])

    def subset(u):
        return df[df["uid"].isin(set(u["uid"]))].reset_index(drop=True)

    return subset(train_uid), subset(val_uid), subset(test_uid)


# --------------------------------------------------------------------------------------
# Dataset PyTorch
# --------------------------------------------------------------------------------------
def get_image_transform(size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


class OpenIDataset(Dataset):
    """Renvoie (image_tensor, token_ids, label) pour chaque exemple."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, vocab: dict[str, int],
                 size: int = 128, max_len: int = 200):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.vocab = vocab
        self.max_len = max_len
        self.transform = get_image_transform(size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = Image.open(self.images_dir / row["filename"]).convert("L")
        image = self.transform(img)
        ids = torch.tensor(encode_text(row["text"], self.vocab, self.max_len), dtype=torch.long)
        label = torch.tensor(row["label"], dtype=torch.float32)
        return image, ids, label


def collate_multimodal(batch):
    """Regroupe un batch : empile les images, padde les séquences de tokens, crée le masque."""
    images, ids, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.stack(labels)
    ids_padded = pad_sequence(ids, batch_first=True, padding_value=PAD)
    mask = (ids_padded != PAD)  # True où il y a un vrai token
    return images, ids_padded, mask, labels


def get_openi_dataloaders(config: dict[str, Any], smoke_test: bool = False):
    """Construit les DataLoaders OpenI + renvoie le vocabulaire (nécessaire au modèle texte)."""
    from torch.utils.data import DataLoader

    m = config["multimodal"]
    df = load_openi_table(m["data_dir"])
    train_df, val_df, test_df = split_by_uid(df, seed=config["seed"])

    if smoke_test:
        train_df, val_df, test_df = train_df.head(64), val_df.head(32), test_df.head(32)

    vocab = build_vocab(train_df["text"].tolist(), min_freq=m["min_freq"])

    images_dir = Path(m["data_dir"]) / "images"
    common = dict(images_dir=images_dir, vocab=vocab, size=m["size"], max_len=m["max_len"])
    train_ds = OpenIDataset(train_df, **common)
    val_ds = OpenIDataset(val_df, **common)
    test_ds = OpenIDataset(test_df, **common)

    loader_args = dict(batch_size=m["batch_size"], collate_fn=collate_multimodal,
                       num_workers=config["data"]["num_workers"])
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader, vocab