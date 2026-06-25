"""Modèle multimodal image + texte (brique 3).

On compare trois stratégies de fusion (cours 4.8, slides 147-151) :
- fusion PRÉCOCE (early)        : on concatène les représentations image et texte tôt, puis un MLP.
- fusion TARDIVE (late)         : chaque modalité produit son propre logit, combinés par une
                                  somme pondérée apprise.
- fusion INTERMÉDIAIRE (interm.): le texte "interroge" l'image via une attention croisée
                                  (cross-attention), avant la décision.
On fournit aussi les deux références unimodales (image seule, texte seul) pour mesurer l'apport
réel de la fusion. Tâche : classification binaire normal/anormal -> 1 logit (BCEWithLogitsLoss).
"""
from __future__ import annotations

import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# Encodeur image : petit CNN -> vecteur global + carte de régions (pour la cross-attention)
# --------------------------------------------------------------------------------------
class ImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(128, out_dim, 3, padding=1), nn.BatchNorm2d(out_dim), nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor):
        feat = self.features(x)                       # (B, out_dim, h, w)
        vec = self.gap(feat).flatten(1)               # (B, out_dim) : vecteur image global
        regions = feat.flatten(2).transpose(1, 2)     # (B, h*w, out_dim) : tokens de régions
        return vec, regions


# --------------------------------------------------------------------------------------
# Encodeur texte : embeddings + moyenne masquée -> vecteur ; renvoie aussi la séquence
# --------------------------------------------------------------------------------------
class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.proj = nn.Linear(emb_dim, out_dim)
        self.out_dim = out_dim

    def forward(self, ids: torch.Tensor, mask: torch.Tensor):
        emb = self.embedding(ids)                     # (B, L, emb_dim)
        emb = self.proj(emb)                          # (B, L, out_dim)
        # moyenne sur les vrais tokens uniquement (masque)
        m = mask.unsqueeze(-1).float()
        summed = (emb * m).sum(dim=1)
        counts = m.sum(dim=1).clamp(min=1)
        vec = summed / counts                         # (B, out_dim) : vecteur texte global
        return vec, emb


# --------------------------------------------------------------------------------------
# Réseau multimodal avec les différents modes de fusion
# --------------------------------------------------------------------------------------
class MultimodalNet(nn.Module):
    """mode ∈ {image, text, early, late, intermediate}."""

    def __init__(self, vocab_size: int, mode: str = "early", dim: int = 256,
                 emb_dim: int = 128, n_heads: int = 4):
        super().__init__()
        self.mode = mode.lower()
        self.image_encoder = ImageEncoder(out_dim=dim)
        self.text_encoder = TextEncoder(vocab_size, emb_dim=emb_dim, out_dim=dim)

        if self.mode == "image":
            self.head = nn.Linear(dim, 1)
        elif self.mode == "text":
            self.head = nn.Linear(dim, 1)
        elif self.mode == "early":
            # concaténation des deux vecteurs -> MLP
            self.head = nn.Sequential(
                nn.Linear(dim * 2, dim), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(dim, 1))
        elif self.mode == "late":
            # un logit par modalité + poids de fusion appris (alpha)
            self.head_img = nn.Linear(dim, 1)
            self.head_txt = nn.Linear(dim, 1)
            self.alpha = nn.Parameter(torch.tensor(0.5))
        elif self.mode == "intermediate":
            # le texte (query) interroge les régions de l'image (key/value) via cross-attention
            self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(dim * 2, dim), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(dim, 1))
        else:
            raise ValueError(f"mode inconnu : {mode!r}")

    def forward(self, image: torch.Tensor, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        img_vec, img_regions = self.image_encoder(image)
        txt_vec, txt_seq = self.text_encoder(ids, mask)

        if self.mode == "image":
            return self.head(img_vec).squeeze(1)
        if self.mode == "text":
            return self.head(txt_vec).squeeze(1)
        if self.mode == "early":
            fused = torch.cat([img_vec, txt_vec], dim=1)
            return self.head(fused).squeeze(1)
        if self.mode == "late":
            logit = self.alpha * self.head_img(img_vec) + (1 - self.alpha) * self.head_txt(img_vec * 0 + txt_vec)
            return logit.squeeze(1)
        if self.mode == "intermediate":
            # query = vecteur texte (1 token), key/value = régions image
            q = txt_vec.unsqueeze(1)                  # (B, 1, dim)
            attended, _ = self.cross_attn(q, img_regions, img_regions)
            attended = attended.squeeze(1)            # (B, dim) : info image pertinente pour le texte
            fused = torch.cat([attended, txt_vec], dim=1)
            return self.head(fused).squeeze(1)
        raise RuntimeError("mode non géré")


def build_multimodal(vocab_size: int, mode: str = "early", dim: int = 256,
                     emb_dim: int = 128) -> nn.Module:
    return MultimodalNet(vocab_size, mode=mode, dim=dim, emb_dim=emb_dim)