"""Détection d'anomalies : autoencodeur (AE) et autoencodeur variationnel (VAE).

Brique 2 de l'énoncé. Principe (cours slides 33, 106-108) :
- L'AE apprend à compresser une image dans un espace latent réduit puis à la reconstruire.
  Entraîné uniquement sur des images NORMALES, il reconstruit mal les images atypiques
  -> l'erreur de reconstruction sert de score d'anomalie.
- Le VAE est une version probabiliste : l'encodeur produit une moyenne (mu) et un écart-type
  (via logvar) définissant une distribution latente, d'où l'on échantillonne (reparamétrage).
  Sa perte ajoute un terme KL qui régularise l'espace latent (cours slide 107-108).

Les deux modèles travaillent sur des images 1 canal en 64x64, valeurs dans [0, 1]
(sortie sigmoïde du décodeur), ce qui rend l'erreur de reconstruction directement interprétable.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvAE(nn.Module):
    """Autoencodeur convolutif pour images 1x64x64.

    Encodeur : 4 convolutions stride 2 (64 -> 32 -> 16 -> 8 -> 4), puis projection
    dense vers un goulot latent de dimension latent_dim (compression réelle).
    Décodeur : projection dense inverse puis 4 convolutions transposées (4 -> 64).
    """

    def __init__(self, in_channels: int = 1, base: int = 32, latent_dim: int = 64):
        super().__init__()
        self._feat = base * 8   # canaux à la couche la plus profonde
        self._spatial = 4       # carte 4x4 pour une entrée 64x64
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, stride=2, padding=1), nn.ReLU(inplace=True),       # 32x32
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True),  # 16x16
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1), nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True),  # 8x8
            nn.Conv2d(base * 4, base * 8, 4, stride=2, padding=1), nn.BatchNorm2d(base * 8), nn.ReLU(inplace=True),  # 4x4
        )
        flat = self._feat * self._spatial * self._spatial  # 4096
        # Goulot d'étranglement réel : 4096 -> latent_dim -> 4096 (sinon pas de compression)
        self.fc_enc = nn.Linear(flat, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, flat)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, stride=2, padding=1), nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True),  # 8x8
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1), nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True),  # 16x16
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),  # 32x32
            nn.ConvTranspose2d(base, in_channels, 4, stride=2, padding=1), nn.Sigmoid(),  # 64x64, sortie [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).flatten(1)
        z = self.fc_enc(h)
        h = self.fc_dec(z).view(-1, self._feat, self._spatial, self._spatial)
        return self.decoder(h)


class ConvVAE(nn.Module):
    """Autoencodeur variationnel convolutif pour images 1x64x64.

    L'encodeur produit mu et logvar (dimension latente = latent_dim). On échantillonne par
    reparamétrage z = mu + sigma * epsilon, puis on décode. forward renvoie (reconstruction, mu, logvar).
    """

    def __init__(self, in_channels: int = 1, base: int = 32, latent_dim: int = 128):
        super().__init__()
        self.latent_dim = latent_dim
        self._feat = base * 8  # canaux à la couche la plus profonde
        self._spatial = 4      # carte 4x4 pour une entrée 64x64

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 4, stride=2, padding=1), nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, stride=2, padding=1), nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, stride=2, padding=1), nn.BatchNorm2d(base * 8), nn.ReLU(inplace=True),
        )
        flat = self._feat * self._spatial * self._spatial
        self.fc_mu = nn.Linear(flat, latent_dim)
        self.fc_logvar = nn.Linear(flat, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, flat)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, stride=2, padding=1), nn.BatchNorm2d(base * 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1), nn.BatchNorm2d(base * 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1), nn.BatchNorm2d(base), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, in_channels, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z).view(-1, self._feat, self._spatial, self._spatial)
        return self.decoder(h)

    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon: torch.Tensor, x: torch.Tensor, mu: torch.Tensor,
             logvar: torch.Tensor, beta: float = 1.0):
    """Perte VAE = reconstruction (MSE) + beta * divergence KL (cours slide 107-108)."""
    recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon_loss + beta * kl, recon_loss, kl


def build_autoencoder(name: str, latent_dim: int = 128) -> nn.Module:
    """Fabrique : 'ae' -> ConvAE, 'vae' -> ConvVAE."""
    name = name.lower()
    if name == "ae":
        return ConvAE(latent_dim=latent_dim)
    if name == "vae":
        return ConvVAE(latent_dim=latent_dim)
    raise ValueError(f"Modèle d'anomalie inconnu : {name!r} (choix : 'ae', 'vae')")