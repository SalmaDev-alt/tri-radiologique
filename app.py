"""Démonstrateur du système d'aide au tri radiologique (brique 4).

Application Streamlit : on charge une radiographie thoracique, le système affiche
(1) les pathologies prédites par le modèle de classification (probabilité par pathologie),
(2) un score d'anomalie issu de l'autoencodeur (erreur de reconstruction vs seuil).

Lancement :
    streamlit run app.py

Pré-requis : avoir au moins un modèle entraîné dans ./checkpoints (entraînement fait sur Colab,
puis checkpoints récupérés). L'application détecte automatiquement les modèles disponibles.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.data import IMAGENET_MEAN, IMAGENET_STD, PATHOLOGIES
from src.models import build_model, get_in_channels
from src.models.autoencoder import build_autoencoder

CHECKPOINTS = Path("checkpoints")

st.set_page_config(page_title="Aide au tri radiologique", layout="wide")


# --------------------------------------------------------------------------------------
# Découverte des modèles disponibles
# --------------------------------------------------------------------------------------
def list_classification_models() -> dict[str, Path]:
    """Modèles de classification présents : {nom_modele: chemin}."""
    found = {}
    if CHECKPOINTS.exists():
        for f in CHECKPOINTS.glob("*_best.pt"):
            stem = f.stem.replace("_best", "")
            if not stem.startswith(("anomaly_", "multimodal_")):
                found[stem] = f
    return found


def list_anomaly_models() -> dict[str, Path]:
    """Modèles d'anomalie présents : {ae|vae: chemin}."""
    found = {}
    if CHECKPOINTS.exists():
        for f in CHECKPOINTS.glob("anomaly_*_best.pt"):
            name = f.stem.replace("anomaly_", "").replace("_best", "")
            found[name] = f
    return found


# --------------------------------------------------------------------------------------
# Chargement des modèles (mis en cache pour ne pas recharger à chaque interaction)
# --------------------------------------------------------------------------------------
@st.cache_resource
def load_classifier(path_str: str, model_name: str):
    ckpt = torch.load(path_str, map_location="cpu")
    size = ckpt.get("config", {}).get("data", {}).get("size", 64)
    in_channels = get_in_channels(model_name)
    model = build_model(model_name, num_classes=len(PATHOLOGIES),
                        in_channels=in_channels, pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, size, in_channels


@st.cache_resource
def load_anomaly(path_str: str, model_name: str):
    ckpt = torch.load(path_str, map_location="cpu")
    size = ckpt.get("config", {}).get("data", {}).get("size", 64)
    latent = ckpt.get("config", {}).get("anomaly", {}).get("latent_dim", 128)
    threshold = ckpt.get("threshold", None)
    model = build_autoencoder(model_name, latent_dim=latent)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, size, threshold


# --------------------------------------------------------------------------------------
# Pré-traitement de l'image chargée
# --------------------------------------------------------------------------------------
def preprocess_classification(img: Image.Image, size: int, in_channels: int) -> torch.Tensor:
    ops = [transforms.Grayscale(num_output_channels=1), transforms.Resize((size, size))]
    if in_channels == 3:
        ops.append(transforms.Grayscale(num_output_channels=3))
        ops += [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    else:
        ops += [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    return transforms.Compose(ops)(img).unsqueeze(0)


def preprocess_anomaly(img: Image.Image, size: int) -> torch.Tensor:
    ops = [transforms.Grayscale(1), transforms.Resize((size, size)), transforms.ToTensor()]
    return transforms.Compose(ops)(img).unsqueeze(0)


# --------------------------------------------------------------------------------------
# Interface
# --------------------------------------------------------------------------------------
st.title("Système d'aide au tri radiologique")
st.caption("Outil d'aide à la décision — ne remplace pas l'avis d'un radiologue.")

clf_models = list_classification_models()
ano_models = list_anomaly_models()

if not clf_models and not ano_models:
    st.warning(
        "Aucun modèle entraîné trouvé dans le dossier checkpoints/. "
        "Entraînez d'abord un modèle (sur Colab), puis placez les fichiers .pt dans checkpoints/."
    )
    st.stop()

with st.sidebar:
    st.header("Configuration")
    clf_choice = st.selectbox("Modèle de classification",
                              options=list(clf_models) or ["(aucun)"],
                              disabled=not clf_models)
    ano_choice = st.selectbox("Modèle d'anomalie",
                              options=list(ano_models) or ["(aucun)"],
                              disabled=not ano_models)
    threshold = st.slider("Seuil de détection (probabilité)", 0.0, 1.0, 0.5, 0.05,
                          help="Une pathologie est signalée si sa probabilité dépasse ce seuil.")

uploaded = st.file_uploader("Charger une radiographie thoracique", type=["png", "jpg", "jpeg"])

if uploaded is None:
    st.info("Chargez une image pour lancer l'analyse.")
    st.stop()

image = Image.open(uploaded).convert("L")
col_img, col_res = st.columns([1, 1.4])

with col_img:
    st.subheader("Image analysée")
    st.image(image, use_container_width=True)

with col_res:
    # --- Classification ---
    if clf_models:
        st.subheader("Pathologies détectées")
        model, size, in_ch = load_classifier(str(clf_models[clf_choice]), clf_choice)
        x = preprocess_classification(image, size, in_ch)
        with torch.no_grad():
            probs = torch.sigmoid(model(x)).squeeze(0).numpy()
        order = np.argsort(probs)[::-1]
        detected = [(PATHOLOGIES[i], float(probs[i])) for i in order if probs[i] >= threshold]
        if detected:
            for name, p in detected:
                st.write(f"**{name}** — {p:.0%}")
                st.progress(min(p, 1.0))
        else:
            st.write("Aucune pathologie au-dessus du seuil. Les trois probabilités les plus élevées :")
            for i in order[:3]:
                st.write(f"{PATHOLOGIES[i]} — {probs[i]:.0%}")
                st.progress(min(float(probs[i]), 1.0))

    # --- Détection d'anomalie ---
    if ano_models:
        st.subheader("Score d'anomalie")
        amodel, asize, athr = load_anomaly(str(ano_models[ano_choice]), ano_choice)
        ax = preprocess_anomaly(image, asize)
        with torch.no_grad():
            recon = amodel(ax)[0] if ano_choice == "vae" else amodel(ax)
            err = float(F.mse_loss(recon, ax).item())
        if athr is not None:
            verdict = "Image atypique (anomalie probable)" if err >= athr else "Image dans la norme"
            st.metric("Erreur de reconstruction", f"{err:.5f}",
                      delta=f"seuil = {athr:.5f}", delta_color="off")
            st.write(f"Verdict : **{verdict}**")
        else:
            st.metric("Erreur de reconstruction", f"{err:.5f}")
            st.caption("Seuil indisponible dans ce modèle ; réentraînez l'autoencodeur pour l'enregistrer.")
        with st.expander("Voir la reconstruction"):
            rec_img = recon.squeeze().numpy()
            st.image(rec_img, caption="Reconstruction par l'autoencodeur",
                     use_container_width=True, clamp=True)

st.divider()
st.caption(
    "Les prédictions sont indicatives et dépendent de la qualité de l'entraînement. "
    "Ce démonstrateur illustre la chaîne technique ; il n'a pas de valeur diagnostique."
)