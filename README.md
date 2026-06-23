# Système d'aide au tri radiologique

Projet Deep Learning — Master Data Engineering & IA (EFREI).
Binôme : Salma DISSI, Jason Mampouya.

Système d'aide au tri de radiographies thoraciques combinant quatre briques :

1. **Classification supervisée multi-label** (14 pathologies) — comparaison de trois architectures profondes.
2. **Détection d'anomalies** par autoencodeur (AE) puis autoencodeur variationnel (VAE).
3. **Composante multimodale** image + texte (fusion précoce / tardive / intermédiaire).
4. **Démonstrateur applicatif** (Streamlit), avec suivi expérimental **MLflow**.

Dataset principal : **ChestMNIST / ChestMNIST+** (sous-ensemble de MedMNIST), classification multi-label.

---

## Pré-requis et workflow matériel

L'entraînement de réseaux profonds nécessite un GPU. Le poste de développement ne dispose
que d'un GPU intégré (Intel Iris Xe), inexploitable pour l'entraînement. Le workflow retenu est donc :

- **Développement local (VS Code)** : écriture du code, exploration des données (EDA), tests
  rapides sur CPU (1-2 époques sur un sous-échantillon pour vérifier que la chaîne tourne).
- **Entraînement (Google Colab)** : les runs réels tournent sur GPU gratuit (T4, 16 Go).
  Colab clone ce dépôt, lance l'entraînement, et renvoie modèles + runs MLflow vers Drive/GitHub.

> Configuration matérielle à documenter dans le rapport (section 6 de l'énoncé) :
> GPU NVIDIA T4 16 Go (Colab), résolutions ChestMNIST 64×64 puis 128×128.

---

## Structure du dépôt

```
tri-radiologique/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml          # hyperparamètres et chemins centralisés
├── src/
│   ├── config.py             # chargement de la configuration + seed
│   ├── data.py               # ChestMNIST : datasets, dataloaders multi-label
│   ├── eda.py                # analyse exploratoire (distribution, co-occurrences)
│   ├── metrics.py            # métriques multi-label (AUC, AP, F1, par classe)
│   ├── mlflow_utils.py       # helpers de tracking MLflow
│   ├── train.py              # boucle d'entraînement classification (sigmoïde + BCE)
│   └── models/
│       └── cnn_scratch.py    # CNN entraîné from scratch (brique 1, modèle 1)
└── notebooks/
    └── (notebooks Colab à venir)
```

(Les briques suivantes — transfer learning, ViT, AE/VAE, multimodal, démonstrateur — seront
ajoutées dans `src/models/` et `src/` au fur et à mesure.)

---

## Installation locale (VS Code)

```bash
python -m venv .venv
# Windows PowerShell :
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ordre d'exécution

```bash
# 1. Analyse exploratoire (local, rapide)
python -m src.eda

# 2. Test rapide de la chaîne sur CPU (1 époque, petit échantillon)
python -m src.train --model cnn_scratch --epochs 1 --smoke-test

# 3. Entraînement réel (sur Colab GPU)
python -m src.train --model cnn_scratch --epochs 50
```

## Suivi MLflow

```bash
mlflow ui            # puis ouvrir http://127.0.0.1:5000
```

Toutes les expériences (hyperparamètres, métriques, courbes, meilleur modèle) sont tracées
dans `./mlruns`.

---

## Répartition indicative du travail (binôme)

| Brique | Pilote suggéré | Notes |
|--------|----------------|-------|
| Données + EDA + pipeline + MLflow | commun | fondation partagée |
| CNN from scratch + transfer learning | — | classification supervisée |
| ViT + comparaison architectures | — | section vision |
| AE / VAE détection d'anomalies | — | brique générative |
| Multimodal (fusion) | — | OpenI image+texte |
| Démonstrateur Streamlit + rapport | commun | livrable final |

> À répartir entre Salma et Jason selon vos préférences ; Jason présentant, il a intérêt à
> co-écrire les sections qu'il devra défendre.
