# Système d'aide au tri radiologique

Détection de pathologies thoraciques par apprentissage profond — Projet de Deep Learning, Master Data Engineering & Intelligence Artificielle (EFREI Paris), module Machine & Deep Learning pour l'ingénierie des données.

Binôme : Salma DISSI et Jason Mampouya.

Le projet construit une chaîne complète d'aide au tri de radiographies thoraciques : classification multi-label des pathologies, détection d'anomalies non supervisée, modélisation multimodale image + texte, et démonstrateur applicatif, le tout sous suivi expérimental MLflow. C'est un outil d'aide à la décision destiné à assister le radiologue, sans valeur diagnostique.

---

## 1. Aperçu des briques

Le système répond à quatre besoins, chacun implémenté comme une brique indépendante.

1. **Classification supervisée** (`ChestMNIST`, multi-label, 14 pathologies) : comparaison de trois architectures profondes.
2. **Détection d'anomalies** (autoencodeur AE / VAE) : score d'anomalie fondé sur l'erreur de reconstruction, entraînement sur images normales uniquement.
3. **Multimodalité image + texte** (`OpenI`) : classification binaire normal / anormal, comparaison de deux modalités seules et de trois stratégies de fusion.
4. **Démonstrateur** (Streamlit) : interface de test chargeant une radiographie et affichant pathologies et score d'anomalie.

---

## 2. Résultats

### Brique 1 — Classification (ensemble de test ChestMNIST)

| Modèle | Architecture | AUC macro | AP macro | Statut |
|---|---|---|---|---|
| CNN from scratch | CNN (résolution 64) | 0,781 | 0,173 | Entraîné |
| DenseNet121 | Transfer learning (128) | 0,812 | 0,214 | Entraîné |
| ViT-B/16 | Vision Transformer (224) | — | — | Implémenté, non entraîné (contrainte GPU) |

DenseNet121 est le meilleur modèle entraîné et celui recommandé pour le démonstrateur. Le ViT-B/16 est entièrement implémenté et intégré au pipeline, mais son entraînement complet (passe sur 78 468 images en 224 px par époque) dépassait les ressources GPU disponibles dans les délais ; il reste à entraîner sur un GPU dédié.

### Brique 2 — Détection d'anomalies (test, espace latent 64, seuil au 95e percentile)

| Modèle | AUC | AP | F1 au seuil | Err. moy. normal | Err. moy. anomalie |
|---|---|---|---|---|---|
| Autoencodeur (AE) | 0,5222 | 0,4877 | 0,089 | 0,00111 | 0,00114 |
| Autoencodeur variationnel (VAE) | 0,5375 | 0,4978 | 0,097 | 0,00576 | 0,00601 |

Le VAE est retenu. Le signal est faible mais réel (erreur de reconstruction plus élevée sur les anomalies) : à 64 px, une pathologie est un changement local et peu contrasté qui pèse peu dans une erreur moyennée sur toute l'image. La détection par reconstruction est donc structurellement limitée sur ce jeu de données.

### Brique 3 — Multimodal (test OpenI)

| Configuration | AUC | AP | F1 |
|---|---|---|---|
| Image seule | 0,7466 | 0,8425 | 0,7851 |
| Texte seul | 0,9792 | 0,9869 | 0,9595 |
| Fusion précoce (early) | 0,9772 | 0,9865 | 0,9354 |
| Fusion tardive (late) | 0,9769 | 0,9873 | 0,9509 |
| Fusion intermédiaire (intermediate) | 0,9735 | 0,9838 | 0,9469 |

Le texte domine très largement (le compte-rendu nomme explicitement les anomalies). Les trois fusions égalent le texte seul sans le dépasser : la fusion n'apporte un gain que lorsque les modalités sont complémentaires, ce qui n'est pas le cas ici où l'image est dominée.

---

## 3. Structure du dépôt

```
tri-radiologique/
├── app.py                       # démonstrateur Streamlit
├── configs/
│   └── default.yaml             # configuration centralisée (hyperparamètres, chemins)
├── requirements.txt
├── notebooks/
│   └── entrainement_colab.ipynb # notebook d'entraînement GPU (Colab)
├── src/
│   ├── __init__.py              # import torch en premier + KMP_DUPLICATE_LIB_OK (fix Windows)
│   ├── config.py                # chargement de la config YAML
│   ├── data.py                  # DataLoaders ChestMNIST (multi-label)
│   ├── data_openi.py            # chargement OpenI (image + texte), split anti-fuite par rapport
│   ├── eda.py                   # analyse exploratoire (distribution, déséquilibre)
│   ├── metrics.py               # AUC, AP, F1 (macro/micro)
│   ├── mlflow_utils.py          # utilitaires de tracking MLflow
│   ├── train.py                 # entraînement classification supervisée
│   ├── train_anomaly.py         # entraînement AE / VAE
│   ├── train_multimodal.py      # entraînement des 5 configurations multimodales
│   └── models/
│       ├── cnn_scratch.py       # CNN entraîné depuis zéro
│       ├── transfer.py          # DenseNet121, ResNet50, ViT-B/16 (transfer learning)
│       ├── autoencoder.py       # ConvAE et ConvVAE (avec goulot latent réel)
│       └── multimodal.py        # encodeurs image/texte et stratégies de fusion
├── checkpoints/                 # meilleurs modèles sauvegardés (généré)
├── figures/                     # sorties EDA et courbes (généré)
└── mlflow.db                    # base de tracking MLflow SQLite (généré)
```

---

## 4. Environnement et installation

Le code a été développé en local (Windows, Python 3.11, pas de GPU NVIDIA) pour l'écriture et les tests rapides, et les entraînements lourds exécutés sur Google Colab (GPU T4).

### Installation

```bash
git clone https://github.com/SalmaDev-alt/tri-radiologique.git
cd tri-radiologique
pip install -r requirements.txt
```

### Particularités Windows (déjà gérées dans le code, à connaître)

- `medmnist` s'importe via `from medmnist.dataset import ChestMNIST`.
- `torch` est importé en premier dans `src/__init__.py`, avec `KMP_DUPLICATE_LIB_OK=TRUE`, pour éviter un conflit de bibliothèque OpenMP sous Windows.
- Lancer les outils en module : `python -m streamlit ...` et `python -m mlflow ...` (et non `streamlit` / `mlflow` directement).

---

## 5. Données

### ChestMNIST (briques 1 et 2)

Téléchargé automatiquement par `medmnist` au premier lancement (`download: true` dans la config), mis en cache dans `./data`. Les splits officiels train / validation / test de MedMNIST sont utilisés tels quels (aucun mélange manuel) pour éviter toute fuite.

### OpenI (brique 3)

À récupérer manuellement. Le code attend l'arborescence suivante :

```
data/openi/
├── images/                     # fichiers .png
├── indiana_reports.csv         # uid, Problems, MeSH, findings, impression, ...
└── indiana_projections.csv     # uid, filename, projection
```

Récupération via l'API Kaggle (dataset `raddar/chest-xrays-indiana-university`) :

```bash
pip install -q kaggle
# authentification : variable KAGGLE_API_TOKEN ou fichier ~/.kaggle/kaggle.json
kaggle datasets download -d raddar/chest-xrays-indiana-university
unzip -q chest-xrays-indiana-university.zip -d openi_raw
mkdir -p data/openi/images
cp openi_raw/indiana_reports.csv openi_raw/indiana_projections.csv data/openi/
find openi_raw -name '*.png' -exec cp {} data/openi/images/ \;
```

OpenI compte environ 7 470 radiographies. Le découpage en train / val / test est effectué par identifiant de rapport (anti-fuite), et le vocabulaire textuel est construit sur le seul ensemble d'entraînement.

---

## 6. Utilisation

Toutes les commandes se lancent depuis la racine du dépôt. Sur Colab, préfixer par `!` dans une cellule. Chaque script accepte `--config` (fichier YAML alternatif) et `--smoke-test` (exécution minimale de vérification).

### Brique 1 — Classification

```bash
# CNN from scratch (résolution 64)
python -m src.train --model cnn_scratch --epochs 40 --size 64

# DenseNet121 (transfer learning, résolution 128, learning rate réduit)
python -m src.train --model densenet121 --epochs 20 --lr 0.0001 --size 128

# ViT-B/16 (résolution 224) — run allégé : geler le backbone et réduire le batch
sed -i 's/freeze_backbone: false/freeze_backbone: true/' configs/default.yaml
sed -i 's/batch_size: 128/batch_size: 32/' configs/default.yaml
python -m src.train --model vit_b16 --epochs 12 --lr 0.0003 --size 224
```

Options de `train.py` : `--model {cnn_scratch|densenet121|resnet50|vit_b16}`, `--epochs`, `--lr`, `--size {64|128|224}`.

### Brique 2 — Détection d'anomalies

Les deux modèles s'entraînent sur les images normales uniquement. L'espace latent doit rester un goulot réel (dimension nettement inférieure à l'image) ; la dimension 64 a été retenue pour les runs finaux.

```bash
sed -i 's/latent_dim: 128/latent_dim: 64/' configs/default.yaml
python -m src.train_anomaly --model ae --epochs 20
python -m src.train_anomaly --model vae --epochs 20
```

Options : `--model {ae|vae}`, `--epochs`. Le seuil de décision est le 95e percentile des erreurs de reconstruction sur les images normales de validation (`threshold_percentile` dans la config).

### Brique 3 — Multimodal

Cinq configurations à entraîner. Le texte converge très vite, quelques époques suffisent pour les fusions ; l'image seule, plus difficile, mérite davantage d'époques.

```bash
python -m src.train_multimodal --mode text --epochs 8
python -m src.train_multimodal --mode image --epochs 8
python -m src.train_multimodal --mode early --epochs 4
python -m src.train_multimodal --mode late --epochs 4
python -m src.train_multimodal --mode intermediate --epochs 4
```

Options : `--mode {image|text|early|late|intermediate}`, `--epochs`.

### Démonstrateur

```bash
python -m streamlit run app.py
```

L'interface charge automatiquement les modèles présents dans `checkpoints/`. Sélectionner `densenet121` (classification) et `vae` (anomalie) pour le meilleur rendu. Charger une radiographie affiche les pathologies détectées avec leur probabilité et un score d'anomalie ajustable par seuil.

### Suivi MLflow

```bash
python -m mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Chaque run enregistre hyperparamètres, métriques par époque (pertes, AUC, AP, F1 macro/micro), meilleur modèle et artefacts (figures, reconstructions). Interface accessible sur `http://localhost:5000`.

---

## 7. Reproductibilité

- Graine aléatoire fixe (`seed: 42`) pour toutes les expériences.
- Configuration centralisée dans `configs/default.yaml` : tout hyperparamètre y est tracé.
- Splits officiels MedMNIST ; découpage OpenI par identifiant de rapport (anti-fuite) ; vocabulaire texte construit sur le seul train.
- Sauvegarde systématique du meilleur modèle (`monitor_metric: auc_macro`).
- Suivi complet via MLflow (backend SQLite, portable).

### Conseil d'exécution sur Colab

Le disque de la VM Colab est limité et effacé entre sessions. Conserver une copie de référence des données et des modèles sur Google Drive, et nettoyer les fichiers temporaires (archive et dossier décompressé OpenI) après mise en place. Sauvegarder `checkpoints/`, `figures/` et `mlflow.db` vers Drive après chaque vague d'entraînement.

---

## 8. Limites et perspectives

- **Déséquilibre des classes** : average precision faible sur les pathologies rares ; pistes : rééchantillonnage, focal loss, augmentation ciblée.
- **Résolution** : ChestMNIST est volontairement réduit ; des images plus grandes amélioreraient la classification fine et la détection d'anomalies.
- **Détection d'anomalies** : remplacer l'erreur de reconstruction globale par un score localisé, mieux adapté aux anomalies locales.
- **Multimodalité** : sur OpenI le texte domine ; la fusion ne devient bénéfique qu'avec des modalités complémentaires.
- **ViT-B/16** : implémenté mais non entraîné faute de GPU dans les délais ; perspective directe : entraînement par linear probing (backbone gelé) sur GPU dédié.
- **Portée clinique** : outil d'aide au tri, non un dispositif de diagnostic ; toute décision médicale demeure de la responsabilité du praticien.

---

## 9. Dépôt et rapport

- Code : https://github.com/SalmaDev-alt/tri-radiologique
- Rapport détaillé : `rapport_tri_radiologique.pdf`