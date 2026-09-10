# FlexiMax

Projet de stage Data Science, MINES Paris PSL / ARMINES (programme France 2030, ADEME).

## Objectif

Prédire la consommation énergétique des bâtiments résidentiels à partir des données
ResStock 2025 Release 1 (NREL), puis analyser la structure de cette consommation
(déterminants climatiques, enveloppe, équipements) et sa dynamique temporelle.

Le périmètre d'étude est restreint aux **bâtiments tout électriques** : l'électricité
est la cible principale, à l'échelle annuelle comme à l'échelle horaire.

## Données

Les données ResStock 2025 Release 1 (NREL) ne sont pas versionnées (volume trop
important). Elles se placent dans `data/raw/` et `data/processed/`.

| Fichier | Description |
|---|---|
| `upgrade0.parquet` | Métadonnées nationales, 549 971 bâtiments x 771 colonnes |
| `metadata_clean.parquet` | Métadonnées nettoyées, produites par `02_nettoyage/` |
| `{bldg_id}-{upgrade}.parquet` | Séries temporelles individuelles, pas 15 min, 35 040 lignes par an |

Références externes (dictionnaires d'attributs, table d'énumération, géométrie des
États) : `data/external/`.

## Structure du dépôt

```
FlexiMax/
├── data/
│   ├── raw/           # Données brutes ResStock (non versionnées)
│   ├── processed/     # Jeux nettoyés et features (non versionnés)
│   └── external/      # Dictionnaires et références ResStock
│
├── notebooks/
│   ├── 01_exploration/    # Exploration : enveloppe, HVAC, occupants, climat, groupes
│   ├── 02_nettoyage/      # Nettoyage et sélection des attributs de métadonnées
│   ├── 03_visualisation/  # Visualisations métadonnées, classification des variables
│   ├── 04_features/       # Feature engineering : encodage, transformations, features physiques
│   ├── 05_deeplearning/   # Modèles annuels : LightGBM, MLP, clustering stratifié, SHAP
│   ├── 06_timeseries/     # Séries temporelles : extraction, clustering, RNN/MLP hybride
│   └── 07_flexibilite/    # Gisement de flexibilité
│
├── experiments/      # Scripts de banc d'essai (entraînements en lot, génération de figures)
├── reports/
│   ├── figures/      # Figures exportées pour le rapport et la soutenance
│   ├── rapport_stage.tex
│   └── soutenance.tex
├── docs/
│   └── PIPELINE.md   # Enchaînement détaillé des étapes, entrées et sorties
└── README.md
```

## Pipeline

L'enchaînement complet des étapes, avec pour chacune ses entrées, ses sorties et les
notebooks concernés, est décrit dans [docs/PIPELINE.md](docs/PIPELINE.md).

Vue d'ensemble :

1. Exploration des métadonnées (`01_exploration/`).
2. Nettoyage et sélection des attributs (`02_nettoyage/`).
3. Visualisation et classification des variables (`03_visualisation/`).
4. Feature engineering (`04_features/`), produit `X.parquet` et `Y.parquet`.
5. Modèles de consommation annuelle et clustering (`05_deeplearning/`).
6. Extraction et modélisation des séries temporelles (`06_timeseries/`).
7. Analyse du gisement de flexibilité (`07_flexibilite/`).

## Environnement

Python 3.12. Principales dépendances : `pandas`, `numpy`, `pyarrow`, `scikit-learn`,
`lightgbm`, `optuna`, `tensorflow`, `torch`, `matplotlib`, `seaborn`, `shap`,
`geopandas`.

## Rapports

- `reports/rapport_stage.tex` : rapport de stage (source LaTeX) et `rapport_stage.pdf`.
- `reports/soutenance.tex` : support de soutenance et `soutenance.pdf`.
