# FlexiMax

Projet de stage Data Science — MINES Paris–PSL / ARMINES  
Analyse de la flexibilité de la demande résidentielle en électricité (France 2030, ADEME).

## Structure du projet

```
FlexiMax/
├── data/
│   ├── raw/            # Données brutes (parquet, csv) — ne pas versionner
│   ├── processed/      # Données nettoyées et prêtes à l'emploi
│   └── external/       # Références externes (data_dictionary, geographic_info)
│
├── notebooks/
│   ├── 01_exploration/         # Exploration initiale des données
│   ├── 02_nettoyage/           # Nettoyage et sélection des attributs
│   ├── 03_visualisation/       # Visualisations des metadata et séries temporelles
│   ├── 04_feature_engineering/ # Construction des variables pour les modèles
│   ├── 05_flexibilite/         # Analyse du potentiel de flexibilité HVAC
│   └── 06_modeles/             # Modèles ML/DL de prédiction
│
├── src/                # Fonctions Python réutilisables entre notebooks
├── reports/
│   └── figures/        # Graphiques exportés pour les rapports
└── README.md
```

## Données

Les données ResStock 2025 Release 1 (NREL) ne sont pas versionnées dans ce repo car trop volumineuses.

| Fichier | Description |
|---------|-------------|
| `upgrade0.parquet` | Metadata nationale — 549 971 bâtiments × 771 colonnes |
| `metadata_clean.parquet` | Metadata nettoyée — produite par `02_nettoyage` |
| `{bldg_id}-{upgrade}.parquet` | Séries temporelles individuelles (15 min, 35 040 lignes) |

## Missions

1. Exploration et nettoyage des données
2. Feature engineering
3. Détection des appareils
4. Analyse de la flexibilité HVAC (upgrades dr_001–dr_005)
5. Prédiction de la consommation
6. Modèles ML/DL
7. Évaluation des modèles
8. Interprétation (SHAP)
