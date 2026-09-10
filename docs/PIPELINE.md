# Pipeline FlexiMax

Enchaînement des étapes, du jeu brut ResStock jusqu'aux modèles et à l'analyse de
flexibilité. Chaque notebook lit ses entrées dans `data/raw/` ou `data/processed/`
et écrit ses sorties dans `data/processed/`.

Conventions de chemins utilisées dans les notebooks :

- `DATA_RAW` : `data/raw/`
- `DATA_PROCESSED` : `data/processed/`
- `FIGURES` : `reports/figures/`

## 1. Exploration (`notebooks/01_exploration/`)

Prise en main des métadonnées nationales, groupe d'attributs par groupe.

| Notebook | Rôle |
|---|---|
| `exploration_enveloppe_thermique.ipynb` | Isolation, fenêtres, murs, inertie |
| `exploration_hvac.ipynb` | Systèmes de chauffage et climatisation, consignes |
| `exploration_occupants.ipynb` | Occupants, revenus, usage |
| `exploration_localisation_climat.ipynb` | État, comté, zone climatique |
| `exploration_in_groups.ipynb` | Vue transversale par groupe d'attributs |

Entrée : `upgrade0.parquet`. Sortie : constats, aucune donnée dérivée.

## 2. Nettoyage (`notebooks/02_nettoyage/`)

`nettoyage_metadata.ipynb` : filtrage des colonnes inutiles ou redondantes,
typage, restriction au périmètre d'étude.

Entrée : `upgrade0.parquet`
Sortie : `data/processed/metadata_clean.parquet`

## 3. Visualisation et classification des variables (`notebooks/03_visualisation/`)

| Notebook | Rôle |
|---|---|
| `visualisation_metadata.ipynb` | Distributions et croisements des métadonnées |
| `visual.ipynb` | Visualisations exploratoires complémentaires |
| `classification_variables.ipynb` | Classement métier des variables, décomposition de la consommation électrique par usage |

Entrée : `metadata_clean.parquet`, `upgrade0.parquet` (colonnes `out.electricity.*`)
Sortie : figures dans `reports/figures/`, table de classification des variables

## 4. Feature engineering (`notebooks/04_features/`)

| Notebook | Rôle |
|---|---|
| `encodage_categoriel.ipynb` | Encodage des variables catégorielles |
| `transformations_numeriques.ipynb` | Transformations et cadrage des variables numériques |
| `physical_feature_engineering.ipynb` | Variables physiques dérivées (enveloppe, degrés jours) |
| `preparation_finale.ipynb` | Assemblage de la matrice finale et de la cible, split train/test |

Entrée : `metadata_clean.parquet`
Sorties :

- `data/processed/X.parquet` : matrice des variables explicatives
- `data/processed/Y.parquet` : cibles de consommation (total et par usage)
- `data/processed/X_physical_engineered.parquet` : variante avec features physiques
- `data/processed/idx_train.npy`, `idx_test.npy` : indices du split

## 5. Modèles de consommation annuelle (`notebooks/05_deeplearning/`)

| Notebook | Rôle |
|---|---|
| `baseline_lgbm.ipynb` | LightGBM de référence |
| `lgbm_consommation_annuelle.ipynb`, `lgbm_electricity*.ipynb` | LightGBM sur la consommation électrique, variantes de périmètre et de jeu de features |
| `lgbm_stratified.ipynb`, `lightgbm_stratified.ipynb` | Entraînement stratifié multi cibles et évaluation |
| `mlp_stratified.ipynb` | Perceptron multicouche, même protocole |
| `clustering_stratifie.ipynb` | Clustering du parc pour la stratification |
| `lgbm_shap.ipynb`, `analyse_exploratoires.ipynb` | Interprétation SHAP, analyses de résidus |

Entrées : `X.parquet`, `Y.parquet`, `idx_train.npy`, `idx_test.npy`
Sorties : métriques (`R2`, RMSE) par cible, figures, `cluster_labels.parquet`

## 6. Séries temporelles (`notebooks/06_timeseries/`)

| Notebook | Rôle |
|---|---|
| `extraction_timeseries_oedi.ipynb` | Téléchargement des profils 15 min depuis OEDI, météo par comté |
| `etude_parc_503.ipynb` | Constitution du panel de bâtiments pour la modélisation temporelle |
| `timeseries_clustering.ipynb` | Jours types d'un bâtiment de référence |
| `timeseries_clustering_multi.ipynb` | Clustering à l'échelle du panel (forme, amplitude, météo) |
| `timeseries_net.ipynb`, `timeseries_conv.ipynb` | Réseaux récurrents et convolutifs |
| `rnn_mlp_hybrid_electricite.ipynb`, `rnn_mlp_hybrid_electricite_full.ipynb` | Modèle hybride RNN + MLP, prédiction horaire et extrapolation annuelle |

Entrées : profils `{bldg_id}-0.parquet` dans `data/raw/`, `metadata_clean.parquet`
Sorties : `clustering_multivarie_jours_types.parquet`, `cluster_labels_sub.parquet`,
`static_preds_oos.parquet`, figures

## 7. Flexibilité (`notebooks/07_flexibilite/`)

`flexibilite.ipynb` : estimation du gisement de flexibilité (chauffage,
climatisation) à partir des profils et des sorties de modèles.

## Scripts de banc d'essai (`experiments/`)

Scripts autonomes pour les entraînements en lot et la génération des figures du
rapport et de la soutenance :

- `lgbm_exp.py`, `ts_data.py`, `ts_train.py` : entraînements paramétrés
- `vague*.sh` : campagnes d'expériences
- `figures*.py`, `flex_*.py`, `courbes.py` : production des figures
- `results/*.json` : métriques consolidées des campagnes
