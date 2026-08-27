# FlexiMax — Contexte du projet (pour claude.ai Project)

## Objectif
Stage Data Science — MINES Paris–PSL / ARMINES (programme France 2030, ADEME).
Analyse de la flexibilité énergétique résidentielle, concrètement : **prédire la consommation
énergétique des bâtiments** à partir des données ResStock 2025 Release 1 (NREL) — metadata,
séries temporelles, fichiers météo, et le guide technique ResStock (`ResStockTechnicalReferenceGuide_2025_1.pdf`).
(La liste de missions ci-dessous vient du README du repo et mentionne aussi la flexibilité HVAC —
mais l'objectif confirmé par l'utilisatrice est la prédiction de consommation, pas spécifiquement
les upgrades HVAC dr_001–dr_005.)

## Données
- `upgrade0.parquet` : métadonnées nationales — 549 971 bâtiments × 771 colonnes.
- `metadata_clean.parquet` : métadonnées nettoyées.
- Séries temporelles individuelles par bâtiment (pas 15 min, 35 040 lignes/an).
- Données brutes non versionnées dans git (trop volumineuses) — voir `data/raw`, `data/processed`, `data/external`.

## Pipeline / structure des notebooks
1. `01_exploration/` — exploration initiale (enveloppe thermique, HVAC, occupants, climat/localisation).
2. `02_nettoyage/` — nettoyage des métadonnées.
3. `03_visualisation/` — visualisation metadata + séries temporelles.
4. `04_features/` — feature engineering (encodage catégoriel, transformations numériques, préparation finale).
5. `05_deeplearning/` — modèles ML/DL (LightGBM, MLP), analyse SHAP, clustering stratifié.
6. `06_timeseries/` — extraction de séries temporelles (OEDI) + clustering des profils de consommation.

## État actuel (dernier commits + travail en cours)
- Clustering univarié des séries temporelles (sans PCA, deux types de normalisation) : `d88424d`.
- Clustering multivarié (forme + amplitude + météo) : `168e17b`.
- `save local changes about the clustering multi buildings` : `daf472f`.
- Figures produites : profils journaliers par cluster, PCA 2D, elbow/silhouette, heatmap résumé
  multivarié — voir `reports/figures/`.

### Travail en cours, non commité (`timeseries_clustering.ipynb`)
- Ajout d'un bâtiment de comparaison (10672-0) pour valider la méthode sur un second cas.
- Nettoyage : suppression des jours avec NaN avant clustering.
- Normalisation en deux temps : forme par z-score intra-jour, amplitude par StandardScaler global.

### Travail en cours, non commité (`timeseries_clustering_multi.ipynb`)
- Passage au pas de temps **journalier (24 points/heure)** au lieu de 15 min (96 points), pour le
  clustering multi-bâtiments.
- Sélection de 100 bâtiments **tout-électriques**, avec téléchargement des séries temporelles
  manquantes et une fonction d'extraction jour × features par bâtiment, étendue au multi-bâtiments.
- Météo : gardée **sans PCA**, standardisée globalement, pour rester interprétable dans les
  corrélations.
- PCA appliquée uniquement sur la **forme** des courbes ; amplitude et météo restent en variables
  brutes standardisées, concaténées au PCA de la forme pour le choix de k et le clustering final.
- Nouvelles visualisations interactives : toutes les courbes journalières (pas seulement la
  moyenne par cluster), inspection d'un bâtiment particulier (ses journées colorées par cluster),
  scatter PC1 (forme) vs température extérieure moyenne.
- Analyse de corrélation : quelles variables (dont météo) expliquent le mieux l'amplitude de
  consommation, avec scatter interactif + droite de tendance sur la variable la plus corrélée.
- Graphique final avec palette **adaptée aux daltoniens** (protan/deutan/tritan) et styles de
  trait redondants (l'identité d'un cluster ne dépend jamais de la couleur seule).

## `05_deeplearning/` — corrections récentes (fichiers de l'utilisatrice uniquement)

Contrainte importante : le repo est partagé avec un collègue (Boubaker AMDYOUN). L'utilisatrice ne
modifie que ses propres fichiers : `analyse_exploratoires.ipynb`, `lgbm_shap.ipynb`,
`lgbm_stratified.ipynb`, `lightgbm_stratified.ipynb`, `mlp_stratified.ipynb`,
`physical_feature_engineering.ipynb`. Les fichiers de Boubaker (`baseline_lgbm.ipynb`,
`lgbm_electricity*.ipynb`, `clustering_stratifie.ipynb`) servent de référence en lecture seule.

Bugs trouvés et corrigés :
- **`mlp_stratified.ipynb`** : `pd.read_parquet("y.parquet")` (minuscule) → `"Y.parquet"` — ne
  correspondait à aucun fichier sur un système sensible à la casse.
- **`lgbm_shap.ipynb`** : `idx_train`/`idx_test` (positions calculées sur les 549 971 lignes brutes
  dans `clustering_stratifie.ipynb`) étaient appliqués après un filtrage + `reset_index` qui
  changeait les positions des lignes → mauvais bâtiments sélectionnés, tous les résultats du
  notebook étaient invalides. Fix : remapping des positions via `np.cumsum` avant le filtrage.
  Deuxième bug découvert ensuite : une fois filtré, certains strata de `stratum_train` (hérités du
  dataset complet) n'ont plus qu'1 membre dans le sous-ensemble filtré → `train_test_split(stratify=...)`
  plantait. Fix : fusion locale des strata rares (<2 membres) dans un groupe `"RARE"` avant de
  stratifier, avec repli sur un split non stratifié en dernier recours.
- **`lgbm_stratified.ipynb`** : remplacé sa propre stratification `qcut` (qui construisait DEUX
  splits différents pour `X` et `X_physical`, rendant la comparaison Global vs Physique non
  comparable) par le split partagé `idx_train.npy`/`idx_test.npy`/`cluster_labels.parquet` de
  `clustering_stratifie.ipynb`, comme dans `lightgbm_stratified.ipynb` et `mlp_stratified.ipynb`.
- **`physical_feature_engineering.ipynb`** : `KeyError: 'in.insulation_slab'` — cette colonne a été
  éclatée en `in.slab_perimeter_r` / `in.slab_under_r` par `transformations_numeriques.ipynb`
  (fichier du collègue, non modifié) avant que `X.parquet` soit construit. Fix : `R_cols`/`R_weights`/
  `drop_cols` utilisent maintenant les deux nouvelles colonnes.

Pipeline de régénération requis avant d'exécuter les notebooks de `05_deeplearning/` (sauf
`analyse_exploratoires.ipynb`, autonome) : `clustering_stratifie.ipynb` (génère `idx_train.npy`,
`idx_test.npy`, `cluster_labels.parquet`) → `physical_feature_engineering.ipynb` (génère
`X_physical_engineered.parquet`) → notebooks de modélisation.

### Nettoyage (code mort / dupliqué / inutile) — les 5 notebooks sont passés en revue
- **`analyse_exploratoires.ipynb`** (15 → 11 cellules) : supprimé une heatmap de corrélation
  dupliquée (recalculait `Y[TARGETS].corr()` une deuxième fois sous un autre nom), un
  `#print()` mort, et surtout un **pipeline de split orphelin** (elbow KMeans, `K_FINAL=50`,
  sauvegarde `X_train.npy`/`X_val.npy`/`X_test.npy`/`scaler.pkl`/`kmeans.pkl`) : aucun autre
  notebook du repo ne lisait ces fichiers, le split officiel étant `idx_train.npy`/`idx_test.npy`
  de `clustering_stratifie.ipynb`.
- **`lightgbm_stratified.ipynb`** : déjà propre — juste un import `DummyRegressor` inutilisé
  et deux variables de split (`strat_train`/`strat_val`) jamais lues, supprimés.
- **`mlp_stratified.ipynb`** (17 → 16 cellules) : supprimé un bloc de code mort laissé en
  triple-quote (ancienne version du split, non exécutée), un titre markdown vide (`##`), et des
  `print()` de debug résiduels. L'import `tensorflow as tf` était inutilisé — plutôt que le
  retirer, ajouté `tf.random.set_seed(42)` (numpy/sklearn étaient seedés mais pas Keras/TF, donc
  les poids et le dropout du MLP changeaient à chaque exécution).
- **`lgbm_shap.ipynb`** / **`lgbm_stratified.ipynb`** : nettoyage déjà fait lors des fixes de bugs
  ci-dessus (cellules mortes supprimées, dicts `results = {}` jamais lus supprimés).

## Missions du stage
1. Exploration et nettoyage des données
2. Feature engineering
3. Détection des appareils
4. Analyse de la flexibilité HVAC (upgrades dr_001–dr_005)
5. Prédiction de la consommation
6. Modèles ML/DL
7. Évaluation des modèles
8. Interprétation (SHAP)
