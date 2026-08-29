# Émulation de courbes de charge résidentielles et gisement de flexibilité

**Fiche technique des modèles** — 24 août 2026

Source : ResStock 2025.1, année météo réelle 2018 (AMY 2018), upgrade 0.
Sous-ensemble : maisons individuelles plain-pied, chauffage électrique, sans véhicule
électrique, piscine ni photovoltaïque.
Cibles : électricité totale, chauffage, climatisation, eau chaude sanitaire.

---

## Sommaire

- [0. Chaîne complète](#0-chaîne-complète)
- [1. Modèle annuel — LightGBM](#1-modèle-annuel--lightgbm)
- [2. Réseau horaire — GRU bidirectionnel](#2-réseau-horaire--gru-bidirectionnel)
- [3. Gisement de flexibilité](#3-gisement-de-flexibilité)
- [4. Pistes testées et écartées](#4-pistes-testées-et-écartées)
- [5. Reproduire](#5-reproduire)

---

## 0. Chaîne complète

Le problème est séparé en deux questions de natures différentes. **Combien** un logement
consomme dans l'année se prédit très bien à partir de ses seules caractéristiques statiques :
c'est le travail du LightGBM. **Comment** cette consommation se répartit heure par heure
demande la météo et les usages : c'est le travail du réseau, qui reçoit le niveau annuel en
entrée plutôt que de le redécouvrir.

```
        Métadonnées ResStock — 64 062 logements × 61 features
                              │
                              ▼
        ① LightGBM annuel — 4 modèles, un par usage
           sortie : 4 × kWh/an, hors échantillon (KFold 5)
           R² 0.942 / 0.951 / 0.955 / 0.981
                              │
                              │  niveau annuel injecté dans le vecteur statique
                              ▼
   Séries horaires ────► ② GRU bidirectionnel
   2 005 bâtiments        f(t) 32 entrées + s 51 colonnes → 4 × kWh/h
   × 8 760 h              R² 0.896 / 0.912 / 0.928 / 0.852
                              │
                              │  la consigne de thermostat est une entrée : on peut la modifier
                              ▼
        ③ Simulation contrefactuelle
           ΔP(t) = référence − scénario → puissance effacée et report
```

| Grandeur | Valeur |
|---|---|
| Logements, modèle annuel | 64 062 |
| Bâtiments, séries horaires | 2 005 |
| Heures simulées apprises | 17,6 M |
| Paramètres du réseau | 164 356 |
| Bâtiments de validation, jamais vus | 101 |

---

## 1. Modèle annuel — LightGBM

Quatre régresseurs indépendants, un par usage, entraînés sur les mêmes features et le même
découpage stratifié par zone climatique ASHRAE. Les hyperparamètres sont réglés **séparément
pour chaque usage** : le chauffage et l'eau chaude n'ont ni la même asymétrie ni le même taux
de zéros, et leur imposer les réglages de `total` coûtait de la précision.

### 1.1 Jeu de features — 61 colonnes en 5 blocs

| Bloc | Cols | Contenu | Origine |
|---|---:|---|---|
| Enveloppe agrégée | 5 | `UA`, `H_ve`, `C`, `A_solaire`, `compacite` | Modèle réduit d'enveloppe |
| Distribution | 1 | `DSE` | ASHRAE 152 simplifié |
| Non-enveloppe | 41 | HVAC, occupants, électroménager, socio-économique | ResStock encodé |
| Système de chauffage | 3 | `pac`, `cop_chauffage`, `mshp` | `in.hvac_heating_efficiency` |
| Climat | 11 | `HDD18`, `CDD18`, `T_moy`, …, `UA×HDD`, `UA×CDD` | Météo par comté |
| **Total** | **61** | | |

Définitions des agrégats d'enveloppe :

- **`UA`** — déperditions par les parois (murs, toiture, plancher, vitrages, portes), en W/K.
- **`H_ve`** — déperditions par ventilation et infiltrations, en W/K.
- **`C`** — inertie thermique : chaleur stockée dans la masse du bâtiment, en J/K.
- **`A_solaire`** — surface vitrée équivalente de captation solaire, pondérée par orientation.
- **`compacite`** — surface d'enveloppe rapportée au volume chauffé.
- **`DSE`** — rendement de distribution des gaines : les pertes ne comptent que pour la part
  des gaines hors volume chauffé.

### 1.2 Le levier décisif — le système de chauffage

Les 47 features d'origine ne contenaient **ni** `in.hvac_heating_type`, **ni**
`in.hvac_heating_efficiency`, **ni** `in.heating_fuel`. Le modèle ne pouvait donc pas
distinguer une pompe à chaleur d'un convecteur — alors qu'à besoin thermique identique une PAC
consomme **trois fois moins** (3 175 kWh/an contre 9 854 sur le parc horaire). Il en faisait la
moyenne, avec un biais mesuré de −16 % sur les convecteurs et +37 % sur les PAC les plus
efficaces.

Trois colonnes suffisent à le corriger :

```python
hspf = eff.str.extract(r'([\d.]+)\s*HSPF')[0].astype(float)   # NaN = résistance
pac  = hspf.notna()                                           # 1 = pompe à chaleur
cop  = np.where(pac, hspf / 3.412, 1.0)                       # HSPF → COP saisonnier
mshp = eff.str.startswith('MSHP')                             # mini-split, sans gaines
```

**Gain : +0,153 de R² sur le chauffage.**

Le second bloc, le climat, était disponible depuis le début dans `weather_static.parquet` mais
n'était joint nulle part. Les deux produits `UA × HDD18` et `UA × CDD18` encodent la forme
physique de la déperdition annuelle, qu'un arbre de décision ne peut pas fabriquer à partir des
deux facteurs pris séparément.

### 1.3 Objectif par usage

L'*objective* est la formule d'erreur minimisée. Par défaut `l2` — la moyenne des écarts au
carré — qui suppose des erreurs symétriques autour de la moyenne.

Le chauffage ne ressemble pas à cela : de nombreux logements du Sud consomment exactement zéro,
et les autres s'étalent jusqu'à 85 000 kWh/an. C'est une distribution **zéro-gonflée**.
**Tweedie** est une famille de lois faite pour ce cas : une masse de probabilité en zéro plus
une queue continue positive. Le paramètre `tweedie_variance_power = 1.4` place le curseur entre
Poisson (1,0) et Gamma (2,0).

Mesure sur le jeu de test, features enrichies :

| Usage | `l2` | `tweedie` | Retenu |
|---|---:|---:|---|
| total | 0.9408 | 0.9408 | `l2` |
| chauffage | 0.9453 | **0.9526** | `tweedie` |
| clim | 0.9503 | **0.9550** | `tweedie` |
| eau chaude | **0.9738** | 0.9714 | `l2` |

D'où une application **par usage** et non en bloc : un objectif zéro-gonflé n'aide que sur les
cibles qui ont réellement beaucoup de zéros.

### 1.4 Hyperparamètres retenus — Optuna, 20 essais par usage

| Paramètre | total | chauffage | clim | eau chaude |
|---|---:|---:|---:|---:|
| `objective` | l2 | tweedie | tweedie | l2 |
| `tweedie_variance_power` | — | 1.40 | 1.40 | — |
| `learning_rate` | 0.0402 | 0.0173 | 0.0260 | 0.0293 |
| `num_leaves` | 23 | 36 | 20 | 99 |
| `min_child_samples` | 13 | 77 | 202 | 11 |
| `colsample_bytree` | 0.859 | 0.405 | 0.402 | 0.623 |
| `subsample` | 0.641 | 0.810 | 0.792 | 0.740 |
| `reg_lambda` | 1.13 | 15.72 | 1.57 | 0.92 |
| `reg_alpha` | 3.13 | 1.39 | 2.09 | 2.87 |
| arbres retenus | 3 992 | 3 980 | 4 000 | 3 012 |

Rôle de chaque réglage :

- **`learning_rate`** — poids de chaque nouvel arbre dans la somme. Petit = apprentissage lent
  mais robuste.
- **`num_leaves`** — nombre de feuilles par arbre, donc sa complexité.
- **`min_child_samples`** — minimum de logements par feuille ; empêche une règle bâtie sur
  quelques cas isolés.
- **`colsample_bytree`, `subsample`** — fraction des colonnes et des lignes vue par chaque
  arbre ; force la diversité de l'ensemble.
- **`reg_lambda`, `reg_alpha`** — pénalités L2 et L1 sur les valeurs de feuilles.

Découpage : 64 % entraînement / 16 % validation / 20 % test, stratifié par zone ASHRAE,
graine 42. L'arrêt anticipé se fait sur la **validation**.

> **Correction apportée.** L'arrêt anticipé se faisait auparavant sur `eval_set=[(X_test,
> Y_test)]`, c'est-à-dire que le nombre d'arbres était choisi sur les données servant à juger le
> modèle. Le jeu de test n'était donc plus un jeu de test.

### 1.5 Performance — jeu de test, 12 813 logements

| Usage | R² · 47 feat. | R² · final | Δ | RMSE finale | RMSE avant |
|---|---:|---:|---:|---:|---:|
| total | 0.8862 | 0.9427 | **+0.0565** | 2 377 | 3 350 |
| chauffage | 0.8004 | 0.9534 | **+0.1530** | 1 649 | 3 410 |
| clim | 0.9159 | 0.9570 | +0.0411 | 645 | 902 |
| eau chaude | 0.9746 | 0.9750 | +0.0004 | 241 | 243 |

RMSE en kWh/an.

### 1.6 Export hors échantillon

Le réseau horaire reçoit ces prédictions dans son vecteur statique. Si elles étaient
*in-sample*, ses bâtiments de validation obtiendraient un niveau annuel qu'aucun bâtiment
inconnu n'atteindrait, et son R² serait optimiste.

Le fichier est donc régénéré en **KFold à 5 plis** : le parc est coupé en cinq, on entraîne sur
quatre parts et on prédit la cinquième, cinq fois. Chaque logement est ainsi prédit par un
modèle qui ne l'a jamais vu. Contrôle : le R² hors échantillon doit retomber sur le R² de test.

| Jeu | total | chauffage | clim | eau chaude |
|---|---:|---:|---:|---:|
| test — 12 813 logements | 0.9427 | 0.9534 | 0.9570 | 0.9750 |
| KFold — 64 062 logements | 0.9415 | 0.9508 | 0.9552 | 0.9809 |
| écart | 0.001 | 0.003 | 0.002 | 0.006 |

Les deux coïncident. L'ancien fichier, lui, affichait **0.949** sur `total` quand un modèle
honnête sur les mêmes features plafonnait à **0.886** — l'écart signait bien des prédictions
in-sample.

---

## 2. Réseau horaire — GRU bidirectionnel

Le modèle lit une **semaine** de conditions (168 pas horaires) et rend la consommation de
chacune de ces heures, pour les quatre usages. Il est **non causal** : à l'heure *t* il voit
toute la semaine, passé et futur. C'est légitime pour un émulateur de simulation sur une année
météo connue d'avance, et ne le serait pas pour de la prévision temps réel. Aucune consommation
passée n'entre en entrée — c'est ce qui permet de l'appliquer à un bâtiment jamais mesuré.

### 2.1 Architecture

```
   f(t) dynamique                            s  statique
   (168, 32)                                 (51,)
   5 météo · 16 schedules · 2 consignes      4 niveaux annuels · 42 physique
   6 calendrier · 3 écarts                   2 équipement · 3 chauffage
        │                                         │
        ▼  standardisation                        ▼  standardisation
   ┌──────────────────────────────┐                │
   │  GRU BIDIRECTIONNEL          │                │
   │  hidden = 128                │                │
   │                              │                │
   │  sens avant   h₁→…→h₁₆₈  128 │                │
   │  sens arrière h₁←…←h₁₆₈  128 │                │
   │                              │                │
   │  384 = 3 × 128 : portes      │                │
   │  reset · update · candidat   │                │
   └──────────────┬───────────────┘                │
                  │                                │
             (168, 256)          s répété 168 fois │
                  │              (51,) → (168, 51) │
                  └──────────────┬─────────────────┘
                                 ▼
                        concat → (168, 307)
                                 │
                  ┌──────────────▼──────────────┐
                  │  TÊTE MLP, partagée         │
                  │  Linear(307 → 128) + ReLU   │
                  │  Linear(128 → 4) + Softplus │
                  └──────────────┬──────────────┘
                                 │
                    × masque d'équipement
                    × échelle des cibles
                                 ▼
                     (168, 4) en kWh
              [total, chauffage, clim, eau chaude]
```

Points de lecture :

- **GRU** (*gated recurrent unit*) — unité récurrente qui lit les heures une par une en
  entretenant une mémoire de 128 nombres. C'est ce qui permet à l'inertie thermique d'exister
  dans le modèle : ce qui se passe à 18 h influence encore 20 h. Trois portes contrôlent cette
  mémoire — `reset` (quelle part du passé oublier), `update` (quelle part de l'état conserver),
  `candidat` (quel nouvel état proposer) — d'où le facteur 3 sur les dimensions.
- **Bidirectionnel** — deux GRU indépendants, l'un du lundi au dimanche, l'autre à l'envers,
  concaténés : `128 × 2 = 256`. Gain mesuré : chauffage 0.805 → 0.835. Coût : ~6,6× plus lent.
- **`s` ne traverse pas le GRU** : il est recopié à chaque pas de temps et agrafé juste avant la
  tête.
- **Tête partagée** — ce ne sont pas 168 réseaux mais le même MLP appliqué 168 fois. C'est aussi
  la limite structurelle connue : un `Linear` après concaténation calcule une **somme**
  `a·h + b·s`, alors que la relation physique est un **produit** (le même écart de température
  produit deux fois plus de kWh dans un bâtiment deux fois moins isolé).
- **`ReLU`** — `max(0, x)`. Sans elle, empiler deux couches linéaires équivaudrait à une seule.
- **`Softplus`** — `log(1 + eˣ)`, toujours positif : une consommation ne peut pas être négative.

### 2.2 Budget de paramètres — 164 356

| Bloc | Tenseur | Forme | Paramètres | Part |
|---|---|---|---:|---:|
| GRU sens avant | `weight_ih` | (384, 32) | 12 288 | 37,9 % |
| | `weight_hh` + biais | (384, 128) | 49 920 | |
| GRU sens arrière | `weight_ih` | (384, 32) | 12 288 | 37,9 % |
| | `weight_hh` + biais | (384, 128) | 49 920 | |
| Tête, couche 1 | `head.0` | (128, 307) | 39 424 | 24,0 % |
| Tête, sortie | `head.2` | (4, 128) | 516 | 0,3 % |
| **Total** | | | **164 356** | 100 % |

### 2.3 Les 32 entrées dynamiques

| Rang | Groupe | n | Colonnes |
|---|---|---:|---|
| 0–4 | Météo | 5 | T° extérieure, humidité relative, vent, solaire direct, solaire diffus |
| 5–20 | Schedules d'usage | 16 | occupants, vacance, éclairage ×2, prises ×2, lave-linge, sèche-linge, lave-vaisselle, cuisson, ventilateur, ECS ×3, `no_cooling`, `no_heating` |
| 21–22 | Consignes | 2 | chauffage, climatisation — **reconstruites**, vides dans la release |
| 23–28 | Calendrier cyclique | 6 | `h_sin/cos`, `d_sin/cos`, `m_sin/cos` |
| 29–31 | Écarts dérivés | 3 | `ecart_chauffage`, `ecart_clim`, `besoin_elec_chauffage` |

**Le calendrier en sinus/cosinus.** Coder l'heure par un entier 0→23 apprendrait au réseau que
23 h et 0 h sont distants de 23 — le maximum possible — alors qu'ils se suivent. Sur un cercle,
chaque heure devient un point (sin, cos) et minuit est voisin de 23 h. Deux colonnes sont
nécessaires par cycle : avec le sinus seul, 6 h et 18 h auraient la même valeur.

**Les écarts rectifiés.** La puissance de chauffage est à peu près proportionnelle à
`consigne − T_ext`, et nulle dès qu'il fait plus chaud dehors — d'où le `clip(lower=0)`. C'est
la relation physique injectée à la main plutôt que laissée à découvrir.

**Le besoin électrique.** Le COP d'une pompe à chaleur chute avec le froid ; diviser le besoin
thermique par ce rendement instantané donne l'énergie réellement appelée :

```python
cop_t = np.clip(cop_bat * (0.6 + 0.02 * t_ext), 1.0, cop_bat)
besoin_elec_chauffage = ecart_chauffage / cop_t

# cop_bat = HSPF / 3.412 pour une PAC ; 1.0 pour une résistance électrique
```

### 2.4 Le vecteur statique — 51 colonnes

| Bloc | Cols | Contenu |
|---|---:|---|
| Niveaux annuels | 4 | prédictions LightGBM hors échantillon, un par usage |
| Physique | 42 | agrégats d'enveloppe, constante de temps `tau`, features ResStock |
| Présence d'équipement | 2 | climatisation présente, chauffe-eau électrique |
| Système de chauffage | 3 | `pac`, `cop`, `mshp` |

`tau = C / (UA + H_ve)` est la **constante de temps** du bâtiment, en heures : le temps
caractéristique de refroidissement quand le chauffage s'arrête. Une maison lourde et bien
isolée tient toute la nuit ; une maison légère et mal isolée refroidit en quelques heures.
C'est la grandeur qui gouverne le potentiel de flexibilité.

### 2.5 Le masque de sortie

16 % des logements ont un chauffe-eau non électrique et 5 % n'ont pas de climatisation : leur
consommation de cet usage est strictement nulle toute l'année. La sortie est donc **multipliée
par zéro** pour ces usages.

Ce n'est pas quelque chose à apprendre : `Softplus` tend vers zéro sans jamais l'atteindre.
Même parfaitement informé, le réseau laisserait quelques watts, 8 760 heures par an. Le total
n'est pas masqué — une maison sans climatisation consomme quand même de l'électricité.

### 2.6 Protocole d'entraînement

| | |
|---|---|
| Découpage | par **bâtiment** — 1 904 entraînement / 101 validation, graine 42 |
| Fenêtres | 99 008 entraînement · 5 252 validation — 168 h, sans recouvrement |
| Optimiseur | Adam, `lr = 1e-3`, batch 64 |
| Perte | MSE sur cibles mises à l'échelle, sans centrage (`Yn ≥ 0`) |
| Arrêt | 40 époques maximum, patience 8 — meilleure époque : 37 |
| Durée | ≈ 3 h sur 20 cœurs CPU, sans GPU |

Le découpage se fait **par bâtiment, jamais par fenêtre** : découper par fenêtre laisserait le
modèle voir janvier d'un bâtiment et être jugé sur son février.

Les statistiques de standardisation (moyenne, écart-type) sont calculées **sur l'entraînement
seul** — les calculer sur tout le parc ferait fuiter de l'information de validation.

### 2.7 Performance — 101 bâtiments jamais vus

| Usage | R² départ | R² final | Δ | \|NMBE\| médian | avant | CV(RMSE) médian |
|---|---:|---:|---:|---:|---:|---:|
| total | 0.828 | 0.896 | **+0.068** | 8,3 % | 10,6 % | 24,3 % |
| chauffage | 0.805 | 0.912 | **+0.107** | 15,5 % | 29,8 % | 59,1 % |
| clim | 0.883 | 0.928 | +0.046 | 12,1 % | 14,6 % | 33,7 % |
| eau chaude | 0.831 | 0.852 | +0.021 | 5,7 % | 9,1 % | 58,7 % |

Les métriques, et ce qu'elles disent chacune :

- **R²** — part de la variabilité expliquée, calculée sur toutes les heures confondues. Défaut :
  dominée par les gros consommateurs, elle masque le biais de niveau.
- **NMBE** — `(Σprédit − Σréel) / Σréel`. Mesure le **biais de niveau** : le modèle a-t-il mis
  la bonne quantité d'énergie sur l'année ?
- **CV(RMSE)** — erreur horaire typique rapportée à la moyenne. Mesure la **dispersion** : les
  pics tombent-ils au bon moment ?

Les deux derniers sont calculés **par bâtiment** puis médianés, selon l'ASHRAE Guideline 14
(critères au pas horaire : |NMBE| ≤ 10 %, CV(RMSE) ≤ 30 %).

Lecture : seul le `total` passe le seuil de dispersion. Le modèle **place à peu près la bonne
énergie sur l'année** mais **ne reproduit pas encore chaque pic** de chauffage.

### 2.8 Comportement sur quatre bâtiments contrastés

Semaine du 15 janvier et semaine du 16 juillet, sur des bâtiments de validation choisis par
climat et par qualité de prédiction — et non au hasard.

| Bâtiment | Situation | Hiver | Été |
|---|---|---|---|
| 170214 · CO, zone 7B | HDD 5 808, résistance, 19 832 kWh/an de chauffage | jusqu'à −22,5 °C ; chauffage 795 réel / 939 prédit | pas de climatisation |
| 420474 · AZ, zone 2B | HDD 505, PAC COP 2.26 | chauffage 130 / 176 | 42 °C ; clim 401 réel / 407 prédit |
| 423584 · TX, zone 2A | cas médian | chauffage 299,5 / 301,9 | clim 265 / 234 |
| 393604 · CO, zone 5B | pire biais du jeu de validation | chauffage 146 réel / **279 prédit** | clim 107 / 168 |

Cumuls hebdomadaires en kWh. Le modèle suit correctement la **forme journalière** et se trompe
surtout sur le **niveau**, ce qui est exactement ce que disent NMBE et CV(RMSE).

> Le quatrième bâtiment est le pire cas du jeu de validation, retenu volontairement. Le graphe
> de contrôle initial traçait systématiquement la première fenêtre du premier bâtiment, qui
> s'est avérée être un cas extrême choisi par hasard — d'où le principe de tracer désormais un
> cas médian, un bon cas et le pire.

### 2.9 Le résultat le plus instructif

Le levier dominant est le **nombre de bâtiments**, pas le modèle. À configuration identique,
passer de 503 à 2 005 bâtiments rapporte +0,043 sur le total et +0,032 sur le chauffage — plus
que toutes les variantes d'architecture réunies, qui sont *toutes négatives*.

Le contraste avec les fenêtres glissantes est net : 7× plus de fenêtres sur les **mêmes**
bâtiments ne donne rien ; 4× plus de **bâtiments** donne tout. Ce qui manquait au réseau n'était
ni la capacité, ni le nombre d'heures, mais la **diversité de bâtiments** — le ratio bâtiments
d'entraînement / colonnes de `s` passe de 8,4 à 37.

Le parc reste extensible : 55 732 bâtiments passent le filtre de sélection, 2 005 sont
téléchargés à ce jour.

---

## 3. Gisement de flexibilité

La consigne de thermostat est une **entrée** du réseau. On peut donc la modifier et relire la
courbe de charge : c'est une expérience contrefactuelle au sens propre — « qu'aurait consommé
ce logement si son thermostat avait été baissé de 2 °C entre 18 h et 21 h ».

```
consignes réelles     →  f(t)   →  modèle  →  courbe de référence
consignes décalées    →  f'(t)  →  modèle  →  courbe scénario

ΔP(t) = référence − scénario      ΔP > 0 : effacement    ΔP < 0 : report
```

### 3.1 Le point critique de l'implémentation

`f(t)` contient la consigne **et trois colonnes qui en dérivent** : les deux écarts rectifiés et
le besoin électrique de chauffage. Décaler la consigne sans recalculer les trois présenterait au
réseau une entrée physiquement incohérente — et le test d'identité ne le verrait pas, puisqu'à
décalage nul toutes ces colonnes sont inchangées.

### 3.2 Scénarios

On borne **l'action**, pas l'effet : le modèle ne prédit pas la température intérieure, donc le
confort n'est pas vérifié.

| Scénario | Saison | Heures | Décalage | Cible |
|---|---|---|---:|---|
| Effacement du soir | déc–fév | 18–21 h | −2 °C | chauffage |
| Effacement du matin | déc–fév | 7–10 h | −2 °C | chauffage |
| Pointe d'été | juin–août | 15–19 h | +2 °C | climatisation |

Les ±2 °C sont du même ordre que les réduits que ResStock applique déjà lui-même, et une partie
du parc en possède — le réseau a donc appris sur de vraies variations de consigne
intra-journalières, pas sur des extrapolations pures.

### 3.3 Contrôles de validité

| Test | Attendu | Mesuré | Verdict |
|---|---|---|---|
| Identité — décalage nul | prédiction inchangée | écart 0,00 e+00 | conforme |
| Test négatif — chauffage décalé en été | effacement ≈ 0 | 0,3 % de l'hiver | conforme |
| Effacement croissant avec `UA` | Spearman nettement > 0 | +0,11 | faible |
| Report étalé avec `tau` | corrélation marquée | −0,07 | absent |

Les deux premiers passent nettement : la machinerie contrefactuelle ne fabrique rien et n'agit
pas là où la physique l'interdit. Les deux derniers sont faibles — le signal existe mais ne
s'ordonne pas encore selon les agrégats d'enveloppe. C'est la limite actuelle de l'étude.

### 3.4 Points ouverts

**Résultats de gisement à re-mesurer.** Les valeurs disponibles (pointe médiane 0,54 à 0,74 kW,
report de 22 % à 35 % selon les scénarios) proviennent d'une version antérieure du modèle. Le
balayage du parc doit être rejoué avec le modèle actuel avant toute citation.

**Décalage horaire à intégrer aux scénarios.** Les séries ResStock sont horodatées dans un
**fuseau unique** pour tout le pays, cohérent avec l'heure de l'Est, et non en heure locale.

Vérification sur le bâtiment 70818 (Californie) contre le fichier météo de son comté : moyenne
annuelle de température identique au millième (15,009 °C des deux côtés), corrélation 0,996 une
fois les séries recalées. Le pic de rayonnement du 15 janvier apparaît à 12 h dans le fichier
comté et à 14 h 30 dans la série horaire, soit **3 h d'écart exactement** une fois corrigées les
conventions d'étiquetage — l'écart PST → EST. Confirmation à l'échelle du parc : le pic solaire
moyen dérive de 10 h (côte Est) à 16 h (Alaska) dans l'horloge du fichier, ce qui n'arriverait
pas si chaque bâtiment était en heure locale.

Conséquence : le modèle reste cohérent avec lui-même, mais la fenêtre « 18–21 h » vise en
réalité 15–18 h locales en Californie. Sur le bâtiment 70818, la pointe hivernale réelle est à
20 h locales, soit **23 h dans l'horloge du fichier** : le scénario la manque entièrement et
n'agit que sur 1,27 kWh/h au lieu des 2,35 du pic. Le gisement est donc sous-estimé pour tout
l'Ouest. Correctif : décaler la fenêtre de `round((longitude + 75) / 15)` heures par bâtiment.

---

## 4. Pistes testées et écartées

Chaque variante a été mesurée **seule**, sur le même parc, la même graine et les mêmes
bâtiments de validation.

| Piste | Modèle | Effet | Lecture |
|---|---|---:|---|
| Tête multiplicative — forme × niveau annuel | réseau | −0.027 ch. | Transforme un problème local en problème global, sans fournir l'information globale |
| FiLM — `s` module les sorties du GRU | réseau | −0.008 | La concaténation suffisait |
| GRU à 2 couches + dropout | réseau | −0.009 | 2× le temps, aucun gain : le plafond n'est pas la capacité |
| Perte de Huber + `ReduceLROnPlateau` | réseau | −0.002 | — |
| Moyennes glissantes de T° (24 h, 72 h) | réseau | −0.007 | L'inertie est déjà portée par la mémoire du GRU |
| Apports solaires `A_solaire × GHI` | réseau | −0.008 | Le rayonnement brut est déjà dans `f(t)` |
| Fenêtres glissantes, `stride` 24 | réseau | +0.000 | 7× plus de fenêtres, mais les mêmes bâtiments |
| Produit enveloppe `(UA+H_ve) × écart` | réseau | +0.002 | Le réseau construit déjà cette interaction |
| Climat dans `s` | réseau | +0.008 | Décisif pour LightGBM, redondant ici |
| Cible en `log1p` | LightGBM | −0.015 ch. | Déplace l'erreur vers les gros consommateurs |
| Objectif tweedie sur l'eau chaude | LightGBM | −0.003 | N'aide que les cibles réellement zéro-gonflées |

Deux enseignements de méthode :

1. **Ajouter de la structure ne remplace pas l'information manquante.** La tête multiplicative
   encodait une loi physique exacte (`conso = niveau × forme`) et a pourtant dégradé le modèle,
   parce qu'elle déplaçait la difficulté vers une information absente du vecteur statique.
2. **Le R² poolé ne suffisait pas à voir le problème.** Ce sont le NMBE par bâtiment et le tracé
   d'un bâtiment nommé qui ont rendu le défaut lisible. Instrumenter d'abord, modifier ensuite.

---

## 5. Reproduire

### 5.1 Fichiers produits

| Fichier | Contenu |
|---|---|
| `X_features_v2.parquet` | 64 062 × 61 — jeu d'entrée du modèle annuel |
| `static_preds_oos.parquet` | 64 062 × 4 — niveaux annuels hors échantillon |
| `nn_buildings_elargi.csv` | 2 005 bâtiments téléchargés, consignes injectées |
| `loadnet.pt` | réseau + normalisations + `s`, masque et niveau par bâtiment |
| `experiments/cache/*.npy` | séries horaires pré-agrégées — 13 s de chargement au lieu de 3 min |

### 5.2 Commandes

```bash
# modèle annuel : réglage par usage, puis export hors échantillon
python lgbm_exp.py --nom final --feats cop,clim --tweedie chauffage,clim \
                   --optuna 20 --arbres 4000 --export --kfold 5

# réseau horaire : configuration retenue
python ts_train.py --nom final --feats cop,copt,oos \
                   --parc nn_buildings_elargi.csv --cache elargi \
                   --val_ref --save loadnet.pt

# courbes réel vs prédit sur bâtiments contrastés
python courbes.py

# tableau comparatif de toutes les expériences
python tableau.py
```

`--val_ref` rejoue exactement les 101 bâtiments de validation du parc initial de 503 : sans
cette option, agrandir le parc change aussi le jeu de validation et les écarts mesurés ne
veulent plus rien dire. Le banc refuse par assertion un cache incohérent avec le parc demandé —
garde-fou ajouté après qu'une exécution annoncée à 2 005 bâtiments en a silencieusement
utilisé 503.

### 5.3 Suite

1. Rejouer le balayage de flexibilité avec le modèle actuel.
2. Intégrer la correction de fuseau horaire aux fenêtres de scénario.
3. Reporter NMBE et CV(RMSE) par bâtiment dans le notebook d'évaluation.
4. Étendre le parc au-delà de 2 005 bâtiments — c'est le levier dont le rendement est mesuré et
   non encore épuisé.
5. Re-tester la tête multiplicative, maintenant que le système de chauffage figure dans `s`.
