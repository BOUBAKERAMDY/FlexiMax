# Plan d'amélioration — `timeseries_net.ipynb`

Date : 2026-08-03
Point de départ : GRU bidirectionnel + statique, 503 bâtiments, validation par bâtiment.

```
total       R² = 0.836   RMSE = 0.983 kWh/h
chauffage   R² = 0.815   RMSE = 0.873 kWh/h
clim        R² = 0.883   RMSE = 0.324 kWh/h
eau_chaude  R² = 0.822   RMSE = 0.142 kWh/h
```

---

## Diagnostic

Le graphe « réel vs prédit » de la cellule c10 trace `w = 0`, c'est-à-dire la
**première semaine du premier bâtiment de validation**. Ce bâtiment est le **530669** :

| | réel | LightGBM (déjà dans `s`) | réseau, semaine tracée |
|---|---|---|---|
| total | 14 927 kWh/an → **1,70 kWh/h** de moyenne | 15 815 | ~**2,0 kWh/h** |
| chauffage | **106 kWh/an** | 95 | ~100 kWh sur la **seule** semaine |
| clim | 5 428 kWh/an | 6 127 | non nul en janvier |

Caractéristiques : Floride, `Ducted Heat Pump`, `ASHP, SEER 15, 8.5 HSPF`.
La semaine tracée (1–7 janvier) concentre 49 kWh, soit **47 % du chauffage annuel**,
et sa moyenne de total (1,358 kWh/h) est **en dessous** de la moyenne annuelle (1,704).

Deux causes, indépendantes l'une de l'autre.

### Cause A — le réseau n'exploite pas le niveau annuel qu'on lui donne

LightGBM prédit 95 kWh/an de chauffage pour ce bâtiment (réel : 106) — l'information
est excellente et elle est **déjà** dans `s`. Le réseau brûle pourtant tout ce budget
en une semaine.

La relation physique est **multiplicative** : `conso(t) = niveau_annuel × forme(t)`.
Or dans l'architecture actuelle le niveau est une colonne parmi 54, centrée-réduite,
concaténée à 256 sorties de GRU, puis passée dans un `Linear`. Un `Linear` après
concaténation représente très mal un produit. Résultat : le réseau régresse vers le
niveau moyen du parc au lieu de suivre le niveau du bâtiment.

### Cause B — le système de chauffage est absent du vecteur statique

`X_47features.parquet` ne contient **ni** `in.hvac_heating_type`, **ni**
`in.hvac_heating_efficiency`, **ni** `in.heating_fuel`
(`in.hvac_cooling_efficiency` est là, encodé en SEER — c'est uniquement le côté
chauffage qui manque).

Sur les 503 bâtiments : 242 pompes à chaleur, 261 résistances.
Consommation de chauffage moyenne : **3 175 kWh/an pour une PAC contre 9 854 pour une
résistance — facteur 3 à besoin thermique identique.**

Biais du LightGBM annuel « chauffage » par type d'équipement (55 732 bâtiments) :

```
Electric Furnace, 100% AFUE          −16 %
Electric Baseboard, 100%              −0 %
ASHP, SEER 13, 7.7 HSPF              +24 %
ASHP, SEER 15, 8.5 HSPF              +37 %   <- le 530669
MSHP, SEER 14.5, 8.2 HSPF            +62 %
```

Le modèle ne peut pas distinguer une PAC d'un convecteur : il fait la moyenne des deux.

---

## Partie 0 — Hygiène des données — FAIT le 05/08

Deux corrections d'entrée, sans effet attendu sur les métriques, mais qui conditionnent la
lisibilité de tous les A/B suivants.

### 0a — Arbitrage des consignes incohérentes (`inject_setpoints`)

Guide technique ResStock 2025.1, **p. 135, §8.4.5** :

> *"If a sampled heating setpoint is greater than the cooling setpoint, the values are averaged
> and kept constant across heating and cooling seasons."*

La règle n'apparaît qu'une fois dans 278 pages et n'était pas implémentée. « Kept constant »
signifie **le même profil pour le chauffage et la clim** (constant d'une saison à l'autre), pas
« constant dans le temps » : les deux schedules sont moyennés **heure par heure**, offsets compris.

Vérifié contre `out.indoor_temperature.conditioned_space..c`, au centième de degré :

| bâtiment | consignes échantillonnées | T_int simulée | = |
|---|---|---|---|
| 4434 | 75F / 65F, sans offset | 21.11 °C constant | (75+65)/2 |
| 149091 | 80F offset 3F « Night +3h » / 70F | 23.89 °C le jour, 23.06 °C de 3h à 9h | (80+70)/2 et (80−3+70)/2 |

**Portée :** 108/503 bâtiments (21 %), dont 21 des 101 de validation. Décalage moyen appliqué
sur la consigne de chauffage : **1,18 K** (max 3,54 K). Effet secondaire : leur bande morte
devient **nulle**, ce qui explique les 14,1 % d'heures chauffage+clim simultanés mesurés chez
eux contre 5,9 % ailleurs.

Le déclencheur porte sur les consignes **échantillonnées**. Un bâtiment dont les bases sont
dans le bon ordre mais qu'un « Night Setback » clim fait passer sous la consigne de chauffage
la nuit n'est **pas** arbitré (ex. 530669 : 75F / 76F) — c'est du ResStock légitime.

**Gain mesuré** (régression linéaire poolée, 108 bâtiments, 946 000 h) : chauffage
0.435 → 0.448 (**+0.012**), clim 0.578 → 0.576 (−0.002). Soit, extrapolé au R² final,
+0.002 à +0.005 — **sous le plancher de bruit** du tirage de graine (±10 % sur le chauffage
moyen de validation, cf. `etude_parc_503` §4.3). Ce n'est pas un levier, c'est une correction
de justesse.

Ré-injection appliquée en place sur les 108 fichiers, sans re-téléchargement (les consignes se
recalculent entièrement depuis `metadata_clean` + `options_lookup`). Opération idempotente,
vérifiée par une seconde passe à blanc.

### 0b — Consignes retirées de `s`

Les 6 colonnes (`in.heating_setpoint`, `in.cooling_setpoint`, et leurs 4 colonnes d'offset)
sont **redondantes** dans le vecteur statique : `f(t)` contient déjà les deux consignes heure
par heure plus les deux écarts qui en dérivent.

Elles restent dans `X_47features.parquet` : LightGBM en a besoin — `in.heating_setpoint` y est
la **3e feature sur 108** (gain 8.99) précisément parce qu'il n'a aucune série temporelle. La
suppression se fait à la construction de `S_brut`, pas dans le fichier partagé.

`s` : 54 → **48 colonnes**. Ratio bâtiments d'entraînement / colonnes de `s` : 7,4 → **8,4**.
Bénéfice annexe : plus rien à corriger côté statique pour les 108 arbitrés.

**Bug trouvé au passage, sans conséquence.** Les `offset_magnitude` de `X_47features` sont
convertis avec la formule **absolue** `(F−32)×5/9` au lieu de la formule d'**écart** `F×5/9` —
d'où des « magnitudes » de −17,8 °C. C'est le piège que `rapport_setpoints.md` documente,
attrapé dans `inject_setpoints` mais resté dans le pipeline d'encodage statique. Numériquement
inoffensif : `(F−32)×5/9 = F×5/9 − 17.78`, une translation constante, absorbée à l'identique
par la standardisation du réseau et par les seuils d'arbre de LightGBM. À corriger pour la
lisibilité des tableaux de `reports/`, pas pour la performance.

**A/B restant :** redondant ne veut pas dire inutile — la tête lit `s` directement à chaque pas
de temps, alors que le niveau de consigne présent dans `f(t)` doit survivre à la récurrence du
GRU. Mesurer avant de conclure.

---

## Partie 1 — Tête multiplicative (ancrer le niveau)

**Cible :** cause A. Cellules `c2` (modèle), `c8` (données), `c9` (boucle), `c10` (éval), `c11` (sauvegarde).

Le réseau ne prédit plus des kWh mais une **forme sans dimension**, remise à l'échelle
par le niveau annuel LightGBM.

```python
def forward(self, x_time, static, niveau):
    h, _ = self.gru(x_time)
    s = static.unsqueeze(1).expand(-1, h.size(1), -1)
    forme = self.head(torch.cat([h, s], dim=-1))   # Softplus >= 0, ~1 en moyenne
    return forme * niveau.unsqueeze(1)             # kWh/h
```

- biais de la dernière couche initialisé à `0.5413` → `softplus(0.5413) = 1`, on démarre
  pile sur le niveau annuel ;
- `NIV`, même ordre que `TGT = [total, chauffage, clim, eau_chaude]`, assertion à l'appui ;
- le niveau est fourni au réseau **divisé par `ys`**, donc dans l'échelle de `Yn` : la loss
  garde exactement la même définition qu'avant et la `val_loss` reste directement comparable
  au 0,1711 du 31/07 ;
- la loss reste inchangée (`ys` global) : on ne fait varier **que** la structure, pour que
  l'A/B soit interprétable ;
- le `GATE` s'applique toujours par-dessus.

### Plancher sur le niveau — corrigé pendant l'implémentation

LightGBM n'est pas contraint et produit des annuels négatifs ou quasi nuls : 18 sur chauffage
(min −695 kWh/an), 16 sur clim, 45 sur ECS. Le plancher initialement prévu (`1e-3` kWh/h,
soit ~9 kWh/an) écrasait **définitivement** le chauffage de 18 bâtiments dont le réel médian
est 135 kWh/an (max 963) : il aurait fallu une forme soutenue de 15 à 110 pour rattraper.

L'erreur est **asymétrique** — un plancher trop bas est irrécupérable, un plancher trop haut
se corrige, puisque Softplus descend aussi bas qu'on veut.

Retenu : **1 % de la médiane du parc (train), par usage**, soit 45 kWh/an sur le chauffage.
La forme maximale requise retombe de 110 à 21, et les bâtiments planchés sur clim/ECS ont de
toute façon un annuel réel nul et sont gatés.

### Résultat — TESTÉE LE 04/08, REJETÉE

503 bâtiments, graine 42, tout le reste identique :

| | total | chauffage | clim | eau_chaude |
|---|---|---|---|---|
| additive (31/07) | **0.836** | **0.815** | 0.883 | 0.822 |
| multiplicative | 0.822 | 0.788 | **0.885** | **0.835** |

**Pourquoi ça échoue.** J'ai mesuré l'amplitude de forme que la tête exige du réseau
(`forme = conso(t) / moyenne annuelle`) et sa dispersion sur le parc. Le classement des
dégradations suit exactement celui de l'hétérogénéité :

| usage | forme max médiane | p10 | p90 | **p90/p10** | ΔR² |
|---|---|---|---|---|---|
| eau_chaude | 9.8 | 6.7 | 15.6 | 2.3 | **+0.013** |
| clim | 5.6 | 3.8 | 9.3 | 2.5 | +0.002 |
| total | 6.3 | 4.3 | 9.8 | 2.3 | −0.014 |
| chauffage | 11.8 | 4.8 | 65.0 | **13.5** | **−0.027** |

En tête additive, le réseau apprend une relation **locale et physique** — « T_ext = 5 °C,
UA = 360 → ~1 kWh/h » — identique en Floride et dans le Minnesota. En multiplicative, il doit
prédire un *rapport à la moyenne annuelle*, donc connaître le nombre d'heures de chauffe de
l'année, une propriété **globale** du climat :

```
HDD18  <500 (n=57) :  673 h de chauffe/an -> 69 % de l'annuel en 1 semaine, forme p99 28.8
HDD18 >4000 (n=12) : 5229 h de chauffe/an ->  7 % de l'annuel en 1 semaine, forme p99  3.8
```

Or `log(HDD18)` explique **72 %** de cette amplitude (corrélation −0,850) et **HDD18 n'est pas
dans `s`**. Seule `in.weather_file_latitude` y figure, qui n'en explique que 57 % — et
n'apporte rien une fois HDD18 présent (0,725 contre 0,722).

Autrement dit : le problème local a été transformé en problème global sans donner au modèle
l'information qui résout la partie globale. Les deux échecs — la sur-estimation d'avant et la
sous-estimation d'après — ont **la même cause racine : le climat absent de `s`**.

### État — reverté

`TETE = 'additive'` par défaut (cellule de configuration). Vérifié : en mode additif la classe
`LoadNet` est identique **au bit près** au modèle du 31/07 (mêmes paramètres à l'initialisation
— le biais `BIAIS_UN` n'est appliqué qu'en multiplicatif — et mêmes sorties).

La variante multiplicative reste accessible d'une ligne, avec `NIV` / `NIVn` conservés. À
re-tester **une fois le climat dans `s`**, pas avant. Elle a déjà prouvé qu'elle fonctionne là
où la forme est homogène (`eau_chaude` +0,013).

Conservé du passage : le bloc **NMBE par bâtiment** en `c10` et l'affichage du bâtiment tracé.
Ils n'ont jamais touché au modèle et c'est ce qui a rendu le problème lisible.

---

## Partie 2 — Ce qui manque à `s` : le climat et le système de chauffage

**Cible :** cause B, plus la cause révélée par l'échec de la partie 1.
Cellules `c6` (chargement) et `c8` (vecteur statique).

### 2a — Le climat (nouveau, priorité haute)

`weather_static.parquet` existe déjà et se joint par `in.county` : `HDD18`, `CDD18`, `T_moy`,
`T_design_min`, `T_design_max`, `GHI_an`, `GHI_hiver`, `vent_moy`, `RH_moy`.

Aucune de ces colonnes n'est dans `s` aujourd'hui. `log(HDD18)` explique à lui seul 72 % de
l'amplitude de forme du chauffage. C'est l'information qui manquait aux deux versions de la
tête, et elle sert **quelle que soit** la tête retenue : savoir qu'un bâtiment est en Floride
ou dans le Minnesota conditionne le niveau *et* la concentration de son chauffage.

*Point d'attention : 1 bâtiment sans météo statique (AK, Wrangell City and Borough) —
prévoir un repli, à défaut l'exclure explicitement.*

### 2b — Le système de chauffage

Trois colonnes ajoutées à `s` (54 → 57), extraites de `in.hvac_heating_efficiency` :

```python
hspf = eff.str.extract(r'([\d.]+)\s*HSPF')[0].astype(float)   # NaN = résistance
pac  = hspf.notna()                                           # 1 = pompe à chaleur
cop  = np.where(pac, hspf / 3.412, 1.0)                       # HSPF -> COP saisonnier
mshp = eff.str.startswith('MSHP')                             # mini-split, sans gaines
```

Et une entrée temporelle supplémentaire dans `f(t)` (31 → 32) : le COP d'une PAC **chute
avec le froid**, c'est la non-linéarité que le réseau essaie actuellement de deviner sans
savoir de quel équipement il s'agit.

```python
cop_t = np.clip(cop_bat * (0.6 + 0.02 * t_ext), 1.0, cop_bat)   # cop_bat à 20 °C, ~1 vers -10 °C
besoin_elec_chauffage = ecart_chauffage / cop_t
```

`load_building` reçoit alors le `cop_bat` du bâtiment en argument.

**Attendu :** avec 2a, c'est désormais le levier principal sur le R² « chauffage ». Les deux
blocs répondent à des questions différentes que le réseau ne peut pas trancher aujourd'hui :
2a lui dit *combien d'heures par an ce bâtiment chauffe*, 2b *avec quel rendement*.

**Méthode :** mesurer 2a seul, puis 2b seul, puis les deux — jamais empilés d'un coup. C'est
précisément l'empilement qui a rendu la partie 1 illisible au départ.

---

## Partie 3 — Évaluer par bâtiment

**Cible :** la métrique elle-même. Cellule `c10`.

Le R² actuel est calculé sur toutes les heures de tous les bâtiments confondus : il est
dominé par les gros consommateurs et masque complètement le biais de niveau visible à l'œil.

Pour un simulateur, les critères ASHRAE Guideline 14 (pas horaire) sont
`|NMBE| ≤ 10 %` et `CV(RMSE) ≤ 30 %`, calculés **par bâtiment** :

```python
NMBE   = (p.sum() - t.sum()) / t.sum() * 100
CVRMSE = sqrt(((p - t)**2).mean()) / t.mean() * 100
```

Rapporter les déciles par usage, plus un nuage « annuel prédit vs annuel réel » par bâtiment.

Et tracer **3 bâtiments** — médian, pire NMBE, meilleur — au lieu de `w = 0`, qui s'est
avéré être un cas extrême choisi par hasard.

---

## Leviers secondaires (par rapport gain/effort)

| Levier | Détail |
|---|---|
| Fenêtres glissantes | `stride = 24` au lieu de 168 → 359 fenêtres/bâtiment au lieu de 52 (~7× plus de données) et récupère les 24 h jetées par `len(f) // L`. Déjà fait dans `timeseries_conv`. |
| Plus de bâtiments | 503 pour un `s` de 57 colonnes reste peu. La note c8 rappelle que passer de 153 à 503 a inversé le signe de l'apport des 48 features. |
| FiLM | `s` produit un couple (γ, β) qui module les sorties du GRU (`h * γ + β`) au lieu d'être concaténé. Version générique de la partie 1 : toutes les interactions statique×temps deviennent représentables. |
| Cohérence physique | Rien ne garantit `total ≥ chauffage + clim + eau_chaude`. Sur le 530669, 57 % du total n'est pas décomposé (appareils, éclairage) → prédire ce talon séparément et sommer serait plus contraint que 4 sorties indépendantes. |
| Loss et optimisation | Huber au lieu de MSE (chauffage zéro-inflaté et à pics), `ReduceLROnPlateau`, 2 couches GRU + dropout, gradient clipping. |

---

## À vérifier avant de conclure quoi que ce soit

`static_preds.parquet` couvre les 55 732 bâtiments avec un R² de **0,949** sur `total`,
alors que le split test de `lgbm_consommation_annuelle.ipynb` tourne autour de 0,85
(référence citée en c15 : 0,8446).

L'écart suggère que ~80 % de ces prédictions sont **in-sample** : les bâtiments de
validation du réseau recevraient dans `s` un niveau annuel bien meilleur que celui qu'ils
auraient sur un vrai bâtiment inconnu. Le R² de validation serait donc optimiste — et il
le devient d'autant plus avec la tête multiplicative de la partie 1, qui met tout son
poids sur ce niveau.

Correctif : régénérer `static_preds` avec `cross_val_predict` (KFold), pour que chaque
bâtiment ait une prédiction hors échantillon.

Statut : **présomption forte, pas une certitude** — aucun commit du dépôt ne contient le
code qui a produit ce fichier, il n'a donc pas pu être relu.

---

## Ordre d'exécution

1. ~~**Partie 1** — tête multiplicative~~ — *testée le 04/08, **rejetée**, revertée
   (`TETE = 'additive'`). Retour à la référence 0.836 / 0.815.*
2. ~~**Partie 0** — hygiène des données~~ — *faite le 05/08 : arbitrage des consignes sur 108
   bâtiments + retrait des 6 colonnes de consigne de `s`.*
3. ~~Le COP / système de chauffage~~ — *fait le 21/08, **+0.059** sur le chauffage*
4. ~~Le produit enveloppe `(UA + H_ve) × ecart` dans `f(t)`~~ — *testé le 21/08, **+0.002**,
   écarté : le gain annoncé de +0.073 était mesuré sur régression linéaire poolée, où le
   produit remplace une interaction que le réseau construit déjà tout seul*
5. ~~Le climat (`HDD18`) dans `s`~~ — *testé le 21/08, **+0.008**, écarté seul*
6. ~~Vérification `static_preds` hors échantillon~~ — *faite le 21/08 : la présomption était
   juste, fichier régénéré en KFold*
7. ~~Leviers secondaires~~ — *tous testés le 21/08, voir le tableau ci-dessous*
8. **Reste : Partie 3**, l'évaluation par bâtiment dans le notebook (NMBE et CV(RMSE) sont
   dans le banc `experiments/`, pas encore reportés en `c10`), et le re-test de la tête
   multiplicative maintenant que `s` contient le système de chauffage.

---

## Campagne du 21/08 — 15 configurations

Banc d'essai `experiments/` (cache `.npy`, une expérience par ligne de commande, résultat
JSON). Même graine, **mêmes 101 bâtiments de validation** dans tous les cas, y compris pour
les runs à 2005 bâtiments — sinon agrandir le parc changerait aussi le jeu de validation.

### Ce qui marche

| levier | total | chauffage | clim | eau_chaude |
|---|---|---|---|---|
| référence (503 bât., `s` à 48 colonnes) | 0.828 | 0.805 | 0.883 | 0.831 |
| `pac`, `cop`, `mshp` dans `s` | +0.021 | **+0.059** | +0.010 | +0.002 |
| ... plus `besoin_elec_chauffage` dans `f(t)` | +0.020 | **+0.068** | +0.013 | +0.002 |
| ... plus niveaux annuels hors échantillon | +0.024 | +0.077 | +0.004 | +0.006 |
| ... plus **parc élargi à 2005 bâtiments** | **+0.067** | **+0.109** | **+0.040** | **+0.021** |

**Résultat : 0.895 / 0.914 / 0.923 / 0.852.** NMBE médian par bâtiment sur le chauffage :
29.8 % → 15.9 %. CV(RMSE) médian : total 23.8 % (sous le seuil ASHRAE Guideline 14 de 30 %),
chauffage 59.5 % — encore loin du compte.

### Ce qui ne marche pas

| levier | Δ total | pourquoi |
|---|---|---|
| FiLM | −0.008 | la modulation `h·γ + β` n'aide pas là où la concaténation suffisait |
| GRU 2 couches + dropout | −0.009 | 2× le temps, aucun gain : le plafond n'est pas la capacité |
| perte de Huber + `ReduceLROnPlateau` | −0.002 | |
| `log1p` sur la cible (LightGBM) | −0.015 sur le chauffage | déplace l'erreur vers les gros consommateurs |
| fenêtres glissantes `stride=24` | +0.000 | 7× plus de fenêtres, mais les **mêmes** bâtiments |
| produit enveloppe `(UA+H_ve) × ecart` | +0.002 | |
| climat (`HDD18`, `CDD18`, …) dans `s` | +0.008 | utile à LightGBM, redondant pour le réseau |

### La leçon principale

**Le levier dominant est le nombre de bâtiments, pas le modèle.** À configuration identique,
passer de 503 à 2005 bâtiments rapporte +0.043 sur le total et +0.032 sur le chauffage — plus
que toutes les variantes d'architecture réunies, qui sont toutes négatives. Et le contraste
avec `stride=24` est net : 7× plus de fenêtres sur le **même** parc ne donne rien, 4× plus de
**bâtiments** donne tout. Ce qui manquait au réseau n'était ni de la capacité, ni des heures,
c'était de la **diversité de bâtiments** — cohérent avec le ratio bâtiments/colonnes de `s`
suivi depuis le début, passé de 8.4 à 37.

Le parc reste extensible : 55 732 bâtiments passent le filtre, 2005 sont téléchargés.

## Ce que l'échec de la partie 1 a appris

1. **Le R² pooled ne suffisait pas à voir le problème.** C'est le bloc NMBE et le graphe
   d'un bâtiment nommé qui ont montré ce qui se passait. Instrumenter d'abord, modifier ensuite.
2. **Ajouter de la structure à un modèle ne remplace pas l'information manquante.** La tête
   multiplicative encodait une vraie loi physique (`conso = niveau × forme`), et elle a quand
   même dégradé — parce qu'elle déplaçait la difficulté vers une information absente de `s`.
3. **Vérifier qu'un revert est un vrai revert.** Le biais `BIAIS_UN` initialisé
   inconditionnellement aurait laissé un modèle différent de l'original sous une étiquette
   « additive ». Comparaison au bit près avant de conclure.
