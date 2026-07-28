# Consignes de thermostat, séries temporelles et température intérieure

*Récupération des consignes non-stochastiques absentes des séries temporelles ResStock, et validation de leur effet sur la température et la consommation de climatisation. Illustration sur le bâtiment 57826 (Iowa).*

---

## 1. Le constat

*Les consignes de thermostat sont présentes dans le fichier mais entièrement vides.*

Dans la release ResStock 2025, les colonnes de consigne du time series existent mais ne contiennent aucune valeur. Le fichier détaillé `in.schedules.csv` ne comporte, lui, aucune colonne de consigne.

| Colonne du time series | Valeurs non nulles |
|---|---|
| `out.schedules.cooling_setpoint..c` | **0 / 35 040** |
| `out.schedules.heating_setpoint..c` | **0 / 35 040** |
| `out.indoor_temperature.conditioned_space..c` | 35 040 / 35 040 |

**À retenir :** la consigne horaire, qui pilote le chauffage et la climatisation, est introuvable telle quelle dans les données.

---

## 2. Ce que dit le guide

*Les consignes sont des schedules non-stochastiques, déterministes et tabulés.*

Le guide technique (Table 2, §8.4.5) classe les consignes comme schedules **non-stochastiques** : contrairement à l'occupation ou aux appareils, elles ne sont pas générées aléatoirement mais définies par des **profils 24 h déterministes** dans `options_lookup.tsv`. Chaque consigne dépend de 4 caractéristiques statiques : la **base**, la présence d'un **offset**, sa **magnitude** et sa **période**.

**À retenir :** la donnée n'est pas perdue — elle est entièrement redéductible du statique.

---

## 3. La reconstruction

*Une formule simple, un masque horaire signé, un piège d'unité à éviter.*

La consigne à l'heure `h` s'écrit :

$$\text{consigne}(h) = \text{base} + \text{masque}(h) \times \text{magnitude}$$

Le **masque** vient de `options_lookup.tsv` : un vecteur de 24 valeurs signées (`+1` = *setup*, consigne relevée ; `−1` = *setback*, consigne abaissée ; `0` = base), **différent entre semaine et week-end**.

Profil clim reconstruit pour le bâtiment 57826 (base 76 °F, offset 9 °F, « Day Setup and Night Setback ») :

| Période | Masque | Consigne |
|---|---|---|
| Nuit (0–6 h, 22–23 h) | −1 | **19.4 °C** |
| Transition | 0 | 24.4 °C |
| Jour (9–16 h, semaine) | +1 | **29.4 °C** |

> **Piège °F → °C.** Une consigne est une température *absolue* → `(F − 32) × 5/9`. Une magnitude d'offset est un *écart* → `F × 5/9` **sans** le −32. Appliquer la mauvaise formule transforme un offset de `0F` en `−17.8 °C` au lieu de `0 °C`.

**À retenir :** `base + masque × magnitude`, avec les bonnes conversions et la distinction semaine / week-end.

---

## 4. Validation sur la température

*La température intérieure suit la consigne reconstruite quand le système est actif.*

![Semaine d'été : consigne clim reconstruite et température intérieure](figures_setpoints/fig_semaine.png)

Sur une semaine d'été, la température intérieure (rouge) est tenue au plancher nocturne de 19.4 °C par la climatisation, puis flotte vers le haut le jour lorsque la consigne monte à 29.4 °C. Le week-end (21–22/07), l'absence de *setup* diurne est bien visible : la consigne reste à 24.4 °C. La corrélation consigne ↔ température sur la saison de clim (mai–sept) est de **r = +0.62**.

**À retenir :** la reconstruction est correcte — la température réelle réagit au profil injecté.

---

## 5. Deux nuances physiques

*Les deux consignes forment une bande morte ; laquelle agit dépend de la saison.*

![Vue annuelle : température encadrée par les deux consignes](figures_setpoints/fig_saison.png)

Les consignes ResStock sont **constantes sur l'année** (seule variation : jour / nuit). Un thermostat à double consigne fait **flotter** la température entre le plancher (chauffage) et le plafond (climatisation) ; **la borne active change avec la saison** : le chauffage tient le plancher en hiver, la clim tient le plafond en été.

Surtout, la consigne est une **borne, pas une valeur** : la température ne l'égale que lorsque le système lutte contre elle.

| Consigne clim | Température moyenne | Écart | Clim active |
|---|---|---|---|
| 19.4 °C (nuit) | 19.69 °C | **+0.25** | 92 % |
| 24.4 °C (base) | 22.92 °C | −1.52 | 48 % |
| 29.4 °C (jour) | 22.94 °C | **−6.50** | 0 % |

**À retenir :** quand la consigne mord, la température s'y colle (à la bande morte près, +0.25 °C) ; sinon elle flotte librement, très loin en dessous.

---

## 6. Consommation et écart moteur

*Ce qui déclenche et dose la climatisation, c'est l'écart entre l'extérieur et la consigne.*

![Consommation clim et déclenchement en fonction de l'écart](figures_setpoints/fig_conso.png)

On définit l'**écart moteur** = `T_extérieure − consigne_clim`. À gauche, la consommation de clim croît de façon quasi-monotone avec cet écart ; à droite, la clim se déclenche lorsque l'écart franchit ~0 °C. La corrélation écart ↔ consommation sur la saison de clim est de **r = +0.60**.

**À retenir :** la consigne fixe le seuil à partir duquel la météo coûte de l'énergie. C'est l'écart, pas la consigne brute, qui porte le signal physique.

---

## 7. Implication pour la modélisation

*La consigne reconstruite est une feature d'entrée légitime, à combiner avec la météo.*

La chaîne de causalité est **consigne (entrée, échantillonnée depuis l'enquête RECS) → simulation → température et consommation (sorties)**. Utiliser la consigne reconstruite comme variable explicative n'introduit **aucune circularité** : elle est connue avant la simulation, et l'écart résiduel non nul entre consigne et température le confirme.

**À retenir :** retenir comme feature l'**écart `T_extérieure − consigne`** (et son symétrique `consigne − T_extérieure` pour le chauffage), plutôt que la consigne seule.
