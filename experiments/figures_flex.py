"""Figures du gisement de flexibilite du chauffage.

Quatre lectures complementaires :
  1. le PROFIL moyen d'effacement sur 24 h — la signature d'un effacement, creux
     pendant l'evenement puis bosse de rebond ;
  2. les COURBES DE REPONSE — effacement en fonction de l'amplitude et de la duree,
     qui repondent a « jusqu'ou peut-on pousser ? » ;
  3. la SEGMENTATION du parc — quels logements recruter en priorite ;
  4. la DISTRIBUTION par batiment — un gisement median cache une forte dispersion.

Palette : les deux memes slots categoriels que les courbes reel/predit, plus un gris
de contexte. Le remplissage distingue effacement et report, doublement encode par le
signe et par la couleur.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures import REEL, PREDIT, CONTEXTE, ENCRE, ENCRE2, GRILLE, ACCENT, FIG, sauver

RES = Path(__file__).resolve().parent / 'results'
EFFACE, REPORT = '#2a78d6', '#eb6834'      # bleu = efface, orange = report


def charger(nom='flex_chauffage.json'):
    return json.loads((RES / nom).read_text(encoding='utf-8'))


def df(lignes):
    return pd.DataFrame(lignes)


# ------------------------------------------------------------------ 1. profils
def figure_profils(d):
    noms = ['soir', 'matin', 'nuit']
    fenetres = {'soir': (18, 21), 'matin': (7, 10), 'nuit': (0, 6)}
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), sharey=True)

    for ax, nom in zip(axes, noms):
        bloc = d['scenarios'][nom]
        prof = np.array([b['profil_chauffage'] for b in bloc['batiments']]).mean(0)
        h = np.arange(24)
        ax.bar(h[prof >= 0], prof[prof >= 0], color=EFFACE, width=.82, zorder=3)
        ax.bar(h[prof < 0], prof[prof < 0], color=REPORT, width=.82, zorder=3)
        h0, h1 = fenetres[nom]
        ax.axvspan(h0 - .5, h1 - .5, color=ACCENT, alpha=.09, zorder=1)
        ax.axhline(0, color=ENCRE2, lw=.9, zorder=4)
        ax.set_title(bloc['libelle'].split(',')[0], fontsize=9, color=ENCRE, pad=6)
        ax.set_xlabel('heure locale')
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xlim(-.7, 23.7)
        ax.grid(axis='y'); ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('effacement moyen\nkWh/h par logement')
    fig.text(.5, -.10, 'barres bleues : puissance effacée   ·   barres orange : report après l\'événement'
             '   ·   zone teintée : fenêtre d\'action',
             ha='center', fontsize=8, color=ENCRE2)
    fig.subplots_adjust(wspace=.12)
    sauver(fig, 'fig_flex_profils')


# ------------------------------------------------- 2. courbes de reponse
def figure_reponse(d):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.1))

    abs_f = lambda k: abs(float(k))        # 1, 2, 3, 4 et non -4, -3, -2, -1
    amps = sorted(d['amplitudes'], key=abs_f)
    x = [abs(float(a)) for a in amps]
    for ax, (cles, xs, xlabel, titre) in zip(axes, [
            (amps, x, 'amplitude du décalage de consigne (°C)',
             'Jusqu\'où pousser le décalage ?'),
            (sorted(d['durees'], key=float), [float(k) for k in sorted(d['durees'], key=float)],
             'durée de l\'effacement (h)', 'Jusqu\'où allonger la fenêtre ?')]):
        source = d['amplitudes'] if ax is axes[0] else d['durees']
        med, q1, q3, pointe = [], [], [], []
        for c in cles:
            e = df(source[c])['efface_kWh_chauffage']
            med.append(e.median()); q1.append(e.quantile(.25)); q3.append(e.quantile(.75))
            pointe.append(df(source[c])['pointe_kW_total'].median())
        ax.fill_between(xs, q1, q3, color=EFFACE, alpha=.16, linewidth=0, zorder=2)
        ax.plot(xs, med, 'o-', color=EFFACE, lw=2, ms=6, mec='white', mew=1.5,
                zorder=4, label='énergie effacée (médiane)')
        ax.set_xlabel(xlabel); ax.set_ylabel('énergie effacée sur l\'hiver\nkWh par logement')
        ax.set_title(titre, fontsize=9.5, color=ENCRE, pad=7)
        ax.set_xticks(xs); ax.grid(axis='y'); ax.grid(axis='x', visible=False)
        ax.set_ylim(bottom=0)
        for xi, m in zip(xs, med):
            ax.annotate(f'{m:.0f}', (xi, m), textcoords='offset points', xytext=(0, 9),
                        ha='center', fontsize=7.6, color=ENCRE2)
    axes[0].legend(loc='lower right', fontsize=8.2)
    fig.text(.5, -.07, 'bande claire : intervalle interquartile du parc',
             ha='center', fontsize=8, color=ENCRE2)
    fig.subplots_adjust(wspace=.30)
    sauver(fig, 'fig_flex_reponse')


# ------------------------------------------------------- 3. segmentation
def figure_segmentation(d):
    b = df(d['scenarios']['soir']['batiments'])
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.0))

    # a. type de chauffage
    ax = axes[0]
    grp = [b.loc[~b['pac'], 'pointe_kW_total'], b.loc[b['pac'], 'pointe_kW_total']]
    bp = ax.boxplot(grp, tick_labels=['résistance', 'PAC'], widths=.55,
                    patch_artist=True, medianprops=dict(color=ENCRE, lw=1.6),
                    flierprops=dict(marker='.', ms=3, mfc=CONTEXTE, mec='none'))
    for p, c in zip(bp['boxes'], [CONTEXTE, EFFACE]):
        p.set(facecolor=c, alpha=.45, edgecolor=c, lw=1.2)
    ax.set_ylabel('pointe effacée (kW)')
    ax.set_title('Par système de chauffage', fontsize=9.5, color=ENCRE, pad=7)
    ax.grid(axis='y'); ax.grid(axis='x', visible=False)

    # b. zone climatique
    ax = axes[1]
    z = b.groupby('zone')['pointe_kW_total'].agg(['median', 'count'])
    z = z[z['count'] >= 3].sort_values('median')
    ax.barh(range(len(z)), z['median'], color=EFFACE, height=.66, zorder=3)
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels([f'{i}  (n={int(n)})' for i, n in zip(z.index, z['count'])], fontsize=8)
    ax.set_xlabel('pointe effacée médiane (kW)')
    ax.set_title('Par zone climatique', fontsize=9.5, color=ENCRE, pad=7)
    ax.grid(axis='x'); ax.grid(axis='y', visible=False)

    # c. inertie
    ax = axes[2]
    ax.scatter(b['tau'], b['pointe_kW_total'], s=16, color=EFFACE, alpha=.55,
               edgecolors='none', zorder=3)
    r = b['tau'].corr(b['pointe_kW_total'], method='spearman')
    ax.set_xlabel('constante de temps τ (h)')
    ax.set_ylabel('pointe effacée (kW)')
    ax.set_title(f'Par inertie · Spearman {r:+.2f}', fontsize=9.5, color=ENCRE, pad=7)
    ax.grid(True)
    fig.subplots_adjust(wspace=.42)
    sauver(fig, 'fig_flex_segmentation')


# -------------------------------------------------------- 4. distribution
def figure_distribution(d):
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.0))
    b = df(d['scenarios']['soir']['batiments'])

    ax = axes[0]
    v = b['pointe_kW_total'].values
    ax.hist(v, bins=22, color=EFFACE, alpha=.75, zorder=3)
    for q, style in [(np.median(v), '-'), (np.percentile(v, 90), ':')]:
        ax.axvline(q, color=ENCRE, lw=1.3, ls=style, zorder=4)
    ax.text(np.median(v), ax.get_ylim()[1] * .96, f'  médiane {np.median(v):.2f} kW',
            fontsize=8, color=ENCRE, va='top')
    ax.text(np.percentile(v, 90), ax.get_ylim()[1] * .80,
            f'  p90 {np.percentile(v, 90):.2f} kW', fontsize=8, color=ENCRE2, va='top')
    ax.set_xlabel('pointe effacée par logement (kW)'); ax.set_ylabel('nombre de logements')
    ax.set_title('Dispersion du gisement — soir, −2 °C', fontsize=9.5, color=ENCRE, pad=7)
    ax.grid(axis='y'); ax.grid(axis='x', visible=False)

    # courbe de concentration : quelle part du gisement pour quelle part du parc
    ax = axes[1]
    tri = np.sort(v)[::-1]
    part_parc = np.arange(1, len(tri) + 1) / len(tri) * 100
    part_gis = np.cumsum(tri) / tri.sum() * 100
    ax.plot(part_parc, part_gis, color=EFFACE, lw=2, zorder=4)
    ax.plot([0, 100], [0, 100], color=CONTEXTE, lw=1, ls='--', zorder=3,
            label='parc homogène')
    i30 = np.searchsorted(part_parc, 30)
    ax.annotate(f'30 % des logements\n→ {part_gis[i30]:.0f} % du gisement',
                (30, part_gis[i30]), textcoords='offset points', xytext=(12, -30),
                fontsize=8, color=ENCRE, arrowprops=dict(arrowstyle='->', color=ENCRE2, lw=.9))
    ax.set_xlabel('part du parc, du plus effaçable au moins (%)')
    ax.set_ylabel('part du gisement total (%)')
    ax.set_title('Concentration du gisement', fontsize=9.5, color=ENCRE, pad=7)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.grid(True)
    ax.legend(loc='lower right', fontsize=8.2)
    fig.subplots_adjust(wspace=.32)
    sauver(fig, 'fig_flex_distribution')


if __name__ == '__main__':
    d = charger()
    print('figures ->', FIG)
    figure_profils(d)
    figure_reponse(d)
    figure_segmentation(d)
    figure_distribution(d)
