"""Figures du gisement : chauffage et climatisation cote a cote.

Deux lectures :
  1. les PROFILS compares — c'est la figure qui montre que le REPORT n'a rien a voir
     d'un usage a l'autre : le chauffage ne rattrape rien, la climatisation rattrape
     pres de la moitie ;
  2. le RECAPITULATIF — pointe et report par scenario, qui designe le pre-refroidissement
     comme la seule strategie d'ete tenable.

L'eau chaude est absente : voir l'en-tete de flex_usages.py.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figures_soutenance import (BLEU_FONCE, BLEU_CLAIR, ORANGE, VERT, GRIS,
                                ENCRE, ENCRE2, sauver, FIG)

RES = Path(__file__).resolve().parent / 'results'


def charger():
    ch = json.loads((RES / 'flex_chauffage.json').read_text(encoding='utf-8'))
    us = json.loads((RES / 'flex_usages.json').read_text(encoding='utf-8'))
    return ch, us


def taux_report(bloc):
    d = pd.DataFrame(bloc['batiments'])
    return 100 * d['rebond_kWh_chauffage'].sum() / max(d['efface_kWh_chauffage'].sum(), 1e-9)


def profils_deux_usages(ch, us):
    blocs = [
        ('Chauffage — hiver, −2 °C de 18 à 21 h', ch['scenarios']['soir'], (18, 21)),
        ("Climatisation — été, +2 °C de 15 à 19 h", us['clim']['pointe_ete'], (15, 19)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6), sharey=True)
    for ax, (titre, bloc, (h0, h1)) in zip(axes, blocs):
        prof = np.array([b['profil_chauffage'] for b in bloc['batiments']]).mean(0)
        h = np.arange(24)
        ax.bar(h[prof >= 0], prof[prof >= 0], color=BLEU_FONCE, width=.82, zorder=3)
        ax.bar(h[prof < 0], prof[prof < 0], color=ORANGE, width=.82, zorder=3)
        ax.axvspan(h0 - .5, h1 - .5, color=BLEU_CLAIR, alpha=.55, zorder=1)
        ax.axhline(0, color=ENCRE2, lw=1, zorder=4)
        rep = taux_report(bloc)
        ax.set_title(titre, fontsize=10, color=ENCRE, pad=6)
        ax.text(.02, .05, f'report {rep:+.0f} %', transform=ax.transAxes, fontsize=11,
                fontweight='bold', color=ORANGE if rep > 20 else VERT)
        ax.set_xlabel('heure locale'); ax.set_xticks(range(0, 24, 6))
        ax.set_xlim(-.7, 23.7)
        ax.grid(axis='y'); ax.grid(axis='x', visible=False)
    axes[0].set_ylabel('effacement moyen\nkWh/h par logement')
    fig.text(.5, -.07, "barres bleues : puissance effacée   ·   barres orange : report   "
             "·   zone teintée : fenêtre d'action",
             ha='center', fontsize=8.5, color=ENCRE2)
    fig.subplots_adjust(wspace=.10)
    sauver(fig, 'fig_flex_deux_usages')


def recapitulatif(ch, us):
    lignes = [
        ('Chauffage', 'hiver, −2 °C, 18–21 h', ch['scenarios']['soir']),
        ('Chauffage', 'hiver, −2 °C, 7–10 h', ch['scenarios']['matin']),
        ('Chauffage', 'hiver, −3 °C, 0–6 h', ch['scenarios']['nuit']),
        ('Climatisation', 'été, +2 °C, 15–19 h', us['clim']['pointe_ete']),
        ('Climatisation', 'été, +2 °C, 18–21 h', us['clim']['soir_ete']),
        ('Climatisation', 'pré-refroidissement', us['clim']['prerefroid']),
    ]
    tab = []
    for usage, scen, bloc in lignes:
        d = pd.DataFrame(bloc['batiments'])
        tab.append({'usage': usage, 'scénario': scen,
                    'pointe': d['pointe_kW_total'].median(),
                    'énergie': d['efface_kWh_chauffage'].median(),
                    'report': taux_report(bloc)})
    t = pd.DataFrame(tab)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8),
                             gridspec_kw=dict(width_ratios=[1.2, 1]))
    y = np.arange(len(t))[::-1]
    coul = [BLEU_FONCE if u == 'Chauffage' else '#2a78d6' for u in t['usage']]

    ax = axes[0]
    ax.barh(y, t['pointe'], color=coul, height=.62, zorder=3)
    for k, v in enumerate(t['pointe']):
        ax.text(v + .05, y[k], f'{v:.2f}', va='center', fontsize=9, color=ENCRE)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['usage']}\n{r['scénario']}" for _, r in t.iterrows()],
                       fontsize=8.5)
    ax.set_xlabel('pointe effacée médiane (kW)')
    ax.grid(axis='x'); ax.grid(axis='y', visible=False)
    ax.set_title('Puissance mobilisable', fontsize=10.5, color=ENCRE, pad=6)

    ax = axes[1]
    rep = t['report'].values
    ax.barh(y, rep, color=[ORANGE if r > 20 else VERT for r in rep], height=.62, zorder=3)
    for k, v in enumerate(rep):
        ax.text(v + (1.6 if v >= 0 else -1.6), y[k], f'{v:+.0f} %', va='center',
                ha='left' if v >= 0 else 'right', fontsize=9, color=ENCRE)
    ax.axvline(0, color=ENCRE2, lw=1)
    ax.set_yticks(y); ax.set_yticklabels([])
    ax.set_xlabel("part de l'énergie rattrapée après la fenêtre (%)")
    ax.grid(axis='x'); ax.grid(axis='y', visible=False)
    ax.set_title('Report', fontsize=10.5, color=ENCRE, pad=6)

    fig.subplots_adjust(wspace=.06)
    sauver(fig, 'fig_flex_recap')
    print(t.round(2).to_string(index=False))


if __name__ == '__main__':
    ch, us = charger()
    print('figures ->', FIG)
    profils_deux_usages(ch, us)
    recapitulatif(ch, us)
