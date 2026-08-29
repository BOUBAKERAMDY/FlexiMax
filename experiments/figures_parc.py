"""Deux figures de description du parc, pour la soutenance.

  parc_metadonnees — ce que sont les logements : climat, enveloppe, equipement.
  parc_timeseries  — ce que sont leurs courbes : rythme journalier, saison,
                     thermosensibilite.

Les deux repondent a la question qu'un jury pose en premier : « sur quoi as-tu
travaille exactement ? ».
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ts_data as D
from figures_soutenance import (BLEU_FONCE, BLEU_CLAIR, ORANGE, VERT, GRIS,
                                ENCRE, ENCRE2, sauver, FIG)

USAGE = {'total': BLEU_FONCE, 'chauffage': '#eb6834',
         'clim': '#2a78d6', 'eau_chaude': '#1baf7a'}
LIB = {'total': 'total', 'chauffage': 'chauffage', 'clim': 'climatisation',
       'eau_chaude': 'eau chaude'}


def charger():
    ids = pd.read_csv(D.DATA / 'nn_buildings.csv')['bldg_id'].values
    cols = ['bldg_id', 'in.state', 'in.county', 'in.ashrae_iecc_climate_zone_2004',
            'in.vintage', 'in.sqft..ft2', 'in.hvac_cooling_type',
            'in.water_heater_fuel', 'in.hvac_heating_type']
    meta = pd.read_parquet(D.DATA / 'metadata_clean.parquet',
                           columns=cols).set_index('bldg_id').loc[ids]
    wx = pd.read_parquet(D.DATA / 'weather_static.parquet').set_index('in.county')
    meta = meta.join(wx, on='in.county')
    return ids, meta


def parc_metadonnees():
    ids, meta = charger()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.6))

    ax = axes[0]
    z = meta['in.ashrae_iecc_climate_zone_2004'].value_counts().sort_index()
    ax.bar(range(len(z)), z.values, color=BLEU_FONCE, width=.68, zorder=3)
    ax.set_xticks(range(len(z)))
    ax.set_xticklabels(z.index, fontsize=8.5, rotation=45, ha='right')
    ax.set(xlabel='zone climatique ASHRAE', ylabel='bâtiments')
    ax.set_title(f'{len(ids)} bâtiments, {z.size} zones climatiques',
                 fontsize=10, color=ENCRE, pad=6)
    ax.grid(axis='y'); ax.grid(axis='x', visible=False)

    ax = axes[1]
    pac = meta['in.hvac_heating_type'].str.contains('Heat Pump').fillna(False).values
    for sel, lib, c in [(~pac, 'résistance', ORANGE), (pac, 'pompe à chaleur', BLEU_FONCE)]:
        ax.scatter(meta['HDD18'][sel], meta['CDD18'][sel], s=16, alpha=.55,
                   color=c, edgecolors='none', label=lib, zorder=3)
    ax.set(xlabel='degrés-jours de chauffe (HDD18)',
           ylabel='degrés-jours de froid (CDD18)')
    ax.set_title('Du désert au climat rigoureux', fontsize=10, color=ENCRE, pad=6)
    ax.legend(fontsize=8.5)

    ax = axes[2]
    parts = {
        'pompe à chaleur': pac.mean(),
        'climatisation': (meta['in.hvac_cooling_type'] != 'None').mean(),
        'eau chaude électrique': (meta['in.water_heater_fuel'] == 'Electricity').mean(),
    }
    y = np.arange(len(parts))[::-1]
    ax.barh(y, [100 * v for v in parts.values()], color=BLEU_FONCE, height=.55, zorder=3)
    for k, v in enumerate(parts.values()):
        ax.text(100 * v + 1.5, y[k], f'{100*v:.0f} %', va='center', fontsize=10,
                color=ENCRE, fontweight='bold')
    ax.set_yticks(y); ax.set_yticklabels(parts.keys(), fontsize=9)
    ax.set(xlim=(0, 112), xlabel='part du parc (%)')
    ax.set_title("L'équipement décide quelles cibles existent",
                 fontsize=10, color=ENCRE, pad=6)
    ax.grid(axis='x'); ax.grid(axis='y', visible=False)

    fig.subplots_adjust(wspace=.46)
    sauver(fig, 'parc_metadonnees')
    print(f'    {len(ids)} bâtiments | {100*pac.mean():.0f} % de PAC | '
          f"{meta['HDD18'].min():.0f} à {meta['HDD18'].max():.0f} HDD18")


def parc_timeseries():
    ids, meta = charger()
    X, Y, ids_c = D.build_cache('503', D.parc())
    Y = np.asarray(Y)
    T = np.asarray(X)[:, :, D.I_TEXT]

    idx = pd.date_range('2018-01-01', periods=D.N_H, freq='1h')
    heure, mois = idx.hour.values, idx.month.values
    hiver = np.isin(mois, (12, 1, 2))
    ete = np.isin(mois, (6, 7, 8))

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 3.6))

    ax = axes[0]
    for sel, lib, c, st in [(hiver, 'hiver', BLEU_FONCE, '-'), (ete, 'été', ORANGE, '--')]:
        prof = [Y[:, sel & (heure == h), 0].mean() for h in range(24)]
        ax.plot(range(24), prof, color=c, lw=2.2, ls=st, label=lib)
    ax.set(xlabel='heure de la journée', ylabel='kWh/h par logement', xlim=(0, 23))
    ax.set_xticks(range(0, 24, 6))
    ax.set_title('Le rythme journalier s\'inverse', fontsize=10, color=ENCRE, pad=6)
    ax.legend(fontsize=9)

    ax = axes[1]
    x = np.arange(1, 13)
    larg = .26
    for k, u in enumerate(['chauffage', 'clim', 'eau_chaude']):
        i = list(USAGE).index(u)
        m = [Y[:, mois == mo, i].mean() for mo in range(1, 13)]
        ax.bar(x + (k - 1) * larg, m, width=larg * .9, color=USAGE[u], label=LIB[u],
               zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(list('JFMAMJJASOND'))
    ax.set(xlabel='mois', ylabel='kWh/h par logement')
    ax.set_title('Deux saisons, deux usages', fontsize=10, color=ENCRE, pad=6)
    ax.legend(fontsize=8.5)
    ax.grid(axis='y'); ax.grid(axis='x', visible=False)

    ax = axes[2]
    t_parc = T.mean(0)
    bornes = [-100, 0, 5, 10, 15, 20, 25, 100]
    etiq = ['< 0', '0–5', '5–10', '10–15', '15–20', '20–25', '> 25']
    cat = np.digitize(t_parc, bornes[1:-1])
    xs = np.arange(len(etiq))
    for u in ['total', 'chauffage', 'clim']:
        i = list(USAGE).index(u)
        ax.plot(xs, [Y[:, cat == k, i].mean() for k in range(len(etiq))],
                'o-', color=USAGE[u], lw=2, ms=5, label=LIB[u])
    ax.set_xticks(xs); ax.set_xticklabels(etiq, fontsize=8.5)
    ax.set(xlabel='température extérieure (°C)', ylabel='kWh/h par logement')
    ax.set_title('La signature en V du parc', fontsize=10, color=ENCRE, pad=6)
    ax.legend(fontsize=8.5)

    fig.subplots_adjust(wspace=.30)
    sauver(fig, 'parc_timeseries')

    pointe = np.array([Y[:, heure == h, 0].mean() for h in range(24)])
    print(f'    pointe journalière moyenne {pointe.max():.2f} kWh/h à {pointe.argmax()} h')


if __name__ == '__main__':
    print('figures ->', FIG)
    parc_metadonnees()
    parc_timeseries()
