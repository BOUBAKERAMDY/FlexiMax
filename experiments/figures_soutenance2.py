"""Deux figures supplementaires pour la soutenance.

1. flex_avant_apres — la courbe de reference et la courbe contrefactuelle d'UN batiment
   sur une semaine. C'est l'image qui rend concret « mesurer par difference » : on voit
   le creux pendant l'evenement et la bosse de report apres. Un profil moyen sur 24 h ne
   montre ni l'un ni l'autre a l'echelle d'une vraie journee.

2. fuseau_horaire — l'heure du pic solaire en fonction de la longitude, sur tout le parc.
   Preuve visuelle que les series ResStock sont horodatees dans un fuseau UNIQUE et non
   en heure locale : si elles l'etaient, le nuage serait horizontal.
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


def flex_avant_apres(bldg=None, debut='2018-01-15'):
    from flex_chauffage import Banc, SEQ

    banc = Banc()
    if bldg is None:                       # un batiment qui chauffe vraiment
        cand = [(k, banc.Y[k, :, 1].sum()) for k in banc.val]
        cand.sort(key=lambda x: -x[1])
        k = cand[len(cand) // 6][0]        # gros consommateur, sans etre l'extreme
    else:
        k = int(np.flatnonzero(banc.ids == bldg)[0])
    bid = int(banc.ids[k])

    m_evt = banc.masque_heures(k, (18, 19, 20))
    ref, sc = banc.scenario(k, [(m_evt, -2.0)])
    ref, sc = ref.reshape(-1, 4), sc.reshape(-1, 4)
    h_abs = np.concatenate([np.arange(d, d + SEQ) for d in banc.f_hiver])

    idx = pd.date_range('2018-01-01', periods=D.N_H, freq='1h')[h_abs]
    deb = pd.Timestamp(debut)
    sel = (idx >= deb) & (idx < deb + pd.Timedelta(days=3))
    t = idx[sel]
    r, s = ref[sel, 0], sc[sel, 0]
    evt = m_evt[h_abs][sel]

    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    ax.fill_between(t, s, r, where=r >= s, color=BLEU_FONCE, alpha=.22, linewidth=0,
                    label='énergie effacée')
    ax.fill_between(t, s, r, where=r < s, color=ORANGE, alpha=.30, linewidth=0,
                    label='report')
    ax.plot(t, r, color=BLEU_FONCE, lw=2, label='consigne réelle')
    ax.plot(t, s, color=ORANGE, lw=2, ls=(0, (5, 2)), label='consigne −2 °C de 18 à 21 h')

    for j in np.flatnonzero(evt & ~np.r_[False, evt[:-1]]):
        fin = j + int(evt[j:].argmin()) if (~evt[j:]).any() else len(evt)
        ax.axvspan(t[j], t[min(fin, len(t) - 1)], color=BLEU_CLAIR, alpha=.6, zorder=0)

    ax.set(xlim=(t[0], t[-1]), ylim=(0, max(r.max(), s.max()) * 1.22),
           ylabel='puissance appelée\nkWh/h', xlabel='')
    ax.legend(ncol=2, fontsize=9, loc='upper left')
    ax.set_title(f'Bâtiment {bid} — trois jours de janvier, avec et sans effacement',
                 fontsize=11, color=ENCRE, pad=8)
    fig.autofmt_xdate()
    sauver(fig, 'flex_avant_apres')

    efface = (r - s)[evt].sum()
    reb = -(r - s)[~evt].sum()
    print(f'    bâtiment {bid} | effacé {efface:.1f} kWh sur 3 jours | '
          f'report {reb:+.1f} kWh | pointe {(r - s)[evt].max():.2f} kW')


def fuseau_horaire():
    X, Y, ids = D.build_cache('503', D.parc())
    X = np.asarray(X)
    i_dni = D.COLONNES.index(D.WEA[3])
    heure = np.tile(np.arange(24), 365)[:X.shape[1]]

    pic = []
    for k in range(X.shape[0]):
        moy = np.array([X[k, heure == h, i_dni].mean() for h in range(24)])
        pic.append(moy.argmax())
    pic = np.array(pic)

    meta = pd.read_parquet(D.DATA / 'metadata_clean.parquet',
                           columns=['bldg_id', 'in.weather_file_longitude',
                                    'in.state']).set_index('bldg_id')
    lon = meta.loc[ids, 'in.weather_file_longitude'].values

    fig, ax = plt.subplots(figsize=(9.0, 3.9))
    ax.scatter(lon, pic + np.random.default_rng(0).uniform(-.18, .18, len(pic)),
               s=22, color=BLEU_FONCE, alpha=.45, edgecolors='none', zorder=3)

    # Droite THEORIQUE d'une horloge unique calee sur l'heure de l'Est : le midi solaire
    # derive de 1 h par 15 degres de longitude. On ne l'ajuste pas aux donnees — la
    # moyenne annuelle du rayonnement direct est biaisee par la nebulosite, plus forte
    # a l'est l'apres-midi, ce qui creuse la pente observee au-dela de la geometrie.
    xs = np.array([lon.min() - 2, lon.max() + 2])
    ax.plot(xs, 12.5 - (xs + 75) / 15, color=ORANGE, lw=2, zorder=4,
            label="attendu si horloge unique (heure de l'Est)")
    ax.axhline(12.5, color=VERT, ls=':', lw=1.8, zorder=2,
               label='attendu si heures locales — le nuage serait plat')

    ax.set(xlabel='longitude du bâtiment (°)', ylabel="heure du pic solaire\ndans le fichier",
           ylim=(8, 18))
    ax.legend(fontsize=9, loc='upper left')
    ax.set_title('Les séries sont horodatées dans un fuseau unique, pas en heure locale',
                 fontsize=11, color=ENCRE, pad=8)
    sauver(fig, 'fuseau_horaire')

    r = np.corrcoef(lon, pic)[0, 1]
    a = np.polyfit(lon, pic, 1)[0]
    print(f'    corrélation longitude / heure du pic : {r:+.2f}')
    print(f'    pente observée {a:.3f} h/degré contre {-1/15:.3f} attendu — '
          "l'écart vient de la nébulosité, pas de la géométrie")


if __name__ == '__main__':
    print('figures ->', FIG)
    fuseau_horaire()
    flex_avant_apres()
