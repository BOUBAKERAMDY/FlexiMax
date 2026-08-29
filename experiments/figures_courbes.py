"""Figures detaillees reel vs predit : UNE figure par batiment.

Le format precedent — six panneaux dans une seule figure — ecrasait les courbes a une
taille illisible une fois inseree dans un document. Ici chaque batiment a sa figure,
et chaque figure porte trois niveaux de lecture :

  ligne 1  consommation TOTALE, reel vs predit, avec la bande d'ecart coloree ;
  ligne 2  l'usage DOMINANT de la saison — chauffage en janvier, climatisation en
           juillet — chaque panneau etant identifie par son etiquette d'axe ;
  ligne 3  temperature exterieure, qui explique la forme des deux lignes du dessus.

Le nom de l'usage est porte par l'ETIQUETTE D'AXE et non par un titre interne : mis
en haut du panneau, il entrait en collision avec le cumul affiche a droite.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from figures import REEL, PREDIT, CONTEXTE, ENCRE, ENCRE2, GRILLE, ACCENT, FIG, sauver

CHOIX = [
    (420474, 'Arizona, zone 2B — pompe à chaleur, COP 2,26',
     "climat désertique · HDD18 505 · CDD18 2 470 · jusqu'à 42 °C en juillet"),
    (170214, 'Colorado, zone 7B — résistance électrique',
     "climat rigoureux · HDD18 5 808 · jusqu'à −22 °C en janvier · 19 832 kWh/an de chauffage"),
    (393604, 'Colorado, zone 5B — pire biais du jeu de validation',
     "HDD18 3 280 · pompe à chaleur COP 2,49 · NMBE total +45 % sur l'année"),
]
SAISONS = [('hiver', 'Semaine du 15 au 21 janvier', 'chauffage', 'Chauffage'),
           ('ete', 'Semaine du 16 au 22 juillet', 'clim', 'Climatisation')]
JOURS = ['lun', 'mar', 'mer', 'jeu', 'ven', 'sam', 'dim', '']


def r2(reel, pred):
    reel, pred = np.asarray(reel), np.asarray(pred)
    denom = ((reel - reel.mean()) ** 2).sum()
    return 1 - ((reel - pred) ** 2).sum() / denom if denom > 0 else np.nan


def grille_jours(ax):
    ax.set_xticks(np.arange(0, 169, 24))
    for k in range(0, 169, 24):
        ax.axvline(k, color=GRILLE, lw=.8, zorder=0)
    ax.grid(axis='y'); ax.grid(axis='x', visible=False)


def trace(ax, reel, pred, ymax, libelle):
    """Deux courbes, la bande d'ecart coloree selon le signe, et les cumuls."""
    reel, pred = np.asarray(reel), np.asarray(pred)
    h = np.arange(len(reel))
    ax.set_ylabel(f'{libelle}\nkWh/h', fontsize=8.6, color=ENCRE, labelpad=6)

    if reel.max() == 0 and pred.max() == 0:
        ax.text(.5, .5, 'usage absent de ce logement', transform=ax.transAxes,
                ha='center', va='center', fontsize=8.5, color=ENCRE2, style='italic')
        ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_xlim(0, len(h) - 1); grille_jours(ax); ax.set_xticklabels([])
        return

    ax.fill_between(h, reel, pred, where=pred >= reel, interpolate=True,
                    color=PREDIT, alpha=.17, linewidth=0, zorder=2)
    ax.fill_between(h, reel, pred, where=pred < reel, interpolate=True,
                    color=REEL, alpha=.17, linewidth=0, zorder=2)
    ax.plot(h, reel, color=REEL, lw=1.5, label='réel (simulation)', zorder=4)
    ax.plot(h, pred, color=PREDIT, lw=1.5, ls=(0, (5, 2)), label='prédit (modèle)', zorder=5)

    ax.set_xlim(0, len(h) - 1); ax.set_ylim(0, ymax if ymax > 0 else 1)
    grille_jours(ax); ax.set_xticklabels([])

    cr, cp = reel.sum(), pred.sum()
    ecart = (cp - cr) / cr * 100 if cr > 0 else np.nan
    lignes = [f'réel {cr:,.0f}  ·  prédit {cp:,.0f} kWh'.replace(',', ' ')]
    if np.isfinite(ecart):
        lignes.append(f'écart {ecart:+.0f} %   ·   R² {r2(reel, pred):.3f}')
    ax.text(.988, .955, '\n'.join(lignes), transform=ax.transAxes, ha='right', va='top',
            fontsize=7.8, color=ENCRE2, family='DejaVu Sans Mono', linespacing=1.5)


def bande_temperature(ax, text_ext):
    t = np.asarray(text_ext)
    h = np.arange(len(t))
    ax.plot(h, t, color=CONTEXTE, lw=1.4, zorder=3)
    ax.fill_between(h, t.min() - 1, t, color=CONTEXTE, alpha=.13, linewidth=0, zorder=2)
    if t.min() < 0 < t.max():
        ax.axhline(0, color=ACCENT, lw=.9, ls=':', zorder=4)
    ax.set_xlim(0, len(h) - 1); ax.set_ylim(t.min() - 1.5, t.max() + 3.5)
    ax.set_ylabel('T° ext.\n°C', fontsize=8.6, color=ENCRE, labelpad=6)
    grille_jours(ax)
    ax.set_xticklabels(JOURS)
    ax.text(.988, .93, f'min {t.min():.0f}  ·  max {t.max():.0f} °C',
            transform=ax.transAxes, ha='right', va='top', fontsize=7.8,
            color=ENCRE2, family='DejaVu Sans Mono')


def figure_batiment(don, bid, titre, sous_titre):
    b = next(x for x in don['batiments'] if x['bldg_id'] == bid)
    fig, axes = plt.subplots(3, 2, figsize=(7.6, 6.4),
                             gridspec_kw=dict(height_ratios=[3, 3, 1.3]))

    ymax_tot = max(max(max(b['semaines'][s]['reel']['total']),
                       max(b['semaines'][s]['predit']['total']))
                   for s, *_ in SAISONS) * 1.24

    for j, (cle, stitre, usage, lib) in enumerate(SAISONS):
        s = b['semaines'][cle]
        axes[0, j].set_title(stitre, fontsize=9.8, color=ENCRE, pad=9)
        trace(axes[0, j], s['reel']['total'], s['predit']['total'], ymax_tot, 'Total')
        pic = max(max(s['reel'][usage]), max(s['predit'][usage]))
        trace(axes[1, j], s['reel'][usage], s['predit'][usage], pic * 1.34, lib)
        bande_temperature(axes[2, j], s['text_ext'])

    poignees, etiquettes = axes[0, 0].get_legend_handles_labels()
    fig.legend(poignees, etiquettes, loc='lower center', ncol=2, fontsize=9,
               bbox_to_anchor=(.5, -.035))
    fig.text(0, 1.035, f'Bâtiment {bid} — {titre}', fontsize=11.5, fontweight='bold',
             color=ENCRE, va='bottom')
    fig.text(0, 1.002, sous_titre, fontsize=8.6, color=ENCRE2, style='italic', va='bottom')
    fig.subplots_adjust(hspace=.20, wspace=.26)
    sauver(fig, f'fig_courbe_{bid}')


if __name__ == '__main__':
    don = json.loads(Path(__file__).with_name('courbes.json').read_text(encoding='utf-8'))
    print('figures ->', FIG)
    for bid, titre, sous in CHOIX:
        figure_batiment(don, bid, titre, sous)
