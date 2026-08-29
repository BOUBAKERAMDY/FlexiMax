"""Figures manquantes de la presentation de soutenance.

Palette accordee au theme Beamer (BleuFonce, Orange, VertOK...) pour que les figures
ne detonnent pas sur les diapos. Les series multiples utilisent les slots categoriels
valides : separation daltonien dE >= 8 sur fond blanc, verifiee par valid_palette.

Sortie en PNG 200 dpi (les \\includegraphics de la presentation attendent du .png) et
en PDF vectoriel pour ceux qui prefereraient le format sans perte.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'processed'
FIG = ROOT / 'reports' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

BLEU_FONCE = '#004A75'
BLEU_CLAIR = '#DEEBF7'
ORANGE = '#E69138'
VERT = '#5F8A3F'
GRIS = '#B8C4CC'
ENCRE = '#1B2A33'
ENCRE2 = '#4A5A63'
GRILLE = '#E4EBEE'

USAGE = {'total': BLEU_FONCE, 'chauffage': '#eb6834',
         'clim': '#2a78d6', 'eau_chaude': '#1baf7a'}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.edgecolor': GRIS, 'axes.labelcolor': ENCRE2,
    'xtick.color': ENCRE2, 'ytick.color': ENCRE2,
    'axes.grid': True, 'grid.color': GRILLE, 'grid.linewidth': .8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'legend.frameon': False,
})


def sauver(fig, nom):
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{nom}.{ext}', dpi=200, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)
    print(f'  {nom}.png / .pdf')


# ------------------------------------------------------ 1. profil par usage
def courbe_exemple():
    """Profil journalier moyen du parc, decompose par usage."""
    Y = np.load(ROOT / 'experiments' / 'cache' / 'Y_503.npy', mmap_mode='r')
    Y = np.asarray(Y)                                    # (n_bat, 8760, 4)
    heure = np.tile(np.arange(24), 365)[:Y.shape[1]]
    prof = {u: np.array([Y[:, heure == h, i].mean() for h in range(24)])
            for i, u in enumerate(USAGE)}

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    autres = prof['total'] - (prof['chauffage'] + prof['clim'] + prof['eau_chaude'])
    bas = np.zeros(24)
    for u, lib in [('chauffage', 'chauffage'), ('clim', 'climatisation'),
                   ('eau_chaude', 'eau chaude')]:
        ax.fill_between(range(24), bas, bas + prof[u], color=USAGE[u], alpha=.85,
                        label=lib, linewidth=0)
        bas = bas + prof[u]
    ax.fill_between(range(24), bas, bas + autres, color=GRIS, alpha=.75,
                    label='autres usages', linewidth=0)
    ax.plot(range(24), prof['total'], color=BLEU_FONCE, lw=2, label='total')

    ax.set(xlim=(0, 23), ylim=(0, prof['total'].max() * 1.18),
           xlabel='heure de la journée', ylabel='kWh/h par logement')
    ax.set_xticks(range(0, 24, 3))
    ax.legend(ncol=2, fontsize=9, loc='upper left')
    ax.set_title('Profil journalier moyen, décomposé par usage',
                 fontsize=11, color=ENCRE, pad=8)
    sauver(fig, 'courbe_exemple')


# -------------------------------------------------------- 2. nettoyage
def nettoyage_colonnes():
    """Entonnoir 771 -> 391 -> 165."""
    etapes = [('Attributs bruts\nde la release', 771, GRIS),
              ('Après nettoyage\n(constants, vides, doublons)', 391, BLEU_CLAIR),
              ('Caractéristiques du logement\n(colonnes in.*)', 165, BLEU_FONCE)]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    y = np.arange(len(etapes))[::-1]
    for k, (lib, n, c) in enumerate(etapes):
        ax.barh(y[k], n, height=.62, color=c,
                edgecolor=BLEU_FONCE if c != BLEU_FONCE else 'none', linewidth=.8)
        ax.text(n + 12, y[k], f'{n}', va='center', fontsize=13, fontweight='bold',
                color=ENCRE)
    ax.set_yticks(y)
    ax.set_yticklabels([e[0] for e in etapes], fontsize=9.5)
    ax.set_xlim(0, 900)
    ax.set_xlabel('nombre de colonnes')
    ax.grid(axis='x'); ax.grid(axis='y', visible=False)
    ax.set_title('Réduction du jeu d\'attributs', fontsize=11, color=ENCRE, pad=8)

    perte = 771 - 391
    ax.annotate(f'−{perte} colonnes sans information', xy=(580, y[0]), xytext=(470, y[0] - .55),
                fontsize=8.5, color=ENCRE2,
                arrowprops=dict(arrowstyle='->', color=ENCRE2, lw=.8))
    sauver(fig, 'nettoyage_colonnes')


# ------------------------------------------------- 3. constante de temps
def enveloppe_inertie():
    """Distribution de tau sur le parc, avec la mediane."""
    feat = pd.read_parquet(DATA / 'X_47features.parquet')
    ids = pd.read_csv(DATA / 'nn_buildings.csv')['bldg_id']
    tau = (feat['C'] / (feat['UA'] + feat['H_ve']) / 3.6).loc[ids]
    med = tau.median()

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5),
                             gridspec_kw=dict(width_ratios=[1.35, 1]))

    ax = axes[0]
    ax.hist(tau, bins=34, color=BLEU_CLAIR, edgecolor=BLEU_FONCE, linewidth=.7, zorder=3)
    ax.axvline(med, color=ORANGE, lw=2.2, zorder=4)
    ax.text(med + .8, ax.get_ylim()[1] * .92, f'médiane {med:.1f} h',
            color=ORANGE, fontsize=10.5, fontweight='bold', va='top')
    ax.set(xlabel='constante de temps  τ = C / (UA + H$_{ve}$)   [h]',
           ylabel='nombre de logements', xlim=(0, tau.quantile(.99)))
    ax.grid(axis='y'); ax.grid(axis='x', visible=False)

    ax = axes[1]
    t = np.linspace(0, 6, 200)
    for lab, tv, c in [('τ = 8 h  (léger)', 8, GRIS),
                       (f'τ = {med:.0f} h  (médian)', med, BLEU_FONCE),
                       ('τ = 31 h  (lourd)', 31, VERT)]:
        ax.plot(t, 100 * np.exp(-t / tv), color=c, lw=2, label=lab)
    ax.axhline(95, color=ORANGE, ls=':', lw=1.4)
    ax.text(6, 95.6, 'dérive de 5 %', ha='right', fontsize=8.5, color=ORANGE)
    ax.set(xlabel='heures après coupure du chauffage',
           ylabel='écart de température restant (%)', xlim=(0, 6), ylim=(60, 102))
    ax.legend(fontsize=8.5, loc='lower left')

    fig.suptitle('Inertie thermique du parc : combien de temps un logement tient',
                 fontsize=11, color=ENCRE, y=1.02)
    fig.subplots_adjust(wspace=.28)
    sauver(fig, 'enveloppe_inertie')


# --------------------------------------------------- 4. profil de flexibilite
def profil_flexibilite():
    """Profil moyen d'effacement du scenario du soir."""
    d = json.loads((ROOT / 'experiments' / 'results' / 'flex_chauffage.json')
                   .read_text(encoding='utf-8'))
    bloc = d['scenarios']['soir']
    prof = np.array([b['profil_chauffage'] for b in bloc['batiments']]).mean(0)
    bat = pd.DataFrame(bloc['batiments'])
    h = np.arange(24)

    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    ax.bar(h[prof >= 0], prof[prof >= 0], color=BLEU_FONCE, width=.8, zorder=3,
           label='puissance effacée')
    ax.bar(h[prof < 0], prof[prof < 0], color=ORANGE, width=.8, zorder=3,
           label='report après l\'événement')
    ax.axvspan(17.5, 20.5, color=BLEU_CLAIR, alpha=.55, zorder=1)
    ax.axhline(0, color=ENCRE2, lw=1, zorder=4)

    pointe = prof.max()
    ax.annotate(f'{pointe:.2f} kWh/h à {int(prof.argmax())} h',
                xy=(prof.argmax(), pointe), xytext=(prof.argmax() - 6.5, pointe * .95),
                fontsize=10, color=ENCRE, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=ENCRE2, lw=1))
    ax.text(19, ax.get_ylim()[0] * .85, 'fenêtre\n18–21 h', ha='center', fontsize=8.5,
            color=ENCRE2)

    ax.set(xlim=(-.7, 23.7), xlabel='heure locale',
           ylabel='effacement moyen\nkWh/h par logement')
    ax.set_xticks(range(0, 24, 3))
    ax.grid(axis='y'); ax.grid(axis='x', visible=False)
    ax.legend(ncol=2, fontsize=9, loc='upper left')
    ax.set_title(f'Effacement du chauffage — consigne −2 °C de 18 à 21 h, '
                 f'{len(bat)} bâtiments de validation', fontsize=11, color=ENCRE, pad=8)
    sauver(fig, 'profil_flexibilite')

    print(f"    pointe médiane par bâtiment : {bat['pointe_kW_total'].median():.2f} kW")
    print(f"    énergie effacée médiane     : {bat['efface_kWh_chauffage'].median():.0f} kWh")
    print(f"    taux de report              : "
          f"{100*bat['rebond_kWh_chauffage'].sum()/bat['efface_kWh_chauffage'].sum():+.0f} %")


if __name__ == '__main__':
    print('figures ->', FIG)
    courbe_exemple()
    nettoyage_colonnes()
    enveloppe_inertie()
    profil_flexibilite()
