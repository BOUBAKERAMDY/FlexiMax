"""Genere les figures du rapport : architecture du reseau + courbes reel/predit.

Sortie en PDF vectoriel (net a toute echelle dans un document LaTeX) et en PNG 300 dpi
pour les previsualisations. Destination : reports/figures/.

Palette : slots categoriels 1 et 2 d'une palette validee (separation daltonien dE 24.7,
vision normale dE 33.6, contraste >= 3:1 sur fond blanc). Le reel et le predit sont aussi
distingues par le style de trait, pour rester lisibles en impression noir et blanc.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / 'reports' / 'figures'
FIG.mkdir(parents=True, exist_ok=True)

REEL, PREDIT, CONTEXTE = '#2a78d6', '#eb6834', '#8C9CA3'
ENCRE, ENCRE2, TRAIT, GRILLE = '#10171A', '#40525A', '#CBD6DA', '#E4EBEE'
ACCENT = '#0E6B76'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.edgecolor': TRAIT, 'axes.labelcolor': ENCRE2, 'axes.titlesize': 9.5,
    'xtick.color': ENCRE2, 'ytick.color': ENCRE2,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.grid': True, 'grid.color': GRILLE, 'grid.linewidth': .7,
    'axes.spines.top': False, 'axes.spines.right': False,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'legend.frameon': False, 'pdf.fonttype': 42,
})


def sauver(fig, nom):
    for ext in ('pdf', 'png'):
        fig.savefig(FIG / f'{nom}.{ext}', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'  {nom}.pdf / .png')


# ============================================================ 1. architecture
STYLE_BOITE = {'titre': dict(fontsize=8.8, fontweight='bold', color=ENCRE),
               'dim':   dict(fontsize=7.8, family='DejaVu Sans Mono', color=ACCENT),
               'note':  dict(fontsize=6.9, color=ENCRE2)}
INTERLIGNE_PT = {'titre': 13.0, 'dim': 12.0, 'note': 10.5}


def _unite_point(ax):
    """Hauteur d'un point typographique, exprimee en unites de donnees de l'axe.

    L'interligne etait fixe en unites d'axe : la meme valeur donnait un espacement
    correct sur une figure haute et des lignes superposees sur une figure basse.
    On le calcule desormais a partir de la hauteur reelle de l'axe. L'ylim doit
    donc etre fixe AVANT d'appeler boite().
    """
    h_pouces = ax.figure.get_size_inches()[1] * ax.get_position().height
    y0, y1 = ax.get_ylim()
    return (y1 - y0) / (h_pouces * 72)


def boite(ax, x, y, w, h, lignes, couleur=ENCRE, fond='white'):
    """Boite a coins arrondis, texte empile depuis le haut, interligne en points."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.004,rounding_size=0.010',
                                linewidth=1.1, edgecolor=couleur, facecolor=fond, zorder=2))
    u = _unite_point(ax)
    esp = {k: v * u for k, v in INTERLIGNE_PT.items()}
    haut = sum(esp[s] for _, s in lignes)
    cy = y + h / 2 + haut / 2
    for txt, st in lignes:
        cy -= esp[st] / 2
        ax.text(x + w / 2, cy, txt, ha='center', va='center', zorder=3, **STYLE_BOITE[st])
        cy -= esp[st] / 2


def fleche(ax, x1, y1, x2, y2, texte=None, cote='droite'):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=10,
                                 linewidth=1.0, color=ENCRE2, zorder=1, shrinkA=0, shrinkB=1))
    if texte:
        dx = .013 if cote == 'droite' else -.013
        ax.text(x1 + dx, (y1 + y2) / 2, texte, ha='left' if cote == 'droite' else 'right',
                va='center', fontsize=7.0, color=ENCRE2, style='italic')


def figure_architecture():
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    ax.set_xlim(-.01, 1.01); ax.set_ylim(-.12, 1.10); ax.axis('off')

    boite(ax, .00, .84, .47, .17, [
        ('f(t) — entrées dynamiques', 'titre'), ('(168, 32)', 'dim'),
        ('5 météo · 16 usages · 2 consignes', 'note'),
        ('6 calendrier · 3 écarts', 'note')], couleur=REEL)
    boite(ax, .53, .84, .47, .17, [
        ('s — vecteur statique', 'titre'), ('(51,)', 'dim'),
        ('4 niveaux annuels · 42 physique', 'note'),
        ('2 équipement · 3 chauffage', 'note')], couleur=ACCENT)

    fleche(ax, .235, .84, .235, .655, 'standardisation')
    fleche(ax, .765, .84, .765, .545)

    boite(ax, .00, .48, .47, .175, [
        ('GRU bidirectionnel — hidden 128', 'titre'),
        ('(168, 32)  →  (168, 256)', 'dim'),
        ('deux passes, avant et arrière', 'note'),
        ('384 = 3 × 128 : portes du GRU', 'note')], couleur=REEL)
    boite(ax, .53, .40, .47, .145, [
        ('s répété sur le temps', 'titre'), ('(51,)  →  (168, 51)', 'dim'),
        ('ne traverse pas le GRU', 'note')], couleur=ACCENT)

    fleche(ax, .235, .48, .235, .335)
    fleche(ax, .765, .40, .765, .335)
    ax.plot([.235, .235, .765, .765], [.335, .315, .315, .335], color=ENCRE2, lw=1.0, zorder=1)
    ax.text(.50, .325, 'concaténation  →  (168, 307)', ha='center', va='bottom',
            fontsize=7.2, color=ENCRE2, style='italic')

    fleche(ax, .50, .315, .50, .235)
    boite(ax, .13, .075, .74, .16, [
        ('Tête MLP — partagée entre les 168 pas de temps', 'titre'),
        ('Linear(307 → 128) + ReLU   →   Linear(128 → 4) + Softplus', 'dim'),
        ('les mêmes poids à chaque heure · sortie contrainte ≥ 0', 'note')])

    fleche(ax, .50, .075, .50, -.005, "× masque d'équipement,  × échelle des cibles")
    boite(ax, .18, -.10, .64, .095, [
        ('conso(t) en kWh — (168, 4)', 'titre'),
        ('total · chauffage · climatisation · eau chaude', 'note')], couleur=PREDIT)

    ax.text(0, 1.07, 'Réseau horaire — GRU bidirectionnel, 164 356 paramètres',
            fontsize=10, fontweight='bold', color=ENCRE, va='bottom')
    sauver(fig, 'fig_architecture_reseau')


# ============================================================ 2. cascade
def figure_pipeline():
    fig, ax = plt.subplots(figsize=(7.6, 3.3))
    ax.set_xlim(-.01, 1.01); ax.set_ylim(-.05, 1.05); ax.axis('off')

    boite(ax, .00, .56, .30, .40, [
        ('Métadonnées ResStock', 'titre'), ('64 062 × 61 features', 'dim'),
        ('enveloppe · chauffage · climat', 'note')], couleur=ENCRE2)
    boite(ax, .35, .56, .30, .40, [
        ('① LightGBM annuel', 'titre'), ('4 modèles → kWh/an', 'dim'),
        ('R² 0.942 / 0.951 / 0.955', 'note')], couleur=ACCENT)
    boite(ax, .70, .56, .30, .40, [
        ('③ Contrefactuel', 'titre'), ('ΔP = réf. − scénario', 'dim'),
        ('effacement et report', 'note')], couleur=PREDIT)
    boite(ax, .00, .02, .30, .40, [
        ('Séries horaires', 'titre'), ('2 005 bât. × 8 760 h', 'dim'),
        ('météo · usages · consignes', 'note')], couleur=ENCRE2)
    boite(ax, .35, .02, .30, .40, [
        ('② GRU bidirectionnel', 'titre'), ('f(t) 32 + s 51 → kWh/h', 'dim'),
        ('R² 0.896 / 0.912 / 0.928', 'note')], couleur=REEL)

    fleche(ax, .30, .76, .35, .76)
    fleche(ax, .30, .22, .35, .22)
    fleche(ax, .50, .56, .50, .42)
    ax.text(.515, .49, 'niveau annuel → s', ha='left', va='center',
            fontsize=7.0, color=ENCRE2, style='italic')
    ax.plot([.65, .85, .85], [.22, .22, .50], color=ENCRE2, lw=1.0, zorder=1)
    fleche(ax, .85, .50, .85, .56)
    sauver(fig, 'fig_pipeline')


# ============================================================ 3. courbes
# Descriptions FACTUELLES du batiment, pas de la qualite d'ajustement : la figure se
# decline sur plusieurs usages, et un commentaire vrai pour la climatisation serait faux
# pour le total. Les cumuls affiches dans chaque panneau disent deja la qualite.
CHOIX = [
    (420474, 'Arizona, zone 2B — pompe à chaleur, COP 2,26',
     "HDD18 505 · jusqu'à 42 °C en juillet · climatisation dominante"),
    (170214, 'Colorado, zone 7B — résistance électrique',
     "HDD18 5 808 · jusqu'à −22 °C en janvier · 19 832 kWh/an de chauffage"),
    (393604, 'Colorado, zone 5B — pire biais du jeu de validation',
     'HDD18 3 280 · pompe à chaleur COP 2,49 · NMBE total +45 %'),
]
SAISONS = [('hiver', 'Semaine du 15 janvier'), ('ete', 'Semaine du 16 juillet')]


def figure_courbes(usage='total', nom='fig_courbes_total'):
    d = json.loads((Path(__file__).with_name('courbes.json')).read_text(encoding='utf-8'))
    par_id = {b['bldg_id']: b for b in d['batiments']}

    fig, axes = plt.subplots(3, 2, figsize=(7.4, 7.0), sharex='col')
    for i, (bid, titre, commentaire) in enumerate(CHOIX):
        b = par_id[bid]
        ymax = max(max(max(b['semaines'][s]['reel'][usage]), max(b['semaines'][s]['predit'][usage]))
                   for s, _ in SAISONS) * 1.30 or 1
        for j, (cle, stitre) in enumerate(SAISONS):
            ax = axes[i, j]
            s = b['semaines'][cle]
            h = np.arange(len(s['reel'][usage]))
            ax.plot(h, s['reel'][usage], color=REEL, lw=1.3, label='réel (simulation)', zorder=3)
            ax.plot(h, s['predit'][usage], color=PREDIT, lw=1.3, ls='--',
                    label='prédit (modèle)', zorder=4)
            ax.set_xlim(0, len(h) - 1); ax.set_ylim(0, ymax)
            ax.set_xticks(np.arange(0, 169, 24))
            ax.set_xticklabels([f'J{k}' for k in range(8)])
            ax.grid(axis='y'); ax.grid(axis='x', alpha=.5)
            for k in range(0, 169, 24):
                ax.axvline(k, color=GRILLE, lw=.7, zorder=0)
            # Saison a l'INTERIEUR du panneau : en titre d'axe, elle entrait en
            # collision avec le sous-titre de la ligne.
            ax.text(.015, .96, stitre, transform=ax.transAxes, ha='left', va='top',
                    fontsize=8, color=ENCRE2)
            if j == 0:
                ax.set_ylabel('kWh/h')
            r = sum(s['reel'][usage]); p = sum(s['predit'][usage])
            # Sur une SECONDE ligne : au meme niveau que le libelle de saison, les deux
            # textes se chevauchaient au milieu du panneau.
            ax.text(.985, .855, f'réel {r:,.0f} · prédit {p:,.0f} kWh'.replace(',', ' '),
                    transform=ax.transAxes, ha='right', va='top', fontsize=7.6,
                    color=ENCRE2, family='DejaVu Sans Mono')
        axes[i, 0].text(0, 1.145, f'Bâtiment {bid} — {titre}',
                        transform=axes[i, 0].transAxes, fontsize=9.3, fontweight='bold',
                        color=ENCRE, va='bottom')
        axes[i, 0].text(0, 1.035, commentaire,
                        transform=axes[i, 0].transAxes, fontsize=7.8, color=ENCRE2,
                        va='bottom', style='italic')

    for ax in axes[2]:
        ax.set_xlabel('jour de la semaine')
    poignees, etiquettes = axes[0, 0].get_legend_handles_labels()
    fig.legend(poignees, etiquettes, loc='lower center', ncol=2, fontsize=8.6,
               bbox_to_anchor=(.5, -.035))
    fig.subplots_adjust(hspace=.60, wspace=.17)
    sauver(fig, nom)


# ============================================================ 4. gains
def figure_gains():
    usages = ['total', 'chauffage', 'clim', 'eau chaude']
    avant_l = [0.886, 0.800, 0.916, 0.975]
    apres_l = [0.943, 0.953, 0.957, 0.975]
    avant_r = [0.828, 0.805, 0.883, 0.831]
    apres_r = [0.896, 0.912, 0.928, 0.852]

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), sharey=True)
    for ax, (av, ap, titre) in zip(axes, [(avant_l, apres_l, 'Modèle annuel — LightGBM'),
                                          (avant_r, apres_r, 'Réseau horaire — GRU')]):
        y = np.arange(len(usages))
        ax.barh(y + .19, av, height=.32, color=CONTEXTE, label='avant', zorder=3)
        ax.barh(y - .19, ap, height=.32, color=REEL, label='après', zorder=3)
        for k, (a, b) in enumerate(zip(av, ap)):
            ax.text(b + .006, k - .19, f'{b:.3f}', va='center', fontsize=7.6, color=ENCRE)
            ax.text(a + .006, k + .19, f'{a:.3f}', va='center', fontsize=7.6, color=ENCRE2)
        ax.set_yticks(y); ax.set_yticklabels(usages)
        ax.set_xlim(0, 1.12); ax.set_xlabel('R²')
        ax.set_title(titre, fontsize=9.3, color=ENCRE, pad=6)
        ax.grid(axis='x'); ax.grid(axis='y', visible=False)
    # sharey=True : inverser dans la boucle annulait l'inversion au second passage.
    axes[0].invert_yaxis()
    poignees, etiquettes = axes[0].get_legend_handles_labels()
    fig.legend(poignees, etiquettes, loc='lower center', ncol=2, fontsize=8.6,
               bbox_to_anchor=(.5, -.10))
    fig.subplots_adjust(wspace=.08)
    sauver(fig, 'fig_gains')


if __name__ == '__main__':
    print('figures ->', FIG)
    figure_architecture()
    figure_pipeline()
    figure_courbes('total', 'fig_courbes_total')
    figure_courbes('chauffage', 'fig_courbes_chauffage')
    figure_gains()
