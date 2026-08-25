"""Tableau comparatif des experiences, trie par R2 total, delta contre la reference."""
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parent / 'results'
NOMS = ['total', 'chauffage', 'clim', 'eau_chaude']


def lire(prefixe=''):
    out = []
    for f in sorted(RES.glob(f'{prefixe}*.json')):
        d = json.loads(f.read_text(encoding='utf-8'))
        if 'r2' in d:
            out.append(d)
    return out


def table(prefixe='', ref='base'):
    ex = lire(prefixe)
    if not ex:
        print('aucun resultat'); return
    r = next((e for e in ex if e['nom'] == ref), None)
    larg = max(len(e['nom']) for e in ex) + 1
    print(f"{'experience':{larg}} " + ''.join(f'{n:>12}' for n in NOMS)
          + f"{'ep':>4}{'min':>6}")
    print('-' * (larg + 12 * 4 + 10))
    for e in sorted(ex, key=lambda x: -x['r2']['total']):
        ligne = f"{e['nom']:{larg}} "
        for n in NOMS:
            v = e['r2'][n]
            d = f"{v - r['r2'][n]:+.3f}" if r and e['nom'] != ref else '     '
            ligne += f'{v:7.3f}{d:>5}'
        ligne += f"{e.get('epoques', -1):>4}{e.get('minutes', 0):>6.0f}"
        print(ligne)
    print()
    print(f"{'experience':{larg}} " + ''.join(f'{n:>12}' for n in NOMS) + '   (|NMBE| median %)')
    for e in sorted(ex, key=lambda x: -x['r2']['total']):
        if 'nmbe_median' in e:
            print(f"{e['nom']:{larg}} " + ''.join(f"{e['nmbe_median'][n]:12.1f}" for n in NOMS))


if __name__ == '__main__':
    table(*(sys.argv[1:] or ['']))
