"""Extrait des semaines reelles vs predites pour quelques batiments de validation.

Choix des batiments : on ne prend pas les 4 premiers (le graphe du notebook tracait
w = 0, qui s'est avere etre un cas extreme choisi par hasard). On echantillonne le
parc de validation sur deux axes : le CLIMAT (degres-jours de chauffe) et la QUALITE
de la prediction (NMBE par batiment), pour montrer aussi bien un bon cas qu'un mauvais.

Sortie : courbes.json, consomme par la page de visualisation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import ts_data as D
from ts_train import LoadNet, static_vector, time_extra

SEQ = 168
NOMS = ['total', 'chauffage', 'clim', 'eau_chaude']
torch.set_num_threads(8)

ck = torch.load(D.DATA / 'loadnet.pt', weights_only=False)
print(f"checkpoint : f(t)={ck['n_time']}  s={ck['n_static']}  tete={ck['tete']}")

bats = D.parc(fichier='nn_buildings_elargi.csv')
X, Y, ids = D.build_cache('elargi', bats)
feats = ck['feats']
S, GATE, cop, noms_s = static_vector(ids, feats)
extra, noms_extra = time_extra(X, ids, cop, feats)

Xf = np.empty((*X.shape[:2], X.shape[2] + extra.shape[2]), dtype='float32')
Xf[:, :, :X.shape[2]] = X
Xf[:, :, X.shape[2]:] = extra
X = Xf
Y = np.array(Y, dtype='float32')

xm, xs, sm, ss, ys = ck['xm'], ck['xs'], ck['sm'], ck['ss'], ck['ys']
Xn = (X - xm) / xs
Sn = (S - sm) / ss

model = LoadNet(ck['n_time'], ck['n_static'], ck['hidden'], 4)
model.load_state_dict(ck['state_dict'])
model.eval()

val = set(ck['val_b'])
idx_val = [k for k, b in enumerate(ids) if int(b) in val]
print(f'{len(idx_val)} batiments de validation')


def predire(k):
    """Annee complete predite pour le batiment d'indice k : (8736, 4) en kWh."""
    n = D.N_H // SEQ
    xb = torch.from_numpy(Xn[k, :n*SEQ].reshape(n, SEQ, -1))
    sb = torch.from_numpy(np.tile(Sn[k], (n, 1)))
    with torch.no_grad():
        p = model(xb, sb).numpy()
    return (p * GATE[k] * ys).reshape(-1, 4)


# --- caracteriser chaque batiment de validation : climat + qualite -------------------
meta = pd.read_parquet(D.DATA / 'metadata_clean.parquet',
                       columns=['bldg_id', 'in.county', 'in.state',
                                'in.ashrae_iecc_climate_zone_2004']).set_index('bldg_id')
wea = pd.read_parquet(D.DATA / 'weather_static.parquet').set_index('in.county')

lignes = []
preds = {}
for k in idx_val:
    b = int(ids[k])
    p = predire(k); t = Y[k, :len(p)]
    preds[b] = p
    nmbe_ch = (p[:, 1].sum() - t[:, 1].sum()) / max(t[:, 1].sum(), 1e-9) * 100
    nmbe_to = (p[:, 0].sum() - t[:, 0].sum()) / t[:, 0].sum() * 100
    cty = meta.loc[b, 'in.county']
    lignes.append(dict(k=k, bldg_id=b, etat=meta.loc[b, 'in.state'],
                       zone=meta.loc[b, 'in.ashrae_iecc_climate_zone_2004'],
                       hdd=float(wea.loc[cty, 'HDD18']) if cty in wea.index else np.nan,
                       cdd=float(wea.loc[cty, 'CDD18']) if cty in wea.index else np.nan,
                       ch_an=float(t[:, 1].sum()), cl_an=float(t[:, 2].sum()),
                       to_an=float(t[:, 0].sum()),
                       nmbe_ch=float(nmbe_ch), nmbe_to=float(nmbe_to),
                       cop=float(cop[k]), pac=bool(cop[k] > 1.0)))
d = pd.DataFrame(lignes).dropna(subset=['hdd'])

# --- selection : 4 batiments qui racontent quatre situations differentes -------------
choix = {}
froid = d.sort_values('hdd', ascending=False)
choix['climat rigoureux'] = froid.iloc[0]
chaud = d[d['cl_an'] > d['cl_an'].median()].sort_values('cdd', ascending=False)
choix['climat chaud'] = chaud.iloc[0]
med = d.iloc[(d['nmbe_to'].abs() - d['nmbe_to'].abs().median()).abs().argsort()]
choix['cas median'] = med.iloc[0]

# Le pire cas se cherche parmi les batiments qui CHAUFFENT vraiment : sur un batiment
# a 0 kWh de chauffage annuel, le NMBE n'est pas defini et le classement est domine
# par des cas degeneres.
chauffe = d[(d['ch_an'] > 1000) & (~d.index.isin([choix[c].name for c in choix]))]
pire = chauffe.sort_values('nmbe_ch', key=lambda s: s.abs(), ascending=False)
choix['pire biais chauffage'] = pire.iloc[0]

# --- extraction des deux semaines ----------------------------------------------------
i0 = pd.date_range('2018-01-01', periods=D.N_H, freq='1h')
sem = {'hiver': int(np.flatnonzero((i0.month == 1) & (i0.day == 15) & (i0.hour == 0))[0]),
       'ete':   int(np.flatnonzero((i0.month == 7) & (i0.day == 16) & (i0.hour == 0))[0])}

sortie = {'batiments': [], 'usages': NOMS,
          'heures': [f'{h.day:02d}/{h.month:02d} {h.hour:02d}h' for h in i0[:SEQ]]}
for titre, r in choix.items():
    k, b = int(r['k']), int(r['bldg_id'])
    p, t = preds[b], Y[k]
    bloc = {'titre': titre, 'bldg_id': b, 'etat': r['etat'], 'zone': r['zone'],
            'hdd': round(r['hdd']), 'cdd': round(r['cdd']),
            'pac': bool(r['pac']), 'cop': round(r['cop'], 2),
            'annuel': {'reel': round(r['to_an']), 'chauffage': round(r['ch_an']),
                       'clim': round(r['cl_an'])},
            'nmbe_total': round(r['nmbe_to'], 1), 'nmbe_chauffage': round(r['nmbe_ch'], 1),
            'semaines': {}}
    for saison, s0 in sem.items():
        bloc['semaines'][saison] = {
            'texte': [f'{h.day:02d}/{h.month:02d} {h.hour:02d}h' for h in i0[s0:s0+SEQ]],
            'reel': {n: [round(float(v), 3) for v in t[s0:s0+SEQ, j]] for j, n in enumerate(NOMS)},
            'predit': {n: [round(float(v), 3) for v in p[s0:s0+SEQ, j]] for j, n in enumerate(NOMS)},
            'text_ext': [round(float(v), 1) for v in X[k, s0:s0+SEQ, D.I_TEXT]],
        }
    sortie['batiments'].append(bloc)
    print(f"{titre:24} {b:>7} {r['etat']}  zone {r['zone']}  HDD {r['hdd']:.0f}  "
          f"chauffage {r['ch_an']:.0f} kWh/an  NMBE tot {r['nmbe_to']:+.1f}%")

Path(__file__).with_name('courbes.json').write_text(json.dumps(sortie), encoding='utf-8')
print('-> courbes.json')
