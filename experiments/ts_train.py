"""Banc d'essai du reseau time series : une experience = une ligne de commande.

Toutes les variantes partagent le meme cache, le meme split par batiment et la meme
graine, pour que les deltas soient lisibles. Resultat ecrit en JSON dans results/.

  python ts_train.py --nom base
  python ts_train.py --nom stride24 --stride 24
  python ts_train.py --nom cop --feats cop
"""
import argparse, copy, json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import ts_data as D

torch.set_num_threads(int(os.environ.get('TS_THREADS', 20)))
ROOT = D.ROOT
RES = ROOT / 'experiments' / 'results'
RES.mkdir(exist_ok=True)
NOMS = ['total', 'chauffage', 'clim', 'eau_chaude']
SEQ_LEN = 168


# --------------------------------------------------------------------------- features
def static_vector(ids, feats):
    """s par batiment. Base = 4 LightGBM + 42 features + 2 presence (48 colonnes)."""
    def _preds(chemin):
        p = pd.read_parquet(chemin)
        return p.set_index('bldg_id') if 'bldg_id' in p.columns else p

    preds = _preds(D.DATA / 'static_preds.parquet')
    if 'oos' in feats:
        p2 = D.DATA / 'static_preds_oos.parquet'
        assert p2.exists(), 'lancer lgbm_exp.py --export avant --feats oos'
        preds = _preds(p2)
    assert list(preds.columns) == ['total', 'chauffage', 'clim', 'eau_chaude'], list(preds.columns)
    feat = pd.read_parquet(D.DATA / 'X_47features.parquet')
    feat = feat.assign(tau=feat['C'] / (feat['UA'] + feat['H_ve']) / 3.6)
    feat = feat.drop(columns=[c for c in feat.columns if 'setpoint' in c])

    equip = pd.read_parquet(D.DATA / 'metadata_clean.parquet',
                            columns=['bldg_id', 'in.hvac_cooling_type', 'in.water_heater_fuel',
                                     'in.hvac_heating_efficiency']).set_index('bldg_id')
    a_clim = (equip.loc[ids, 'in.hvac_cooling_type'] != 'None').values.astype('float32')
    ecs_el = (equip.loc[ids, 'in.water_heater_fuel'] == 'Electricity').values.astype('float32')

    blocs = [preds.loc[ids].values, feat.loc[ids].values, a_clim[:, None], ecs_el[:, None]]
    noms = ['lgbm']*4 + list(feat.columns) + ['a_clim', 'ecs_el']

    cop = np.ones(len(ids), dtype='float32')
    if 'cop' in feats or 'copt' in feats:
        eff = equip.loc[ids, 'in.hvac_heating_efficiency'].astype(str)
        hspf = eff.str.extract(r'([\d.]+)\s*HSPF')[0].astype(float)
        pac = hspf.notna().values.astype('float32')
        cop = np.where(hspf.notna(), hspf.fillna(3.412) / 3.412, 1.0).astype('float32')
        mshp = eff.str.startswith('MSHP').values.astype('float32')
        if 'cop' in feats:
            blocs += [pac[:, None], cop[:, None], mshp[:, None]]
            noms += ['pac', 'cop', 'mshp']

    if 'hdd' in feats:
        w = pd.read_parquet(D.DATA / 'weather_static.parquet')
        w = w.set_index('in.county') if 'in.county' in w.columns else w
        cty = pd.read_parquet(D.DATA / 'metadata_clean.parquet',
                              columns=['bldg_id', 'in.county']).set_index('bldg_id')
        j = cty.loc[ids, 'in.county'].map(lambda c: c).values
        wc = w.reindex(j)
        wc = wc.fillna(wc.median(numeric_only=True))
        blocs.append(wc.values.astype('float32'))
        noms += ['w_' + c for c in wc.columns]

    S = np.hstack(blocs).astype('float32')
    un = np.ones_like(a_clim)
    GATE = np.stack([un, un, a_clim, ecs_el], axis=-1).astype('float32')
    return S, GATE, cop, noms


def time_extra(X, ids, cop, feats):
    """Colonnes ajoutees a f(t), calculees par-dessus le cache : (tableau, noms)."""
    out, noms = [], []
    t_ext = X[:, :, D.I_TEXT]
    ech, ecc = X[:, :, D.I_ECH], X[:, :, D.I_ECC]
    if 'copt' in feats:
        cop_t = np.clip(cop[:, None] * (0.6 + 0.02 * t_ext), 1.0, cop[:, None])
        out += [ech / cop_t]; noms += ['besoin_elec_chauffage']
    if 'env' in feats:
        feat = pd.read_parquet(D.DATA / 'X_47features.parquet')
        ua = feat.loc[ids, 'UA'].values.astype('float32')
        hv = feat.loc[ids, 'H_ve'].values.astype('float32')
        g = ((ua + hv) / 1000.0)[:, None]
        out += [g * ech, g * ecc]; noms += ['env_x_ecart_ch', 'env_x_ecart_cl']
    if 'lag' in feats:
        # moyennes glissantes de T_ext : l'inertie thermique fait dependre la conso de
        # l'histoire recente, que le GRU doit sinon reconstruire pas a pas.
        def roll(a, w):
            c = np.cumsum(np.pad(a, ((0, 0), (w, 0)), mode='edge'), axis=1, dtype='float64')
            return ((c[:, w:] - c[:, :-w]) / w).astype('float32')
        out += [roll(t_ext, 24), roll(t_ext, 72), roll(ech, 24)]
        noms += ['t_ext_24h', 't_ext_72h', 'ecart_ch_24h']
    if 'solar' in feats:
        feat = pd.read_parquet(D.DATA / 'X_47features.parquet')
        asol = feat.loc[ids, 'A_solaire'].values.astype('float32')[:, None]
        ghi = X[:, :, D.COLONNES.index(D.WEA[3])] + X[:, :, D.COLONNES.index(D.WEA[4])]
        out += [asol * ghi / 1000.0]; noms += ['apport_solaire']
    if not out:
        return None, []
    return np.stack(out, axis=-1).astype('float32'), noms


# --------------------------------------------------------------------------- dataset
class Fenetres(Dataset):
    def __init__(self, X, S, Y, G, idx):
        self.X, self.S, self.Y, self.G, self.idx = X, S, Y, G, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        b, t = self.idx[k]
        return (torch.from_numpy(self.X[b, t:t+SEQ_LEN]), torch.from_numpy(self.S[b]),
                torch.from_numpy(self.Y[b, t:t+SEQ_LEN]), torch.from_numpy(self.G[b]))


def fenetres(bat_idx, stride, n_h=D.N_H):
    starts = list(range(0, n_h - SEQ_LEN + 1, stride))
    return np.array([(b, t) for b in bat_idx for t in starts])


# --------------------------------------------------------------------------- modeles
class LoadNet(nn.Module):
    def __init__(self, n_time, n_static, hidden=128, n_out=4, arch='base',
                 couches=1, dropout=0.0):
        super().__init__()
        self.arch = arch
        self.gru = nn.GRU(n_time, hidden, num_layers=couches, batch_first=True,
                          bidirectional=True, dropout=dropout if couches > 1 else 0.0)
        if arch == 'film':
            self.film = nn.Sequential(nn.Linear(n_static, hidden), nn.ReLU(),
                                      nn.Linear(hidden, hidden * 4))
            self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(),
                                      nn.Linear(hidden, n_out), nn.Softplus())
        else:
            self.head = nn.Sequential(nn.Linear(hidden * 2 + n_static, hidden), nn.ReLU(),
                                      nn.Linear(hidden, n_out), nn.Softplus())

    def forward(self, x, s):
        h, _ = self.gru(x)
        if self.arch == 'film':
            gb = self.film(s)
            g, b = gb.chunk(2, dim=-1)
            h = h * (1 + g).unsqueeze(1) + b.unsqueeze(1)
            return self.head(h)
        se = s.unsqueeze(1).expand(-1, h.size(1), -1)
        return self.head(torch.cat([h, se], dim=-1))


# --------------------------------------------------------------------------- run
def main(a):
    t0 = time.time()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    bats = D.parc(a.n_bat, seed=0, fichier=a.parc)
    assert ('elargi' in a.cache) == ('elargi' in a.parc), (
        f'cache « {a.cache} » incoherent avec le parc « {a.parc} »')
    X, Y, ids = D.build_cache(a.cache if a.cache else f'p{len(bats)}', bats)

    # MEMOIRE : a 2000 batiments le cache pese 2,2 Go. Chaque copie intermediaire
    # (concat des features en plus, puis standardisation) en ajoutait autant. On
    # alloue le tableau final UNE fois et on standardise EN PLACE.
    feats = [f for f in a.feats.split(',') if f]
    S, GATE, cop, noms_s = static_vector(ids, feats)
    extra, noms_extra = time_extra(X, ids, cop, feats)
    COLONNES_FT = D.COLONNES + noms_extra
    if extra is None:
        X = np.array(X, dtype='float32')
    else:
        Xf = np.empty((*X.shape[:2], X.shape[2] + extra.shape[2]), dtype='float32')
        Xf[:, :, :X.shape[2]] = X
        Xf[:, :, X.shape[2]:] = extra
        X, extra = Xf, None
    Y = np.array(Y, dtype='float32')

    # Split par batiment. Avec --val_ref on rejoue EXACTEMENT les 101 batiments de
    # validation du parc de 503 : sans ca, agrandir le parc change aussi le jeu de
    # validation, et le delta ne veut plus rien dire.
    rng = np.random.default_rng(42)
    if a.val_ref:
        ref = np.load(D.CACHE / 'B_503.npy')
        val_b = set(rng.choice(np.unique(ref), size=round(0.2 * len(ref)),
                               replace=False).tolist())
    else:
        val_b = set(rng.choice(np.unique(ids), size=round(0.2 * len(ids)),
                               replace=False).tolist())
    is_val = np.array([b in val_b for b in ids])
    itr, iva = np.flatnonzero(~is_val), np.flatnonzero(is_val)

    def stats(A, idx):
        """Moyenne et ecart-type exacts sur les heures des batiments d'entrainement.

        Par accumulation batiment par batiment : `A[idx]` en ferait une copie, soit
        1,7 Go de plus a 2000 batiments.
        """
        n = len(idx) * A.shape[1]
        s1 = np.zeros(A.shape[-1], dtype='float64')
        s2 = np.zeros(A.shape[-1], dtype='float64')
        for b in idx:
            a = A[b].astype('float64')
            s1 += a.sum(0); s2 += (a * a).sum(0)
        m = s1 / n
        return m.astype('float32'), (np.sqrt(np.maximum(s2 / n - m * m, 0)) + 1e-8).astype('float32')

    xm, xs = stats(X, itr)
    sm = S[itr].mean(0); ss = S[itr].std(0) + 1e-8
    _, ys = stats(Y, itr)                              # cibles : mise a l'echelle sans centrage
    X -= xm; X /= xs                                   # en place : pas de second tableau
    Y /= ys
    Xn, Yn = X, Y
    Sn = ((S - sm) / ss).astype('float32')

    ftr, fva = fenetres(itr, a.stride), fenetres(iva, SEQ_LEN)
    dl = DataLoader(Fenetres(Xn, Sn, Yn, GATE, ftr), batch_size=a.batch, shuffle=True)
    dlv = DataLoader(Fenetres(Xn, Sn, Yn, GATE, fva), batch_size=256)

    model = LoadNet(Xn.shape[-1], Sn.shape[1], a.hidden, 4, a.arch, a.couches, a.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    lossf = nn.HuberLoss(delta=a.delta) if a.loss == 'huber' else nn.MSELoss()
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
             if a.sched == 'plateau' else None)
    mse = nn.MSELoss()

    def evaluer(dl_):
        model.eval(); P, T, B = [], [], []
        with torch.no_grad():
            for xb, sb, yb, gb in dl_:
                P.append((model(xb, sb) * gb.unsqueeze(1)).numpy()); T.append(yb.numpy())
        return np.concatenate(P), np.concatenate(T)

    best, best_state, best_ep, hist = float('inf'), None, -1, []
    for ep in range(a.epochs):
        model.train()
        for xb, sb, yb, gb in dl:
            opt.zero_grad()
            lossf(model(xb, sb) * gb.unsqueeze(1), yb).backward()
            if a.clip:
                nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
        P, T = evaluer(dlv)
        vloss = float(((P - T) ** 2).mean())
        hist.append(vloss)
        if sched:
            sched.step(vloss)
        if vloss < best:
            best, best_ep, best_state = vloss, ep, copy.deepcopy(model.state_dict())
        print(f'  ep {ep:3d}  val_mse {vloss:.4f}  ({time.time()-t0:.0f}s)', flush=True)
        if ep - best_ep >= a.patience:
            break

    model.load_state_dict(best_state)
    P, T = evaluer(dlv)
    P, T = P * ys, T * ys                      # retour en kWh
    from sklearn.metrics import r2_score
    r2 = {n: float(r2_score(T[..., i].ravel(), P[..., i].ravel())) for i, n in enumerate(NOMS)}
    rmse = {n: float(np.sqrt(((P[..., i] - T[..., i]) ** 2).mean())) for i, n in enumerate(NOMS)}

    bva = np.array([ids[b] for b, _ in fva])
    nmbe, cvr = {}, {}
    for i, n in enumerate(NOMS):
        a_, c_ = [], []
        for b in np.unique(bva):
            k = bva == b
            t, p = T[k, :, i].sum(), P[k, :, i].sum()
            if t > 0:
                a_.append(abs((p - t) / t * 100))
                c_.append(np.sqrt(((P[k, :, i] - T[k, :, i]) ** 2).mean()) / T[k, :, i].mean() * 100)
        nmbe[n] = float(np.median(a_)); cvr[n] = float(np.median(c_))

    if a.save:
        # Meme format que la derniere cellule de timeseries_net, pour que
        # 07_flexibilite/flexibilite.ipynb le relise sans modification.
        torch.save({
            'state_dict': model.state_dict(),
            'archi': 'gru_bidirectionnel', 'tete': 'additive',
            'n_time': int(Xn.shape[-1]), 'n_static': int(Sn.shape[1]),
            'hidden': a.hidden, 'n_out': 4,
            'xm': xm, 'xs': xs, 'sm': sm, 'ss': ss, 'ys': ys,
            'colonnes': COLONNES_FT, 'colonnes_s': noms_s, 'feats': feats,
            'buildings': [(int(b), st) for b, st in bats if b in set(ids.tolist())],
            'val_b': sorted(int(b) for b in val_b),
            'gate': {int(b): GATE[k].tolist() for k, b in enumerate(ids)},
            'statique': {int(b): S[k].tolist() for k, b in enumerate(ids)},
            'niveau': {int(b): (S[k][:4] / 8760.0).tolist() for k, b in enumerate(ids)},
        }, D.DATA / a.save)
        print('modele sauvegarde ->', D.DATA / a.save)

    out = dict(nom=a.nom, args=vars(a), n_bat=len(ids), n_static=int(Sn.shape[1]),
               n_time=int(Xn.shape[-1]), fenetres_train=len(ftr), epoques=best_ep,
               val_mse=best, r2=r2, rmse=rmse, nmbe_median=nmbe, cvrmse_median=cvr,
               minutes=round((time.time() - t0) / 60, 1))
    (RES / f'{a.nom}.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    print(json.dumps({k: out[k] for k in ['nom', 'r2', 'nmbe_median', 'minutes']}, indent=1))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--nom', required=True)
    p.add_argument('--cache', default='503')
    p.add_argument('--parc', default='nn_buildings.csv',
                   help='nn_buildings.csv (503) ou nn_buildings_elargi.csv (2005)')
    p.add_argument('--n_bat', type=int, default=None)
    p.add_argument('--feats', default='')
    p.add_argument('--arch', default='base', choices=['base', 'film'])
    p.add_argument('--couches', type=int, default=1)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--stride', type=int, default=168)
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--loss', default='mse', choices=['mse', 'huber'])
    p.add_argument('--delta', type=float, default=1.0)
    p.add_argument('--sched', default='none', choices=['none', 'plateau'])
    p.add_argument('--clip', type=float, default=0.0)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save', default='', help='nom du checkpoint dans data/processed')
    p.add_argument('--val_ref', action='store_true',
                   help='rejouer les 101 batiments de validation du parc de 503')
    main(p.parse_args())
