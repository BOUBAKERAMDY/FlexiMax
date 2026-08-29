"""Banc d'essai du LightGBM annuel (4 usages).

Repart de X_47features.parquet (deja construit par le notebook) pour ne pas refaire
les agregats a chaque essai. Trois corrections structurelles par rapport au notebook :

  1. l'arret anticipe se fait sur VAL, pas sur TEST (le notebook passait X_test en
     eval_set : le nombre d'arbres etait donc choisi sur le jeu de test) ;
  2. chaque usage a droit a ses propres hyperparametres, au lieu de reprendre ceux
     regles sur `total` ;
  3. les predictions exportees pour le reseau sont HORS ECHANTILLON (KFold).
"""
import argparse, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_squared_error

warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'processed'
RAW = ROOT / 'data' / 'raw'
RES = ROOT / 'experiments' / 'results'
RES.mkdir(exist_ok=True)
SEED = 42
USAGES = ['total', 'chauffage', 'clim', 'eau_chaude']
TGT = {'total': 'out.electricity.total.energy_consumption..kwh',
       'chauffage': 'out.electricity.heating.energy_consumption..kwh',
       'clim': 'out.electricity.cooling.energy_consumption..kwh',
       'eau_chaude': 'out.electricity.hot_water.energy_consumption..kwh'}


def charger(feats):
    X = pd.read_parquet(DATA / 'X_47features.parquet')
    ids = X.index

    cols = ['bldg_id'] + list(TGT.values())
    Y = pd.read_parquet(RAW / 'upgrade0.parquet', columns=cols).set_index('bldg_id')
    Y = Y.rename(columns={v: k for k, v in TGT.items()}).loc[ids]

    meta = pd.read_parquet(DATA / 'metadata_clean.parquet',
                           columns=['bldg_id', 'in.ashrae_iecc_climate_zone_2004', 'in.county',
                                    'in.hvac_heating_efficiency']).set_index('bldg_id').loc[ids]
    zone = meta['in.ashrae_iecc_climate_zone_2004'].astype(str)

    if 'cop' in feats:
        eff = meta['in.hvac_heating_efficiency'].astype(str)
        hspf = eff.str.extract(r'([\d.]+)\s*HSPF')[0].astype(float)
        X = X.assign(pac=hspf.notna().astype(int).values,
                     cop_chauffage=np.where(hspf.notna(), hspf / 3.412, 1.0),
                     mshp=eff.str.startswith('MSHP').astype(int).values)
    if 'clim' in feats:
        w = pd.read_parquet(DATA / 'weather_static.parquet').set_index('in.county')
        wc = w.reindex(meta['in.county'].values)
        wc = wc.fillna(wc.median(numeric_only=True))
        for c in wc.columns:
            X['w_' + c] = wc[c].values
        if 'UA' in X.columns:
            X['UA_x_HDD'] = X['UA'] * X['w_HDD18']
            X['UA_x_CDD'] = X['UA'] * X['w_CDD18']
    return X.astype('float32'), Y, zone


def split(X, Y, zone):
    z = zone.copy()
    rares = z.value_counts()[lambda s: s < 5].index
    z = z.where(~z.isin(rares), 'RARE')
    itv, ite = train_test_split(np.arange(len(X)), test_size=0.2, random_state=SEED, stratify=z)
    itr, iva = train_test_split(itv, test_size=0.2, random_state=SEED, stratify=z.iloc[itv])
    return itr, iva, ite


def params(usage, a, best=None):
    p = dict(random_state=SEED, n_jobs=a.jobs, verbose=-1, n_estimators=a.arbres,
             learning_rate=0.03, num_leaves=63, min_child_samples=40,
             colsample_bytree=0.8, subsample=0.8, subsample_freq=1,
             reg_lambda=5.0, reg_alpha=0.5)
    if best:
        p.update(best)
    # Tweedie par usage, pas en bloc : mesure du 21/08 (features cop+climat),
    # chauffage +0.007 et clim +0.005 contre l2, mais eau_chaude -0.002 et total 0.
    # Un objectif zero-gonfle n'aide que sur les cibles qui ont vraiment des zeros.
    if usage in a.tweedie.split(','):
        p.update(objective='tweedie', tweedie_variance_power=1.4)
    return p


def entrainer(X, y, itr, iva, ite, p, log1p):
    yt = np.log1p(y) if log1p else y
    m = lgb.LGBMRegressor(**p)
    m.fit(X.iloc[itr], yt.iloc[itr], eval_set=[(X.iloc[iva], yt.iloc[iva])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    pred = m.predict(X.iloc[ite])
    if log1p:
        pred = np.expm1(pred)
    return m, np.clip(pred, 0, None)


def main(a):
    feats = [f for f in a.feats.split(',') if f]
    X, Y, zone = charger(feats)
    itr, iva, ite = split(X, Y, zone)
    print(f'{len(X):,} logements | {X.shape[1]} features | '
          f'train {len(itr):,} val {len(iva):,} test {len(ite):,}')

    best_par = {}
    if a.optuna:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        for u in USAGES:
            def obj(t):
                p = params(u, a, dict(
                    learning_rate=t.suggest_float('learning_rate', 0.005, 0.08, log=True),
                    num_leaves=t.suggest_int('num_leaves', 20, 200),
                    min_child_samples=t.suggest_int('min_child_samples', 10, 400),
                    colsample_bytree=t.suggest_float('colsample_bytree', 0.4, 1.0),
                    subsample=t.suggest_float('subsample', 0.5, 1.0),
                    reg_lambda=t.suggest_float('reg_lambda', 0.1, 100.0, log=True),
                    reg_alpha=t.suggest_float('reg_alpha', 0.01, 20.0, log=True)))
                yt = np.log1p(Y[u]) if a.log1p else Y[u]
                m = lgb.LGBMRegressor(**p)
                m.fit(X.iloc[itr], yt.iloc[itr], eval_set=[(X.iloc[iva], yt.iloc[iva])],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
                return np.sqrt(mean_squared_error(yt.iloc[iva], m.predict(X.iloc[iva])))
            st = optuna.create_study(direction='minimize',
                                     sampler=optuna.samplers.TPESampler(seed=SEED))
            st.optimize(obj, n_trials=a.optuna, show_progress_bar=False)
            best_par[u] = st.best_params
            print(f'  optuna {u:11} val_rmse {st.best_value:.1f}')

    out = {'nom': a.nom, 'args': vars(a), 'n': len(X), 'n_feat': int(X.shape[1]), 'r2': {},
           'rmse': {}, 'arbres': {}, 'best_params': best_par}
    for u in USAGES:
        m, pred = entrainer(X, Y[u], itr, iva, ite, params(u, a, best_par.get(u)), a.log1p)
        yte = Y[u].iloc[ite]
        out['r2'][u] = float(r2_score(yte, pred))
        out['rmse'][u] = float(np.sqrt(mean_squared_error(yte, pred)))
        out['arbres'][u] = int(m.best_iteration_ or a.arbres)
        print(f'{u:11} R2={out["r2"][u]:.4f}  RMSE={out["rmse"][u]:7.0f} kWh  '
              f'arbres={out["arbres"][u]}')

    if a.export:
        print('\nexport hors echantillon (KFold) ...')
        kf = KFold(n_splits=a.kfold, shuffle=True, random_state=SEED)
        oos = pd.DataFrame(index=X.index, columns=USAGES, dtype='float64')
        for k, (i_tr, i_te) in enumerate(kf.split(X)):
            i_tr2, i_va2 = train_test_split(i_tr, test_size=0.15, random_state=SEED)
            for u in USAGES:
                _, p = entrainer(X, Y[u], i_tr2, i_va2, i_te, params(u, a, best_par.get(u)), a.log1p)
                oos.iloc[i_te, USAGES.index(u)] = p
            print(f'  fold {k+1}/{a.kfold}')
        oos.index.name = 'bldg_id'
        oos.to_parquet(DATA / 'static_preds_oos.parquet')
        out['r2_oos'] = {u: float(r2_score(Y[u], oos[u])) for u in USAGES}
        print('R2 hors echantillon :', {k: round(v, 4) for k, v in out['r2_oos'].items()})

    (RES / f'lgbm_{a.nom}.json').write_text(json.dumps(out, indent=1), encoding='utf-8')
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--nom', required=True)
    p.add_argument('--feats', default='')
    p.add_argument('--log1p', action='store_true')
    p.add_argument('--tweedie', default='', help="usages en objectif tweedie, ex. 'chauffage,clim'")
    p.add_argument('--optuna', type=int, default=0)
    p.add_argument('--arbres', type=int, default=4000)
    p.add_argument('--export', action='store_true')
    p.add_argument('--kfold', type=int, default=5)
    p.add_argument('--jobs', type=int, default=6)
    main(p.parse_args())
