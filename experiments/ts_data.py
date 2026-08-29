"""Cache des series temporelles + construction de f(t) pour les A/B du reseau.

Le notebook relit 503 parquets et re-echantillonne a chaque execution (~3 min).
Ici on le fait UNE fois, on stocke en .npy, et chaque experience repart du cache.

f(t) de base = les 31 colonnes du modele en production. Les features additionnelles
testees en A/B sont calculees PAR-DESSUS le cache, jamais dedans : le cache ne doit
pas dependre de l'experience en cours.
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'processed'
CACHE = ROOT / 'experiments' / 'cache'
CACHE.mkdir(parents=True, exist_ok=True)

WEA = ['out.outdoor_air_drybulb_temp..c', 'out.outdoor_air_relative_humidity..percentage',
       'out.weather.wind_speed..meter_per_second',
       'out.weather.direct_normal_solar_radiation..watt_per_m2',
       'out.weather.diffuse_solar_radiation..watt_per_m2']
SCHED = ['out.schedules.' + s for s in [
    'occupants', 'vacancy', 'lighting_interior', 'lighting_garage', 'plug_loads_other',
    'plug_loads_tv', 'clothes_dryer', 'clothes_washer', 'dishwasher', 'cooking_range',
    'ceiling_fan', 'hot_water_fixtures', 'hot_water_clothes_washer', 'hot_water_dishwasher',
    'no_space_cooling', 'no_space_heating']]
SETP = ['out.schedules.heating_setpoint..c', 'out.schedules.cooling_setpoint..c']
TGT = ['out.electricity.' + t + '.energy_consumption..kwh' for t in
       ['total', 'heating', 'cooling', 'hot_water']]
CAL = ['h_sin', 'h_cos', 'd_sin', 'd_cos', 'm_sin', 'm_cos']
ECA = ['ecart_chauffage', 'ecart_clim']

COLONNES = WEA + SCHED + SETP + CAL + ECA          # 31, ordre du modele en production
I_TEXT = COLONNES.index(WEA[0])
I_HEAT, I_COOL = COLONNES.index(SETP[0]), COLONNES.index(SETP[1])
I_ECH, I_ECC = COLONNES.index('ecart_chauffage'), COLONNES.index('ecart_clim')
N_H = 8760


def load_building(bldg_id, state):
    """f(t) horaire (8760, 31) et cibles (8760, 4). Identique au notebook."""
    path = DATA / f'{bldg_id}-0.parquet'
    ts = pd.read_parquet(path)
    ts['timestamp'] = pd.to_datetime(ts['timestamp'])
    brut = ts.set_index('timestamp').reindex(columns=WEA + SCHED + SETP + TGT)
    h = brut[WEA + SCHED + SETP].resample('1h').mean().iloc[:N_H]
    y = brut[TGT].resample('1h').sum().iloc[:N_H]
    h[SCHED] = h[SCHED].fillna(0.0)
    if h[SETP].isna().any().any():
        raise ValueError(f'batiment {bldg_id} : consignes absentes')

    i = h.index
    cal = pd.DataFrame({
        'h_sin': np.sin(2*np.pi*i.hour/24),      'h_cos': np.cos(2*np.pi*i.hour/24),
        'd_sin': np.sin(2*np.pi*i.dayofweek/7),  'd_cos': np.cos(2*np.pi*i.dayofweek/7),
        'm_sin': np.sin(2*np.pi*(i.month-1)/12), 'm_cos': np.cos(2*np.pi*(i.month-1)/12),
    }, index=i)
    t_ext = h[WEA[0]]
    eca = pd.DataFrame({
        'ecart_chauffage': (h[SETP[0]] - t_ext).clip(lower=0),
        'ecart_clim':      (t_ext - h[SETP[1]]).clip(lower=0),
    }, index=i)
    f = pd.concat([h[WEA + SCHED + SETP], cal, eca], axis=1)
    assert list(f.columns) == COLONNES, list(f.columns)
    return f.values.astype('float32'), y.values.astype('float32')


def build_cache(tag, buildings, n_jobs=10):
    """(n_bat, 8760, 31) et (n_bat, 8760, 4) sur disque. Idempotent.

    Le tag seul ne suffit pas a identifier un cache : un `--n_bat 60` avait ecrase
    le cache des 503. On compare la LISTE d'identifiants, pas sa longueur.
    """
    fx, fy, fb = CACHE / f'X_{tag}.npy', CACHE / f'Y_{tag}.npy', CACHE / f'B_{tag}.npy'
    voulu = np.array([b for b, _ in buildings])
    if fx.exists() and fb.exists():
        ids = np.load(fb)
        if len(ids) == len(voulu) and set(ids.tolist()) == set(voulu.tolist()):
            return np.load(fx, mmap_mode='r'), np.load(fy, mmap_mode='r'), ids
        print(f'cache {tag} present mais different ({len(ids)} vs {len(voulu)}) -> reconstruction')

    from concurrent.futures import ThreadPoolExecutor
    from numpy.lib.format import open_memmap

    # Ecriture INCREMENTALE sur disque : empiler 2000 batiments en RAM demandait
    # 4,4 Go de pic (la liste puis le np.stack), pour 4,7 Go disponibles.
    n = len(buildings)
    Xm = open_memmap(fx, mode='w+', dtype='float32', shape=(n, N_H, len(COLONNES)))
    Ym = open_memmap(fy, mode='w+', dtype='float32', shape=(n, N_H, len(TGT)))

    def one(job):
        bid, st = job
        try:
            return bid, load_building(bid, st)
        except Exception as e:
            return bid, e

    ok = []
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:
        for bid, res in ex.map(one, buildings):
            if isinstance(res, Exception):
                print(f'  skip {bid} : {type(res).__name__} {res}')
                continue
            k = len(ok)
            Xm[k], Ym[k] = res
            ok.append(bid)
    Xm.flush(); Ym.flush(); del Xm, Ym
    ids = np.array(ok)
    if len(ok) < n:                       # tronquer les lignes non remplies
        for f, k in [(fx, len(COLONNES)), (fy, len(TGT))]:
            a = np.load(f, mmap_mode='r')[:len(ok)]
            np.save(f, np.array(a))
    np.save(fb, ids)
    print(f'cache {tag} : {len(ids)} batiments')
    return np.load(fx, mmap_mode='r'), np.load(fy, mmap_mode='r'), ids


def parc(n=None, seed=0, fichier='nn_buildings.csv'):
    """Liste (bldg_id, etat). `fichier` choisit le parc : 503 ou elargi.

    Piege corrige : `--cache elargi` ne changeait que le NOM du cache, pas la liste
    lue ici. Le job « 2005 batiments » tournait donc sur les 503 et reconstruisait
    le cache elargi a 503 lignes, en silence.
    """
    p = pd.read_csv(DATA / fichier)
    if n is not None:
        p = p.sample(n=n, random_state=seed)
    return list(p.itertuples(index=False, name=None))
