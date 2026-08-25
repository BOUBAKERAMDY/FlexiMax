"""Elargit le parc telecharge, en reutilisant la logique de extraction_timeseries_oedi.

Meme filtre de candidats, meme stratification par zone climatique, memes colonnes,
et surtout la MEME reconstruction de consignes (inject_setpoints, regle ResStock
p.135 comprise) : les nouveaux fichiers doivent etre indiscernables des 503 actuels.

Les 503 deja presents sont conserves tels quels et exclus du tirage.
"""
import argparse, functools, re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'processed'
RAW = ROOT / 'data' / 'raw'
LOOKUP = ROOT / 'data' / 'external' / 'options_lookup.tsv'
OEDI = ('https://oedi-data-lake.s3.amazonaws.com/nrel-pds-building-stock/'
        'end-use-load-profiles-for-us-building-stock/2025/resstock_amy2018_release_1/'
        'timeseries_individual_buildings/by_state/upgrade=0')

COL_COOL = 'out.schedules.cooling_setpoint..c'
COL_HEAT = 'out.schedules.heating_setpoint..c'
WEA = ['out.outdoor_air_drybulb_temp..c', 'out.outdoor_air_relative_humidity..percentage',
       'out.weather.wind_speed..meter_per_second',
       'out.weather.direct_normal_solar_radiation..watt_per_m2',
       'out.weather.diffuse_solar_radiation..watt_per_m2']
SCHED = ['out.schedules.' + s for s in [
    'occupants', 'vacancy', 'lighting_interior', 'lighting_garage', 'plug_loads_other',
    'plug_loads_tv', 'clothes_dryer', 'clothes_washer', 'dishwasher', 'cooking_range',
    'ceiling_fan', 'hot_water_fixtures', 'hot_water_clothes_washer', 'hot_water_dishwasher',
    'no_space_cooling', 'no_space_heating']]
TGT = ['out.electricity.' + t + '.energy_consumption..kwh' for t in
       ['total', 'heating', 'cooling', 'hot_water']]
NN_COLS = ['timestamp'] + WEA + SCHED + TGT


def _f_abs_to_c(t):
    return (float(re.search(r'[-\d.]+', str(t)).group()) - 32) * 5 / 9


def _f_delta_to_c(t):
    return float(re.search(r'[-\d.]+', str(t)).group()) * 5 / 9


@functools.lru_cache(maxsize=1)
def _lines():
    return LOOKUP.read_text(encoding='utf-8').splitlines()


@functools.lru_cache(maxsize=None)
def _masks(carac, periode):
    if periode is None or str(periode).strip().lower() == 'none':
        return tuple(np.zeros(24)), tuple(np.zeros(24))
    pre = f'{carac}\t{periode}\t'
    line = next((l for l in _lines() if l.startswith(pre)), None)
    if line is None:
        raise ValueError(f'"{carac}" / "{periode}" introuvable')
    wk = re.search(r'weekday_setpoint_schedule=([-\d,\s]+)', line).group(1)
    we = re.search(r'weekend_setpoint_schedule=([-\d,\s]+)', line).group(1)
    arr = lambda s: tuple(int(x) for x in s.split(',')[:24])
    return arr(wk), arr(we)


@functools.lru_cache(maxsize=1)
def _meta():
    cols = ['bldg_id',
            'in.cooling_setpoint', 'in.cooling_setpoint_has_offset',
            'in.cooling_setpoint_offset_magnitude', 'in.cooling_setpoint_offset_period',
            'in.heating_setpoint', 'in.heating_setpoint_has_offset',
            'in.heating_setpoint_offset_magnitude', 'in.heating_setpoint_offset_period']
    return pd.read_parquet(DATA / 'metadata_clean.parquet', columns=cols).set_index('bldg_id')


def inject_setpoints(df, bldg_id):
    ts = pd.DatetimeIndex(df['timestamp'] if 'timestamp' in df.columns else df.index)
    hours, wknd = ts.hour.values, (ts.dayofweek.values >= 5)
    row = _meta().loc[bldg_id]
    for col, pfx, param in [
            (COL_COOL, 'in.cooling_setpoint', 'Cooling Setpoint Offset Period'),
            (COL_HEAT, 'in.heating_setpoint', 'Heating Setpoint Offset Period')]:
        has = row[f'{pfx}_has_offset'] == 'Yes'
        base = _f_abs_to_c(row[pfx])
        mag = _f_delta_to_c(row[f'{pfx}_offset_magnitude']) if has else 0.0
        wk, we = _masks(param, row[f'{pfx}_offset_period'] if has else None)
        m = np.where(wknd, np.asarray(we, float)[hours], np.asarray(wk, float)[hours])
        df[col] = base + m * mag
    if _f_abs_to_c(row['in.heating_setpoint']) > _f_abs_to_c(row['in.cooling_setpoint']):
        moy = (df[COL_HEAT] + df[COL_COOL]) / 2
        df[COL_HEAT] = moy
        df[COL_COOL] = moy
    return df


def main(a):
    raw = pd.read_parquet(RAW / 'upgrade0.parquet', columns=[
        'bldg_id', 'in.state', 'in.ashrae_iecc_climate_zone_2004',
        'in.geometry_building_type_recs', 'in.geometry_stories', 'in.heating_fuel',
        'in.electric_vehicle_ownership', 'in.misc_pool', 'in.has_pv', 'in.vacancy_status'])
    m = ((raw['in.geometry_building_type_recs'] == 'Single-Family Detached')
         & (raw['in.geometry_stories'] == '1')
         & (raw['in.heating_fuel'] == 'Electricity')
         & (raw['in.electric_vehicle_ownership'] == 'No')
         & (raw['in.misc_pool'] == 'None') & (raw['in.has_pv'] == 'No')
         & (raw['in.vacancy_status'] == 'Occupied'))
    cand = raw[m][['bldg_id', 'in.state', 'in.ashrae_iecc_climate_zone_2004']]

    deja = {int(p.name.split('-')[0]) for p in DATA.glob('*-0.parquet')}
    libre = cand[~cand['bldg_id'].isin(deja)]
    print(f'{len(cand):,} candidats | {len(deja)} deja locaux | {len(libre):,} disponibles')

    z = 'in.ashrae_iecc_climate_zone_2004'
    parts = [g.sample(min(len(g), max(1, round(a.n * len(g) / len(libre)))), random_state=7)
             for _, g in libre.groupby(z)]
    ech = pd.concat(parts).reset_index(drop=True)
    print(f'tirage : {len(ech)} batiments sur {ech[z].nunique()} zones')

    def un(job):
        bid, st = job
        out = DATA / f'{bid}-0.parquet'
        if out.exists():
            return None
        try:
            df = pd.read_parquet(f'{OEDI}/state={st}/{bid}-0.parquet', columns=NN_COLS)
            df = inject_setpoints(df, bid)
            assert {COL_COOL, COL_HEAT} <= set(df.columns)
            df.to_parquet(out)
        except Exception as e:
            return f'skip {bid} : {type(e).__name__} {e}'

    jobs = list(zip(ech['bldg_id'].astype(int), ech['in.state']))
    fait = 0
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        for msg in ex.map(un, jobs):
            fait += 1
            if msg:
                print(msg, flush=True)
            if fait % 100 == 0:
                print(f'  {fait}/{len(jobs)}', flush=True)

    ok = [int(b) for b in ech['bldg_id']
          if (DATA / f'{b}-0.parquet').exists()
          and {COL_COOL, COL_HEAT} <= set(pq.ParquetFile(DATA / f'{b}-0.parquet').schema.names)]
    print(f'telecharges avec consignes : {len(ok)}/{len(ech)}')

    anciens = pd.read_csv(DATA / 'nn_buildings.csv')
    neuf = ech[ech['bldg_id'].isin(ok)][['bldg_id', 'in.state']]
    tout = pd.concat([anciens, neuf]).drop_duplicates('bldg_id').reset_index(drop=True)
    tout.to_csv(DATA / 'nn_buildings_elargi.csv', index=False)
    print(f'parc elargi : {len(tout)} batiments -> nn_buildings_elargi.csv')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1500)
    p.add_argument('--jobs', type=int, default=8)
    main(p.parse_args())
