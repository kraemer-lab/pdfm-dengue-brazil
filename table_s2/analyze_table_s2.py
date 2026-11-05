#!/usr/bin/env python3
"""Read-only analysis for Supplementary Table S2; run from table_s2/.

Reads existing Thucia baseline WIS outputs and the cases_with_climate lake.
Writes a compact CSV of model mean WIS for five macroregions (all models)
and three southern states (TCN and TimesFM), at all four horizons.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from thucia.core.fs import read_db

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = (HERE.parent.parent / 'data' / 'PDFM_analysis' / 'BRA_full').resolve()
MODELS = {'SARIMA':'sarima', 'TCN':'tcn', 'XGBoost':'xgboost',
          'N-BEATS':'nbeats', 'TimesFM':'timesfm'}
REGIONS = ['North','Northeast','Central-West','Southeast','South']
STATE_REGIONS = {
    1:'North',3:'North',4:'North',14:'North',22:'North',23:'North',27:'North',
    2:'Northeast',5:'Northeast',6:'Northeast',10:'Northeast',15:'Northeast',
    17:'Northeast',18:'Northeast',20:'Northeast',26:'Northeast',
    7:'Central-West',9:'Central-West',11:'Central-West',12:'Central-West',
    8:'Southeast',13:'Southeast',19:'Southeast',25:'Southeast',
    16:'South',21:'South',24:'South',
}
SOUTH_STATES = {'BRA.16_1':'Paraná','BRA.21_1':'Rio Grande do Sul',
                'BRA.24_1':'Santa Catarina'}
EXCLUDED = {'BRA.21.229_2','BRA.21.231_2'}
MONTHS = pd.period_range('2020-01','2025-08',freq='M')
HORIZONS = (1,3,6,12)


def sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def month_period(series):
    if isinstance(series.dtype,pd.PeriodDtype):
        return series.dt.asfreq('M')
    return pd.PeriodIndex(series.astype(str),freq='M')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--output-dir',type=Path,default=HERE/'data')
    args = parser.parse_args()
    root = args.data_root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    if len(STATE_REGIONS)!=27:
        raise ValueError('State mapping must contain 27 states')
    lake = read_db(str(root/'cases_with_climate')).df
    required = {'GID_2','GID_1','Date','pop_count'}
    if not required.issubset(lake.columns):
        raise ValueError(f'Lake missing columns: {required-set(lake.columns)}')
    geo = lake[list(required)].copy()
    geo['Date'] = month_period(geo['Date'])
    geo = geo.loc[geo.Date.isin(MONTHS)&~geo.GID_2.isin(EXCLUDED)].copy()
    if geo.duplicated(['GID_2','Date']).any():
        raise ValueError('Duplicate municipality-month in lake')
    if len(geo)!=5570*68 or geo.GID_2.nunique()!=5570:
        raise ValueError('Unexpected lake coverage')
    if set(geo.Date.unique())!=set(MONTHS):
        raise ValueError('Unexpected lake months')
    pop = pd.to_numeric(geo.pop_count,errors='coerce')
    if not np.isfinite(pop).all() or (pop<=0).any():
        raise ValueError('Invalid population values')
    if (geo.groupby('GID_2',observed=True).GID_1.nunique()!=1).any():
        raise ValueError('Inconsistent municipality-to-state mapping')
    geo = geo[['GID_2','GID_1']].drop_duplicates()
    geo['macroregion'] = geo.GID_1.map({f'BRA.{k}_1':v for k,v in STATE_REGIONS.items()})
    if geo.macroregion.isna().any() or geo.GID_1.nunique()!=27:
        raise ValueError('Unmapped state identifiers')
    records = []
    sources = {}
    for model,stem in MODELS.items():
        path = root/f'{stem}_h12_wis.csv'
        sources[path.name] = {'sha256':sha256(path),'bytes':path.stat().st_size}
        f = pd.read_csv(path,usecols=['GID_2','Date','horizon','WIS','Cases'])
        f['Date'] = month_period(f['Date'])
        f = f.loc[f.Date.isin(MONTHS)&~f.GID_2.isin(EXCLUDED)].copy()
        if f.duplicated(['GID_2','Date','horizon']).any():
            raise ValueError(f'{model}: duplicate forecast keys')
        if len(f)!=5570*68*4 or f.GID_2.nunique()!=5570:
            raise ValueError(f'{model}: unexpected forecast coverage')
        if set(f.horizon.unique())!=set(HORIZONS):
            raise ValueError(f'{model}: incorrect horizons')
        if not np.isfinite(f[['WIS','Cases']].to_numpy(dtype=float)).all():
            raise ValueError(f'{model}: nonfinite WIS or cases')
        if (f.WIS<0).any() or (f.Cases<0).any():
            raise ValueError(f'{model}: negative WIS or cases')
        if not f.groupby(['GID_2','horizon'],observed=True).Date.nunique().eq(68).all():
            raise ValueError(f'{model}: incomplete monthly series')
        f = f.merge(geo,on='GID_2',how='left',validate='many_to_one',indicator=True)
        if not f['_merge'].eq('both').all():
            raise ValueError(f'{model}: unmatched municipality')
        for (region,horizon),g in f.groupby(['macroregion','horizon'],observed=True):
            records.append({'geography':region,'level':'macroregion','model':model,
                            'horizon':int(horizon),'n_municipalities':g.GID_2.nunique(),
                            'mean_WIS':g.WIS.mean()})
        if model in ('TCN','TimesFM'):
            south = f.loc[f.GID_1.isin(SOUTH_STATES)].copy()
            south['state'] = south.GID_1.map(SOUTH_STATES)
            for (state,horizon),g in south.groupby(['state','horizon'],observed=True):
                records.append({'geography':state,'level':'state','model':model,
                                'horizon':int(horizon),'n_municipalities':g.GID_2.nunique(),
                                'mean_WIS':g.WIS.mean()})
        print(f'{model}: {len(f):,} validated rows',flush=True)
    summary = pd.DataFrame(records)
    if len(summary)!=5*5*4+3*2*4:
        raise ValueError('Incorrect number of summary records')
    if summary.duplicated(['geography','model','horizon']).any():
        raise ValueError('Duplicate summary keys')
    counts = summary.query('level == "macroregion" and model == "TCN" and horizon == 1')
    if counts.n_municipalities.sum()!=5570:
        raise ValueError('Macroregions do not total 5570')
    for h in HORIZONS:
        south_n = summary.query('geography == "South" and model == "TCN" and horizon == @h').n_municipalities.iloc[0]
        state_n = summary.query('level == "state" and model == "TCN" and horizon == @h').n_municipalities.sum()
        if south_n!=state_n:
            raise ValueError('Southern state counts do not sum to South')
    out.mkdir(parents=True,exist_ok=True)
    target = out/'regional_model_wis.csv'
    summary.to_csv(target,index=False)
    script = Path(__file__).resolve()
    manifest = {'created_utc':datetime.now(timezone.utc).isoformat(),
                'script':str(script),'script_sha256':sha256(script),
                'effective_paths':{'data_root':str(root),'output_dir':str(out)},
                'evaluation_period':['2020-01','2025-08'],
                'excluded_gids':sorted(EXCLUDED),'state_to_macroregion':STATE_REGIONS,
                'models':MODELS,'horizons':list(HORIZONS),
                'source_files':sources,
                'outputs':{target.name:sha256(target)},
                'notes':['Read-only lake access','No model fitting','Equal municipality-month weighting']}
    (out/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(f'Wrote {target} and analysis_manifest.json')

if __name__=='__main__':
    main()
