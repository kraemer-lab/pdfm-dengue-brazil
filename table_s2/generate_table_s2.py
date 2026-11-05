#!/usr/bin/env python3
"""Generate compact Supplementary Table S2 from table_s2/data/regional_model_wis.csv.

Rows: five models nested under each macroregion; then TCN and TimesFM
nested under each of the three southern states. Columns: mean WIS at
1, 3, 6 and 12 months. Lowest mean WIS per geography/horizon is marked
with an asterisk. No exploratory analysis folder required.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'data'/'regional_model_wis.csv'
OUTPUT = HERE/'table_s2.tsv'
MODELS = ['SARIMA','TCN','XGBoost','N-BEATS','TimesFM']
REGIONS = ['North','Northeast','Central-West','Southeast','South']
STATES = ['Paraná','Rio Grande do Sul','Santa Catarina']
HORIZONS = [1,3,6,12]
COUNTS = {'North':450,'Northeast':1794,'Central-West':467,
          'Southeast':1668,'South':1191,'Paraná':399,
          'Rio Grande do Sul':497,'Santa Catarina':295}


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    d = pd.read_csv(SOURCE)
    required = {'geography','level','model','horizon','n_municipalities','mean_WIS'}
    if not required.issubset(d.columns):
        raise ValueError(f'Missing columns: {required-set(d.columns)}')
    if d.duplicated(['geography','model','horizon']).any():
        raise ValueError('Duplicate summary keys')
    if not np.isfinite(d.mean_WIS.to_numpy(dtype=float)).all() or (d.mean_WIS<0).any():
        raise ValueError('Invalid WIS values')
    expected = {(geo,model,h) for geo in REGIONS for model in MODELS for h in HORIZONS}
    expected |= {(state,model,h) for state in STATES for model in ('TCN','TimesFM') for h in HORIZONS}
    actual = set(zip(d.geography,d.model,d.horizon))
    if actual!=expected:
        raise ValueError(f'Incorrect summary coverage: missing={expected-actual}, extra={actual-expected}')
    for geo,n in COUNTS.items():
        x=d.loc[d.geography==geo]
        if not x.n_municipalities.eq(n).all():
            raise ValueError(f'Unexpected municipality count for {geo}')
        expected_level = 'macroregion' if geo in REGIONS else 'state'
        if not x.level.eq(expected_level).all():
            raise ValueError(f'Incorrect geography level for {geo}')
    lookup = d.set_index(['geography','model','horizon'])['mean_WIS']
    rows=[]
    for geo in REGIONS+STATES:
        models = MODELS if geo in REGIONS else ['TCN','TimesFM']
        for i,model in enumerate(models):
            row={'Geography':geo if i==0 else '',
                 'Model':model,'Municipalities':COUNTS[geo] if i==0 else ''}
            for h in HORIZONS:
                score=float(lookup.loc[(geo,model,h)])
                best=min(float(lookup.loc[(geo,m,h)]) for m in models)
                row[f'{h} month WIS']=f'{score:.3f}'+('*' if np.isclose(score,best,rtol=0,atol=1e-12) else '')
            rows.append(row)
    table=pd.DataFrame(rows)
    table.to_csv(OUTPUT,sep='\t',index=False,encoding='utf-8')
    script=Path(__file__).resolve()
    manifest={'created_utc':datetime.now(timezone.utc).isoformat(),
              'script':str(script),'script_sha256':sha256(script),
              'source':str(SOURCE),'source_sha256':sha256(SOURCE),
              'output':str(OUTPUT),'output_sha256':sha256(OUTPUT),
              'notation':'* denotes lowest mean WIS within the displayed models for a geography and horizon',
              'note':'Southern states show TCN and TimesFM only; no claim about other models at state level.'}
    (HERE/'run_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(f'Wrote {OUTPUT} ({len(table)} rows)')
    print(table.to_string(index=False))

if __name__=='__main__':
    main()
