#!/usr/bin/env python3
"""Read-only Figure 2F sensitivity analysis; run from figure_2/ beside
analyze_pdfm_relative.py and analyze_fig2f_strata.py.

Outputs only new CSVs under figure_2/data/. Does not alter existing outputs.
Uses identical paired forecasts for case-count and incidence stratification.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import analyze_pdfm_relative as core
from analyze_fig2f_strata import boot_month_means
from thucia.core.fs import read_db

OUT = Path(__file__).resolve().parent / 'data'
OUT.mkdir(exist_ok=True)
DATA = Path(__file__).resolve().parent.parent.parent / "data" / "PDFM_analysis" / "BRA_full"
BLOCKS = (3, 6, 12)
STRATA = ('zero', 'low', 'medium', 'high')
EXCLUDED = {'BRA.21.229_2', 'BRA.21.231_2'}
START, END = pd.Period('2020-01', 'M'), pd.Period('2025-08', 'M')


def population_lookup():
    p = read_db(DATA / 'cases_with_climate').df[['GID_2', 'Date', 'pop_count']].copy()
    p['Date'] = pd.PeriodIndex(p['Date'].astype(str), freq='M')
    p = p.loc[p['Date'].between(START, END) & ~p['GID_2'].isin(EXCLUDED)]
    if p.duplicated(['GID_2', 'Date']).any():
        raise ValueError('Duplicate municipality-month population records')
    if not np.isfinite(p['pop_count']).all() or (p['pop_count'] <= 0).any():
        raise ValueError('Missing, nonfinite or nonpositive population')
    print(f'Population lookup: {len(p)} rows, {p.GID_2.nunique()} units, {p.Date.nunique()} months', flush=True)
    return p


def stratify(d, variable):
    """Use exact Figure 2F rule: positive values <= q33 low, <= q67 medium."""
    result = pd.Series('zero', index=d.index, dtype='object')
    nz = d[variable] > 0
    if nz.any():
        cutoffs = d.loc[nz].groupby('Date', observed=True)[variable].quantile([1/3, 2/3]).unstack()
        cutoffs.columns = ['q33', 'q67']
        sub = d.loc[nz, ['Date', variable]].join(cutoffs, on='Date')
        result.loc[nz] = np.select([sub[variable] <= sub.q33, sub[variable] <= sub.q67],
                                    ['low', 'medium'], default='high')
    return result


def main():
    pop = population_lookup()
    effects, contrasts = [], []
    for mi, (model, stem) in enumerate(core.MODELS.items()):
        paired = core.load_model_pair(stem)
        paired = paired.loc[~paired.GID_2.isin(EXCLUDED)].copy()
        paired = paired.merge(pop, on=['GID_2', 'Date'], how='left', validate='many_to_one', indicator=True)
        if not paired['_merge'].eq('both').all():
            raise ValueError(f'{model}: {paired._merge.value_counts().to_dict()} unmatched population rows')
        paired.drop(columns='_merge', inplace=True)
        # Match the established paired-analysis policy: omit unusable paired
        # forecasts from BOTH variants, without imputing or altering source files.
        required = ['Cases', 'WIS_Base', 'WIS_PDFM', 'pop_count', 'DeltaWIS']
        valid = np.isfinite(paired[required].to_numpy(dtype=float)).all(axis=1)
        n_invalid = int((~valid).sum())
        if n_invalid:
            print(f'{model}: excluding {n_invalid} nonfinite paired rows '
                  f'(both baseline and PDFM; no imputation)', flush=True)
            paired = paired.loc[valid].copy()
        if paired.empty:
            raise ValueError(f'{model}: no valid paired observations')
        if (paired[['WIS_Base', 'WIS_PDFM']] < 0).any().any():
            raise ValueError(f'{model}: negative WIS prevents log1p skill')
        paired['incidence_100k'] = 1e5 * paired['Cases'] / paired['pop_count']
        paired['log_relative_skill'] = np.log1p(paired['WIS_Base']) - np.log1p(paired['WIS_PDFM'])
        for horizon in core.HORIZONS:
            d = paired.loc[paired.horizon.eq(horizon)].copy()
            if d.empty:
                raise ValueError(f'{model} h{horizon}: no paired observations')
            # Reclassify the SAME paired observations for both stratifications.
            for strat_name, variable in [('case_count', 'Cases'), ('incidence_100k', 'incidence_100k')]:
                d['stratum'] = stratify(d, variable)
                for metric, col in [('absolute_DeltaWIS', 'DeltaWIS'),
                                    ('log_relative_skill', 'log_relative_skill')]:
                    monthly = d.groupby(['Date', 'stratum'], observed=True)[col].mean().unstack()
                    for si, stratum in enumerate(STRATA):
                        g = d.loc[d.stratum.eq(stratum), col]
                        m = monthly[stratum].dropna() if stratum in monthly else pd.Series(dtype=float)
                        if g.empty or len(m) != 68:
                            raise ValueError(f'{model} h{horizon} {strat_name} {stratum}: incomplete monthly strata')
                        for block in BLOCKS:
                            seed = 10000 + mi*10000 + horizon*500 + (0 if strat_name == 'case_count' else 200) + (0 if metric == 'absolute_DeltaWIS' else 100) + si*10 + block
                            lo, hi = boot_month_means(m.to_numpy(), block, seed)
                            effects.append(dict(model=model, horizon=horizon, stratification=strat_name,
                                metric=metric, stratum=stratum, n_forecasts=len(g), n_months=len(m),
                                mean_forecast_weighted=g.mean(), median_forecast_weighted=g.median(),
                                fraction_improved=(g < 0).mean() if metric == 'absolute_DeltaWIS' else (g > 0).mean(),
                                mean_month_weighted=m.mean(), block_months=block,
                                CI95_lower=lo, CI95_upper=hi))
                    diff = (monthly['high'] - monthly['low']).dropna()
                    if len(diff) != 68:
                        raise ValueError(f'{model} h{horizon}: high/low months incomplete')
                    for block in BLOCKS:
                        seed = 50000 + mi*10000 + horizon*500 + (0 if strat_name == 'case_count' else 200) + (0 if metric == 'absolute_DeltaWIS' else 100) + block
                        lo, hi = boot_month_means(diff.to_numpy(), block, seed)
                        contrasts.append(dict(model=model, horizon=horizon, stratification=strat_name,
                            metric=metric, contrast='high_minus_low', n_months=len(diff),
                            mean_difference=diff.mean(), block_months=block,
                            CI95_lower=lo, CI95_upper=hi))
        print(f'Processed {model}: {len(paired)} valid paired rows', flush=True)
    a, b = pd.DataFrame(effects), pd.DataFrame(contrasts)
    assert len(a) == 5*4*2*2*4*3 and len(b) == 5*4*2*2*3
    a.to_csv(OUT / 'fig2f_sensitivity_effects.csv', index=False)
    b.to_csv(OUT / 'fig2f_sensitivity_contrasts.csv', index=False)
    print('\nN-BEATS, 6 months, high vs low (3-month blocks):')
    print(b.query("model == 'N-BEATS' and horizon == 6 and block_months == 3").to_string(index=False))
    print('\nSaved:', OUT / 'fig2f_sensitivity_effects.csv', 'and', OUT / 'fig2f_sensitivity_contrasts.csv')


if __name__ == '__main__':
    main()
