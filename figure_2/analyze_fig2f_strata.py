"""Read-only Figure 2F follow-up. Run beside analyze_pdfm_relative.py.
Writes only new CSVs into figure_2/data. Does not alter model outputs.
"""
import numpy as np
import pandas as pd
from pathlib import Path
import analyze_pdfm_relative as core

OUT = Path(__file__).resolve().parent / 'data'
OUT.mkdir(exist_ok=True)
BLOCKS = (3, 6, 12)
NBOOT = 2000
ORDER = ('zero', 'low', 'medium', 'high')


def boot_month_means(month_values, block, seed):
    x = np.asarray(month_values, dtype=float)
    n = len(x)
    if n < block:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(NBOOT, int(np.ceil(n / block))))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets).reshape(NBOOT, -1)[:, :n]
    means = x[indices].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def main():
    rows = []
    contrasts = []
    for mi, (model, stem) in enumerate(core.MODELS.items()):
        df = core.load_model_pair(stem)
        for horizon in core.HORIZONS:
            d = df.loc[df.horizon.eq(horizon)].copy()
            d = core.assign_case_burden_by_month(d)
            # Each month has its own within-month stratum mean, avoiding
            # treating individual municipalities as independent bootstrap units.
            month = d.groupby(['Date', 'burden'], observed=True)['DeltaWIS'].mean().unstack()
            for si, stratum in enumerate(ORDER):
                g = d.loc[d.burden.eq(stratum), 'DeltaWIS'].dropna()
                if g.empty:
                    continue
                m = month[stratum].dropna() if stratum in month else pd.Series(dtype=float)
                for block in BLOCKS:
                    lo, hi = boot_month_means(m.to_numpy(), block, 1000 + mi*100 + horizon*10 + si + block)
                    rows.append(dict(model=model, horizon=horizon, stratum=stratum,
                                     n_forecasts=len(g), n_months=len(m),
                                     mean_DeltaWIS_forecast_weighted=g.mean(),
                                     median_DeltaWIS=g.median(), fraction_improved=(g < 0).mean(),
                                     mean_DeltaWIS_month_weighted=m.mean(), block_months=block,
                                     CI95_lower=lo, CI95_upper=hi,
                                     CI_excludes_zero=bool(lo > 0 or hi < 0) if np.isfinite(lo) else False))
            # High vs low: matched month-by-month contrast, not independent CIs.
            if {'high', 'low'}.issubset(month.columns):
                difference = (month['high'] - month['low']).dropna()
                for block in BLOCKS:
                    lo, hi = boot_month_means(difference.to_numpy(), block, 5000 + mi*100 + horizon*10 + block)
                    contrasts.append(dict(model=model, horizon=horizon, contrast='high_minus_low',
                                          n_months=len(difference), mean_difference=difference.mean(),
                                          block_months=block, CI95_lower=lo, CI95_upper=hi,
                                          CI_excludes_zero=bool(lo > 0 or hi < 0) if np.isfinite(lo) else False))
        print('Processed', model, flush=True)
    a = pd.DataFrame(rows)
    b = pd.DataFrame(contrasts)
    a.to_csv(OUT / 'fig2f_stratum_effects_bootstrap.csv', index=False)
    b.to_csv(OUT / 'fig2f_high_vs_low_bootstrap.csv', index=False)
    print('\nN-BEATS 6-month strata (3-month blocks):')
    print(a.query("model == 'N-BEATS' and horizon == 6 and block_months == 3").to_string(index=False))
    print('\nN-BEATS 6-month high vs low (3-month blocks):')
    print(b.query("model == 'N-BEATS' and horizon == 6 and block_months == 3").to_string(index=False))
    print('\nSaved:', OUT / 'fig2f_stratum_effects_bootstrap.csv', 'and', OUT / 'fig2f_high_vs_low_bootstrap.csv')


if __name__ == '__main__':
    main()
