#!/usr/bin/env python3
"""Prepare Figure 2 Panel A inputs from prepared data stores.

This is the only Figure 2 preparation script that accesses the large prepared
DuckDB stores. The public plotting script reads only the resulting CSVs.
"""
from pathlib import Path
import pandas as pd
from thucia.core.cases import read_db

HERE = Path(__file__).resolve().parent
BRA_FULL = (HERE.parent.parent / "data" / "PDFM_analysis" / "BRA_full").resolve()
OUT = HERE / "data"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "SARIMA": "sarima_h12",
    "TCN": "tcn_h12",
    "XGBoost": "xgboost_h12",
    "N-BEATS": "nbeats_h12",
    "TimesFM": "timesfm_h12",
}

def month_end(series):
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.dt.to_timestamp() + pd.offsets.MonthEnd()
    return pd.to_datetime(series)

def load_observed():
    df = read_db(str(BRA_FULL / "cases_with_climate")).df.copy()
    df["Date"] = month_end(df["Date"])
    out = df.groupby("Date", as_index=False)["Cases"].sum().sort_values("Date")
    return out

def load_median_forecast(model, horizon):
    df = read_db(str(BRA_FULL / f"{MODELS[model]}_cases_quantiles")).df.copy()
    df["Date"] = month_end(df["Date"])
    d = df[(df["quantile"] == 0.5) & (df["horizon"] == horizon)]
    return d.groupby("Date", as_index=False)["prediction"].sum().sort_values("Date")

def main():
    observed = load_observed()
    observed.to_csv(OUT / "figure2_observed_national.csv", index=False)
    rows=[]
    for model,h in [("TCN",1),("TimesFM",12)]:
        d=load_median_forecast(model,h).copy()
        d["model"]=model
        d["horizon"]=h
        rows.append(d[["model","horizon","Date","prediction"]])
    reps=pd.concat(rows, ignore_index=True)
    reps.to_csv(OUT / "figure2_representative_forecasts.csv", index=False)
    print(f"Wrote {OUT / 'figure2_observed_national.csv'} ({len(observed):,} rows)")
    print(f"Wrote {OUT / 'figure2_representative_forecasts.csv'} ({len(reps):,} rows)")

if __name__ == "__main__":
    main()
