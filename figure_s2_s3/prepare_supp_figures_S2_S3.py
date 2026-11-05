#!/usr/bin/env python3
"""
Prepare compact publication data for Supplementary Figures S2 and S3.

S2 = 6-month horizon
S3 = 12-month horizon

IMPORTANT
---------
The SARIMA >1,000,000 rule is a VISUALIZATION-ONLY rule for Panel A.
It is applied to each municipality-level median prediction BEFORE those
predictions are summed to the national monthly total. It does not modify
or feed into the WIS analysis.

Outputs:
    data/figure_s2_s3_observed.csv
    data/figure_s2_s3_forecasts.csv
    data/figure_s2_s3_wis.csv
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd
from thucia.core.cases import read_db

HERE = Path(__file__).resolve().parent
BRA_FULL = (HERE.parent.parent / "data" / "PDFM_analysis" / "BRA_full").resolve()
OUT = HERE / "data"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Period("2020-01", freq="M")
END = pd.Period("2025-08", freq="M")
HORIZONS = [6, 12]
PLOT_THRESHOLD = 1_000_000
EXCLUDED_GIDS = {"BRA.21.229_2", "BRA.21.231_2"}

MODELS = {
    "SARIMA": "sarima_h12",
    "TCN": "tcn_h12",
    "XGBoost": "xgboost_h12",
    "N-BEATS": "nbeats_h12",
    "TimesFM": "timesfm_h12",
}

BASE_Q_PATTERNS = [
    "{stem}_cases_quantiles",
    "{stem}_cases_quantiles.duckdb",
]
PDFM_Q_PATTERNS = [
    "{stem}_pdfmrr_pinball_cases_quantiles",
    "{stem}_pdfmrr_pinball_cases_quantiles.duckdb",
    "{stem}_pdfmrr_2025_pinball_cases_quantiles",
    "{stem}_pdfmrr_2025_pinball_cases_quantiles.duckdb",
]


def resolve_path(candidates):
    for name in candidates:
        p = BRA_FULL / name
        if p.exists():
            return p

    lower_map = {
        p.name.lower(): p
        for p in BRA_FULL.iterdir()
        if p.is_file()
    }
    for name in candidates:
        p = lower_map.get(name.lower())
        if p is not None:
            return p

    raise FileNotFoundError(
        "Could not resolve any of:\n" + "\n".join(candidates)
    )


def load_observed():
    db = BRA_FULL / "cases_with_climate"
    df = read_db(str(db)).df.copy()

    df["Date"] = pd.PeriodIndex(df["Date"], freq="M")
    df = df[
        (df["Date"] >= START)
        & (df["Date"] <= END)
        & df["Cases"].notna()
    ].copy()

    out = (
        df.groupby("Date", as_index=False)["Cases"]
        .sum()
        .sort_values("Date")
    )
    out["Date"] = out["Date"].dt.to_timestamp(how="end")
    return out


def load_median_quantiles(stem, pdfm=False):
    patterns = PDFM_Q_PATTERNS if pdfm else BASE_Q_PATTERNS
    path = resolve_path([p.format(stem=stem) for p in patterns])

    logging.info("Loading quantiles: %s", path)
    df = read_db(str(path)).df.copy()

    required = {"GID_2", "Date", "horizon", "quantile", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    df["Date"] = pd.PeriodIndex(df["Date"], freq="M")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")

    return df[
        ~df["GID_2"].isin(EXCLUDED_GIDS)
        & (df["Date"] >= START)
        & (df["Date"] <= END)
        & df["horizon"].isin(HORIZONS)
        & np.isclose(df["quantile"], 0.5)
    ][["GID_2", "Date", "horizon", "prediction"]].copy()


def national_forecast(model, stem, horizon, pdfm):
    df = load_median_quantiles(stem, pdfm=pdfm)
    df = df[df["horizon"] == horizon].copy()

    # VISUALIZATION ONLY:
    # remove individual municipality median predictions >1M BEFORE aggregation.
    # This preserves the rest of the target month's municipalities.
    df["prediction_plot"] = df["prediction"]
    invalid = (
        ~np.isfinite(df["prediction_plot"])
        | (df["prediction_plot"] > PLOT_THRESHOLD)
    )
    removed = int(invalid.sum())

    df.loc[invalid, "prediction_plot"] = np.nan

    out = (
        df.groupby("Date", as_index=False)["prediction_plot"]
        .sum(min_count=1)
    )
    out["Date"] = out["Date"].dt.to_timestamp(how="end")
    out["model"] = model
    out["horizon"] = horizon
    out["configuration"] = "PDFM" if pdfm else "Base"
    out["n_municipalities_filtered"] = removed

    return out[
        [
            "model",
            "horizon",
            "Date",
            "configuration",
            "prediction_plot",
            "n_municipalities_filtered",
        ]
    ]


def load_wis_pair(model, stem, horizon):
    base_path = BRA_FULL / f"{stem}_wis.csv"
    pdfm_path = BRA_FULL / f"{stem}_pdfmrr_pinball_wis.csv"

    base = pd.read_csv(base_path)
    pdfm = pd.read_csv(pdfm_path)

    for df in (base, pdfm):
        df["Date"] = pd.PeriodIndex(df["Date"], freq="M")
        df["WIS"] = pd.to_numeric(df["WIS"], errors="coerce")
        df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")

    base = base[
        (base["Date"] >= START)
        & (base["Date"] <= END)
        & (base["horizon"] == horizon)
        & base["Cases"].notna()
    ][["GID_2", "Date", "horizon", "WIS", "Cases"]].copy()

    pdfm = pdfm[
        (pdfm["Date"] >= START)
        & (pdfm["Date"] <= END)
        & (pdfm["horizon"] == horizon)
        & pdfm["Cases"].notna()
    ][["GID_2", "Date", "horizon", "WIS", "Cases"]].copy()

    keys = ["GID_2", "Date", "horizon"]
    merged = pdfm.merge(
        base,
        on=keys,
        suffixes=("_PDFM", "_Base"),
        how="inner",
        validate="one_to_one",
    )

    if not np.allclose(
        merged["Cases_PDFM"].to_numpy(),
        merged["Cases_Base"].to_numpy(),
        equal_nan=True,
    ):
        raise ValueError(
            f"{model}, h{horizon}: Cases differ between paired WIS inputs."
        )

    if merged["GID_2"].isin(EXCLUDED_GIDS).any():
        raise ValueError(f"{model}, h{horizon}: excluded geographic units in paired WIS")

    merged["DeltaWIS"] = merged["WIS_PDFM"] - merged["WIS_Base"]
    merged["model"] = model
    merged["horizon"] = horizon

    monthly = (
        merged.groupby(["Date"], as_index=False)
        .agg(
            base_WIS=("WIS_Base", "mean"),
            pdfm_WIS=("WIS_PDFM", "mean"),
            DeltaWIS=("DeltaWIS", "mean"),
            n_base_finite=("WIS_Base", "count"),
            n_pdfm_finite=("WIS_PDFM", "count"),
            n_delta_finite=("DeltaWIS", "count"),
        )
        .sort_values("Date")
    )
    monthly["model"] = model
    monthly["horizon"] = horizon
    monthly["Date"] = monthly["Date"].dt.to_timestamp(how="end")
    return monthly


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 90)
    print("PREPARE SUPPLEMENTARY FIGURES S2 AND S3")
    print("=" * 90)
    print(f"BRA_FULL: {BRA_FULL}")
    print(f"Evaluation: {START} to {END}")
    print(f"Horizons: {HORIZONS}")
    print(f"Panel A visualization threshold: {PLOT_THRESHOLD:,}")
    print()

    observed = load_observed()
    observed.to_csv(OUT / "figure_s2_s3_observed.csv", index=False)

    forecast_rows = []
    wis_rows = []

    for model, stem in MODELS.items():
        for horizon in HORIZONS:
            for pdfm in (False, True):
                forecast_rows.append(
                    national_forecast(
                        model=model,
                        stem=stem,
                        horizon=horizon,
                        pdfm=pdfm,
                    )
                )
            wis_rows.append(load_wis_pair(model, stem, horizon))

    forecasts = pd.concat(forecast_rows, ignore_index=True)
    wis = pd.concat(wis_rows, ignore_index=True)

    forecasts.to_csv(OUT / "figure_s2_s3_forecasts.csv", index=False)
    wis.to_csv(OUT / "figure_s2_s3_wis.csv", index=False)

    print(f"Wrote {OUT / 'figure_s2_s3_observed.csv'}")
    print(f"Wrote {OUT / 'figure_s2_s3_forecasts.csv'}")
    print(f"Wrote {OUT / 'figure_s2_s3_wis.csv'}")

    print("\nPanel A filtered municipality records by model/horizon/configuration:")
    print(
        forecasts.groupby(
            ["model", "horizon", "configuration"],
            as_index=False,
        )["n_municipalities_filtered"].max().to_string(index=False)
    )


if __name__ == "__main__":
    main()
