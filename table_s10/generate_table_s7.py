#!/usr/bin/env python3
"""
Recreate Supplementary Table S10 from the canonical Run3a hold-out summary.

Input
-----
run3a_holdout_wis_summary.csv

Expected location, in order:
1. ./run3a_holdout_wis_summary.csv
2. ../ERA_model/run3a/run3a_holdout_wis_summary.csv
3. ../../data/PDFM_analysis/ERA_model/run3a/run3a_holdout_wis_summary.csv

The input is already a constructed evaluation table. No raw data, DuckDB
forecast stores, or model fitting is performed here.

Table S10 reports WIS across municipality-month forecasts during the
independent September 2023-August 2025 hold-out.

Outputs
-------
table_s10.csv
table_s10.tsv
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent

CANDIDATES = [
    HERE / "run3a_holdout_wis_summary.csv",
    HERE.parent / "ERA_model" / "run3a" / "run3a_holdout_wis_summary.csv",
    HERE.parent.parent / "data" / "PDFM_analysis" / "ERA_model" / "run3a"
    / "run3a_holdout_wis_summary.csv",
]

OUTPUT_CSV = HERE / "table_s10.csv"
OUTPUT_TSV = HERE / "table_s10.tsv"

MODELS = ["ERA", "SARIMA", "TCN", "N-BEATS", "TimesFM", "XGBoost"]
HORIZONS = [1, 3, 6, 12]
EXPECTED_N_FORECASTS = 133680
EXPECTED_N_MUNICIPALITIES = 5570
EXPECTED_N_MONTHS = 24


def locate_input():
    for path in CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find run3a_holdout_wis_summary.csv. Checked:\n"
        + "\n".join(str(p) for p in CANDIDATES)
    )


def load_input(path):
    df = pd.read_csv(path)

    required = {
        "model",
        "horizon",
        "n_forecasts",
        "mean_WIS",
        "median_WIS",
        "P90_WIS",
        "P95_WIS",
        "n_municipalities",
        "n_months",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name}: missing columns {sorted(missing)}; "
            f"found {list(df.columns)}"
        )

    # S10 compares ERA with the five primary forecasting models.
    # Other Run3a models are excluded from this table.
    df = df[
        df["model"].isin(MODELS)
        & df["horizon"].isin(HORIZONS)
    ].copy()

    expected_keys = {
        (model, horizon)
        for model in MODELS
        for horizon in HORIZONS
    }
    observed_keys = set(zip(df["model"], df["horizon"]))

    if observed_keys != expected_keys:
        missing_keys = sorted(expected_keys - observed_keys)
        extra_keys = sorted(observed_keys - expected_keys)
        raise ValueError(
            f"Unexpected S10 model/horizon combinations. "
            f"Missing: {missing_keys}; Extra: {extra_keys}"
        )

    if len(df) != len(expected_keys):
        raise ValueError(
            f"Expected {len(expected_keys)} S10 rows, got {len(df)}"
        )

    checks = {
        "n_forecasts": EXPECTED_N_FORECASTS,
        "n_municipalities": EXPECTED_N_MUNICIPALITIES,
        "n_months": EXPECTED_N_MONTHS,
    }
    for col, expected in checks.items():
        observed = set(pd.to_numeric(df[col], errors="raise").unique())
        if observed != {expected}:
            raise ValueError(
                f"{col}: expected {expected} throughout S10; "
                f"found {sorted(observed)}"
            )

    return df


def build_table(df):
    out = df[
        [
            "model",
            "horizon",
            "mean_WIS",
            "median_WIS",
            "P90_WIS",
            "P95_WIS",
        ]
    ].copy()

    out = out.rename(
        columns={
            "model": "Model",
            "horizon": "Horizon (months)",
            "mean_WIS": "Mean WIS",
            "median_WIS": "Median WIS",
            "P90_WIS": "P90 WIS",
            "P95_WIS": "P95 WIS",
        }
    )

    out["Model"] = pd.Categorical(
        out["Model"], categories=MODELS, ordered=True
    )
    out["Horizon (months)"] = pd.Categorical(
        out["Horizon (months)"],
        categories=HORIZONS,
        ordered=True,
    )
    out = out.sort_values(
        ["Model", "Horizon (months)"]
    ).reset_index(drop=True)

    # Match manuscript display precision.
    out["Mean WIS"] = out["Mean WIS"].round(1)
    out["Median WIS"] = out["Median WIS"].round(2)
    out["P90 WIS"] = out["P90 WIS"].round(1)
    out["P95 WIS"] = out["P95 WIS"].round(1)

    return out


def main():
    print("=" * 80)
    print("TABLE S10 — INDEPENDENT HOLD-OUT PERFORMANCE")
    print("=" * 80)

    path = locate_input()
    print(f"Input: {path}")

    df = load_input(path)
    out = build_table(df)

    out.to_csv(OUTPUT_CSV, index=False)
    out.to_csv(OUTPUT_TSV, index=False, sep="\t")

    print(out.to_string(index=False))
    print()
    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
