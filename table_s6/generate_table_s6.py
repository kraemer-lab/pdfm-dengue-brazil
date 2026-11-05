#!/usr/bin/env python3
"""
Recreate Supplementary Table S6 from the canonical Figure 3A summary.

Input
-----
../figure_3/data/figure3_A_covariate_summary.csv

This is a constructed analysis output produced by analyze_fig3.py.
No raw data, DuckDB files, or row-level WIS tables are required.

Table S6:
    Model
    Horizon (months)
    Mean ΔWIS
    Median ΔWIS
    P25
    P75
    Municipalities favouring minimal (%)

ΔWIS = WIS_minimal - WIS_full.

Outputs
-------
table_s6.csv
table_s6.tsv
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / "figure_3" / "data" / "figure3_A_covariate_summary.csv"
OUTPUT_CSV = HERE / "table_s6.csv"
OUTPUT_TSV = HERE / "table_s6.tsv"

MODELS = ["N-BEATS", "SARIMA", "TCN", "TimesFM", "XGBoost"]
HORIZONS = [1, 3, 6, 12]
EXPECTED_N_UNITS = 5570


def load_input():
    if not INPUT.exists():
        raise FileNotFoundError(f"Missing constructed input: {INPUT}")

    df = pd.read_csv(INPUT)

    required = {
        "model",
        "horizon",
        "n_units",
        "mean_DeltaWIS",
        "median_DeltaWIS",
        "P25_DeltaWIS",
        "P75_DeltaWIS",
        "units_first_better_pct",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{INPUT.name}: missing columns {sorted(missing)}; "
            f"found {list(df.columns)}"
        )

    expected_keys = {
        (model, horizon)
        for model in MODELS
        for horizon in HORIZONS
    }
    observed_keys = set(
        zip(df["model"].astype(str), pd.to_numeric(df["horizon"]))
    )

    if observed_keys != expected_keys:
        missing_keys = sorted(expected_keys - observed_keys)
        extra_keys = sorted(observed_keys - expected_keys)
        raise ValueError(
            f"Unexpected model/horizon combinations. "
            f"Missing: {missing_keys}; Extra: {extra_keys}"
        )

    if len(df) != len(expected_keys):
        raise ValueError(
            f"Expected {len(expected_keys)} rows, got {len(df)}"
        )

    if set(pd.to_numeric(df["n_units"]).unique()) != {EXPECTED_N_UNITS}:
        raise ValueError(
            f"Expected {EXPECTED_N_UNITS} municipalities in every comparison; "
            f"found {sorted(pd.to_numeric(df['n_units']).unique())}"
        )

    return df


def build_table(df):
    out = df[
        [
            "model",
            "horizon",
            "mean_DeltaWIS",
            "median_DeltaWIS",
            "P25_DeltaWIS",
            "P75_DeltaWIS",
            "units_first_better_pct",
        ]
    ].copy()

    out = out.rename(
        columns={
            "model": "Model",
            "horizon": "Horizon (months)",
            "mean_DeltaWIS": "Mean ΔWIS",
            "median_DeltaWIS": "Median ΔWIS",
            "P25_DeltaWIS": "P25",
            "P75_DeltaWIS": "P75",
            "units_first_better_pct": "Municipalities favouring minimal (%)",
        }
    )

    out["Model"] = pd.Categorical(
        out["Model"],
        categories=MODELS,
        ordered=True,
    )
    out["Horizon (months)"] = pd.Categorical(
        out["Horizon (months)"],
        categories=HORIZONS,
        ordered=True,
    )

    out = out.sort_values(
        ["Model", "Horizon (months)"]
    ).reset_index(drop=True)

    for col in ["Mean ΔWIS", "Median ΔWIS", "P25", "P75"]:
        out[col] = out[col].round(3)

    out["Municipalities favouring minimal (%)"] = (
        out["Municipalities favouring minimal (%)"].round(1)
    )

    return out


def main():
    print("=" * 80)
    print("TABLE S6 — EFFECT OF COVARIATE-SET SIZE")
    print("=" * 80)
    print(f"Input: {INPUT}")

    df = load_input()
    out = build_table(df)

    out.to_csv(OUTPUT_CSV, index=False)
    out.to_csv(OUTPUT_TSV, index=False, sep="\t")

    print(out.to_string(index=False))
    print()
    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
