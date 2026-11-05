#!/usr/bin/env python3
"""
Recreate Supplementary Table S7 from the canonical Figure 3B summary.

Input
-----
../figure_3/data/figure3_B_spatial_summary.csv

The current manuscript labels S7 as the univariate-versus-multivariate
municipality-level comparison. The body of the current document, however,
contains the 27-state aggregation values that belong to the S5 analysis.
This script follows the S7 caption/definition and the canonical Figure 3B
analysis output.

No raw data, DuckDB files, or row-level WIS tables are required.

Outputs
-------
table_s7.csv
table_s7.tsv
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
INPUT = HERE.parent / "figure_3" / "data" / "figure3_B_spatial_summary.csv"
OUTPUT_CSV = HERE / "table_s7.csv"
OUTPUT_TSV = HERE / "table_s7.tsv"

MODELS = ["N-BEATS", "SARIMA", "TCN", "TimesFM", "XGBoost"]
HORIZONS = [1, 3, 6, 12]
EXPECTED_N_UNITS = 20


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
        "W_statistic",
        "p_value",
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
        zip(
            df["model"].astype(str),
            pd.to_numeric(df["horizon"], errors="raise"),
        )
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

    n_values = pd.to_numeric(df["n_units"], errors="raise").unique()
    if set(n_values) != {EXPECTED_N_UNITS}:
        raise ValueError(
            f"Expected {EXPECTED_N_UNITS} municipalities in every comparison; "
            f"found {sorted(n_values)}"
        )

    return df


def build_table(df):
    out = df[
        [
            "model",
            "horizon",
            "n_units",
            "mean_DeltaWIS",
            "median_DeltaWIS",
            "P25_DeltaWIS",
            "P75_DeltaWIS",
            "units_first_better_pct",
            "W_statistic",
            "p_value",
        ]
    ].copy()

    out = out.rename(
        columns={
            "model": "Model",
            "horizon": "Horizon",
            "n_units": "n",
            "mean_DeltaWIS": "Mean ΔWIS",
            "median_DeltaWIS": "Median ΔWIS",
            "P25_DeltaWIS": "P25",
            "P75_DeltaWIS": "P75",
            "units_first_better_pct": "Univariate better (%)",
            "W_statistic": "W",
            "p_value": "p-value",
        }
    )

    out["Model"] = pd.Categorical(
        out["Model"],
        categories=MODELS,
        ordered=True,
    )
    out["Horizon"] = pd.Categorical(
        out["Horizon"],
        categories=HORIZONS,
        ordered=True,
    )

    out = out.sort_values(
        ["Model", "Horizon"]
    ).reset_index(drop=True)

    for col in ["Mean ΔWIS", "Median ΔWIS", "P25", "P75"]:
        out[col] = out[col].round(3)

    out["Univariate better (%)"] = out[
        "Univariate better (%)"
    ].round(1)

    out["p-value"] = out["p-value"].map(
        lambda x: "<0.0001" if x < 0.0001 else f"{x:.4f}"
    )

    return out


def main():
    print("=" * 80)
    print("TABLE S7 — UNIVARIATE VERSUS MULTIVARIATE FITTING")
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
