#!/usr/bin/env python3
"""
Recreate Supplementary Table S1 from the canonical compact baseline summaries.

Input files (all constructed analysis outputs):
    data/baseline_mean_wis.csv
    data/baseline_median_wis.csv
    data/baseline_p95_wis.csv
    data/baseline_mean_r2.csv

These canonical files are wide: one row per model and columns 1, 3, 6, 12
for the forecast horizons.

No raw data, DuckDB files, or row-level forecast tables are required.

Outputs:
    table_s1.csv
    table_s1.tsv
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT_CSV = HERE / "table_s1.csv"
OUTPUT_TSV = HERE / "table_s1.tsv"

MODELS = ["SARIMA", "TCN", "XGBoost", "N-BEATS", "TimesFM"]
HORIZONS = [1, 3, 6, 12]

FILES = {
    "Mean WIS": "baseline_mean_wis.csv",
    "Median WIS": "baseline_median_wis.csv",
    "P95 WIS": "baseline_p95_wis.csv",
    "Mean R2": "baseline_mean_r2.csv",
}


def read_wide_summary(label):
    path = DATA / FILES[label]
    if not path.exists():
        raise FileNotFoundError(f"Missing constructed input: {path}")

    df = pd.read_csv(path)

    required = {"model", *(str(h) for h in HORIZONS)}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name}: missing required columns {sorted(missing)}; "
            f"found {list(df.columns)}"
        )

    # The canonical summary is one row per model, with horizons in columns.
    if len(df) != len(MODELS):
        raise ValueError(
            f"{path.name}: expected {len(MODELS)} model rows, got {len(df)}"
        )

    observed_models = set(df["model"].astype(str))
    if observed_models != set(MODELS):
        raise ValueError(
            f"{path.name}: expected models {MODELS}, "
            f"found {sorted(observed_models)}"
        )

    out = df[["model", *(str(h) for h in HORIZONS)]].copy()
    out = out.melt(
        id_vars="model",
        var_name="horizon",
        value_name="value",
    )
    out["horizon"] = pd.to_numeric(out["horizon"], errors="raise")
    out["value"] = pd.to_numeric(out["value"], errors="raise")

    return out


def main():
    print("=" * 80)
    print("TABLE S1 — BASELINE FORECASTING PERFORMANCE")
    print("=" * 80)
    print(f"Input directory: {DATA}")

    summaries = {
        label: read_wide_summary(label)
        for label in FILES
    }

    out = summaries["Mean WIS"].rename(columns={"value": "Mean WIS"})

    for label in ["Median WIS", "P95 WIS", "Mean R2"]:
        tmp = summaries[label].rename(columns={"value": label})
        out = out.merge(
            tmp,
            on=["model", "horizon"],
            how="outer",
            validate="one_to_one",
        )

    if out.isna().any().any():
        raise ValueError("Merged Table S1 contains missing values.")

    out["model"] = pd.Categorical(
        out["model"],
        categories=MODELS,
        ordered=True,
    )
    out["horizon"] = pd.Categorical(
        out["horizon"],
        categories=HORIZONS,
        ordered=True,
    )

    out = out.sort_values(["model", "horizon"]).reset_index(drop=True)

    for col in ["Mean WIS", "Median WIS", "P95 WIS", "Mean R2"]:
        out[col] = out[col].round(3)

    out.to_csv(OUTPUT_CSV, index=False)
    out.to_csv(OUTPUT_TSV, sep="\t", index=False)

    print(f"Output rows: {len(out)}")
    print()
    print(out.to_string(index=False))
    print()
    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
