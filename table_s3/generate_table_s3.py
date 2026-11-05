#!/usr/bin/env python3
"""
Recreate Supplementary Table S2 from canonical Figure 2 analysis outputs.

Inputs
------
../figure_2/data/pdfm_heterogeneity_block_bootstrap.csv
../figure_2/data/pdfm_heterogeneity_municipality_summary.csv
data/pdfm_heterogeneity_dm.csv

No raw data, DuckDB files, or row-level forecast tables are accessed.

Table S2 combines:
- mean DeltaWIS and 95% temporal bootstrap CI from the 3-month moving-block
  bootstrap;
- median municipality-level DeltaWIS and percentage of municipalities
  improved from the municipality summary;
- complementary two-sided Diebold-Mariano statistic and p-value from the
  chronological target-month mean WIS sequences.

Outputs
-------
table_s3.csv
table_s3.tsv
"""

from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "figure_2" / "data"

BOOTSTRAP = DATA / "pdfm_heterogeneity_block_bootstrap.csv"
MUNICIPALITY = DATA / "pdfm_heterogeneity_municipality_summary.csv"
DM = HERE / "data" / "pdfm_heterogeneity_dm.csv"

OUTPUT_CSV = HERE / "table_s3.csv"
OUTPUT_TSV = HERE / "table_s3.tsv"

MODELS = ["N-BEATS", "SARIMA", "TCN", "TimesFM", "XGBoost"]
HORIZONS = [1, 3, 6, 12]
BLOCK_LENGTH = 3
EXPECTED_N_MUNICIPALITIES = 5568
EXPECTED_N_TARGET_MONTHS = 68


def load_bootstrap():
    df = pd.read_csv(BOOTSTRAP)

    required = {
        "model",
        "horizon",
        "block_length_months",
        "mean_DeltaWIS",
        "CI95_lower",
        "CI95_upper",
        "n_boot",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{BOOTSTRAP.name}: missing columns {sorted(missing)}; "
            f"found {list(df.columns)}"
        )

    df = df[df["block_length_months"] == BLOCK_LENGTH].copy()

    if df.empty:
        raise ValueError(f"No {BLOCK_LENGTH}-month bootstrap rows found.")

    return df


def load_municipality():
    df = pd.read_csv(MUNICIPALITY)

    required = {
        "model",
        "horizon",
        "n_municipalities",
        "median_municipality_DeltaWIS",
        "fraction_municipalities_better",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{MUNICIPALITY.name}: missing columns {sorted(missing)}; "
            f"found {list(df.columns)}"
        )

    return df


def load_dm():
    df = pd.read_csv(DM)

    required = {
        "model",
        "horizon",
        "n_target_months",
        "DM_statistic",
        "DM_p_value",
        "DM_h",
        "DM_variance_estimator",
        "DM_harvey_correction",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{DM.name}: missing columns {sorted(missing)}; "
            f"found {list(df.columns)}"
        )

    return df


def validate_keys(df, label):
    expected = {
        (model, horizon)
        for model in MODELS
        for horizon in HORIZONS
    }
    observed = set(zip(df["model"].astype(str), df["horizon"]))

    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            f"{label}: unexpected model/horizon combinations. "
            f"Missing: {missing}; Extra: {extra}"
        )

    if len(df) != len(expected):
        raise ValueError(
            f"{label}: expected {len(expected)} rows, got {len(df)}"
        )


def main():
    print("=" * 80)
    print("TABLE S2 — HETEROGENEOUS EFFECTS OF PDFM ADJUSTMENT")
    print("=" * 80)

    boot = load_bootstrap()
    muni = load_municipality()
    dm = load_dm()

    validate_keys(boot, "Bootstrap")
    validate_keys(muni, "Municipality summary")
    validate_keys(dm, "Diebold-Mariano")

    if set(boot["n_boot"].unique()) != {2000}:
        raise ValueError(
            f"Expected 2,000 bootstrap replicates; "
            f"found {sorted(boot['n_boot'].unique())}"
        )

    if set(muni["n_municipalities"].unique()) != {
        EXPECTED_N_MUNICIPALITIES
    }:
        raise ValueError(
            f"Expected {EXPECTED_N_MUNICIPALITIES} municipalities; "
            f"found {sorted(muni['n_municipalities'].unique())}"
        )

    if set(dm["n_target_months"].unique()) != {EXPECTED_N_TARGET_MONTHS}:
        raise ValueError(
            f"Expected {EXPECTED_N_TARGET_MONTHS} target months; "
            f"found {sorted(dm['n_target_months'].unique())}"
        )

    if set(dm["DM_h"].unique()) != set(HORIZONS):
        raise ValueError(
            f"Expected DM lag h to equal forecast horizon; "
            f"found {sorted(dm['DM_h'].unique())}"
        )

    if set(dm["DM_variance_estimator"].astype(str).str.lower().unique()) != {"bartlett"}:
        raise ValueError(
            "Expected Bartlett DM variance estimator."
        )

    if not bool(dm["DM_harvey_correction"].astype(bool).all()):
        raise ValueError("Expected Harvey small-sample correction for all DM tests.")

    out = (
        boot[
            [
                "model",
                "horizon",
                "mean_DeltaWIS",
                "CI95_lower",
                "CI95_upper",
            ]
        ]
        .merge(
            muni[
                [
                    "model",
                    "horizon",
                    "median_municipality_DeltaWIS",
                    "fraction_municipalities_better",
                ]
            ],
            on=["model", "horizon"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            dm[
                ["model", "horizon", "DM_statistic", "DM_p_value"]
            ],
            on=["model", "horizon"],
            how="inner",
            validate="one_to_one",
        )
    )

    out["95% temporal bootstrap CI"] = out.apply(
        lambda r: f"[{r['CI95_lower']:.4f}, {r['CI95_upper']:.4f}]",
        axis=1,
    )

    out["Municipalities improved (%)"] = (
        100 * out["fraction_municipalities_better"]
    ).round(1)

    out = out.rename(
        columns={
            "model": "Model",
            "horizon": "Horizon (months)",
            "mean_DeltaWIS": "Mean ΔWIS (PDFM − base)",
            "median_municipality_DeltaWIS":
                "Median municipality-level ΔWIS",
            "DM_statistic": "DM statistic",
            "DM_p_value": "DM p-value",
        }
    )

    out = out[
        [
            "Model",
            "Horizon (months)",
            "Mean ΔWIS (PDFM − base)",
            "95% temporal bootstrap CI",
            "Median municipality-level ΔWIS",
            "Municipalities improved (%)",
            "DM statistic",
            "DM p-value",
        ]
    ]

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

    out["Mean ΔWIS (PDFM − base)"] = (
        out["Mean ΔWIS (PDFM − base)"].round(4)
    )
    out["Median municipality-level ΔWIS"] = (
        out["Median municipality-level ΔWIS"].round(4)
    )
    out["DM statistic"] = out["DM statistic"].round(3)
    out["DM p-value"] = out["DM p-value"].map(
        lambda x: f"{x:.4g}" if x >= 0.0001 else "<0.0001"
    )

    out.to_csv(OUTPUT_CSV, index=False)
    out.to_csv(OUTPUT_TSV, index=False, sep="\t")

    print(out.to_string(index=False))
    print()
    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {OUTPUT_TSV}")


if __name__ == "__main__":
    main()
