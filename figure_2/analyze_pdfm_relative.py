#!/usr/bin/env python3
"""
03_pdfm_heterogeneity.py

Core paper analysis: heterogeneity of the PDFM effect.

Purpose
-------
Reproduce and extend the primary Brazil PDFM analysis without writing
the >1 GB paired municipality-month dataset to disk.

The script evaluates where and when PDFM-adjusted forecasts differ
from the base forecasts, with particular attention to the pattern seen
in the Mexico analysis:

    overall mean improvement
    vs.
    heterogeneous effects concentrated in a subset of observations.

Analyses
--------
1. Overall PDFM effect by model x horizon.
2. Case-burden stratification:
      - zero cases
      - low non-zero cases
      - medium non-zero cases
      - high non-zero cases
   Non-zero groups are defined separately within each target month,
   using tertiles of observed cases, to avoid calendar-period effects
   dominating the burden groups.
3. Municipality-level mean DeltaWIS over time.
4. Target-month mean DeltaWIS.
5. Temporal moving-block bootstrap confidence intervals.
6. Optional sensitivity analysis for block lengths 3, 6 and 12 months.

Important convention
--------------------
DeltaWIS = WIS_PDFM - WIS_Base

Therefore:
    negative DeltaWIS = improvement with PDFM
    positive DeltaWIS = worse with PDFM

Inputs
------
The existing WIS CSVs in:
    ../data/PDFM_analysis/BRA_full/

Outputs
-------
Compact CSVs only, written to:
    ./results/

No row-level merged paired dataset is saved.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from thucia.core.logging import enable_logging


enable_logging(level=logging.INFO)


# =============================================================================
# PATHS / CONFIGURATION
# =============================================================================

PROJECT_DIR = Path(__file__).resolve().parent

BRA_FULL = (
    PROJECT_DIR.parent.parent
    / "data"
    / "PDFM_analysis"
    / "BRA_full"
).resolve()

OUTPUT_DIR = PROJECT_DIR / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_DATE = pd.Period("2020-01", freq="M")
END_DATE = pd.Period("2025-08", freq="M")

HORIZONS = [1, 3, 6, 12]

MODELS = {
    "SARIMA": "sarima_h12",
    "TCN": "tcn_h12",
    "XGBoost": "xgboost_h12",
    "N-BEATS": "nbeats_h12",
    "TimesFM": "timesfm_h12",
}

BASE_SUFFIX = "_wis.csv"
PDFM_SUFFIX = "_pdfmrr_pinball_wis.csv"

BLOCK_LENGTHS = [3, 6, 12]
N_BOOT = 2000
BOOT_SEED = 42


# =============================================================================
# LOADERS
# =============================================================================

def load_wis_csv(path):
    """Load one saved WIS file using the same conventions as Figure 2."""
    df = pd.read_csv(path)

    df["Date"] = pd.PeriodIndex(df["Date"], freq="M")

    df = df[
        (df["Date"] >= START_DATE)
        & (df["Date"] <= END_DATE)
        & df["Cases"].notna()
    ].copy()

    if "R2" in df.columns:
        df["R2"] = df["R2"].clip(lower=0, upper=1)

    return df


def load_model_pair(filestem):
    """
    Load base and PDFM WIS CSVs and immediately return only the columns
    required for this analysis.

    The full paired dataset remains in memory while each model is processed,
    but no row-level merged dataset is written to disk.
    """
    base_path = BRA_FULL / f"{filestem}{BASE_SUFFIX}"
    pdfm_path = BRA_FULL / f"{filestem}{PDFM_SUFFIX}"

    logging.info("Loading base: %s", base_path)
    logging.info("Loading PDFM: %s", pdfm_path)

    base = load_wis_csv(base_path)
    pdfm = load_wis_csv(pdfm_path)

    keys = ["GID_2", "Date", "horizon"]

    base = base[keys + ["WIS", "Cases"]].copy()
    pdfm = pdfm[keys + ["WIS", "Cases"]].copy()

    merged = pd.merge(
        pdfm,
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
            f"{filestem}: observed Cases differ between paired WIS files."
        )

    merged["DeltaWIS"] = (
        merged["WIS_PDFM"] - merged["WIS_Base"]
    )

    merged["Cases"] = merged["Cases_Base"]

    return merged[
        ["GID_2", "Date", "horizon", "Cases", "WIS_Base", "WIS_PDFM", "DeltaWIS"]
    ].copy()


# =============================================================================
# CASE-BURDEN CATEGORIES
# =============================================================================

def assign_case_burden_by_month(df):
    """
    Assign burden groups separately within each target month.

    Categories:
        zero   = observed cases == 0
        low    = >0 and <= month-specific 33rd percentile
        medium = >33rd and <= month-specific 67th percentile
        high   = >67th percentile

    This mirrors the conceptual structure of the Mexico analysis while
    protecting against seasonal/month-specific shifts in the overall
    distribution of dengue incidence.
    """
    d = df.copy()

    d["burden"] = "zero"

    nonzero = d["Cases"] > 0

    if not nonzero.any():
        return d

    nz = d.loc[nonzero].copy()

    q = (
        nz.groupby("Date")["Cases"]
        .quantile([1 / 3, 2 / 3])
        .unstack()
        .rename(columns={
            1 / 3: "q33",
            2 / 3: "q67",
        })
    )

    nz = nz.join(q, on="Date")

    nz["burden"] = np.select(
        [
            nz["Cases"] <= nz["q33"],
            nz["Cases"] <= nz["q67"],
        ],
        [
            "low",
            "medium",
        ],
        default="high",
    )

    d.loc[nz.index, "burden"] = nz["burden"]

    return d


# =============================================================================
# OVERALL SUMMARY
# =============================================================================

def summarize_effect(df, group_cols):
    """Summarize DeltaWIS within arbitrary grouping columns."""
    records = []

    for keys, d in df.groupby(group_cols, observed=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        record = dict(zip(group_cols, keys))

        valid = d[["WIS_Base", "WIS_PDFM", "DeltaWIS"]].dropna()
        x = valid["DeltaWIS"]
        mean_base = valid["WIS_Base"].mean()
        mean_pdfm = valid["WIS_PDFM"].mean()
        relative_change_pct = (
            100.0 * (mean_pdfm - mean_base) / mean_base
            if mean_base != 0 else np.nan
        )

        record.update({
            "n": len(x),
            "mean_WIS_Base": mean_base,
            "mean_WIS_PDFM": mean_pdfm,
            "relative_mean_WIS_change_pct": relative_change_pct,
            "mean_DeltaWIS": x.mean(),
            "median_DeltaWIS": x.median(),
            "P10_DeltaWIS": x.quantile(0.10),
            "P90_DeltaWIS": x.quantile(0.90),
            "P95_DeltaWIS": x.quantile(0.95),
            "fraction_better": (x < 0).mean(),
            "fraction_worse": (x > 0).mean(),
            "fraction_equal": (x == 0).mean(),
        })

        records.append(record)

    return pd.DataFrame(records)


# =============================================================================
# MUNICIPALITY-LEVEL EFFECTS
# =============================================================================

def municipality_summary(df):
    """
    Compute the mean DeltaWIS over the evaluation period for each municipality.

    This is especially useful for detecting the pattern where a negative
    aggregate mean is driven by a subset of municipalities with large gains.
    """
    d = (
        df.groupby(
            ["model", "horizon", "GID_2"],
            observed=True,
        )["DeltaWIS"]
        .agg(
            mean_DeltaWIS="mean",
            median_DeltaWIS="median",
            n="count",
        )
        .reset_index()
    )

    summary = []

    for (model, horizon), g in d.groupby(
        ["model", "horizon"],
        observed=True,
    ):
        x = g["mean_DeltaWIS"]

        summary.append({
            "model": model,
            "horizon": horizon,
            "n_municipalities": len(g),
            "mean_municipality_DeltaWIS": x.mean(),
            "median_municipality_DeltaWIS": x.median(),
            "P25_municipality_DeltaWIS": x.quantile(0.25),
            "P75_municipality_DeltaWIS": x.quantile(0.75),
            "fraction_municipalities_better": (x < 0).mean(),
            "fraction_municipalities_worse": (x > 0).mean(),
        })

    return (
        pd.DataFrame(summary),
        d,
    )


# =============================================================================
# TEMPORAL SUMMARY
# =============================================================================

def temporal_summary(df):
    """Mean DeltaWIS by target month."""
    return summarize_effect(
        df,
        ["model", "horizon", "Date"],
    ).sort_values(
        ["model", "horizon", "Date"]
    )


# =============================================================================
# TEMPORAL MOVING-BLOCK BOOTSTRAP
# =============================================================================

def block_bootstrap_mean_ci(
    df,
    block_length=3,
    n_boot=2000,
    seed=42,
):
    """
    Bootstrap the mean DeltaWIS using contiguous blocks of target months.

    The resampling unit is month, not municipality-month, preserving
    temporal dependence and retaining all municipalities within each
    sampled month.

    Returns:
        mean observed DeltaWIS,
        lower 95% CI,
        upper 95% CI.
    """
    d = (
        df.groupby("Date", observed=True)["DeltaWIS"]
        .mean()
        .dropna()
        .sort_index()
    )

    values = d.to_numpy()

    if len(values) < block_length:
        return np.nan, np.nan, np.nan

    rng = np.random.default_rng(seed)

    n_dates = len(values)
    n_blocks = int(np.ceil(n_dates / block_length))

    samples = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        starts = rng.integers(
            0,
            n_dates - block_length + 1,
            size=n_blocks,
        )

        draw = np.concatenate(
            [
                values[start:start + block_length]
                for start in starts
            ]
        )[:n_dates]

        samples[b] = draw.mean()

    observed = values.mean()
    lower = np.quantile(samples, 0.025)
    upper = np.quantile(samples, 0.975)

    return observed, lower, upper


def bootstrap_summary(df):
    """
    Bootstrap overall temporal mean DeltaWIS separately for every
    model x horizon x block length.
    """
    records = []

    for (model, horizon), d in df.groupby(
        ["model", "horizon"],
        observed=True,
    ):
        for block_length in BLOCK_LENGTHS:
            observed, lower, upper = block_bootstrap_mean_ci(
                d,
                block_length=block_length,
                n_boot=N_BOOT,
                seed=BOOT_SEED + block_length,
            )

            records.append({
                "model": model,
                "horizon": horizon,
                "block_length_months": block_length,
                "mean_DeltaWIS": observed,
                "CI95_lower": lower,
                "CI95_upper": upper,
                "n_boot": N_BOOT,
            })

    return pd.DataFrame(records)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("CORE PAPER ANALYSIS 03 — PDFM HETEROGENEITY")
    print("=" * 90)
    print(f"Data directory: {BRA_FULL}")
    print(f"Evaluation period: {START_DATE} to {END_DATE}")
    print(f"Horizons: {HORIZONS}")
    print(f"Bootstrap blocks: {BLOCK_LENGTHS} months")
    print(f"Bootstrap replicates: {N_BOOT}")
    print("=" * 90)

    all_models = []

    for model, filestem in MODELS.items():
        d = load_model_pair(filestem)
        d["model"] = model

        logging.info(
            "%s: rows=%d, municipalities=%d, dates=%d",
            model,
            len(d),
            d["GID_2"].nunique(),
            d["Date"].nunique(),
        )

        all_models.append(d)

    df = pd.concat(
        all_models,
        ignore_index=True,
    )

    # -----------------------------------------------------------------
    # Add burden groups.
    # -----------------------------------------------------------------

    df = (
        df.groupby(
            ["model", "horizon"],
            group_keys=False,
            observed=True,
        )
        .apply(assign_case_burden_by_month)
        .reset_index(drop=True)
    )

    # -----------------------------------------------------------------
    # 1. Overall model x horizon effect.
    # -----------------------------------------------------------------

    overall = summarize_effect(
        df,
        ["model", "horizon"],
    )

    # -----------------------------------------------------------------
    # 2. Case-burden stratification.
    # -----------------------------------------------------------------

    burden = summarize_effect(
        df,
        ["model", "horizon", "burden"],
    )

    burden_order = [
        "zero",
        "low",
        "medium",
        "high",
    ]

    burden["burden"] = pd.Categorical(
        burden["burden"],
        categories=burden_order,
        ordered=True,
    )

    burden = burden.sort_values(
        ["model", "horizon", "burden"]
    )

    # -----------------------------------------------------------------
    # 3. Municipality-level summary.
    # -----------------------------------------------------------------

    municipality_aggregate, municipality_long = municipality_summary(df)

    # -----------------------------------------------------------------
    # 4. Target-month summary.
    # -----------------------------------------------------------------

    temporal = temporal_summary(df)

    # -----------------------------------------------------------------
    # 5. Temporal moving-block bootstrap.
    # -----------------------------------------------------------------

    bootstrap = bootstrap_summary(df)

    # -----------------------------------------------------------------
    # Print compact results.
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("TABLE 1 — OVERALL PDFM EFFECT")
    print("=" * 90)
    print(
        overall.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n")
    print("=" * 90)
    print("TABLE 2 — EFFECT BY CASE BURDEN")
    print("=" * 90)
    print(
        burden.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n")
    print("=" * 90)
    print("TABLE 3 — MUNICIPALITY-LEVEL EFFECT")
    print("=" * 90)
    print(
        municipality_aggregate.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print("\n")
    print("=" * 90)
    print("TABLE 4 — TEMPORAL MOVING-BLOCK BOOTSTRAP")
    print("=" * 90)
    print(
        bootstrap.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    # -----------------------------------------------------------------
    # Headline checks for the Brazil analogue of the Mexico pattern.
    # -----------------------------------------------------------------

    print("\n")
    print("=" * 90)
    print("HEADLINE CHECKS")
    print("=" * 90)

    for model in MODELS:
        for horizon in HORIZONS:
            g = overall[
                (overall["model"] == model)
                & (overall["horizon"] == horizon)
            ]

            if g.empty:
                continue

            row = g.iloc[0]

            print(
                f"{model:10s} h={horizon:2d}: "
                f"mean={row['mean_DeltaWIS']:+.6f}, "
                f"median={row['median_DeltaWIS']:+.6f}, "
                f"better={row['fraction_better']:.3f}"
            )

    # -----------------------------------------------------------------
    # Save compact outputs.
    # -----------------------------------------------------------------

    overall.to_csv(
        OUTPUT_DIR / "pdfm_heterogeneity_overall.csv",
        index=False,
    )

    burden.to_csv(
        OUTPUT_DIR / "pdfm_heterogeneity_case_burden.csv",
        index=False,
    )

    municipality_aggregate.to_csv(
        OUTPUT_DIR / "pdfm_heterogeneity_municipality_summary.csv",
        index=False,
    )

    municipality_long.to_csv(
        OUTPUT_DIR / "pdfm_heterogeneity_municipality_long.csv",
        index=False,
    )

    temporal.to_csv(
        OUTPUT_DIR / "pdfm_heterogeneity_temporal.csv",
        index=False,
    )

    bootstrap.to_csv(
        OUTPUT_DIR / "pdfm_heterogeneity_block_bootstrap.csv",
        index=False,
    )

    # A small, convenient table for eventual figure generation.
    burden_wide = burden.pivot_table(
        index=["model", "horizon"],
        columns="burden",
        values="mean_DeltaWIS",
        aggfunc="first",
    ).reset_index()

    burden_wide.to_csv(
        OUTPUT_DIR / "pdfm_heterogeneity_burden_mean_wide.csv",
        index=False,
    )

    print("\n")
    print("=" * 90)
    print("SAVED COMPACT OUTPUTS")
    print("=" * 90)

    for p in sorted(
        OUTPUT_DIR.glob("pdfm_heterogeneity_*.csv")
    ):
        print(p)

    print(OUTPUT_DIR / "pdfm_heterogeneity_burden_mean_wide.csv")


if __name__ == "__main__":
    main()
