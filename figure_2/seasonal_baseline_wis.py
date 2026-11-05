#!/usr/bin/env python3
"""
Construct a frozen seasonal-mean benchmark for Figure 2B.

Benchmark definition
--------------------
For each municipality and calendar month, calculate the mean of
log(1 + reported dengue cases) over the fixed January 2010-December 2019
training period. The resulting 12 monthly values per municipality are frozen
and applied to every target month in the January 2020-August 2025 evaluation
period.

The benchmark is scored using the same Thucia WIS implementation used for the
primary model WIS files: observed cases and the benchmark forecast are on the
log(1 + cases) scale, and the same quantile grid is supplied to `wis`. The
seasonal mean is represented as a degenerate predictive distribution (all
quantiles equal the seasonal mean), so no alternative WIS implementation is
introduced.

Outputs
-------
figure2/data/seasonal_baseline_wis.csv
figure2/data/model_vs_seasonal_skill.csv

The second file contains municipality-month-horizon paired model WIS and
seasonal-baseline WIS, plus two municipality-month paired relative-performance measures:
    skill_pct = 100 * (1 - WIS_model / WIS_seasonal)
        defined only where WIS_seasonal > 0
    log1p_skill = log1p(WIS_seasonal) - log1p(WIS_model)
        defined for all finite non-negative WIS pairs

For both measures, positive values indicate lower model WIS than the municipality-
specific seasonal benchmark. The log1p measure is included as a zero-safe
exploratory alternative; it is not a conventional WIS skill score.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from thucia.core.cases import read_db, wis
from thucia.core.logging import enable_logging


enable_logging(level=logging.INFO)


# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
BRA_FULL = (HERE.parent.parent / "data" / "PDFM_analysis" / "BRA_full").resolve()
OUTPUT_DIR = HERE / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CASES_PER_MONTH = BRA_FULL / "cases_with_climate"

TRAIN_START = pd.Period("2010-01", freq="M")
TRAIN_END = pd.Period("2019-12", freq="M")
EVAL_START = pd.Period("2020-01", freq="M")
EVAL_END = pd.Period("2025-08", freq="M")

MODELS = {
    "SARIMA": "sarima_h12_wis.csv",
    "TCN": "tcn_h12_wis.csv",
    "XGBoost": "xgboost_h12_wis.csv",
    "N-BEATS": "nbeats_h12_wis.csv",
    "TimesFM": "timesfm_h12_wis.csv",
}

HORIZONS = [1, 3, 6, 12]
QUANTILES = [
    0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
    0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99,
]

SEASONAL_WIS_FILE = OUTPUT_DIR / "seasonal_baseline_wis.csv"
SKILL_FILE = OUTPUT_DIR / "model_vs_seasonal_skill.csv"

EXCLUDED_GIDS = {
    "BRA.21.229_2",
    "BRA.21.231_2",
}


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_model_wis(filename: str) -> pd.DataFrame:
    path = BRA_FULL / filename
    logging.info("Loading model WIS: %s", path)
    df = pd.read_csv(path)

    required = {"GID_2", "Date", "horizon", "WIS", "Cases"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")

    df["Date"] = pd.PeriodIndex(df["Date"], freq="M")
    df = df[
        (df["Date"] >= EVAL_START)
        & (df["Date"] <= EVAL_END)
        & df["Cases"].notna()
        & ~df["GID_2"].isin(EXCLUDED_GIDS)
    ].copy()
    return df


def build_seasonal_means() -> pd.DataFrame:
    logging.info("Reading monthly case data: %s", CASES_PER_MONTH)
    cases = read_db(CASES_PER_MONTH)
    cases = cases[["GID_2", "Date", "Cases"]].copy()
    cases = cases.loc[
        ~cases["GID_2"].isin(EXCLUDED_GIDS)
    ].copy()
    cases["Date"] = pd.PeriodIndex(cases["Date"], freq="M")

    train = cases[
        (cases["Date"] >= TRAIN_START)
        & (cases["Date"] <= TRAIN_END)
        & cases["Cases"].notna()
    ].copy()

    # The primary WIS analysis is performed on log(1 + cases), so the
    # seasonal benchmark is defined on the same scale.
    train["log_cases"] = np.log1p(train["Cases"].astype(float))
    train["calendar_month"] = train["Date"].dt.month

    seasonal = (
        train.groupby(["GID_2", "calendar_month"], observed=True)["log_cases"]
        .mean()
        .rename("seasonal_mean_log_cases")
        .reset_index()
    )

    expected = cases[["GID_2"]].drop_duplicates().assign(_key=1).merge(
        pd.DataFrame({"calendar_month": range(1, 13), "_key": 1}),
        on="_key",
    ).drop(columns="_key")

    seasonal = expected.merge(
        seasonal,
        on=["GID_2", "calendar_month"],
        how="left",
        validate="one_to_one",
    )

    if seasonal["seasonal_mean_log_cases"].isna().any():
        missing = seasonal.loc[
            seasonal["seasonal_mean_log_cases"].isna(), ["GID_2", "calendar_month"]
        ]
        raise ValueError(
            "Missing frozen seasonal means for some municipality/calendar-month pairs; "
            f"examples: {missing.head().to_dict('records')}"
        )

    return seasonal


def build_seasonal_wis(seasonal: pd.DataFrame) -> pd.DataFrame:
    """Score the frozen seasonal mean with the canonical Thucia WIS function."""
    # Use the canonical SARIMA WIS file only to define the exact evaluation
    # rows. The observed Cases values are the same across model WIS files.
    reference = load_model_wis(MODELS["SARIMA"])
    reference = reference[["GID_2", "Date", "horizon", "Cases"]].copy()
    reference["calendar_month"] = reference["Date"].dt.month

    reference = reference.merge(
        seasonal,
        on=["GID_2", "calendar_month"],
        how="left",
        validate="many_to_one",
    )

    if reference["seasonal_mean_log_cases"].isna().any():
        raise ValueError("Some evaluation rows lack a seasonal benchmark value.")

    # Primary analyses transform both predictions and observations using
    # log(1 + cases) before calling Thucia's canonical WIS implementation.
    # We therefore do exactly the same here. The seasonal mean is represented
    # as a degenerate predictive distribution: every quantile equals the same
    # seasonal mean forecast.
    score_input = []
    for q in QUANTILES:
        part = reference[[
            "GID_2", "Date", "horizon", "Cases", "seasonal_mean_log_cases"
        ]].copy()
        part["quantile"] = q
        part["prediction"] = part["seasonal_mean_log_cases"]
        part["Cases"] = np.log1p(part["Cases"].astype(float))
        score_input.append(
            part[["GID_2", "Date", "horizon", "quantile", "prediction", "Cases"]]
        )

    score_input = pd.concat(score_input, ignore_index=True)

    # Thucia's canonical wis() returns one score per municipality/date and does
    # not carry the horizon column through the scoring function. The primary
    # pipeline adds horizon back after scoring, so we do the same here and
    # score each horizon separately.
    scored_horizons = []
    for horizon in HORIZONS:
        h_input = score_input[score_input["horizon"] == horizon].copy()
        scored = wis(
            h_input,
            "prediction",
            "Cases",
            geo_col="GID_2",
            transform=None,
            df_filter=None,
        )

        required = {"GID_2", "Date", "WIS"}
        missing = required - set(scored.columns)
        if missing:
            raise ValueError(
                "Canonical Thucia wis() did not return the expected municipality-"
                f"month structure; missing columns: {sorted(missing)}"
            )

        scored["horizon"] = horizon
        scored_horizons.append(scored[["GID_2", "Date", "horizon", "WIS"]])

    scored = pd.concat(scored_horizons, ignore_index=True)
    scored["Date"] = pd.PeriodIndex(scored["Date"], freq="M")
    return scored.rename(columns={"WIS": "WIS_Seasonal"})

def build_skill_table(seasonal_wis: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for model, filename in MODELS.items():
        df = load_model_wis(filename)
        df = df[["GID_2", "Date", "horizon", "WIS"]].copy()
        df = df.rename(columns={"WIS": "WIS_Model"})

        merged = df.merge(
            seasonal_wis,
            on=["GID_2", "Date", "horizon"],
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(df):
            raise ValueError(
                f"{model}: seasonal benchmark missing for {len(df) - len(merged)} rows."
            )

        merged["model"] = model

        # Conventional percentage WIS skill is undefined when the paired
        # municipality-month seasonal benchmark has WIS == 0. Preserve those
        # rows, but mark percentage skill as missing rather than +/- infinity.
        positive_denom = merged["WIS_Seasonal"] > 0
        merged["skill_pct"] = np.nan
        merged.loc[positive_denom, "skill_pct"] = 100.0 * (
            1.0
            - merged.loc[positive_denom, "WIS_Model"]
            / merged.loc[positive_denom, "WIS_Seasonal"]
        )

        # Zero-safe paired relative score. This compares each model forecast
        # with the seasonal benchmark for the same municipality and target
        # month, while compressing the long WIS tail. Positive = model better.
        finite_nonnegative = (
            np.isfinite(merged["WIS_Model"])
            & np.isfinite(merged["WIS_Seasonal"])
            & (merged["WIS_Model"] >= 0)
            & (merged["WIS_Seasonal"] >= 0)
        )
        merged["log1p_skill"] = np.nan
        merged.loc[finite_nonnegative, "log1p_skill"] = (
            np.log1p(merged.loc[finite_nonnegative, "WIS_Seasonal"])
            - np.log1p(merged.loc[finite_nonnegative, "WIS_Model"])
        )
        rows.append(merged)

    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    seasonal = build_seasonal_means()
    seasonal_means_path = OUTPUT_DIR / "seasonal_means_log_cases.csv"
    seasonal.to_csv(seasonal_means_path, index=False)
    logging.info("Wrote seasonal means: %s", seasonal_means_path)

    seasonal_wis = build_seasonal_wis(seasonal)
    seasonal_wis.to_csv(SEASONAL_WIS_FILE, index=False)
    logging.info("Wrote seasonal WIS: %s", SEASONAL_WIS_FILE)

    skill = build_skill_table(seasonal_wis)
    # skill.to_csv(SKILL_FILE, index=False)
    # logging.info("Wrote model vs seasonal skill: %s", SKILL_FILE)

    seasonal_summary = (
        skill.loc[
            (skill["horizon"] == 1)
            & (skill["Date"] >= EVAL_START)
            & (skill["Date"] <= EVAL_END)
        ]
        .assign(month=lambda d: d["Date"].dt.month)
        .groupby(["model", "month"], as_index=False)["log1p_skill"]
        .mean()
        .rename(columns={"log1p_skill": "mean_log1p_skill"})
    )

    seasonal_summary.to_csv(
        OUTPUT_DIR / "seasonal_skill_summary.csv",
        index=False,
    )

    zero_summary = (
        skill.assign(seasonal_zero=skill["WIS_Seasonal"] == 0)
        .groupby(["model", "horizon"], observed=True)
        .agg(
            n_pairs=("WIS_Seasonal", "size"),
            n_seasonal_zero=("seasonal_zero", "sum"),
        )
        .reset_index()
    )
    zero_summary["seasonal_zero_pct"] = (
        100.0 * zero_summary["n_seasonal_zero"] / zero_summary["n_pairs"]
    )

    pct_summary = (
        skill.groupby(["model", "horizon"], observed=True)["skill_pct"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )
    log_summary = (
        skill.groupby(["model", "horizon"], observed=True)["log1p_skill"]
        .agg(["mean", "median", "count"])
        .reset_index()
    )

    print("\nSeasonal-baseline zero-WIS diagnostics")
    print(zero_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nPercentage WIS skill (seasonal WIS > 0 only)")
    print(pct_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nPaired log1p WIS skill (all finite non-negative pairs)")
    print(log_summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
