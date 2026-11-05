#!/usr/bin/env python3
"""
Diebold-Mariano validation for the Brazil PDFM analysis.

Purpose
-------
Provide a Mexico-style predictive-accuracy test for the Brazil PDFM
comparison using the same canonical WIS files used by the Figure 2
analysis.

For each model and forecast horizon:
  1. Load paired base and PDFM WIS files.
  2. Pair observations on GID_2 + Date + horizon and verify Cases match.
  3. Average WIS across municipalities within each target month.
  4. Construct the chronological monthly loss-difference series:
         mean(WIS_PDFM) - mean(WIS_base)
  5. Apply a two-sided Diebold-Mariano test to the chronological loss
     sequences using the same implementation/conventions as the Mexico
     dengue analysis:
       - custom loss returning the supplied WIS values
       - h = forecast horizon
       - Bartlett variance estimator
       - Harvey correction

This is a validation analysis only. It does not modify canonical Figure 2
outputs.

Run from the pdfm_paper/figure2 directory or from the repository with the
standard relative paths resolved from this script's location.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from dieboldmariano import dm_test


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

OUTPUT_FILE = OUTPUT_DIR / "pdfm_heterogeneity_dm.csv"


def load_wis_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"GID_2", "Date", "horizon", "WIS", "Cases"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")

    df["Date"] = pd.PeriodIndex(df["Date"], freq="M")
    df = df[
        (df["Date"] >= START_DATE)
        & (df["Date"] <= END_DATE)
        & df["Cases"].notna()
    ].copy()
    return df


def load_model_pair(filestem: str) -> pd.DataFrame:
    base_path = BRA_FULL / f"{filestem}_wis.csv"
    pdfm_path = BRA_FULL / f"{filestem}_pdfmrr_pinball_wis.csv"

    logging.info("Loading base: %s", base_path)
    logging.info("Loading PDFM: %s", pdfm_path)

    if not base_path.exists():
        raise FileNotFoundError(base_path)
    if not pdfm_path.exists():
        raise FileNotFoundError(pdfm_path)

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
        raise ValueError(f"{filestem}: observed Cases differ between paired files.")

    merged["WIS_PDFM"] = merged["WIS_PDFM"].astype(float)
    merged["WIS_Base"] = merged["WIS_Base"].astype(float)

    return merged[["GID_2", "Date", "horizon", "WIS_Base", "WIS_PDFM", "Cases_Base"]].copy()


def make_date_level_wis(df_h: pd.DataFrame) -> pd.DataFrame:
    out = (
        df_h.groupby("Date", observed=True)
        .agg(
            WIS_baseline=("WIS_Base", "mean"),
            WIS_pdfm=("WIS_PDFM", "mean"),
            n_municipalities=("GID_2", "nunique"),
        )
        .sort_index()
        .dropna(subset=["WIS_baseline", "WIS_pdfm"])
    )
    out["loss_diff"] = out["WIS_pdfm"] - out["WIS_baseline"]
    return out


def run_dm(date_level: pd.DataFrame, horizon: int) -> dict:
    if len(date_level) < max(10, 2 * horizon + 1):
        raise ValueError(
            f"Too few target months ({len(date_level)}) for horizon {horizon}."
        )

    actual = np.zeros(len(date_level), dtype=float)
    baseline_wis = date_level["WIS_baseline"].to_numpy(dtype=float)
    pdfm_wis = date_level["WIS_pdfm"].to_numpy(dtype=float)

    def wis_loss(_actual_value, predicted_loss):
        return predicted_loss

    statistic, p_value = dm_test(
        actual,
        baseline_wis,
        pdfm_wis,
        loss=wis_loss,
        h=horizon,
        one_sided=False,
        harvey_correction=True,
        variance_estimator="bartlett",
    )

    return {
        "n_target_months": int(len(date_level)),
        "mean_DeltaWIS": float(date_level["loss_diff"].mean()),
        "median_monthly_DeltaWIS": float(date_level["loss_diff"].median()),
        "DM_statistic": float(statistic),
        "DM_p_value": float(p_value),
        "DM_h": int(horizon),
        "DM_variance_estimator": "bartlett",
        "DM_harvey_correction": True,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 90)
    print("BRAZIL PDFM DIEBOLD-MARIANO VALIDATION")
    print("=" * 90)
    print(f"Data directory: {BRA_FULL}")
    print(f"Evaluation period: {START_DATE} to {END_DATE}")
    print(f"Horizons: {HORIZONS}")
    print("Target-month unit: municipality-average WIS per target month")
    print("DM: two-sided, h=horizon, Bartlett variance, Harvey correction")
    print("=" * 90)

    rows = []

    for model, filestem in MODELS.items():
        df = load_model_pair(filestem)

        for horizon in HORIZONS:
            df_h = df[df["horizon"] == horizon].copy()
            date_level = make_date_level_wis(df_h)

            result = run_dm(date_level, horizon)
            result.update(
                {
                    "model": model,
                    "horizon": horizon,
                    "first_target_date": str(date_level.index.min()),
                    "last_target_date": str(date_level.index.max()),
                    "mean_monthly_municipality_n": float(date_level["n_municipalities"].mean()),
                    "min_monthly_municipality_n": int(date_level["n_municipalities"].min()),
                    "max_monthly_municipality_n": int(date_level["n_municipalities"].max()),
                }
            )
            rows.append(result)

            print(
                f"{model:10s} h={horizon:2d}: "
                f"n={result['n_target_months']:2d}, "
                f"mean ΔWIS={result['mean_DeltaWIS']:+.6f}, "
                f"DM={result['DM_statistic']:+.4f}, "
                f"p={result['DM_p_value']:.6g}"
            )

    out = pd.DataFrame(rows)[
        [
            "model",
            "horizon",
            "n_target_months",
            "first_target_date",
            "last_target_date",
            "mean_DeltaWIS",
            "median_monthly_DeltaWIS",
            "DM_statistic",
            "DM_p_value",
            "DM_h",
            "DM_variance_estimator",
            "DM_harvey_correction",
            "mean_monthly_municipality_n",
            "min_monthly_municipality_n",
            "max_monthly_municipality_n",
        ]
    ].sort_values(["model", "horizon"]).reset_index(drop=True)

    out.to_csv(OUTPUT_FILE, index=False)

    print()
    print("=" * 90)
    print("RESULTS")
    print("=" * 90)
    print(out.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print()
    print(f"Saved: {OUTPUT_FILE}")
    print("No canonical Figure 2 datasets were modified.")


if __name__ == "__main__":
    main()
