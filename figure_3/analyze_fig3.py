#!/usr/bin/env python3
"""
analyze_fig3.py

Figure 3 — canonical data-derived model configuration effects.

This version treats the current saved evaluation outputs as the source of truth
and reports, for every panel and model x horizon combination:

    * mean unit-level DeltaWIS
    * median unit-level DeltaWIS
    * P25 / P75 of unit-level DeltaWIS
    * proportion of units favouring the first configuration
    * Wilcoxon signed-rank statistic and p-value

The statistical unit is the spatial unit after averaging DeltaWIS through time:

    A: municipality (GID_2)
    B: municipality (GID_2), restricted to the top-20 set
    C: state (GID_1)

Panel definitions:

    A  Minimal - full covariates
    B  Univariate - multivariate spatial fitting
    C  Aggregated cases - aggregated forecasts

The figure continues to plot the median unit-level DeltaWIS because this is
robust to the highly right-skewed WIS distribution, while the saved result
tables expose the mean, median, spread, improvement proportion and paired
Wilcoxon test for manuscript/table development.

No manuscript result values are hard-coded.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats


# =============================================================================
# PATHS / CONFIG
# =============================================================================

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "data"
FIGURES = HERE / "figures"
RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

DATA_ROOT = (HERE.parent.parent / "data" / "PDFM_analysis").resolve()

BRA_FULL = DATA_ROOT / "BRA_full"
BRA_TOP20 = DATA_ROOT / "BRA_top20"
BRA_MIN = DATA_ROOT / "BRA_min"
BRA_ADM1 = DATA_ROOT / "BRA_adm1"

MODELS = ["SARIMA", "TCN", "XGBoost", "N-BEATS", "TimesFM"]

MODEL_STEMS = {
    "SARIMA": "sarima",
    "TCN": "tcn",
    "XGBoost": "xgboost",
    "N-BEATS": "nbeats",
    "TimesFM": "timesfm",
}

EXCLUDED_GIDS = {
    "BRA.21.229_2",
    "BRA.21.231_2",
}

FULL_STEMS = {
    model: f"{stem}_h12"
    for model, stem in MODEL_STEMS.items()
}

TOP20_STEMS = {
    model: f"{stem}_top20"
    for model, stem in MODEL_STEMS.items()
}

ADM1_STEMS = {
    model: f"{stem}_adm1"
    for model, stem in MODEL_STEMS.items()
}

HORIZONS = [1, 3, 6, 12]
START_DATE = pd.Period("2020-01", freq="M")
END_DATE = pd.Period("2025-08", freq="M")

cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
MODEL_COLOR = {model: cycle[i] for i, model in enumerate(MODELS)}


# =============================================================================
# LOADERS / VALIDATION
# =============================================================================

def read_wis(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing WIS file: {path}")

    df = pd.read_csv(path)

    required = {"Date", "horizon", "WIS", "Cases"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name} missing required columns: {sorted(missing)}"
        )

    df["Date"] = pd.PeriodIndex(df["Date"], freq="M")
    df = df[
        (df["Date"] >= START_DATE)
        & (df["Date"] <= END_DATE)
        & df["Cases"].notna()
    ].copy()

    if "GID_2" in df.columns:
        df = df.loc[
            ~df["GID_2"].isin(EXCLUDED_GIDS)
        ].copy()
    return df


def validate_cases(
    paired: pd.DataFrame,
    left: str,
    right: str,
    label: str,
):
    # Paired Base/PDFM merges in Figure 4 use suffixes with spaces:
    # "Cases PDFM" and "Cases Base". Other panel merges use underscore
    # suffixes, so accept either convention.
    candidates = [
        (f"Cases_{left}", f"Cases_{right}"),
        (f"Cases {left}", f"Cases {right}"),
        (f"Cases PDFM", f"Cases Base"),
    ]

    for left_col, right_col in candidates:
        if left_col in paired.columns and right_col in paired.columns:
            break
    else:
        raise ValueError(
            f"{label}: could not find paired Cases columns. "
            f"Available columns: {list(paired.columns)}"
        )

    ok = np.isclose(
        paired[left_col].to_numpy(),
        paired[right_col].to_numpy(),
        equal_nan=True,
    )

    if not np.all(ok):
        mismatch = paired.loc[
            ~ok,
            ["Date", "horizon", left_col, right_col],
        ].head(10)

        raise ValueError(
            f"{label}: observed Cases differ in paired rows.\n"
            f"{mismatch.to_string(index=False)}"
        )


def read_figure4_configuration(
    base_path: Path,
    pdfm_path: Path,
    geo_col: str,
    model: str,
) -> pd.DataFrame:
    """
    Reuse the Base/PDFM pairing logic from generate_fig4_v2.py.

    Only Base WIS is returned after confirming a corresponding PDFM row
    exists for the same spatial unit, date and horizon.
    """
    base = read_wis(base_path)
    pdfm = read_wis(pdfm_path)

    for df, name in [(base, "Base"), (pdfm, "PDFM")]:
        if geo_col not in df.columns:
            raise ValueError(
                f"{model}: {name} output {Path(base_path if name == 'Base' else pdfm_path).name}"
                f" lacks {geo_col}."
            )

    paired = pd.merge(
        pdfm,
        base,
        on=[geo_col, "Date", "horizon"],
        suffixes=(" PDFM", " Base"),
        how="inner",
        validate="one_to_one",
    )

    if paired.empty:
        raise ValueError(
            f"{model}: no Base/PDFM pairs for {base_path.name}."
        )

    validate_cases(
        paired,
        "PDFM",
        "Base",
        f"{model} Base/PDFM validation",
    )

    return paired[
        [geo_col, "Date", "horizon", "WIS Base", "Cases Base"]
    ].rename(
        columns={
            "WIS Base": "WIS",
            "Cases Base": "Cases",
        }
    )


# =============================================================================
# STATISTICS
# =============================================================================

def unit_summary(
    paired: pd.DataFrame,
    unit_col: str,
    model: str,
    comparison: str,
) -> pd.DataFrame:
    """
    Convert municipality/month (or state/month) paired WIS to one DeltaWIS
    per unit per horizon, then calculate the full summary for each horizon.
    """
    unit_delta = (
        paired.groupby(
            [unit_col, "horizon"],
            observed=True,
        )["DeltaWIS"]
        .mean()
        .reset_index(name="unit_mean_DeltaWIS")
    )

    rows = []

    for horizon in HORIZONS:
        d = unit_delta[
            unit_delta["horizon"] == horizon
        ]["unit_mean_DeltaWIS"].dropna()

        if len(d) == 0:
            continue

        # Two-sided paired sign-rank test at the spatial-unit level.
        # A zero-difference sample has no valid Wilcoxon statistic.
        if np.allclose(d.to_numpy(), 0.0):
            statistic = 0.0
            p_value = 1.0
        else:
            statistic, p_value = stats.wilcoxon(
                d.to_numpy(),
                alternative="two-sided",
                zero_method="wilcox",
                correction=False,
                method="auto",
            )

        rows.append(
            {
                "model": model,
                "horizon": horizon,
                "n_units": int(d.shape[0]),
                "mean_DeltaWIS": float(d.mean()),
                "median_DeltaWIS": float(d.median()),
                "P25_DeltaWIS": float(d.quantile(0.25)),
                "P75_DeltaWIS": float(d.quantile(0.75)),
                "units_first_better_pct": float(
                    100.0 * (d < 0).mean()
                ),
                "units_second_better_pct": float(
                    100.0 * (d > 0).mean()
                ),
                "ties_pct": float(
                    100.0 * (d == 0).mean()
                ),
                "W_statistic": float(statistic),
                "p_value": float(p_value),
                "comparison": comparison,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# PANEL A — COVARIATES
# =============================================================================

def run_covariates():
    detailed = []
    unit_level = []

    for model in MODELS:
        full = read_wis(
            BRA_FULL / f"{FULL_STEMS[model]}_wis.csv"
        )
        minimal = read_wis(
            BRA_MIN / f"{FULL_STEMS[model]}_wis.csv"
        )

        keys = ["GID_2", "Date", "horizon"]

        paired = pd.merge(
            minimal[
                keys + ["WIS", "Cases"]
            ],
            full[
                keys + ["WIS", "Cases"]
            ],
            on=keys,
            how="inner",
            suffixes=("_minimal", "_full"),
            validate="one_to_one",
        )

        if paired.empty:
            raise ValueError(
                f"{model}: no paired BRA_min/BRA_full rows."
            )

        validate_cases(
            paired,
            "minimal",
            "full",
            f"{model} covariate comparison",
        )

        paired["DeltaWIS"] = (
            paired["WIS_minimal"]
            - paired["WIS_full"]
        )

        unit_level.append(
            paired[
                ["GID_2", "Date", "horizon", "DeltaWIS"]
            ].assign(model=model)
        )

        detailed.append(
            unit_summary(
                paired,
                unit_col="GID_2",
                model=model,
                comparison="Minimal - full",
            )
        )

    return pd.concat(detailed, ignore_index=True), pd.concat(
        unit_level, ignore_index=True
    )


# =============================================================================
# PANEL B — SPATIAL FITTING
# =============================================================================

def run_spatial():
    detailed = []
    unit_level = []

    for model in MODELS:
        full = read_figure4_configuration(
            BRA_FULL / f"{FULL_STEMS[model]}_wis.csv",
            BRA_FULL / f"{FULL_STEMS[model]}_pdfmrr_pinball_wis.csv",
            geo_col="GID_2",
            model=model,
        )

        top20 = read_figure4_configuration(
            BRA_TOP20 / f"{TOP20_STEMS[model]}_wis.csv",
            BRA_TOP20 / f"{TOP20_STEMS[model]}_pdfmrr_pinball_wis.csv",
            geo_col="GID_2",
            model=model,
        )

        top20_ids = set(
            top20["GID_2"].dropna().unique()
        )

        if len(top20_ids) != 20:
            print(
                f"WARNING {model}: top20 configuration contains "
                f"{len(top20_ids)} GID_2 values."
            )

        full = full[
            full["GID_2"].isin(top20_ids)
        ].copy()

        keys = ["GID_2", "Date", "horizon"]

        paired = pd.merge(
            top20[
                keys + ["WIS", "Cases"]
            ],
            full[
                keys + ["WIS", "Cases"]
            ],
            on=keys,
            how="inner",
            suffixes=("_univariate", "_multivariate"),
            validate="one_to_one",
        )

        if paired.empty:
            raise ValueError(
                f"{model}: no paired top20/full rows."
            )

        validate_cases(
            paired,
            "univariate",
            "multivariate",
            f"{model} spatial comparison",
        )

        paired["DeltaWIS"] = (
            paired["WIS_univariate"]
            - paired["WIS_multivariate"]
        )

        unit_level.append(
            paired[
                ["GID_2", "Date", "horizon", "DeltaWIS"]
            ].assign(model=model)
        )

        detailed.append(
            unit_summary(
                paired,
                unit_col="GID_2",
                model=model,
                comparison="Univariate - multivariate",
            )
        )

    return pd.concat(detailed, ignore_index=True), pd.concat(
        unit_level, ignore_index=True
    )


# =============================================================================
# PANEL C — AGGREGATION
# =============================================================================

def run_aggregation():
    detailed = []
    unit_level = []

    for model in MODELS:
        aggregate_cases = read_wis(
            BRA_ADM1 / f"{ADM1_STEMS[model]}_wis.csv"
        )
        aggregate_forecasts = read_wis(
            BRA_FULL / f"{FULL_STEMS[model]}_agg_wis.csv"
        )

        for df, name in [
            (aggregate_cases, "BRA_adm1"),
            (aggregate_forecasts, "BRA_full_agg"),
        ]:
            if "GID_1" not in df.columns:
                raise ValueError(
                    f"{model}: {name} WIS output lacks GID_1."
                )

        keys = ["GID_1", "Date", "horizon"]

        paired = pd.merge(
            aggregate_cases[
                keys + ["WIS", "Cases"]
            ],
            aggregate_forecasts[
                keys + ["WIS", "Cases"]
            ],
            on=keys,
            how="inner",
            suffixes=(
                "_aggregate_cases",
                "_aggregate_forecasts",
            ),
            validate="one_to_one",
        )

        if paired.empty:
            raise ValueError(
                f"{model}: no paired aggregation rows."
            )

        validate_cases(
            paired,
            "aggregate_cases",
            "aggregate_forecasts",
            f"{model} aggregation comparison",
        )

        paired["DeltaWIS"] = (
            paired["WIS_aggregate_cases"]
            - paired["WIS_aggregate_forecasts"]
        )

        unit_level.append(
            paired[
                ["GID_1", "Date", "horizon", "DeltaWIS"]
            ].assign(model=model)
        )

        detailed.append(
            unit_summary(
                paired,
                unit_col="GID_1",
                model=model,
                comparison="Aggregated cases - aggregated forecasts",
            )
        )

    return pd.concat(detailed, ignore_index=True), pd.concat(
        unit_level, ignore_index=True
    )



def main():
    print("=" * 100)
    print("FIGURE 3 — CANONICAL DATA-DERIVED SUMMARY")
    print("=" * 100)
    print(f"Data root: {DATA_ROOT}")
    print(f"Evaluation period: {START_DATE} to {END_DATE}")
    print(f"Horizons: {HORIZONS}")

    cov_summary, cov_unit = run_covariates()
    spatial_summary, spatial_unit = run_spatial()
    agg_summary, agg_unit = run_aggregation()

    for name, df in [
        ("A — COVARIATE SELECTION", cov_summary),
        ("B — SPATIAL FITTING", spatial_summary),
        ("C — AGGREGATION STRATEGY", agg_summary),
    ]:
        print("\n" + "=" * 100)
        print(name)
        print("=" * 100)
        print(df[
            ["model","horizon","n_units","mean_DeltaWIS","median_DeltaWIS",
             "P25_DeltaWIS","P75_DeltaWIS","units_first_better_pct",
             "W_statistic","p_value"]
        ].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    cov_summary.to_csv(RESULTS / "figure3_A_covariate_summary.csv", index=False)
    spatial_summary.to_csv(RESULTS / "figure3_B_spatial_summary.csv", index=False)
    agg_summary.to_csv(RESULTS / "figure3_C_aggregation_summary.csv", index=False)

    combined = pd.concat(
        [cov_summary, spatial_summary, agg_summary],
        ignore_index=True,
    )
    combined.to_csv(RESULTS / "figure3_all_summary.csv", index=False)

    print("\nWrote:")
    print(RESULTS / "figure3_A_covariate_summary.csv")
    print(RESULTS / "figure3_B_spatial_summary.csv")
    print(RESULTS / "figure3_C_aggregation_summary.csv")
    print(RESULTS / "figure3_all_summary.csv")


if __name__ == "__main__":
    main()
