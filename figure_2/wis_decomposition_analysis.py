#!/usr/bin/env python3

"""
Vectorised decomposition of PDFM-minus-baseline WIS.

For each model x forecast horizon, quantify:

    delta_WIS
    delta_median_error
    delta_sharpness / interval-width contribution
    delta_lower_miscoverage
    delta_upper_miscoverage

Important:
    The quantile grid is discovered from each forecast store.

    We do NOT interpolate missing quantiles.

    Thus, if a model contains:
        0.01, 0.05, ..., 0.50, ..., 0.95, 0.99

    but does not contain 0.025/0.975, only the available symmetric
    intervals are used. This matches the actual Thucia WIS implementation.

The script:
  * uses the project's own Thucia WIS implementation;
  * applies log1p exactly once, matching the primary analysis;
  * applies the same monotonic quantile correction;
  * calculates WIS components with vectorised NumPy operations;
  * summarises to model x horizon;
  * uses chronological target-month means for inference;
  * uses a 3-month moving-block bootstrap;
  * reports interval-width and empirical-coverage diagnostics.

Outputs:

    data/wis_decomposition_summary.csv
    data/wis_decomposition_monthly.csv
    data/wis_interval_diagnostics.csv

    figures/wis_decomposition_summary.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thucia.core.cases import read_db
from thucia.core.cases import wis as canonical_wis
from thucia.core.cases.wis import (
    _bracher_wis_from_quantiles,
    _enforce_monotone_quantiles,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

HERE = Path(__file__).resolve().parent

BRA_FULL = (
    HERE.parent.parent
    / "data"
    / "PDFM_analysis"
    / "BRA_full"
).resolve()

OUT_DATA = HERE / "data"
OUT_FIG = HERE / "figures"

OUT_DATA.mkdir(parents=True, exist_ok=True)
OUT_FIG.mkdir(parents=True, exist_ok=True)

MODELS = [
    "SARIMA",
    "TCN",
    "XGBoost",
    "N-BEATS",
    "TimesFM",
]

HORIZONS = [1, 3, 6, 12]

START = pd.Period(
    "2020-01",
    freq="M",
)

END = pd.Period(
    "2025-08",
    freq="M",
)

VALIDATION_TOL = 1e-3

BOOTSTRAP_BLOCK = 3
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------
# File naming
# ---------------------------------------------------------------------

def forecast_stem(model, pdfm=False):

    stems = {
        "SARIMA": "sarima_h12",
        "TCN": "tcn_h12",
        "XGBoost": "xgboost_h12",
        "N-BEATS": "nbeats_h12",
        "TimesFM": "timesfm_h12",
    }

    stem = stems[model]

    if pdfm:
        stem += "_pdfmrr_pinball"

    return stem


# ---------------------------------------------------------------------
# Basic utilities
# ---------------------------------------------------------------------

def month_period(series):

    if isinstance(
        series.dtype,
        pd.PeriodDtype,
    ):
        return series.astype("period[M]")

    return (
        pd.to_datetime(series)
        .dt.to_period("M")
    )


def read_forecasts(
    model,
    pdfm=False,
):
    path = (
        BRA_FULL
        / f"{forecast_stem(model, pdfm)}_cases_quantiles"
    )

    # if not path.exists():
    #     raise FileNotFoundError(
    #         f"Forecast store not found:\n{path}"
    #     )

    print(f"Reading {path}")

    df = read_db(
        str(path)
    ).df.copy()

    required = {
        "GID_2",
        "Date",
        "horizon",
        "quantile",
        "prediction",
        "Cases",
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"{path.name}: missing columns "
            f"{sorted(missing)}"
        )

    df["GID_2"] = (
        df["GID_2"]
        .astype(str)
    )

    df["Date"] = month_period(
        df["Date"]
    )

    df["horizon"] = pd.to_numeric(
        df["horizon"],
        errors="coerce",
    )

    df["quantile"] = pd.to_numeric(
        df["quantile"],
        errors="coerce",
    )

    df["prediction"] = pd.to_numeric(
        df["prediction"],
        errors="coerce",
    )

    df["Cases"] = pd.to_numeric(
        df["Cases"],
        errors="coerce",
    )

    df = df[
        (df["Date"] >= START)
        & (df["Date"] <= END)
        & (df["horizon"].isin(HORIZONS))
    ].copy()

    return df


# ---------------------------------------------------------------------
# Dynamic quantile discovery
# ---------------------------------------------------------------------

def available_quantiles(df):
    """
    Discover the exact quantile grid actually present in a store.

    Do not assume a universal 15-quantile grid.
    """
    q = np.sort(
        df["quantile"]
        .dropna()
        .unique()
        .astype(float)
    )

    if q.size == 0:
        raise ValueError(
            "Forecast store contains no quantiles."
        )

    if not np.any(
        np.isclose(
            q,
            0.5,
            rtol=0,
            atol=1e-12,
        )
    ):
        raise ValueError(
            "Forecast store does not contain q=0.50."
        )

    return q


def symmetric_intervals(q):
    """
    Return all symmetric quantile pairs that actually exist.

    Each row is:

        (q_lo, q_hi, alpha, WIS interval weight)

    with:
        q_hi = 1-q_lo
        alpha = 2*q_lo
        weight = alpha/2 = q_lo
    """
    q = np.asarray(
        q,
        dtype=float,
    )

    intervals = []

    for q_lo in q:
        if not (
            q_lo > 0
            and q_lo < 0.5
        ):
            continue

        q_hi = 1.0 - q_lo

        if not np.any(
            np.isclose(
                q,
                q_hi,
                rtol=0,
                atol=1e-12,
            )
        ):
            continue

        alpha = 2.0 * q_lo
        weight = alpha / 2.0

        intervals.append(
            (
                float(q_lo),
                float(q_hi),
                float(alpha),
                float(weight),
            )
        )

    if not intervals:
        raise ValueError(
            f"No symmetric quantile intervals found "
            f"for grid {q}"
        )

    return intervals


def print_quantile_grid(
    model,
    system,
    q,
):
    intervals = symmetric_intervals(q)

    print(
        f"\n  {model} {system} quantile grid:"
    )

    print(
        "   ",
        ", ".join(
            f"{x:g}"
            for x in q
        ),
    )

    print(
        "  symmetric intervals:"
    )

    for q_lo, q_hi, alpha, weight in intervals:
        print(
            f"    {q_lo:g}-{q_hi:g} "
            f"(coverage={1-alpha:g}, weight={weight:g})"
        )

    denominator = (
        0.5
        + sum(
            weight
            for _, _, _, weight in intervals
        )
    )

    print(
        f"  WIS denominator: {denominator:g}"
    )


# ---------------------------------------------------------------------
# Structure validation
# ---------------------------------------------------------------------

def validate_structure(
    df,
    expected_q,
):
    """
    Validate the store against its own discovered quantile grid.

    This deliberately does NOT impose the same grid on every model.
    """
    keys = [
        "GID_2",
        "Date",
        "horizon",
    ]

    case_counts = (
        df.groupby(
            keys,
            observed=True,
        )["Cases"]
        .nunique(dropna=False)
    )

    if (
        case_counts > 1
    ).any():

        bad = case_counts[
            case_counts > 1
        ].head(10)

        raise ValueError(
            "Observed Cases vary within forecast groups:\n"
            f"{bad}"
        )

    duplicate = df.duplicated(
        keys + ["quantile"],
        keep=False,
    )

    if duplicate.any():
        raise ValueError(
            "Duplicate municipality/date/horizon/"
            "quantile rows found."
        )

    q_count = (
        df.groupby(
            keys,
            observed=True,
        )["quantile"]
        .nunique()
    )

    expected_count = len(
        expected_q
    )

    bad_count = q_count[
        q_count
        != expected_count
    ]

    if not bad_count.empty:
        raise ValueError(
            f"Some forecast groups do not contain "
            f"the expected {expected_count} quantiles:\n"
            f"{bad_count.head(10)}"
        )

    # Verify the actual set of quantiles.
    actual_q = np.sort(
        df["quantile"]
        .dropna()
        .unique()
        .astype(float)
    )

    if not np.allclose(
        actual_q,
        expected_q,
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError(
            "Quantile grid varies across the store."
        )


# ---------------------------------------------------------------------
# Wide representation
# ---------------------------------------------------------------------

def to_wide(df):
    q = available_quantiles(
        df
    )

    validate_structure(
        df,
        q,
    )

    keys = [
        "GID_2",
        "Date",
        "horizon",
    ]

    wide = (
        df.pivot_table(
            index=keys,
            columns="quantile",
            values="prediction",
            aggfunc="first",
        )
        .reset_index()
    )

    missing = [
        value
        for value in q
        if value not in wide.columns
    ]

    if missing:
        raise ValueError(
            f"Missing quantiles after pivot: "
            f"{missing}"
        )

    obs = (
        df.groupby(
            keys,
            observed=True,
        )["Cases"]
        .first()
        .reset_index()
    )

    wide = wide.merge(
        obs,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    return (
        wide,
        q,
    )


# ---------------------------------------------------------------------
# Transformation
# ---------------------------------------------------------------------

def transform_for_wis(
    wide,
    q,
):
    """
    Reproduce the primary pipeline exactly:

        prediction = log1p(prediction)
        Cases      = log1p(Cases)

    followed by the behaviour inside wis_bracher when transform=None:
        clamp negative transformed predictions to zero
        monotone quantile correction
    """
    prediction = (
        wide[q]
        .to_numpy(dtype=float)
    )

    observations = (
        wide["Cases"]
        .to_numpy(dtype=float)
    )

    prediction = np.log1p(
        prediction
    )

    observations = np.log1p(
        observations
    )

    # Match clamp_negative_to_zero=True.
    prediction[
        prediction < 0
    ] = 0.0

    # Same operation as Thucia's monotone correction.
    prediction = np.maximum.accumulate(
        prediction,
        axis=1,
    )

    return (
        prediction,
        observations,
    )


# ---------------------------------------------------------------------
# Exact vectorised decomposition
# ---------------------------------------------------------------------

def decompose_arrays(
    prediction,
    observations,
    q,
):
    """
    Vectorised equivalent of
    _bracher_wis_from_quantiles().

    The interval set and denominator are derived from the actual quantiles
    supplied to this particular forecast system.
    """
    q = np.asarray(
        q,
        dtype=float,
    )

    intervals = symmetric_intervals(
        q
    )

    denominator = (
        0.5
        + sum(
            weight
            for _, _, _, weight
            in intervals
        )
    )

    q_index = {
        float(value): i
        for i, value in enumerate(q)
    }

    median_index = q_index[
        0.5
    ]

    median = prediction[
        :,
        median_index,
    ]

    median_raw = (
        0.5
        * np.abs(
            observations
            - median
        )
    )

    n = len(
        observations
    )

    sharpness_raw = np.zeros(
        n,
        dtype=float,
    )

    lower_raw = np.zeros(
        n,
        dtype=float,
    )

    upper_raw = np.zeros(
        n,
        dtype=float,
    )

    for (
        q_lo,
        q_hi,
        alpha,
        weight,
    ) in intervals:

        lo = prediction[
            :,
            q_index[q_lo],
        ]

        hi = prediction[
            :,
            q_index[q_hi],
        ]

        width = (
            hi - lo
        )

        lower_penalty = np.maximum(
            lo - observations,
            0.0,
        )

        upper_penalty = np.maximum(
            observations - hi,
            0.0,
        )

        sharpness_raw += (
            weight
            * width
        )

        lower_raw += (
            weight
            * (2.0 / alpha)
            * lower_penalty
        )

        upper_raw += (
            weight
            * (2.0 / alpha)
            * upper_penalty
        )

    median_component = (
        median_raw
        / denominator
    )

    sharpness_component = (
        sharpness_raw
        / denominator
    )

    lower_component = (
        lower_raw
        / denominator
    )

    upper_component = (
        upper_raw
        / denominator
    )

    miscoverage_component = (
        lower_component
        + upper_component
    )

    wis_reconstructed = (
        median_component
        + sharpness_component
        + lower_component
        + upper_component
    )

    return {
        "median_error": median_component,
        "sharpness": sharpness_component,
        "lower_miscoverage": lower_component,
        "upper_miscoverage": upper_component,
        "miscoverage": miscoverage_component,
        "WIS": wis_reconstructed,
    }


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_private_helper(
    prediction,
    observations,
    q,
    label,
):
    """
    Compare vectorised decomposition to the actual private helper for a
    small sample.
    """
    n = len(
        observations
    )

    if n == 0:
        raise RuntimeError(
            f"{label}: no forecasts."
        )

    sample_n = min(
        25,
        n,
    )

    if sample_n == n:
        idx = np.arange(n)
    else:
        idx = np.linspace(
            0,
            n - 1,
            sample_n,
            dtype=int,
        )

    vectorised = decompose_arrays(
        prediction[idx],
        observations[idx],
        q,
    )["WIS"]

    direct = np.array(
        [
            float(
                _bracher_wis_from_quantiles(
                    observations[i],
                    q,
                    prediction[i],
                )
            )
            for i in idx
        ]
    )

    diff = float(
        np.max(
            np.abs(
                vectorised
                - direct
            )
        )
    )

    print(
        f"  Private-helper validation "
        f"{label}: max diff = {diff:.3e}"
    )

    if diff > VALIDATION_TOL:
        raise RuntimeError(
            f"{label}: vectorised decomposition "
            f"does not match _bracher_wis_from_quantiles()."
        )


def validate_public_wis(
    raw_df,
    q,
    label,
):
    """
    Validate the vectorised result against public wis().

    We validate one horizon and a modest sample of municipality/date groups.
    """
    h = int(
        raw_df["horizon"]
        .dropna()
        .iloc[0]
    )

    keys = [
        "GID_2",
        "Date",
        "horizon",
    ]

    selected = (
        raw_df[
            raw_df["horizon"] == h
        ][keys]
        .drop_duplicates()
        .head(250)
    )

    sample = raw_df.merge(
        selected,
        on=keys,
        how="inner",
        validate="many_to_many",
    ).copy()

    # Exact primary-pipeline transformation.
    sample["prediction"] = np.log1p(
        sample["prediction"]
    )

    sample["Cases"] = np.log1p(
        sample["Cases"]
    )

    canonical = canonical_wis(
        sample,
        "prediction",
        "Cases",
        geo_col="GID_2",
        transform=None,
        df_filter=None,
    )

    if canonical.empty:
        raise RuntimeError(
            f"{label}: canonical wis() returned no rows."
        )

    canonical_mean = float(
        canonical["WIS"].mean()
    )

    sample_wide = (
        sample.pivot_table(
            index=keys,
            columns="quantile",
            values="prediction",
            aggfunc="first",
        )
        .reset_index()
    )

    sample_obs = (
        sample.groupby(
            keys,
            observed=True,
        )["Cases"]
        .first()
        .reset_index()
    )

    sample_wide = sample_wide.merge(
        sample_obs,
        on=keys,
        how="left",
        validate="one_to_one",
    )

    # Values are already log1p transformed here.
    p = sample_wide[q].to_numpy(
        dtype=float
    )

    y = sample_wide["Cases"].to_numpy(
        dtype=float
    )

    p[p < 0] = 0.0

    p = np.maximum.accumulate(
        p,
        axis=1,
    )

    reconstructed = decompose_arrays(
        p,
        y,
        q,
    )["WIS"]

    reconstructed_mean = float(
        reconstructed.mean()
    )

    diff = abs(
        canonical_mean
        - reconstructed_mean
    )

    print(
        f"  Public-wis validation "
        f"{label} h={h}: "
        f"canonical={canonical_mean:.12f}, "
        f"reconstructed={reconstructed_mean:.12f}, "
        f"diff={diff:.3e}"
    )

    if diff > VALIDATION_TOL:
        raise RuntimeError(
            f"{label}: vectorised reconstruction "
            f"does not match public thucia WIS."
        )


# ---------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------

def moving_block_bootstrap_ci(
    dates,
    values,
    block_length=3,
    n_boot=2000,
    seed=42,
):
    """
    Circular moving-block bootstrap over target months.
    """
    dates = pd.PeriodIndex(
        dates,
        freq="M",
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(dates) != len(values):
        raise ValueError(
            "Dates and values have different lengths."
        )

    order = np.argsort(
        dates.asi8
    )

    values = values[
        order
    ]

    n = len(
        values
    )

    if n == 0:
        raise ValueError(
            "No monthly values."
        )

    rng = np.random.default_rng(
        seed
    )

    if block_length <= 1:
        indices = rng.integers(
            0,
            n,
            size=(n_boot, n),
        )

        boot_means = (
            values[indices]
            .mean(axis=1)
        )

    else:
        n_blocks = int(
            np.ceil(
                n / block_length
            )
        )

        starts = np.arange(
            n
        )

        boot_means = np.empty(
            n_boot,
            dtype=float,
        )

        for b in range(
            n_boot
        ):
            sampled_starts = rng.choice(
                starts,
                size=n_blocks,
                replace=True,
            )

            sample = np.concatenate(
                [
                    values[
                        (
                            start
                            + np.arange(
                                block_length
                            )
                        )
                        % n
                    ]
                    for start
                    in sampled_starts
                ]
            )[:n]

            boot_means[b] = (
                sample.mean()
            )

    return (
        float(
            np.quantile(
                boot_means,
                0.025,
            )
        ),
        float(
            np.quantile(
                boot_means,
                0.975,
            )
        ),
    )


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def summarize_model(
    model,
    paired,
):
    rows = []

    for horizon in HORIZONS:

        d = paired[
            paired["horizon"]
            == horizon
        ].copy()

        if d.empty:
            continue

        monthly = (
            d.groupby(
                "Date",
                observed=True,
            )
            .agg(
                delta_WIS=(
                    "delta_WIS",
                    "mean",
                ),
                delta_median_error=(
                    "delta_median_error",
                    "mean",
                ),
                delta_sharpness=(
                    "delta_sharpness",
                    "mean",
                ),
                delta_lower_miscoverage=(
                    "delta_lower_miscoverage",
                    "mean",
                ),
                delta_upper_miscoverage=(
                    "delta_upper_miscoverage",
                    "mean",
                ),
                delta_miscoverage=(
                    "delta_miscoverage",
                    "mean",
                ),
                n_forecasts=(
                    "delta_WIS",
                    "size",
                ),
            )
            .reset_index()
            .sort_values(
                "Date"
            )
        )

        row = {
            "model": model,
            "horizon": horizon,
            "n_forecasts": len(d),
            "n_target_months": len(monthly),
        }

        components = [
            "delta_WIS",
            "delta_median_error",
            "delta_sharpness",
            "delta_lower_miscoverage",
            "delta_upper_miscoverage",
            "delta_miscoverage",
        ]

        for i, component in enumerate(
            components
        ):

            value = float(
                monthly[
                    component
                ].mean()
            )

            ci_low, ci_high = (
                moving_block_bootstrap_ci(
                    monthly["Date"].to_numpy(),
                    monthly[
                        component
                    ].to_numpy(),
                    block_length=BOOTSTRAP_BLOCK,
                    n_boot=BOOTSTRAP_REPS,
                    seed=(
                        BOOTSTRAP_SEED
                        + horizon
                        + 100 * i
                    ),
                )
            )

            row[component] = value

            row[
                f"{component}_ci_low"
            ] = ci_low

            row[
                f"{component}_ci_high"
            ] = ci_high

        row["component_sum"] = (
            row["delta_median_error"]
            + row["delta_sharpness"]
            + row["delta_lower_miscoverage"]
            + row["delta_upper_miscoverage"]
        )

        row["reconstruction_error"] = (
            row["delta_WIS"]
            - row["component_sum"]
        )

        row[
            "fraction_median_error_improved"
        ] = float(
            np.mean(
                d["delta_median_error"]
                < 0
            )
        )

        row[
            "fraction_sharpness_improved"
        ] = float(
            np.mean(
                d["delta_sharpness"]
                < 0
            )
        )

        row[
            "fraction_miscoverage_improved"
        ] = float(
            np.mean(
                d["delta_miscoverage"]
                < 0
            )
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Interval diagnostics
# ---------------------------------------------------------------------

def interval_diagnostics(
    model,
    system,
    predictions,
    observations,
    horizons,
    q,
):
    rows = []

    intervals = symmetric_intervals(
        q
    )

    for horizon in HORIZONS:

        mask = (
            horizons
            == horizon
        )

        p = predictions[
            mask
        ]

        y = observations[
            mask
        ]

        if len(y) == 0:
            continue

        q_index = {
            float(value): i
            for i, value in enumerate(q)
        }

        for (
            q_lo,
            q_hi,
            alpha,
            _weight,
        ) in intervals:

            lo = p[
                :,
                q_index[q_lo],
            ]

            hi = p[
                :,
                q_index[q_hi],
            ]

            width = (
                hi - lo
            )

            rows.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "system": system,
                    "lower_quantile": q_lo,
                    "upper_quantile": q_hi,
                    "central_coverage": (
                        1.0 - alpha
                    ),
                    "mean_log_interval_width": float(
                        width.mean()
                    ),
                    "median_log_interval_width": float(
                        np.median(width)
                    ),
                    "empirical_coverage": float(
                        np.mean(
                            (y >= lo)
                            & (y <= hi)
                        )
                    ),
                    "lower_miscoverage_rate": float(
                        np.mean(
                            y < lo
                        )
                    ),
                    "upper_miscoverage_rate": float(
                        np.mean(
                            y > hi
                        )
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ---------------------------------------------------------------------
# Model-level analysis
# ---------------------------------------------------------------------

def analyse_model(model):

    print("\n" + "=" * 80)
    print(f"PROCESSING {model}")
    print("=" * 80)

    baseline_df = read_forecasts(
        model,
        pdfm=False,
    )

    pdfm_df = read_forecasts(
        model,
        pdfm=True,
    )

    baseline_wide, baseline_q = (
        to_wide(
            baseline_df
        )
    )

    pdfm_wide, pdfm_q = (
        to_wide(
            pdfm_df
        )
    )

    print_quantile_grid(
        model,
        "baseline",
        baseline_q,
    )

    print_quantile_grid(
        model,
        "PDFM",
        pdfm_q,
    )

    # Baseline and PDFM must have the same grid for a clean paired
    # decomposition. This is true for the TimesFM stores currently observed
    # (both have 13 quantiles), and should remain true for every model.
    if not np.allclose(
        baseline_q,
        pdfm_q,
        rtol=0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"{model}: baseline and PDFM quantile grids differ.\n"
            f"baseline={baseline_q}\n"
            f"PDFM={pdfm_q}"
        )

    q = baseline_q

    baseline_p, baseline_y = (
        transform_for_wis(
            baseline_wide,
            q,
        )
    )

    pdfm_p, pdfm_y = (
        transform_for_wis(
            pdfm_wide,
            q,
        )
    )

    # Exact private helper checks.
    validate_private_helper(
        baseline_p,
        baseline_y,
        q,
        f"{model} baseline",
    )

    validate_private_helper(
        pdfm_p,
        pdfm_y,
        q,
        f"{model} PDFM",
    )

    # Public WIS checks.
    validate_public_wis(
        baseline_df,
        q,
        f"{model} baseline",
    )

    validate_public_wis(
        pdfm_df,
        q,
        f"{model} PDFM",
    )

    # Vectorised decomposition.
    baseline_comp = decompose_arrays(
        baseline_p,
        baseline_y,
        q,
    )

    pdfm_comp = decompose_arrays(
        pdfm_p,
        pdfm_y,
        q,
    )

    keys = [
        "GID_2",
        "Date",
        "horizon",
    ]

    base_meta = baseline_wide[
        keys
    ].copy()

    pdfm_meta = pdfm_wide[
        keys
    ].copy()

    for component in [
        "WIS",
        "median_error",
        "sharpness",
        "lower_miscoverage",
        "upper_miscoverage",
        "miscoverage",
    ]:

        base_meta[component] = (
            baseline_comp[
                component
            ]
        )

        pdfm_meta[component] = (
            pdfm_comp[
                component
            ]
        )

    paired = base_meta.merge(
        pdfm_meta,
        on=keys,
        how="inner",
        suffixes=(
            "_base",
            "_pdfm",
        ),
        validate="one_to_one",
    )

    if paired.empty:
        raise RuntimeError(
            f"{model}: no paired forecasts."
        )

    print(
        f"  baseline forecast groups: "
        f"{len(base_meta):,}"
    )

    print(
        f"  PDFM forecast groups:     "
        f"{len(pdfm_meta):,}"
    )

    print(
        f"  paired forecast groups:    "
        f"{len(paired):,}"
    )

    for component in [
        "WIS",
        "median_error",
        "sharpness",
        "lower_miscoverage",
        "upper_miscoverage",
        "miscoverage",
    ]:

        paired[
            f"delta_{component}"
        ] = (
            paired[
                f"{component}_pdfm"
            ]
            - paired[
                f"{component}_base"
            ]
        )

    paired[
        "delta_component_sum"
    ] = (
        paired[
            "delta_median_error"
        ]
        + paired[
            "delta_sharpness"
        ]
        + paired[
            "delta_lower_miscoverage"
        ]
        + paired[
            "delta_upper_miscoverage"
        ]
    )

    paired[
        "delta_reconstruction_error"
    ] = (
        paired[
            "delta_WIS"
        ]
        - paired[
            "delta_component_sum"
        ]
    )

    max_error = float(
        paired[
            "delta_reconstruction_error"
        ]
        .abs()
        .max()
    )

    print(
        f"  maximum paired decomposition error: "
        f"{max_error:.3e}"
    )

    if max_error > VALIDATION_TOL:
        raise RuntimeError(
            f"{model}: paired component deltas do not "
            f"reconstruct delta WIS."
        )

    # Chronological target-month aggregate.
    monthly = (
        paired.groupby(
            [
                "Date",
                "horizon",
            ],
            observed=True,
        )
        .agg(
            delta_WIS=(
                "delta_WIS",
                "mean",
            ),
            delta_median_error=(
                "delta_median_error",
                "mean",
            ),
            delta_sharpness=(
                "delta_sharpness",
                "mean",
            ),
            delta_lower_miscoverage=(
                "delta_lower_miscoverage",
                "mean",
            ),
            delta_upper_miscoverage=(
                "delta_upper_miscoverage",
                "mean",
            ),
            delta_miscoverage=(
                "delta_miscoverage",
                "mean",
            ),
            n_forecasts=(
                "delta_WIS",
                "size",
            ),
        ).reset_index()
    )

    monthly["model"] = model

    # Store model-month-horizon only, not the enormous forecast-level
    # component table.
    diagnostics_base = (
        interval_diagnostics(
            model,
            "baseline",
            baseline_p,
            baseline_y,
            baseline_wide[
                "horizon"
            ].to_numpy(),
            q,
        )
    )

    diagnostics_pdfm = (
        interval_diagnostics(
            model,
            "PDFM",
            pdfm_p,
            pdfm_y,
            pdfm_wide[
                "horizon"
            ].to_numpy(),
            q,
        )
    )

    diagnostics = pd.concat(
        [
            diagnostics_base,
            diagnostics_pdfm,
        ],
        ignore_index=True,
    )

    summary = summarize_model(
        model,
        paired,
    )

    del (
        baseline_df,
        pdfm_df,
        baseline_wide,
        pdfm_wide,
        baseline_p,
        pdfm_p,
        base_meta,
        pdfm_meta,
        paired,
    )

    return (
        summary,
        monthly,
        diagnostics,
    )


# ---------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------

def plot_summary(summary):

    labels = []
    median = []
    sharpness = []
    miscoverage = []

    for model in MODELS:
        for horizon in HORIZONS:

            d = summary[
                (summary["model"] == model)
                & (summary["horizon"] == horizon)
            ]

            if d.empty:
                continue

            row = d.iloc[0]

            labels.append(
                f"{model}\n{horizon}m"
            )

            median.append(
                row[
                    "delta_median_error"
                ]
            )

            sharpness.append(
                row[
                    "delta_sharpness"
                ]
            )

            miscoverage.append(
                row[
                    "delta_miscoverage"
                ]
            )

    x = np.arange(
        len(labels)
    )

    width = 0.25

    fig, ax = plt.subplots(
        figsize=(14, 6)
    )

    ax.bar(
        x - width,
        median,
        width,
        label="Median error",
    )

    ax.bar(
        x,
        sharpness,
        width,
        label="Sharpness / interval width",
    )

    ax.bar(
        x + width,
        miscoverage,
        width,
        label="Miscoverage penalties",
    )

    ax.axhline(
        0,
        linewidth=0.8,
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        "PDFM minus baseline contribution to WIS"
    )

    ax.set_title(
        "Decomposition of PDFM change in WIS"
    )

    ax.legend(
        frameon=False
    )

    ax.grid(
        axis="y",
        alpha=0.15,
    )

    fig.tight_layout()

    output = (
        OUT_FIG
        / "wis_decomposition_summary.png"
    )

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"\nSaved figure: {output}"
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 88)
    print("DYNAMIC-QUANTILE PDFM WIS DECOMPOSITION")
    print("=" * 88)

    print(
        f"Evaluation: {START} to {END}"
    )

    print(
        f"Models: {MODELS}"
    )

    print(
        f"Horizons: {HORIZONS}"
    )

    print(
        f"Bootstrap: "
        f"{BOOTSTRAP_BLOCK}-month blocks, "
        f"{BOOTSTRAP_REPS} replicates"
    )

    print(
        "Quantile handling: use only quantiles present "
        "in each forecast store; no interpolation."
    )

    print("=" * 88)

    summaries = []
    monthly_outputs = []
    diagnostics_outputs = []

    for model in MODELS:

        (
            summary,
            monthly,
            diagnostics,
        ) = analyse_model(
            model
        )

        summaries.append(
            summary
        )

        monthly_outputs.append(
            monthly
        )

        diagnostics_outputs.append(
            diagnostics
        )

    summary = pd.concat(
        summaries,
        ignore_index=True,
    )

    monthly = pd.concat(
        monthly_outputs,
        ignore_index=True,
    )

    diagnostics = pd.concat(
        diagnostics_outputs,
        ignore_index=True,
    )

    max_summary_error = float(
        summary[
            "reconstruction_error"
        ]
        .abs()
        .max()
    )

    print(
        "\nMaximum summary reconstruction error:",
        f"{max_summary_error:.3e}",
    )

    if max_summary_error > VALIDATION_TOL:
        raise RuntimeError(
            "Summary decomposition does not reconstruct delta WIS."
        )

    # -------------------------------------------------------------
    # Save
    # -------------------------------------------------------------

    summary_path = (
        OUT_DATA
        / "wis_decomposition_summary.csv"
    )

    monthly_path = (
        OUT_DATA
        / "wis_decomposition_monthly.csv"
    )

    diagnostics_path = (
        OUT_DATA
        / "wis_interval_diagnostics.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    monthly.to_csv(
        monthly_path,
        index=False,
    )

    diagnostics.to_csv(
        diagnostics_path,
        index=False,
    )

    # -------------------------------------------------------------
    # Print main result
    # -------------------------------------------------------------

    print("\n" + "=" * 88)
    print("WIS DECOMPOSITION SUMMARY")
    print("=" * 88)

    print(
        summary[
            [
                "model",
                "horizon",
                "n_forecasts",
                "n_target_months",
                "delta_WIS",
                "delta_WIS_ci_low",
                "delta_WIS_ci_high",
                "delta_median_error",
                "delta_sharpness",
                "delta_miscoverage",
                "delta_lower_miscoverage",
                "delta_upper_miscoverage",
                "reconstruction_error",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nWrote:"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {monthly_path}"
    )

    print(
        f"  {diagnostics_path}"
    )

    plot_summary(
        summary
    )


if __name__ == "__main__":
    main()
