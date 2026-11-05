#!/usr/bin/env python3
"""Figure 2 exploratory layout based directly on v13.

A  Observed epidemic trajectory and representative baseline forecasts
B  Baseline probabilistic performance: overall horizon comparison and seasonal profiles
C  Example XGBoost baseline vs PDFM-adjusted forecast (6-month horizon)
D  Decomposition of PDFM change in WIS
E  Proportion of municipality-month forecasts improved by PDFM

This version preserves the v13 municipality example panels, including their
log-scale insets, and replaces the former burden-level mean-DeltaWIS heatmap
with the WIS decomposition. The final burden panel retained is the fraction
of municipality-month forecasts improved by PDFM.

Panel D uses the pre-computed vectorised WIS decomposition output:
    data/wis_decomposition_summary.csv

No forecasting or WIS calculations are performed here.
"""
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from thucia.core.cases import read_db

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "data"
FIGURES = HERE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

MODELS = ["SARIMA", "TCN", "XGBoost", "N-BEATS", "TimesFM"]
HORIZONS = [1, 3, 6, 12]
BURDENS = ["zero", "low", "medium", "high"]
BURDEN_LABELS = ["Zero", "Low", "Moderate", "High"]
SHOW_A_RIBBONS = True
PDFM_EXAMPLE_HORIZON = 6

cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
MODEL_COLOR = {m: cycle[i] for i, m in enumerate(MODELS)}
FONTSIZE = 12.0
PANEL_LABEL_FONTSIZE = 16.0

# Fixed semantic colours for the WIS decomposition.
DECOMP_MEDIAN = "tab:blue"
DECOMP_WIDTH = "tab:orange"
DECOMP_MISC = "tab:green"
DECOMP_TOTAL = "black"

# Evaluation period used for all baseline WIS summaries.
EVAL_START = pd.Period("2020-01", freq="M")
EVAL_END = pd.Period("2025-08", freq="M")
BOOTSTRAP_BLOCK = 3
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 42

EXCLUDED_GIDS = {
    "BRA.21.229_2",
    "BRA.21.231_2",
}


def figure_panel_label(fig, ax, text, x):
    y = ax.get_position().y1 + 0.015
    fig.text(
        x,
        y,
        text,
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
        va="top",
        ha="left",
    )


def clean_axes(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(axis="y", alpha=0.15)


def load_results():
    files = {
        "seasonal_skill": "seasonal_skill_summary.csv",
        "burden": "pdfm_heterogeneity_case_burden.csv",
        "wis_decomposition": "wis_decomposition_summary.csv",
    }
    out = {}
    for key, filename in files.items():
        path = RESULTS / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required result file: {path}")
        out[key] = pd.read_csv(path)
    return out


def load_example_municipality():
    obs_path = RESULTS / "figure2_example_municipality_observed.csv"
    fc_path = RESULTS / "figure2_example_municipality_forecasts.csv"
    if not obs_path.exists() or not fc_path.exists():
        raise FileNotFoundError(
            "Panel A municipality inputs are missing. Run "
            "prepare_panel_a_municipality.py first."
        )
    obs = pd.read_csv(obs_path)
    fc = pd.read_csv(fc_path)
    required_obs = {"GID_2", "Date", "Cases"}
    required_fc = {"GID_2", "model", "horizon", "Date", "q10", "q50", "q90"}
    missing_obs = required_obs - set(obs.columns)
    missing_fc = required_fc - set(fc.columns)
    if missing_obs:
        raise ValueError(f"{obs_path.name}: missing columns {sorted(missing_obs)}")
    if missing_fc:
        raise ValueError(f"{fc_path.name}: missing columns {sorted(missing_fc)}")
    obs["Date"] = pd.to_datetime(obs["Date"])
    fc["Date"] = pd.to_datetime(fc["Date"])
    gids = set(obs["GID_2"].astype(str)) | set(fc["GID_2"].astype(str))
    if len(gids) != 1:
        raise ValueError(f"Panel A/C inputs contain multiple municipalities: {sorted(gids)}")
    return obs.sort_values("Date"), fc.sort_values("Date"), next(iter(gids))


def load_example_pdfm_forecast():
    path = RESULTS / "figure2_example_municipality_pdfm_forecasts.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing PDFM municipality forecasts: {path}. Run "
            "prepare_panel_a_municipality.py first."
        )
    df = pd.read_csv(path)
    required = {"GID_2", "model", "horizon", "Date", "q10", "q50", "q90"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values("Date")


def panel_a(ax):
    obs, forecasts, gid = load_example_municipality()
    obs = obs[
        (obs["Date"] >= pd.Timestamp("2020-01-31"))
        & (obs["Date"] <= pd.Timestamp("2025-08-31"))
    ].copy()

    ax.plot(
        obs["Date"],
        obs["Cases"],
        color="black",
        linewidth=2.1,
        label="Observed",
    )

    examples = [("TCN", 1), ("TimesFM", 1)]

    for model, horizon in examples:
        d = forecasts[
            (forecasts["model"] == model)
            & (forecasts["horizon"].astype(int) == horizon)
        ].copy()
        d = d[
            (d["Date"] >= obs["Date"].min())
            & (d["Date"] <= obs["Date"].max())
        ]

        if SHOW_A_RIBBONS:
            ax.fill_between(
                d["Date"],
                d["q10"],
                d["q90"],
                color=MODEL_COLOR[model],
                alpha=0.20,
                linewidth=0,
                label=f"{model} {horizon}-month",  # 10–90% interval",
            )
        else:
            ax.plot(
                d["Date"],
                d["q50"],
                color=MODEL_COLOR[model],
                linewidth=1.7,
                label=f"{model} median, {horizon}-month",
            )

    ax.set_xlim(pd.Timestamp("2020-01-01"), pd.Timestamp("2025-09-01"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Monthly dengue cases", fontsize=FONTSIZE)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.text(
        0.803,
        1.04,
        # f"Municipality: {gid}",
        "São Paulo municipality",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONTSIZE + 1,
    )
    ax.legend(frameon=False, fontsize=FONTSIZE, loc="upper right")
    clean_axes(ax)

    # Preserve the v13 inset in the open lower-left space.
    inset = inset_axes(
        ax,
        width="55%",
        height="70%",
        loc="upper left",
        bbox_to_anchor=(0.06, -0.05, 1, 1),
        bbox_transform=ax.transAxes,
    )
    inset.plot(
        obs["Date"],
        np.log1p(obs["Cases"]),
        color="black",
        linewidth=1.0,
    )

    for model, horizon in examples:
        d = forecasts[
            (forecasts["model"] == model)
            & (forecasts["horizon"].astype(int) == horizon)
        ].copy()
        d = d[
            (d["Date"] >= obs["Date"].min())
            & (d["Date"] <= obs["Date"].max())
        ]

        if SHOW_A_RIBBONS:
            inset.fill_between(
                d["Date"],
                np.log1p(d["q10"]),
                np.log1p(d["q90"]),
                color=MODEL_COLOR[model],
                alpha=0.18,
                linewidth=0,
            )
        else:
            inset.plot(
                d["Date"],
                np.log1p(d["q50"]),
                color=MODEL_COLOR[model],
                linewidth=0.9,
            )

    inset.yaxis.set_major_locator(MaxNLocator(integer=True))
    inset.set_xlim(ax.get_xlim())
    inset.xaxis.set_major_locator(mdates.YearLocator(1))
    inset.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    inset.tick_params(labelsize=FONTSIZE)
    inset.set_title("")
    inset.set_ylabel("log(1 + cases)", fontsize=FONTSIZE)
    inset.set_xlabel("")
    inset.grid(axis="y", alpha=0.12)



def _forecast_path(model):
    stems = {
        "SARIMA": "sarima_h12",
        "TCN": "tcn_h12",
        "XGBoost": "xgboost_h12",
        "N-BEATS": "nbeats_h12",
        "TimesFM": "timesfm_h12",
    }
    return (
        HERE.parent.parent
        / "data"
        / "PDFM_analysis"
        / "BRA_full"
        / f"{stems[model]}_cases_quantiles"
    ).resolve()


def _month_period_series(series):
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.astype("period[M]")
    return pd.to_datetime(series).dt.to_period("M")


def _available_quantiles(df):
    q = np.sort(
        pd.to_numeric(df["quantile"], errors="coerce")
        .dropna()
        .unique()
        .astype(float)
    )
    if not np.any(np.isclose(q, 0.5, rtol=0, atol=1e-12)):
        raise ValueError("Forecast store does not contain q=0.50.")
    return q


def _symmetric_intervals(q):
    intervals = []
    q = np.asarray(q, dtype=float)
    for q_lo in q:
        if not (q_lo > 0 and q_lo < 0.5):
            continue
        q_hi = 1.0 - q_lo
        if np.any(np.isclose(q, q_hi, rtol=0, atol=1e-12)):
            intervals.append((float(q_lo), float(q_hi), float(1.0 - 2.0 * q_lo)))
    return intervals


def _vectorized_wis(prediction, observations, q):
    """Match Thucia WIS after log1p transform and monotonic correction."""
    q = np.asarray(q, dtype=float)
    q_index = {float(v): i for i, v in enumerate(q)}
    intervals = _symmetric_intervals(q)

    prediction = np.log1p(prediction.astype(float))
    observations = np.log1p(observations.astype(float))
    prediction[prediction < 0] = 0.0
    prediction = np.maximum.accumulate(prediction, axis=1)

    denominator = 0.5 + sum(q_lo for q_lo, _, _ in intervals)

    median = prediction[:, q_index[0.5]]
    total = 0.5 * np.abs(observations - median)

    for q_lo, q_hi, alpha in intervals:
        lo = prediction[:, q_index[q_lo]]
        hi = prediction[:, q_index[q_hi]]
        total += q_lo * (
            hi - lo
            + (2.0 / alpha) * np.maximum(lo - observations, 0.0)
            + (2.0 / alpha) * np.maximum(observations - hi, 0.0)
        )

    return total / denominator


def _read_baseline_wis(model):
    path = _forecast_path(model)
    # read_db() accepts the extensionless store path directly; do not use
    # Path.exists() here because these stores are not ordinary single files.
    print(f"Panel B reading {path}")
    df = read_db(str(path)).df.copy()

    required = {"GID_2", "Date", "horizon", "quantile", "prediction", "Cases"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    df["Date"] = _month_period_series(df["Date"])
    df["horizon"] = pd.to_numeric(df["horizon"], errors="coerce")
    df["quantile"] = pd.to_numeric(df["quantile"], errors="coerce")
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["Cases"] = pd.to_numeric(df["Cases"], errors="coerce")

    df = df[
        (df["Date"] >= EVAL_START)
        & (df["Date"] <= EVAL_END)
        & (df["horizon"].isin(HORIZONS))
    ].copy()
    df = df.loc[
        ~df["GID_2"].isin(EXCLUDED_GIDS)
    ].copy()

    q = _available_quantiles(df)
    print("  quantiles:", ", ".join(f"{v:g}" for v in q))

    records = []
    keys = ["GID_2", "Date", "horizon"]

    for horizon in HORIZONS:
        dh = df[df["horizon"] == horizon]
        if dh.empty:
            continue

        wide = dh.pivot_table(
            index=keys,
            columns="quantile",
            values="prediction",
            aggfunc="first",
        ).reset_index()

        missing_q = [v for v in q if v not in wide.columns]
        if missing_q:
            raise ValueError(f"{model} h={horizon}: missing quantiles {missing_q}")

        obs = (
            dh.groupby(keys, observed=True)["Cases"]
            .first()
            .reset_index()
        )
        wide = wide.merge(obs, on=keys, how="left", validate="one_to_one")

        prediction = wide[q].to_numpy(dtype=float)
        observations = wide["Cases"].to_numpy(dtype=float)
        wis = _vectorized_wis(prediction, observations, q)

        records.append(
            pd.DataFrame(
                {
                    "GID_2": wide["GID_2"].astype(str).to_numpy(),
                    "Date": wide["Date"].to_numpy(),
                    "horizon": horizon,
                    "WIS": wis,
                }
            )
        )

    if not records:
        raise RuntimeError(f"No baseline forecasts for {model} in evaluation period.")

    return pd.concat(records, ignore_index=True)


def _moving_block_ci(monthly_values, block_length=3, n_boot=2000, seed=42):
    """95% circular moving-block bootstrap over chronological target-month means."""
    monthly_values = monthly_values.sort_index()
    values = monthly_values.to_numpy(dtype=float)
    n = len(values)

    if n == 0:
        return np.nan, np.nan
    if n == 1:
        return float(values[0]), float(values[0])

    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block_length))
    starts = np.arange(n)
    boots = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        sampled = rng.choice(starts, size=n_blocks, replace=True)
        sample = np.concatenate(
            [values[(start + np.arange(block_length)) % n] for start in sampled]
        )[:n]
        boots[b] = sample.mean()

    return (
        float(np.quantile(boots, 0.025)),
        float(np.quantile(boots, 0.975)),
    )


def calculate_baseline_wis_summary():
    """Calculate overall mean WIS and temporal bootstrap CI, returning raw WIS for B2."""
    cache = RESULTS / "baseline_wis_summary_for_fig2.csv"
    required = {"model", "horizon", "mean_WIS", "ci_low", "ci_high"}

    rows = []
    raw_by_model = {}

    for model_index, model in enumerate(MODELS):
        raw = _read_baseline_wis(model)
        raw_by_model[model] = raw

        for horizon in HORIZONS:
            d = raw[raw["horizon"] == horizon]
            monthly = d.groupby("Date", observed=True)["WIS"].mean()
            mean_wis = float(monthly.mean())
            ci_low, ci_high = _moving_block_ci(
                monthly,
                block_length=BOOTSTRAP_BLOCK,
                n_boot=BOOTSTRAP_REPS,
                seed=BOOTSTRAP_SEED + horizon + 100 * model_index,
            )
            rows.append(
                {
                    "model": model,
                    "horizon": horizon,
                    "mean_WIS": mean_wis,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                }
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(cache, index=False)
    print(f"Panel B cached overall summary: {cache}")
    return summary, raw_by_model


def panel_b1(ax, summary):
    """Overall mean WIS by horizon with offset 95% temporal bootstrap CIs."""
    # Small horizontal offsets keep the CI bars visually distinct where model
    # values are close, while retaining the actual horizon positions on the axis.
    # Offset in approximate overall performance order so overlapping CIs are
    # displaced consistently from left to right at each horizon.
    offset_order = ["TimesFM", "TCN", "XGBoost", "SARIMA", "N-BEATS"]
    offset_values = np.linspace(0.30, -0.30, len(offset_order))
    model_offset = dict(zip(offset_order, offset_values))

    for model in MODELS:
        offset = model_offset[model]
        d = summary[summary["model"] == model].sort_values("horizon")
        if d.empty:
            continue
        x = d["horizon"].to_numpy(float) + offset
        y = d["mean_WIS"].to_numpy(float)
        err = np.vstack([
            y - d["ci_low"].to_numpy(float),
            d["ci_high"].to_numpy(float) - y,
        ])
        ax.errorbar(
            x,
            y,
            yerr=err,
            fmt="-o",
            color=MODEL_COLOR[model],
            capsize=0,
            linewidth=1.5,
            markersize=4.5,
            label=model,
        )

    ax.set_xticks(HORIZONS)
    ax.set_xlim(0.0, 13.0)
    ax.set_ylim(1.5, 6.25)
    ax.set_xlabel("Forecast horizon (months)", fontsize=FONTSIZE)
    ax.set_ylabel("Mean WIS", fontsize=FONTSIZE)
    ax.text(
        0.015, 0.97, "↓ better", transform=ax.transAxes,
        ha="left", va="top", fontsize=FONTSIZE, color="0.35",
    )
    ax.set_title("")  # "Overall performance", loc="left", fontsize=FONTSIZE, pad=4)
    ax.legend(
        loc="upper right",
        frameon=False,
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.5,
        fontsize=FONTSIZE - 1,
    )
    clean_axes(ax)


def panel_b2(ax, seasonal_skill):
    """Seasonal 1-month skill relative to the frozen seasonal baseline.

    The seasonal baseline itself is defined independently; this panel uses only
    matched forecast scores from the January 2020-August 2025 evaluation period.
    For each municipality-target-month-model pair, ``log1p_skill`` is:

        log(1 + WIS_seasonal) - log(1 + WIS_model)

    Positive values therefore indicate that the forecasting model has lower WIS
    than the seasonal baseline. Values are averaged across all municipalities
    and evaluation years for each calendar month.
    """
    required = {"model", "month", "mean_log1p_skill"}
    missing = required - set(seasonal_skill.columns)
    if missing:
        raise ValueError(f"Missing seasonal skill columns: {sorted(missing)}")

    month_numbers = np.arange(1, 13)
    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    for model in MODELS:
        dm = seasonal_skill[seasonal_skill["model"] == model]

        monthly = (
            dm.set_index("month")["mean_log1p_skill"]
            .reindex(month_numbers)
        )

        ax.plot(
            month_numbers,
            monthly.to_numpy(float),
            color=MODEL_COLOR[model],
            linewidth=1.6,
            marker="o",
            markersize=3.4,
            label=model,
            zorder=3,
        )

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.8)

    # Context: mean national dengue burden for the same evaluation-period
    # calendar months. figure2_observed_national.csv is already aggregated over
    # all municipalities, so no municipality-level averaging is done here.
    observed_path = RESULTS / "figure2_observed_national.csv"
    if not observed_path.exists():
        raise FileNotFoundError(f"Missing national observed-case file: {observed_path}")
    observed = pd.read_csv(observed_path)
    observed["Date"] = pd.to_datetime(observed["Date"])
    observed = observed[
        (observed["Date"] >= EVAL_START.start_time)
        & (observed["Date"] <= EVAL_END.end_time)
    ].copy()
    observed["month"] = observed["Date"].dt.month
    seasonal_cases = (
        observed.groupby("month", observed=True)["Cases"]
        .mean()
        .reindex(month_numbers)
    )
    case_values = seasonal_cases.to_numpy(float)

    ax_case = ax.twinx()
    ax_case.fill_between(
        month_numbers, 0, case_values,
        color="0.82", alpha=0.40, linewidth=0, zorder=0,
    )
    ax_case.set_ylim(0, float(np.nanmax(case_values)) * 1.08)
    ax_case.grid(False)
    ax_case.spines["top"].set_visible(False)
    ax_case.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_case.set_ylabel("Mean national dengue cases", fontsize=FONTSIZE - 1)

    ax.set_xlim(1, 12)
    ax.set_ylim(ax.get_ylim()[0], 0.7)
    ax.set_xticks(month_numbers, month_labels)
    ax.set_xlabel("Forecast target month", fontsize=FONTSIZE)
    ax.set_ylabel(
        "Skill relative to\nseasonal reference",
        fontsize=FONTSIZE,
    )
    ax.text(
        0.015, 0.97, "↑ better", transform=ax.transAxes,
        ha="left", va="top", fontsize=FONTSIZE, color="0.35",
    )
    ax.legend(frameon=False, loc="upper right", fontsize=FONTSIZE - 1)
    ax.grid(axis="y", alpha=0.15)
    ax.spines["top"].set_visible(False)


def panel_c(ax):
    obs, _, gid = load_example_municipality()
    pdfm = load_example_pdfm_forecast()
    horizon = PDFM_EXAMPLE_HORIZON
    model = "XGBoost"

    base_path = RESULTS / "figure2_example_municipality_forecasts.csv"
    base = pd.read_csv(base_path)
    base["Date"] = pd.to_datetime(base["Date"])

    base = base[
        (base["model"] == model)
        & (base["horizon"].astype(int) == horizon)
    ].copy()

    pdfm = pdfm[
        (pdfm["model"] == model)
        & (pdfm["horizon"].astype(int) == horizon)
    ].copy()

    obs = obs[
        (obs["Date"] >= pd.Timestamp("2020-01-31"))
        & (obs["Date"] <= pd.Timestamp("2025-08-31"))
    ].copy()

    lo = max(
        obs["Date"].min(),
        base["Date"].min(),
        pdfm["Date"].min(),
    )
    hi = min(
        obs["Date"].max(),
        base["Date"].max(),
        pdfm["Date"].max(),
    )

    obs = obs[
        (obs["Date"] >= lo)
        & (obs["Date"] <= hi)
    ]
    base = base[
        (base["Date"] >= lo)
        & (base["Date"] <= hi)
    ]
    pdfm = pdfm[
        (pdfm["Date"] >= lo)
        & (pdfm["Date"] <= hi)
    ]

    # Draw baseline first so the PDFM interval remains visible on top.
    ax.plot(
        obs["Date"],
        obs["Cases"],
        color="black",
        linewidth=2.1,
        label="Observed",
    )
    ax.fill_between(
        base["Date"],
        base["q10"],
        base["q90"],
        color="0.55",
        alpha=0.18,
        linewidth=0,
        label=f"{model} baseline {horizon}-month",
    )
    ax.fill_between(
        pdfm["Date"],
        pdfm["q10"],
        pdfm["q90"],
        color=MODEL_COLOR["TCN"],
        alpha=0.18,
        linewidth=0,
        label=f"{model} + PDFM {horizon}-month",
    )

    ax.set_xlim(
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2025-09-01"),
    )
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Monthly dengue cases", fontsize=FONTSIZE)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.text(
        0.803,
        1.04,
        # f"Municipality: {gid}",
        "São Paulo municipality",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=FONTSIZE + 1,
    )
    ax.legend(
        frameon=False,
        fontsize=FONTSIZE,
        loc="upper right",
        bbox_to_anchor=(1.067, 1.0),
    )
    clean_axes(ax)

    # Preserve v13 log-scale inset in open lower-left space.
    inset = inset_axes(
        ax,
        width="55%",
        height="70%",
        loc="upper left",
        bbox_to_anchor=(0.06, -0.05, 1, 1),
        bbox_transform=ax.transAxes,
    )

    inset.fill_between(
        base["Date"],
        np.log1p(base["q10"]),
        np.log1p(base["q90"]),
        color="0.55",
        alpha=0.16,
        linewidth=0,
    )
    inset.fill_between(
        pdfm["Date"],
        np.log1p(pdfm["q10"]),
        np.log1p(pdfm["q90"]),
        color=MODEL_COLOR["TCN"],
        alpha=0.16,
        linewidth=0,
    )
    inset.plot(
        obs["Date"],
        np.log1p(obs["Cases"]),
        color="black",
        linewidth=1.0,
    )

    inset.yaxis.set_major_locator(MaxNLocator(integer=True))
    inset.set_xlim(ax.get_xlim())
    inset.xaxis.set_major_locator(mdates.YearLocator(1))
    inset.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    inset.tick_params(labelsize=FONTSIZE)
    inset.set_title("")
    inset.set_ylabel("log(1 + cases)", fontsize=FONTSIZE)
    inset.set_xlabel("")
    inset.grid(axis="y", alpha=0.12)


def panel_d(ax, decomposition):
    """Diverging stacked decomposition with fixed semantic colours."""
    d = decomposition.copy()

    d["model"] = pd.Categorical(
        d["model"],
        categories=MODELS,
        ordered=True,
    )
    d["horizon"] = pd.Categorical(
        d["horizon"],
        categories=HORIZONS,
        ordered=True,
    )
    d = (
        d.sort_values(["model", "horizon"])
        .reset_index(drop=True)
    )

    required = {
        "model",
        "horizon",
        "delta_WIS",
        "delta_WIS_ci_low",
        "delta_WIS_ci_high",
        "delta_median_error",
        "delta_sharpness",
        "delta_miscoverage",
    }
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            f"WIS decomposition is missing columns: {sorted(missing)}"
        )

    # Verify that the displayed decomposition reconstructs delta WIS.
    reconstructed = (
        d["delta_median_error"]
        + d["delta_sharpness"]
        + d["delta_miscoverage"]
    )
    max_err = float(
        (d["delta_WIS"] - reconstructed).abs().max()
    )
    if max_err > 1e-3:
        raise ValueError(
            f"WIS decomposition does not reconstruct delta WIS; max error={max_err:.3e}"
        )

    labels = [
        (
            f"{row.model} {int(row.horizon)}m"
            if int(row.horizon) == 1
            else f"{int(row.horizon)}m"
        )
        for row in d.itertuples()
    ]

    y = np.arange(
        len(d),
        dtype=float,
    )

    # Alternate subtle background shading between model groups.
    # Each model occupies four consecutive forecast horizons.
    n_horizons = len(HORIZONS)

    for i in range(len(MODELS)):
        if i % 2 == 0:
            ax.axhspan(
                i * n_horizons - 0.5,
                (i + 1) * n_horizons - 0.5,
                facecolor="0.94",
                edgecolor="none",
                zorder=-1,
            )

    median = d["delta_median_error"].to_numpy(float)
    width = d["delta_sharpness"].to_numpy(float)
    miscoverage = d["delta_miscoverage"].to_numpy(float)
    total = d["delta_WIS"].to_numpy(float)

    # -------------------------------------------------------------
    # Sign-consistent component values.
    # -------------------------------------------------------------
    median_pos = np.maximum(median, 0)
    median_neg = np.minimum(median, 0)

    width_pos = np.maximum(width, 0)
    width_neg = np.minimum(width, 0)

    misc_pos = np.maximum(miscoverage, 0)
    misc_neg = np.minimum(miscoverage, 0)

    # Positive stack.
    pos_median_left = np.zeros(len(d))
    pos_width_left = median_pos
    pos_misc_left = median_pos + width_pos

    # Negative stack.
    neg_median_left = np.zeros(len(d))
    neg_width_left = median_neg
    neg_misc_left = median_neg + width_neg

    # -------------------------------------------------------------
    # Median-error contribution: blue in both directions.
    # -------------------------------------------------------------
    ax.barh(
        y,
        median_pos,
        left=pos_median_left,
        color=DECOMP_MEDIAN,
        label="Median error",
    )
    ax.barh(
        y,
        median_neg,
        left=neg_median_left,
        color=DECOMP_MEDIAN,
    )

    # -------------------------------------------------------------
    # Interval-width contribution: orange in both directions.
    # -------------------------------------------------------------
    ax.barh(
        y,
        width_pos,
        left=pos_width_left,
        color=DECOMP_WIDTH,
        label="Interval-width penalty",
    )
    ax.barh(
        y,
        width_neg,
        left=neg_width_left,
        color=DECOMP_WIDTH,
    )

    # -------------------------------------------------------------
    # Miscoverage contribution: green in both directions.
    # -------------------------------------------------------------
    ax.barh(
        y,
        misc_pos,
        left=pos_misc_left,
        color=DECOMP_MISC,
        label="Out-of-interval penalties",
    )
    ax.barh(
        y,
        misc_neg,
        left=neg_misc_left,
        color=DECOMP_MISC,
    )

    # -------------------------------------------------------------
    # Net delta WIS: black point + uncapped horizontal 95% CI.
    # -------------------------------------------------------------
    wis_err = np.vstack([
        total - d["delta_WIS_ci_low"].to_numpy(float),
        d["delta_WIS_ci_high"].to_numpy(float) - total,
    ])

    ax.errorbar(
        total,
        y,
        xerr=wis_err,
        fmt="o",
        color=DECOMP_TOTAL,
        markersize=4.5,
        capsize=0,
        elinewidth=1.0,
        linewidth=0,
        zorder=6,
        label="Total ΔWIS (95% CI)",
    )

    ax.axvline(
        0,
        linewidth=0.9,
        color="black",
    )

    ax.set_yticks(
        y
    )
    ax.set_yticklabels(
        labels
    )
    ax.invert_yaxis()

    ax.set_xlabel(
        "Contribution to PDFM − baseline WIS",
        fontsize=FONTSIZE,
    )
    ax.text(
        0.02, 0.05, "← PDFM better", transform=ax.transAxes,
        ha="left", va="top", fontsize=FONTSIZE - 1, color="0.35",
        clip_on=False,
    )
    ax.text(
        0.98, 0.05, "baseline better →", transform=ax.transAxes,
        ha="right", va="top", fontsize=FONTSIZE - 1, color="0.35",
        clip_on=False,
    )

    ax.legend(
        loc="upper left",
        frameon=False,
        fontsize=FONTSIZE - 1,
    )

    # Equal positive/negative x extent.
    xmin, xmax = ax.get_xlim()
    limit = max(
        abs(xmin),
        abs(xmax),
    ) * 1.05
    ax.set_xlim(
        -limit,
        limit - 0.5,
    )

    clean_axes(ax)
    ax.grid(False)


def panel_e(ax, burden):
    d = burden.copy()
    d["burden"] = pd.Categorical(
        d["burden"],
        categories=BURDENS,
        ordered=True,
    )
    d["model"] = pd.Categorical(
        d["model"],
        categories=MODELS,
        ordered=True,
    )
    d["horizon"] = pd.Categorical(
        d["horizon"],
        categories=HORIZONS,
        ordered=True,
    )

    required = {
        "model",
        "horizon",
        "burden",
        "fraction_better",
    }
    missing = required - set(d.columns)
    if missing:
        raise ValueError(
            f"Burden results are missing columns: {sorted(missing)}"
        )

    values = (
        d.pivot_table(
            index=["model", "horizon"],
            columns="burden",
            values="fraction_better",
            aggfunc="first",
        )
        .reindex(
            pd.MultiIndex.from_product(
                [MODELS, HORIZONS],
                names=["model", "horizon"],
            )
        )
        .reindex(columns=BURDENS)
        * 100.0
    )

    arr = values.to_numpy(dtype=float)

    cmap = plt.get_cmap("viridis")
    image = ax.imshow(
        arr,
        cmap=cmap,
        vmin=0,
        vmax=100,
        aspect="auto",
    )

    labels = [
        (
            f"{model} {h}m"
            if h == 1
            else f"{h}m"
        )
        for model in MODELS
        for h in HORIZONS
    ]

    ax.set_xticks(
        range(len(BURDENS)),
        BURDEN_LABELS,
    )
    ax.set_yticks(
        range(len(labels)),
        labels,
    )
    ax.set_xlabel(
        "Reported dengue cases",
        fontsize=FONTSIZE,
    )

    for yline in [3.5, 7.5, 11.5, 15.5]:
        ax.axhline(
            yline,
            color="black",
            linewidth=0.65,
            alpha=0.45,
        )

    cbar = ax.figure.colorbar(
        image,
        ax=ax,
        fraction=0.028,
        pad=0.02,
    )
    cbar.set_label(
        "Forecasts improved by PDFM (%)"
    )


def main():
    results = load_results()

    # Panel B baseline WIS summary. Raw WIS is returned by the helper for
    # compatibility with the cached calculation, but Panel C now uses the
    # independently computed seasonal-baseline skill file.
    baseline_summary, raw_by_model = calculate_baseline_wis_summary()

    fig = plt.figure(
        figsize=(12.4, 14.8)
    )

    outer = fig.add_gridspec(
        4,
        1,
        height_ratios=[1.0, 0.95, 1.0, 1.20],
        hspace=0.28,
    )

    # -------------------------------------------------------------
    # A: representative high-incidence municipality.
    # -------------------------------------------------------------
    ax_a = fig.add_subplot(outer[0, 0])
    panel_a(ax_a)

    # -------------------------------------------------------------
    # B + C: baseline overall and seasonally adjusted performance.
    # -------------------------------------------------------------
    bc = outer[1, 0].subgridspec(
        1,
        2,
        width_ratios=[1.0, 1.0],
        wspace=0.24,
    )

    ax_b = fig.add_subplot(bc[0, 0])
    ax_c = fig.add_subplot(bc[0, 1])

    panel_b1(
        ax_b,
        baseline_summary,
    )
    panel_b2(
        ax_c,
        results["seasonal_skill"],
    )

    # -------------------------------------------------------------
    # D: representative PDFM adjustment.
    # -------------------------------------------------------------
    ax_d = fig.add_subplot(outer[2, 0])
    panel_c(ax_d)

    # -------------------------------------------------------------
    # E + F: WIS decomposition and burden heterogeneity.
    # -------------------------------------------------------------
    bottom = outer[3, 0].subgridspec(
        1,
        2,
        width_ratios=[1.0, 1.0],
        wspace=0.28,
    )

    ax_e = fig.add_subplot(bottom[0, 0])
    ax_f = fig.add_subplot(bottom[0, 1])

    panel_d(
        ax_e,
        results["wis_decomposition"],
    )

    panel_e(
        ax_f,
        results["burden"],
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.95,
        bottom=0.05,
        top=0.975,
    )

    # Align panel letters by figure column rather than by each axes' local frame.
    left_x = min(
        ax_a.get_position().x0,
        ax_b.get_position().x0,
        ax_e.get_position().x0,
    ) - 0.04
    right_x = min(
        ax_c.get_position().x0,
        ax_f.get_position().x0,
    ) - 0.04

    figure_panel_label(fig, ax_a, "A", left_x)
    figure_panel_label(fig, ax_b, "B", left_x)
    figure_panel_label(fig, ax_c, "C", right_x)
    figure_panel_label(fig, ax_d, "D", left_x)
    figure_panel_label(fig, ax_e, "E", left_x)
    figure_panel_label(fig, ax_f, "F", right_x)

    png = (
        FIGURES
        / "figure2.png"
    )
    svg = (
        FIGURES
        / "figure2.svg"
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        svg,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved {png}")
    print(f"Saved {svg}")


if __name__ == "__main__":
    main()
