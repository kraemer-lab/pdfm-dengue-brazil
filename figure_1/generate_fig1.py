#!/usr/bin/env python3
"""Generate Figure 1 v26.

Revision notes:
- fixed outer panel boxes with explicit title/content bands for A-F;
- no layout negotiation from subplot titles, annotations or insets;
- Panel E uses a genuine São Paulo state predictive distribution built from
  municipality quantiles via the project aggregation rule (Gaussian copula,
  fixed rho=0.3), rather than summing municipality quantiles;
- Panel E separates the forecast graph from the WIS explanation region;
- Panel D separates mini-map labels from the choropleths;
- Panel C confines modality labels and colorbar inside its panel;
- Panel F uses a fixed internal workflow canvas.

This script expects the study data directory used by the project scripts to be
available at the canonical relative location: data/PDFM_analysis/BRA_full.
The uploaded project archive contains source scripts but not the study data.
"""
from pathlib import Path
from statistics import NormalDist

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from thucia.core.fs import read_db

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
FIGURES = HERE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

PROJECT_ROOT = HERE.parent.parent if (HERE.parent.parent / "data").exists() else Path.cwd()
DATA_ROOT = PROJECT_ROOT / "data" / "PDFM_analysis"
EMBEDDINGS_PATH = DATA_ROOT / "BRA_full" / "embeddings.nc"

GROUPS = [
    ("Search Trends", 0, 127),
    ("Maps & Busyness", 128, 255),
    ("Weather & Air Quality", 256, 329),
]

ANCHOR_ORDER = ["log_nightlight", "population", "temperature"]
ANCHOR_LABELS = {
    "log_nightlight": "Log night-light",
    "population": "Population",
    "temperature": "Temperature",
}

EXAMPLE_MODEL = "TCN"
# Canonical Thucia store: the h12 database contains the horizon column for
# all forecast horizons. This matches the existing Figure 2 preparation code.
EXAMPLE_STEM = "tcn_h12"
EXAMPLE_ORIGIN = pd.Timestamp("2019-12-01")
EXAMPLE_HORIZONS = [1, 3, 6, 12]
COPULA_RHO = 0.3

labelsize = 10.0

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
})


def rect_from_frac(rect, x, y, w, h):
    """Convert panel-relative fractions to explicit figure coordinates."""
    return [
        rect.x0 + rect.width * x,
        rect.y0 + rect.height * y,
        rect.width * w,
        rect.height * h,
    ]


def panel_shell(fig, rect, label, title, fontsize=10.0, content_axis_off=False):
    """Create a fixed outer panel slot with an explicit title band."""
    outer = fig.add_axes([rect.x0, rect.y0, rect.width, rect.height])
    outer.set_xlim(0, 1)
    outer.set_ylim(0, 1)
    outer.set_axis_off()
    outer.text(0.018, 0.965, label, transform=outer.transAxes,
               fontsize=10.8, fontweight="bold",
               ha="left", va="top",
               clip_on=True)
    # outer.text(0.075, 0.965, title, transform=outer.transAxes,
    #            fontsize=fontsize, fontweight="bold", ha="left", va="top",
    #            clip_on=True)
    # With panel titles removed, use nearly the full outer panel box for content.
    content_rect = rect_from_frac(rect, 0.045, 0.025, 0.91, 0.92)
    content = fig.add_axes(content_rect)
    for spine in content.spines.values():
        spine.set_visible(False)
    if content_axis_off:
        content.set_axis_off()
    return outer, content

def rounded_box(ax, xy, width, height, text, fontsize=10.0):
    patch = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=0.8, facecolor="white", edgecolor="0.35",
        transform=ax.transAxes, clip_on=True,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2, xy[1] + height / 2,
        text, ha="center", va="center", fontsize=fontsize,
        transform=ax.transAxes, clip_on=True,
    )


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=9,
            linewidth=0.8, color="0.25", transform=ax.transAxes,
        )
    )

def load():
    burden = pd.read_csv(DATA_DIR / "figure1_municipality_burden.csv")
    national = pd.read_csv(DATA_DIR / "figure1_national_cases.csv")
    national["Date"] = pd.PeriodIndex(
        national["Date"], freq="M"
    ).to_timestamp(how="start")

    selected = pd.read_csv(
        DATA_DIR / "figure1_pdfm_selected_dimensions.csv"
    )
    maps = pd.read_csv(
        DATA_DIR / "figure1_pdfm_selected_maps.csv"
    )

    emb = read_db(str(EMBEDDINGS_PATH)).df.copy()
    feature_cols = [f"feature{i}" for i in range(330)]
    missing = [c for c in feature_cols if c not in emb.columns]
    if missing:
        raise ValueError(
            f"embeddings.nc is missing expected dimensions: {missing[:10]}"
        )
    embedding_values = emb[feature_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Standardized pairwise correlation is used for the visual covariance
    # structure because embedding dimensions have different scales.
    embedding_corr = embedding_values.corr(method="pearson").to_numpy()

    example = load_real_forecast_example()
    return burden, national, selected, maps, embedding_corr, example


def _normalise_state_name(value):
    import unicodedata
    text = str(value).strip()
    folded = "".join(
        ch for ch in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(ch)
    ).lower()
    return folded.replace("_", " ").replace("-", " ").strip()


def _infer_sao_paulo_gid1_from_observed(observed):
    """Return the project's GID_1 code for São Paulo from municipality IDs.

    GADM-style IDs encode the parent state in the first two components. The
    project observed-case store is the authoritative geography used elsewhere
    in this figure, so we use its municipality IDs rather than assuming a
    hard-coded GID_1 code in the aggregated forecast store.
    """
    gid2 = observed["GID_2"].astype(str).str.split("_", n=1).str[0]
    parts = gid2.str.split(".")
    gid1 = parts.str[:2].str.join(".")
    # Canonical GADM/BRA state code for São Paulo, retained here only as the
    # geography anchor. The aggregate store may use a project-local code.
    return "BRA.25", gid1.eq("BRA.25")


def load_real_forecast_example():
    """Load a genuine Sao Paulo state-level TCN forecast from one common origin.

    The prepared aggregate store contains rows keyed by target Date and horizon.
    We derive the forecast origin as Date - horizon months, then select one origin
    for which the four intended horizons (1, 3, 6, 12 months) are all present.
    The predictive intervals come directly from the stored state-level quantiles.
    """
    br_full = DATA_ROOT / "BRA_full"
    quantile_db = br_full / "tcn_h12_agg_cases_quantiles.duckdb"
    wis_csv = br_full / "tcn_h12_agg_wis.csv"

    if not quantile_db.exists():
        raise FileNotFoundError(f"State-level forecast store not found: {quantile_db}")
    if not wis_csv.exists():
        raise FileNotFoundError(f"State-level WIS file not found: {wis_csv}")

    sp_gid = "BRA.25_1"
    qdf = read_db(str(quantile_db)).df.copy()
    required = {"GID_1", "Date", "horizon", "quantile", "prediction"}
    missing = required - set(qdf.columns)
    if missing:
        raise ValueError(f"{quantile_db.name}: missing columns {sorted(missing)}")

    qdf["GID_1"] = qdf["GID_1"].astype(str)
    qdf["Date"] = pd.PeriodIndex(qdf["Date"], freq="M")
    qdf["horizon"] = pd.to_numeric(qdf["horizon"], errors="coerce").astype("Int64")
    qdf["quantile"] = pd.to_numeric(qdf["quantile"], errors="coerce")
    qdf["prediction"] = pd.to_numeric(qdf["prediction"], errors="coerce")

    state = qdf[
        qdf["GID_1"].eq(sp_gid)
        & qdf["horizon"].isin(EXAMPLE_HORIZONS)
        & qdf["prediction"].notna()
    ].copy()
    if state.empty:
        raise RuntimeError(f"No Sao Paulo rows found for {sp_gid} in {quantile_db.name}.")

    available_q = np.sort(state["quantile"].dropna().unique())
    desired = {"q10": 0.10, "q50": 0.50, "q90": 0.90}
    selected_q = {
        name: float(available_q[np.argmin(np.abs(available_q - target))])
        for name, target in desired.items()
    }
    if len(set(selected_q.values())) != 3:
        raise RuntimeError(
            "Could not select distinct quantiles near 0.10, 0.50 and 0.90 from "
            f"the available grid: {available_q.tolist()}"
        )

    state = state[state["quantile"].isin(selected_q.values())].copy()
    # Period arithmetic is exact when subtracting an integer number of months.
    # Avoid object-dtype DateOffset arithmetic, which is both slow and unsupported
    # for PeriodArray in recent pandas versions.
    state["origin"] = [
        d - int(h) for d, h in zip(state["Date"], state["horizon"])
    ]
    state["origin"] = pd.PeriodIndex(state["origin"], freq="M")

    coverage = (
        state.groupby("origin", observed=True)
        .agg(
            n_horizons=("horizon", "nunique"),
            n_quantiles=("quantile", "nunique"),
            n_rows=("prediction", "count"),
        )
        .reset_index()
    )
    complete = coverage[
        (coverage["n_horizons"] == len(EXAMPLE_HORIZONS))
        & (coverage["n_quantiles"] == len(selected_q))
        & (coverage["n_rows"] == len(EXAMPLE_HORIZONS) * len(selected_q))
    ].copy()
    if complete.empty:
        raise RuntimeError(
            f"No complete Sao Paulo forecast origin found for horizons {EXAMPLE_HORIZONS}."
        )

    desired_origin = pd.Period("2019-12", freq="M")
    matching = complete[complete["origin"].eq(desired_origin)]
    chosen_origin = desired_origin if not matching.empty else complete["origin"].min()

    ex = state[state["origin"].eq(chosen_origin)].copy()
    rename_q = {q: name for name, q in selected_q.items()}
    forecast = (
        ex.pivot(index="Date", columns="quantile", values="prediction")
        .reset_index()
        .rename(columns=rename_q)
        .sort_values("Date")
    )
    forecast["horizon"] = [
        int((d - chosen_origin).n) for d in forecast["Date"]
    ]
    forecast["Date"] = forecast["Date"].dt.to_timestamp(how="start")

    # The aggregate quantile store carries observed state cases. Use those
    # values directly for the historical series and realised future outcomes.
    if "Cases" not in qdf.columns:
        raise ValueError(
            f"{quantile_db.name} does not contain the observed Cases column required for Panel E."
        )
    observed = (
        qdf[qdf["GID_1"].eq(sp_gid)][["Date", "Cases"]]
        .drop_duplicates("Date")
        .copy()
    )
    observed["Cases"] = pd.to_numeric(observed["Cases"], errors="coerce")
    observed["Date"] = observed["Date"].dt.to_timestamp(how="start")

    wis = pd.read_csv(wis_csv)
    required_wis = {"GID_1", "Date", "horizon", "WIS"}
    missing_wis = required_wis - set(wis.columns)
    if missing_wis:
        raise ValueError(f"{wis_csv.name}: missing columns {sorted(missing_wis)}")
    wis["GID_1"] = wis["GID_1"].astype(str)
    wis["Date"] = pd.PeriodIndex(wis["Date"], freq="M")
    wis["horizon"] = pd.to_numeric(wis["horizon"], errors="coerce").astype("Int64")
    wis["WIS"] = pd.to_numeric(wis["WIS"], errors="coerce")
    wis_example = wis[
        wis["GID_1"].eq(sp_gid)
        & wis["Date"].isin(pd.PeriodIndex(forecast["Date"], freq="M"))
        & wis["horizon"].isin(EXAMPLE_HORIZONS)
    ][["Date", "horizon", "WIS"]].copy()

    return {
        "state": "Sao Paulo",
        "origin": chosen_origin,
        "forecast": forecast,
        "observed": observed,
        "wis": wis_example,
    }

def draw_brazil_map(ax, df, value_col):
    from thucia.viz.maps import choropleth
    choropleth(
        df,
        ax=ax,
        admin_level=2,
        value_col=value_col,
        cmap="viridis",
        aggregation="mean",
        edgecolor="none",
        linewidth=0,
        legend=False,
        colorbar=True,
        colorbar_format="%.2f",
    )
    ax.set_axis_off()



def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=10,
            linewidth=0.9, color="0.25",
        )
    )


def main():
    burden, national, selected, maps, embedding_corr, example = load()

    fig = plt.figure(figsize=(11.7, 8.3))
    # Compact three-row layout after removing panel titles/headings.
    # Fixed outer panel boxes are retained; row spacing is reduced to use the
    # available page area without relying on tight_layout/constrained_layout.
    gs = fig.add_gridspec(
        nrows=3, ncols=2,
        left=0.045, right=0.985, bottom=0.035, top=0.93,
        width_ratios=[1.0, 2.0], height_ratios=[1.0, 1.08, 1.0],
        wspace=0.065, hspace=0.285,
    )
    slots = [[gs[r, c].get_position(fig) for c in range(2)] for r in range(3)]

    # Subtle shaded grouping boxes distinguish the three conceptual panel pairs,
    # with a small title band above each group.
    group_titles = [
        "Brazilian dengue surveillance data",
        "PDFM representation",
        "Probabilistic forecasting",
    ]
    for r in range(3):
        left = slots[r][0].x0 - 0.008
        bottom = slots[r][0].y0 - 0.016
        right = slots[r][1].x1 + 0.008
        top = slots[r][0].y1 + 0.038
        group_box = FancyBboxPatch(
            (left, bottom),
            right - left,
            top - bottom,
            boxstyle="round,pad=0.004,rounding_size=0.006",
            transform=fig.transFigure,
            facecolor="0.97",
            edgecolor="0.82",
            linewidth=0.8,
            zorder=-10,
            clip_on=False,
        )
        fig.add_artist(group_box)

        group_mid = (left + right) / 2.0
        fig.text(
            group_mid,
            top - 0.008,
            group_titles[r],
            ha="center",
            va="top",
            fontsize=12.5,
            fontweight="bold",
            color="0.22",
        )

    from thucia.viz.maps import choropleth

    # ------------------------------- A
    outer_a, abox = panel_shell(
        fig, slots[0][0], "A", "Cumulative dengue burden, 2010–2025", 9.6,
        content_axis_off=True,
    )
    map_a = fig.add_axes(
        rect_from_frac(slots[0][0], 0.055, 0.055, 0.89, 0.82)
    )
    _axes_before_map = set(fig.axes)
    choropleth(
        burden, ax=map_a, admin_level=2, value_col="cumulative_cases",
        value_transform=np.log1p, cmap="viridis", aggregation="sum",
        edgecolor="none", linewidth=0, legend=False, colorbar=True,
    )
    _new_axes = [a for a in fig.axes if a not in _axes_before_map]
    if _new_axes:
        _map_colorbar_ax = _new_axes[-1]
        _map_colorbar_ax.set_ylabel("log(1 + cases)", fontsize=9.5, labelpad=2)
    map_a.set_axis_off()
    map_a.text(
        -33, 2, "Cumulative reported\ndengue cases\nJan 2010–Aug 2025",
        ha="center", va="bottom", fontsize=10.0,
    )

    # ------------------------------- B
    outer_b, ax_b = panel_shell(
        fig, slots[0][1], "B", "Epidemic history and forecast evaluation", 9.6
    )
    ax_b.set_position(
        rect_from_frac(slots[0][1], 0.055, 0.055, 0.89, 0.80)
    )
    ax_b.plot(national["Date"], national["national_cases"], color="black", linewidth=1.35)
    ax_b.fill_between(
        national["Date"], 0, national["national_cases"],
        color="black", alpha=.06, linewidth=0,
    )
    split = pd.Timestamp("2020-01-01")
    ax_b.axvline(split, ls="--", lw=.85, color=".3")
    ymax = national["national_cases"].max()
    ax_b.text(
        pd.Timestamp("2015-01-01"), ymax*1.1, "Training Period\n2010–2019",
        ha="center", va="top", fontsize=10.0, clip_on=True,
    )
    ax_b.text(
        pd.Timestamp("2022-08-01"), ymax*1.1, "Evaluation Period\n2020–Aug 2025",
        ha="center", va="top", fontsize=10.0, clip_on=True,
    )
    ax_b.set_ylabel("Monthly dengue cases", fontsize=10.0)
    ax_b.set_ylim(0, ymax*1.1)
    ax_b.spines["top"].set_visible(False)
    ax_b.spines["right"].set_visible(False)
    ax_b.grid(axis="y", alpha=.12)
    ax_b.tick_params(axis="both", labelsize=labelsize)
    ax_b.set_facecolor("none")

    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((0, 0))
    ax_b.yaxis.set_major_formatter(y_formatter)

    # ------------------------------- C
    outer_c, cbox = panel_shell(
        fig, slots[1][0], "C", "PDFM embedding correlation structure", 9.0,
        content_axis_off=True,
    )
    mat = fig.add_axes(
        rect_from_frac(slots[1][0], 0.09, 0.105, 0.72, 0.75)
    )
    im1 = mat.imshow(
        embedding_corr, aspect="equal", cmap="coolwarm",
        vmin=-1, vmax=1, interpolation="nearest",
    )
    mat.set_xticks([0, 100, 200, 300])
    mat.set_yticks([0, 100, 200, 300])
    mat.tick_params(labelsize=labelsize)
    mat.set_xlabel("Embedding dimension", fontsize=10.0, labelpad=2)
    mat.set_ylabel("Embedding dimension", fontsize=10.0, labelpad=2)
    for boundary in [127.5, 255.5]:
        mat.axvline(boundary, color="black", lw=.65)
        mat.axhline(boundary, color="black", lw=.65)
    cb_ax = fig.add_axes(
        rect_from_frac(slots[1][0], 0.845, 0.25, 0.05, 0.51)
    )
    cb1 = fig.colorbar(im1, cax=cb_ax)
    cb1.set_label("Pearson\ncorrelation", fontsize=10.0, labelpad=2)
    cb1.ax.tick_params(labelsize=labelsize)
    # Deliberately no modality/subset labels: the matrix is the visual object,
    # and the block boundaries provide sufficient structure at this scale.

    # ------------------------------- D
    outer_d, dbox = panel_shell(
        fig, slots[1][1], "D", "Representative PDFM dimensions and spatial context", 9.0,
        content_axis_off=True,
    )
    for i, (_, row) in enumerate(selected.iterrows()):
        x0 = 0.005 + i * 0.333
        r_text = f"{row['pearson_r']:.2f}".replace("-", "−")
        dbox.text(
            x0 + 0.163, 0.82,  # 0.925,
            # f"{row['group']} · {row['feature']}, r={row['pearson_r']:.2f}\nwith {ANCHOR_LABELS[row['anchor']].lower()}",
            (
                rf"$\mathbf{{{row['group'].replace(' ', r'\ ')}}}$" "\n"
                f"{ANCHOR_LABELS[row['anchor']].lower()}\n"
                f"{row['feature'].replace('feature', 'feature ')}, "
                rf"$r$ = {r_text}"
            ),
            transform=dbox.transAxes, ha="center", va="center", fontsize=10.0,
            clip_on=True,
        )
        sub = fig.add_axes(
            rect_from_frac(slots[1][1], x0, 0.025, 0.315, 0.64)
        )
        part = maps[maps["group"] == row["group"]][["GID_2", "embedding_value"]].copy()
        draw_brazil_map(sub, part, "embedding_value")

    # ------------------------------- E
    outer_e, ebox = panel_shell(
        fig, slots[2][0], "E", "WIS scoring of probabilistic forecasts", 9.0,
        content_axis_off=True,
    )

    # Three schematic Gaussian predictive distributions, all scored against the
    # same observed outcome. The quantile grid matches the manuscript
    # specification (0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
    # 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99). WIS values below are
    # illustrative calculations for the displayed Gaussians, not study results.
    wis_q = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4,
                      0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99])
    interval_q = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4])
    z = np.array([NormalDist().inv_cdf(float(q)) for q in wis_q])
    z_interval = np.array([NormalDist().inv_cdf(float(q)) for q in interval_q])

    def illustrative_wis(observed, mu, sigma):
        quantiles = mu + sigma * z
        median = quantiles[np.where(np.isclose(wis_q, 0.5))[0][0]]
        score_terms = []
        for q, zq in zip(interval_q, z_interval):
            lo = mu + sigma * zq
            hi = mu - sigma * zq
            alpha = 2 * q
            interval_score = (hi - lo)
            if observed < lo:
                interval_score += (2 / alpha) * (lo - observed)
            elif observed > hi:
                interval_score += (2 / alpha) * (observed - hi)
            score_terms.append(interval_score)
        K = len(score_terms)
        return float((sum(score_terms) + abs(observed - median)) / (K + 0.5))

    specs = [
        (0.64, "Lower WIS (better)", 0.00, 0.45),
        (0.38, "Intermediate WIS", 0.00, 1.00),
        (0.12, "Higher WIS (worse)", 0.85, 0.45),
    ]
    observed = 0.0
    x = np.linspace(-2.6, 2.6, 320)

    for y0, title, mu, sigma in specs:
        ax = fig.add_axes(
            rect_from_frac(slots[2][0], 0.12, y0, 0.80, 0.19)
        )
        yy = np.exp(-0.5 * ((x - mu) / sigma) ** 2)
        yy /= yy.max()
        ax.plot(x, yy, color=".18", lw=1.6, zorder=3)

        # Central 80% predictive interval, corresponding to the 0.10 and 0.90
        # quantiles in the manuscript's predictive grid.
        z80 = NormalDist().inv_cdf(0.90)
        lo = mu - z80 * sigma
        hi = mu + z80 * sigma
        ax.axvspan(lo, hi, color=".65", alpha=.20, zorder=1)
        ax.vlines([lo, hi], 0, [0.13, 0.13], color=".48", lw=2.0, zorder=2)

        # Same realised outcome in all examples. The missed forecast has the
        # observation clearly outside its 10-90% predictive interval.
        ax.axvline(observed, color=".05", lw=1.1, ls="--", zorder=4)
        ax.scatter([observed], [0], s=22, color=".05", zorder=5)

        ax.text(0.0, 1.04, title, transform=ax.transAxes,
                va="bottom", fontsize=10.0)

        ax.set_xlim(-2.6, 2.6)
        ax.set_ylim(-0.10, 1.08)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor("none")
        for spine in ax.spines.values():
            spine.set_visible(False)

    # ------------------------------- F
    outer_f, fbox = panel_shell(
        fig, slots[2][1], "F", "Example Sao Paulo state forecast", 9.0,
        content_axis_off=True,
    )
    ex = example["forecast"].copy()
    obs = example["observed"].copy()
    origin = example["origin"].to_timestamp(how="start")

    # With the new 1:2 column geometry, reserve most of E for the actual forecast
    # and replace the previous box-and-arrow WIS schematic with one compact sentence.
    plot = fig.add_axes(
        rect_from_frac(slots[2][1], 0.055, 0.17, 0.92, 0.66)
    )
    history_start = origin - pd.DateOffset(months=18)
    last_target = origin + pd.DateOffset(months=12)
    observed_series = obs[
        (obs["Date"] >= history_start) & (obs["Date"] <= last_target)
    ].copy()

    plot.plot(
        observed_series["Date"], observed_series["Cases"],
        color=".15", lw=1.05, zorder=2,
    )
    plot.scatter(
        observed_series["Date"], observed_series["Cases"],
        color=".15", s=5, zorder=2,
    )
    plot.axvline(origin, ls="--", color=".25", lw=.8, zorder=3)
    plot.text(
        origin, 1.01, "Forecast origin: Dec 2019", transform=plot.get_xaxis_transform(),
        ha="center", va="bottom", fontsize=10.0,  # fontweight="bold",
    )
    plot.set_facecolor("none")

    for _, r in ex.iterrows():
        d = r["Date"]
        plot.vlines(d, r["q10"], r["q90"], color=".55", lw=2.6, zorder=4)
        plot.scatter(
            d, r["q50"], s=24, facecolor="white", edgecolor=".05",
            marker="D", lw=.9, zorder=5,
        )
        offset_y = 5
        if r['horizon'] == 1:
            offset_y = 15
        plot.annotate(
            f"{int(r['horizon'])}m",
            xy=(d, r["q90"]), xytext=(0, offset_y), textcoords="offset points",
            ha="center", va="bottom", fontsize=10.0,  # fontweight="bold",
        )

    # plot.set_xlim(history_start, last_target + pd.DateOffset(months=1))
    plot.set_xlim(pd.Timestamp("2019-01-01") - pd.DateOffset(months=1), last_target + pd.DateOffset(months=1))
    plot.set_ylabel("Monthly dengue cases", fontsize=10.0)
    y_formatter = ScalarFormatter(useMathText=True)
    y_formatter.set_scientific(True)
    y_formatter.set_powerlimits((0, 0))
    plot.yaxis.set_major_formatter(y_formatter)
    plot.tick_params(axis="both", labelsize=labelsize)
    plot.grid(axis="y", alpha=.10)
    plot.spines["top"].set_visible(False)
    plot.spines["right"].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D(
            [0], [0], color=".15", lw=1.05, marker="o", markersize=2.7,
            label="Observed cases",
        ),
        Line2D(
            [0], [0], color=".55", lw=2.6,
            label="10–90% predictive interval",
        ),
        Line2D(
            [0], [0], color=".05", marker="D", markerfacecolor="white",
            markeredgewidth=.8, markersize=4.1, linestyle="None",
            label="Forecast median",
        ),
    ]
    plot.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.65, 0.985),
        frameon=False, ncol=1, fontsize=10.0, handlelength=1.5,
        columnspacing=1.0, handletextpad=.45, borderaxespad=0,
    )
    fbox.text(
        0.055, -0.0,  # -0.055
        "São Paulo state · TCN baseline · 1-, 3-, 6- and 12-month forecasts",
        transform=fbox.transAxes, fontsize=10.0, color=".45",
        ha="left", va="top", clip_on=False,
    )

    # fig.suptitle(
    #     "Study setting, PDFM representations, and forecasting design",
    #     fontsize=12.5, fontweight="bold", y=.975,
    # )
    fig.savefig(FIGURES / "figure1.png", dpi=300, bbox_inches=None)
    fig.savefig(FIGURES / "figure1.svg", bbox_inches=None)
    plt.close(fig)
    print("Saved:", FIGURES / "figure1.png")
    print("Saved:", FIGURES / "figure1.svg")


if __name__ == "__main__":
    main()
