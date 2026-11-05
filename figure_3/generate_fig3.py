#!/usr/bin/env python3
"""Generate Figure 3 from constructed publication tables only.

No WIS or raw-data access occurs in this script. The three summary CSVs are
produced by analyze_fig3.py.
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

MODELS = ["SARIMA", "TCN", "XGBoost", "N-BEATS", "TimesFM"]
MODEL_OFFSET_ORDER = [3, 0, 2, 1, 4]  # Order of models for offsetting IQR bars in plots
HORIZONS = [1, 3, 6, 12]

cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
MODEL_COLOR = {model: cycle[i] for i, model in enumerate(MODELS)}

def load_summary(filename):
    path = DATA / filename
    df = pd.read_csv(path)
    required = {
        "model", "horizon", "n_units", "mean_DeltaWIS",
        "median_DeltaWIS", "P25_DeltaWIS", "P75_DeltaWIS",
        "units_first_better_pct", "units_second_better_pct",
        "ties_pct", "W_statistic", "p_value", "comparison",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    return df

def panel_label(ax, label):
    ax.text(
        -0.1,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
    )


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    ax.grid(axis="y", alpha=0.15)


def plot_panel(
    ax,
    summary,
    title,
    top_label,
    bottom_label,
    letter,
):
    # Offset model estimates at each horizon to keep IQR bars visible.
    offsets = [-0.24, -0.12, 0.0, 0.12, 0.24]
    OFFSET_MODEL = [MODELS[i] for i in MODEL_OFFSET_ORDER]
    for model, offset in zip(OFFSET_MODEL, offsets):
        d = summary[
            summary["model"] == model
        ].sort_values("horizon")

        if d.empty:
            continue

        median = d["median_DeltaWIS"]
        lower_iqr = (median - d["P25_DeltaWIS"]).clip(lower=0)
        upper_iqr = (d["P75_DeltaWIS"] - median).clip(lower=0)
        ax.errorbar(
            d["horizon"] + offset,
            median,
            yerr=[lower_iqr, upper_iqr],
            color=MODEL_COLOR[model],
            marker="o",
            markersize=5,
            linewidth=2,
            elinewidth=1,
            capsize=0,
        )

    ax.set_xticks(HORIZONS)
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("Median ΔWIS")
    ax.set_title(
        title,
        loc="center",
    )

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    ax.text(
        0.02,
        0.92,
        top_label,
        transform=ax.transAxes,
        fontsize=12,
    )

    ax.text(
        0.02,
        0.045,
        bottom_label,
        transform=ax.transAxes,
        fontsize=12,
    )

    clean_axes(ax)
    panel_label(ax, letter)

    # Lower y-min by 5%
    y_min, y_max = ax.get_ylim()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.05 * y_range, y_max)



def main():
    cov_summary = load_summary("figure3_A_covariate_summary.csv")
    spatial_summary = load_summary("figure3_B_spatial_summary.csv")
    agg_summary = load_summary("figure3_C_aggregation_summary.csv")

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(13.2, 4.5),
    )

    plot_panel(
        axes[0],
        cov_summary,
        "Covariate selection",
        "↑ Full covariate set better",
        "↓ Minimal covariate set better",
        "A",
    )
    plot_panel(
        axes[1],
        spatial_summary,
        "Spatial fitting",
        "↑ State-grouped fitting better",
        "↓ Independent fitting better",
        "B",
    )
    plot_panel(
        axes[2],
        agg_summary,
        "Aggregation strategy",
        "↑ Aggregated forecasts better",
        "↓ Aggregated cases better",
        "C",
    )

    axes[0].set_ylim(-1, 1*2/3)
    axes[1].set_ylim(-15, 15*2/3)
    axes[2].set_ylim(-8, 8*2/3)

    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_COLOR[model],
            marker="o",
            linewidth=2,
            markersize=5,
            label=model,
        )
        for model in MODELS
    ]

    fig.legend(
        handles=handles,
        frameon=False,
        ncol=1,
        fontsize=10,
        loc="upper left",
        bbox_to_anchor=(0.82, 0.92),
    )

    fig.subplots_adjust(
        left=0.07,
        right=0.80,
        bottom=0.13,
        top=0.92,
        wspace=0.35,
    )

    png = FIGURES / "figure3.png"
    svg = FIGURES / "figure3.svg"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")

    print("Saved:")
    print(png)
    print(svg)
    plt.show()


if __name__ == "__main__":
    main()
