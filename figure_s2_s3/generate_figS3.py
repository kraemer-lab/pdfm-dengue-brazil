#!/usr/bin/env python3
"""
Generate a supplementary PDFM diagnostic figure.

Panel A: observed national dengue cases and summed municipality-level
         median forecasts after the visualization-only >1M municipality
         filter has been applied in the preparation stage.
Panel B: monthly mean WIS, base vs PDFM.
Panel C: monthly mean DeltaWIS = PDFM - base.

This script reads only compact publication tables.
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

HORIZON = 12
FIG_NAME = "S3"
OUT_STEM = "figure_S3_12month"

MODELS = ["SARIMA", "TCN", "XGBoost", "N-BEATS", "TimesFM"]
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
MODEL_COLOR = {m: cycle[i] for i, m in enumerate(MODELS)}


def panel_label(ax, text):
    ax.text(
        -0.06, 1.05, text,
        transform=ax.transAxes,
        fontsize=13, fontweight="bold",
        va="top", ha="left",
    )


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.15)


def main():
    observed = pd.read_csv(DATA / "figure_s2_s3_observed.csv")
    observed["Date"] = pd.to_datetime(observed["Date"])

    forecasts = pd.read_csv(DATA / "figure_s2_s3_forecasts.csv")
    forecasts["Date"] = pd.to_datetime(forecasts["Date"])
    forecasts = forecasts[forecasts["horizon"] == HORIZON].copy()

    wis = pd.read_csv(DATA / "figure_s2_s3_wis.csv")
    wis["Date"] = pd.to_datetime(wis["Date"])
    wis = wis[wis["horizon"] == HORIZON].copy()

    fig, axes = plt.subplots(3, 1, figsize=(12.4, 10.3), sharex=True)
    ax_cases, ax_wis, ax_delta = axes

    # A: national display
    obs = observed.sort_values("Date")
    ax_cases.fill_between(
        obs["Date"], 0, obs["Cases"], alpha=0.08, linewidth=0
    )
    ax_cases.plot(
        obs["Date"], obs["Cases"],
        color="black", linewidth=2.0, label="Observed",
    )

    for model in MODELS:
        d = forecasts[forecasts["model"] == model].sort_values("Date")
        base = d[d["configuration"] == "Base"]
        pdfm = d[d["configuration"] == "PDFM"]

        ax_cases.plot(
            base["Date"], base["prediction_plot"],
            color=MODEL_COLOR[model], linestyle="--", linewidth=1.7,
        )
        ax_cases.plot(
            pdfm["Date"], pdfm["prediction_plot"],
            color=MODEL_COLOR[model], linestyle="-", linewidth=1.7,
        )

    ax_cases.set_ylabel("Monthly dengue cases")
    ax_cases.set_title(
        "Observed national dengue cases and forecast medians",
        loc="left", fontweight="bold",
    )
    ax_cases.xaxis.set_major_locator(mdates.YearLocator())
    ax_cases.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax_cases.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
    clean_axes(ax_cases)
    panel_label(ax_cases, "A")

    # B: WIS
    for model in MODELS:
        d = wis[wis["model"] == model].sort_values("Date")
        ax_wis.plot(
            d["Date"], d["base_WIS"],
            color=MODEL_COLOR[model], linestyle="--", linewidth=1.7,
        )
        ax_wis.plot(
            d["Date"], d["pdfm_WIS"],
            color=MODEL_COLOR[model], linestyle="-", linewidth=1.7,
        )

    ax_wis.set_ylabel("Monthly mean WIS")
    ax_wis.set_title(
        "Temporal probabilistic forecast performance",
        loc="left", fontweight="bold",
    )
    clean_axes(ax_wis)
    panel_label(ax_wis, "B")

    # C: Delta WIS
    ax_delta.axhline(0, color="black", linewidth=0.9)

    for model in MODELS:
        d = wis[wis["model"] == model].sort_values("Date")
        ax_delta.plot(
            d["Date"], d["DeltaWIS"],
            color=MODEL_COLOR[model], linewidth=1.7,
        )

    ax_delta.set_ylabel("Mean ΔWIS")
    ax_delta.set_xlabel("Target month")
    ax_delta.set_title(
        "Temporal effect of PDFM adjustment",
        loc="left", fontweight="bold",
    )
    clean_axes(ax_delta)
    panel_label(ax_delta, "C")

    handles_models = [
        Line2D([0], [0], color=MODEL_COLOR[m], linewidth=2, label=m)
        for m in MODELS
    ]
    handles_style = [
        Line2D([0], [0], color="black", linestyle="--",
               linewidth=1.7, label="Base"),
        Line2D([0], [0], color="black", linestyle="-",
               linewidth=1.7, label="PDFM"),
    ]

    fig.legend(
        handles=handles_models + handles_style,
        frameon=False, ncol=7, fontsize=8.2,
        loc="upper center", bbox_to_anchor=(0.50, 0.985),
    )

    fig.subplots_adjust(
        left=0.07, right=0.97, bottom=0.08, top=0.91, hspace=0.16
    )

    png = FIGURES / f"{OUT_STEM}.png"
    svg = FIGURES / f"{OUT_STEM}.svg"

    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {png}")
    print(f"Saved: {svg}")


if __name__ == "__main__":
    main()
