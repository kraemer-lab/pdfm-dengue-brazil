import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from thucia.viz.maps import choropleth

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "svg.fonttype": "none",
    "axes.titlesize": 11.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
})

METHOD_PALETTE = {
    "WIS Base": "#d9d9d9",
    "WIS PDFM": "#6a51a3",
}
MODEL_ORDER = ["SARIMA", "TCN", "N-BEATS", "XGBoost", "TimesFM"]
HORIZONS = [1, 6, 12]

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIGURES = HERE / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

def panel_label(ax, label):
    ax.text(
        -0.08, 1.04, label, transform=ax.transAxes,
        fontsize=12.5, fontweight="bold",
        va="bottom", ha="right",
    )

def run_pipeline():
    data = pd.read_csv(DATA / "figure4_violin_data.csv")
    top20_regions = pd.read_csv(DATA / "figure4_top20_regions.csv")

    for c in ["horizon"]:
        data[c] = pd.to_numeric(data[c], errors="raise")

    fig, axes = plt.subplots(
        4, 4, figsize=(10.8, 11.0), constrained_layout=False,
    )

    choices = [
        ("Multivariate", 1, True, False),
        ("Univariate", 2, False, True),
        ("Aggregate cases", 3, True, False),
        ("Aggregate forecasts", 4, False, False),
    ]

    config_labels = {
        "Multivariate": "State-grouped",
        "Univariate": "Independent",
        "Aggregate cases": "Aggregate cases",
        "Aggregate forecasts": "Aggregate forecasts",
    }

    for config, col, plot_choropleth, plot_regions in choices:
        df_diff = data[data["configuration"] == config].copy()

        df_long = df_diff.melt(
            id_vars=["model", "horizon"],
            value_vars=["WIS Base", "WIS PDFM"],
            var_name="method",
            value_name="wis",
        )
        df_long["wis"] = np.log1p(np.maximum(df_long["wis"], 0))

        xlims = (-0.05, 3.4)

        for row, horizon in zip([1, 2, 3], HORIZONS):
            ax = axes[row, col - 1]
            subset = df_long[df_long["horizon"] == horizon].copy()

            sns.violinplot(
                data=subset,
                ax=ax,
                y="model",
                x="wis",
                hue="method",
                order=MODEL_ORDER,
                hue_order=["WIS Base", "WIS PDFM"],
                palette=METHOD_PALETTE,
                split=True,
                inner="box",
                cut=0,
                gap=0.08,
                linewidth=0.8,
            )

            ax.legend_.remove()
            ax.set_xlim(xlims)
            ax.set_ylabel(
                f"Horizon {horizon}" if col == 1 else ""
            )
            ax.set_xlabel(
                "Log(1 + WIS)" if horizon == 12 else ""
            )

            if horizon != 12:
                ax.tick_params(axis="x", labelbottom=False)

            ax.tick_params(axis="y", length=0)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if col != 1:
                ax.set_yticklabels([])

        axes[0, col - 1].set_title(
            config_labels[config],
            fontsize=11.5,
            fontweight="bold",
            pad=7,
        )

        if plot_choropleth:
            ax = axes[0, col - 1]
            map_data = df_diff[df_diff["horizon"] == 1].copy()

            # The original v6 maps the same summary data at horizon 1.
            if config == "Aggregate cases":
                admin_level = 1
            else:
                admin_level = 2

            choropleth(
                map_data,
                ax=ax,
                admin_level=admin_level,
                value_col="Cases",
                value_transform=np.log1p,
                cmap="viridis",
                aggregation="sum",
                edgecolor=None,
                linewidth=0,
            )
            ax.set_axis_off()
            ax.set_title(
                config_labels[config],
                fontsize=11.5,
                fontweight="bold",
                pad=4,
            )

        elif plot_regions:
            ax = axes[0, col - 1]

            choropleth(
                top20_regions,
                ax=ax,
                admin_level=2,
                value_col="selected",
                value_transform=None,
                cmap="Greys",
                aggregation="first",
                edgecolor="0.85",
                linewidth=0.05,
                colorbar=False,
            )

            selected = top20_regions[top20_regions["selected"] == 1].copy()
            choropleth(
                selected,
                ax=ax,
                admin_level=2,
                value_col="selected",
                value_transform=None,
                cmap="OrRd",
                aggregation="first",
                edgecolor="0.35",
                linewidth=0.15,
                colorbar=False,
            )

            ax.set_axis_off()
            ax.set_title(
                config_labels[config],
                fontsize=11.5,
                fontweight="bold",
                pad=4,
            )

    axes[0, 3].axis("off")
    axes[0, 3].text(
        0.5, 0.70, "Municipality forecasts",
        ha="center", va="center", fontsize=9.5,
        fontweight="bold", transform=axes[0, 3].transAxes,
    )
    axes[0, 3].annotate(
        "", xy=(0.5, 0.48), xytext=(0.5, 0.63),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", lw=1.2),
    )
    axes[0, 3].text(
        0.5, 0.40, "Aggregate to state level",
        ha="center", va="center", fontsize=9.2,
        transform=axes[0, 3].transAxes,
    )

    legend_handles = [
        plt.Line2D(
            [0], [0], color=METHOD_PALETTE["WIS Base"],
            linewidth=7, label="Base",
        ),
        plt.Line2D(
            [0], [0], color=METHOD_PALETTE["WIS PDFM"],
            linewidth=7, label="PDFM-adjusted",
        ),
    ]

    axes[0, 3].legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        frameon=False,
        title="Forecast",
    )

    for col, label in enumerate(["A", "B", "C", "D"]):
        axes[0, col].text(
            -0.08, 1.08, label,
            transform=axes[0, col].transAxes,
            fontsize=12.5,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    fig.suptitle(
        "Forecast performance across spatial and aggregation configurations",
        fontsize=14,
        fontweight="bold",
        y=0.985,
    )

    fig.text(
        0.5, 0.015,
        "Split violins show the distribution of log(1 + WIS) for base "
        "and PDFM-adjusted probabilistic forecasts. Lower WIS indicates "
        "better forecast performance.",
        ha="center",
        va="bottom",
        fontsize=8.5,
    )

    fig.subplots_adjust(
        left=0.075, right=0.985, bottom=0.065,
        top=0.94, wspace=0.08, hspace=0.20,
    )

    output_base = FIGURES / "figure4"
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"Saved: {output_base.with_suffix('.svg')}")
    print(f"Saved: {output_base.with_suffix('.png')}")
    plt.show()

if __name__ == "__main__":
    run_pipeline()
