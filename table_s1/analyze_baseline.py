import logging
from pathlib import Path

import pandas as pd
from thucia.core.logging import enable_logging


enable_logging(level=logging.INFO)


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
BRA_FULL = (
    PROJECT_DIR.parent.parent
    / "data"
    / "PDFM_analysis"
    / "BRA_full"
).resolve()

START_DATE = pd.Period("2020-01", freq="M")
END_DATE = pd.Period("2025-08", freq="M")

MODELS = {
    "SARIMA": "sarima_h12_wis.csv",
    "TCN": "tcn_h12_wis.csv",
    "XGBoost": "xgboost_h12_wis.csv",
    "N-BEATS": "nbeats_h12_wis.csv",
    "TimesFM": "timesfm_h12_wis.csv",
}

HORIZONS = [1, 3, 6, 12]

OUTPUT_DIR = PROJECT_DIR / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

EXCLUDED_GIDS = {
    "BRA.21.229_2",
    "BRA.21.231_2",
}


# ---------------------------------------------------------------------
# Load one WIS file
# ---------------------------------------------------------------------

def load_wis(filename):
    path = BRA_FULL / filename

    logging.info("Loading %s", path)

    df = pd.read_csv(path)

    # Match generate_fig2.py
    df["Date"] = pd.PeriodIndex(df["Date"], freq="M")

    df = df[
        (df["Date"] >= START_DATE)
        & (df["Date"] <= END_DATE)
        & df["Cases"].notna()
    ].copy()

    df = df.loc[
        ~df["GID_2"].isin(EXCLUDED_GIDS)
    ].copy()

    print(
        "Municipalities:", df["GID_2"].nunique(),
        "Observations:", len(df),
        "Excluded IDs remaining:",
        df["GID_2"].isin(EXCLUDED_GIDS).sum(),
    )

    # Match existing handling of R2 in generate_fig2.py
    if "R2" in df.columns:
        df["R2"] = df["R2"].clip(lower=0, upper=1)

    return df


# ---------------------------------------------------------------------
# Summarise one model
# ---------------------------------------------------------------------

def summarise_model(model, filename):
    df = load_wis(filename)

    logging.info(
        "%s: rows=%d, municipalities=%d, dates=%d",
        model,
        len(df),
        df["GID_2"].nunique() if "GID_2" in df.columns else -1,
        df["Date"].nunique(),
    )

    records = []

    for horizon in HORIZONS:
        d = df[df["horizon"] == horizon].copy()

        record = {
            "model": model,
            "horizon": horizon,
            "n": len(d),
            "n_municipalities": d["GID_2"].nunique()
            if "GID_2" in d.columns
            else None,
            "n_dates": d["Date"].nunique(),
            "mean_WIS": d["WIS"].mean(),
            "median_WIS": d["WIS"].median(),
            "P90_WIS": d["WIS"].quantile(0.90),
            "P95_WIS": d["WIS"].quantile(0.95),
            "max_WIS": d["WIS"].max(),
        }

        if "R2" in d.columns:
            record["mean_R2"] = d["R2"].mean()
            record["median_R2"] = d["R2"].median()
            record["P10_R2"] = d["R2"].quantile(0.10)
            record["P90_R2"] = d["R2"].quantile(0.90)
        else:
            record["mean_R2"] = None
            record["median_R2"] = None
            record["P10_R2"] = None
            record["P90_R2"] = None

        records.append(record)

    return pd.DataFrame(records)



def build_violin_sample(max_per_group=10000, random_state=42):
    """Create a reproducible plotting sample of row-level WIS values.

    Sampling is performed separately within each model x horizon group so that
    the violin panel can show the empirical WIS distribution without writing
    the full multi-million-row evaluation dataset to the public figure inputs.
    """
    rows = []
    for model, filename in MODELS.items():
        df = load_wis(filename)
        for horizon in HORIZONS:
            d = df.loc[df["horizon"] == horizon, ["WIS"]].dropna().copy()
            if len(d) > max_per_group:
                d = d.sample(n=max_per_group, random_state=random_state)
            d["model"] = model
            d["horizon"] = horizon
            rows.append(d[["model", "horizon", "WIS"]])
    return pd.concat(rows, ignore_index=True)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 90)
    print("PRIMARY BASELINE FORECASTING RESULTS")
    print("=" * 90)
    print(f"Data directory: {BRA_FULL}")
    print(f"Evaluation period: {START_DATE} to {END_DATE}")
    print(f"Horizons: {HORIZONS}")
    print("=" * 90)

    summaries = []

    for model, filename in MODELS.items():
        summaries.append(
            summarise_model(model, filename)
        )

    results = pd.concat(
        summaries,
        ignore_index=True,
    )

    # -------------------------------------------------------------
    # Print long table
    # -------------------------------------------------------------

    cols = [
        "model",
        "horizon",
        "mean_WIS",
        "median_WIS",
        "P90_WIS",
        "P95_WIS",
        "max_WIS",
        "mean_R2",
        "median_R2",
        "n",
        "n_municipalities",
        "n_dates",
    ]

    print("\n")
    print("=" * 90)
    print("TABLE 1 — BASELINE PERFORMANCE")
    print("=" * 90)

    print(
        results[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    # -------------------------------------------------------------
    # Mean WIS
    # -------------------------------------------------------------

    mean_wis = results.pivot(
        index="model",
        columns="horizon",
        values="mean_WIS",
    ).reindex(columns=HORIZONS)

    print("\n")
    print("=" * 90)
    print("TABLE 2 — MEAN WIS")
    print("=" * 90)
    print(
        mean_wis.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # -------------------------------------------------------------
    # Median WIS
    # -------------------------------------------------------------

    median_wis = results.pivot(
        index="model",
        columns="horizon",
        values="median_WIS",
    ).reindex(columns=HORIZONS)

    print("\n")
    print("=" * 90)
    print("TABLE 3 — MEDIAN WIS")
    print("=" * 90)
    print(
        median_wis.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # -------------------------------------------------------------
    # P95 WIS
    # -------------------------------------------------------------

    p95_wis = results.pivot(
        index="model",
        columns="horizon",
        values="P95_WIS",
    ).reindex(columns=HORIZONS)

    print("\n")
    print("=" * 90)
    print("TABLE 4 — P95 WIS")
    print("=" * 90)
    print(
        p95_wis.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # -------------------------------------------------------------
    # Mean R2
    # -------------------------------------------------------------

    if results["mean_R2"].notna().any():
        mean_r2 = results.pivot(
            index="model",
            columns="horizon",
            values="mean_R2",
        ).reindex(columns=HORIZONS)

        print("\n")
        print("=" * 90)
        print("TABLE 5 — MEAN R2")
        print("=" * 90)
        print(
            mean_r2.to_string(
                float_format=lambda x: f"{x:.4f}"
            )
        )

    # -------------------------------------------------------------
    # Rank models by mean WIS at each horizon
    # -------------------------------------------------------------

    ranked = results.copy()
    ranked["rank_mean_WIS"] = (
        ranked.groupby("horizon")["mean_WIS"]
        .rank(method="min", ascending=True)
    )

    rank_table = ranked[
        [
            "model",
            "horizon",
            "mean_WIS",
            "median_WIS",
            "rank_mean_WIS",
        ]
    ].sort_values(["horizon", "rank_mean_WIS", "model"])

    print("\n")
    print("=" * 90)
    print("TABLE 6 — MODEL RANKING BY MEAN WIS")
    print("=" * 90)
    print(
        rank_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # -------------------------------------------------------------
    # Horizon degradation
    # -------------------------------------------------------------

    degradation = []

    for model in MODELS:
        d = results[results["model"] == model].set_index("horizon")

        h1 = d.loc[1, "mean_WIS"]
        h12 = d.loc[12, "mean_WIS"]

        degradation.append(
            {
                "model": model,
                "mean_WIS_h1": h1,
                "mean_WIS_h12": h12,
                "absolute_change_h1_to_h12": h12 - h1,
                "relative_change_h1_to_h12": (
                    (h12 - h1) / h1
                    if h1 != 0
                    else None
                ),
            }
        )

    degradation = pd.DataFrame(degradation)

    print("\n")
    print("=" * 90)
    print("TABLE 7 — WIS CHANGE FROM 1 TO 12 MONTHS")
    print("=" * 90)
    print(
        degradation.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    # -------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------

    results.to_csv(
        OUTPUT_DIR / "baseline_results_long.csv",
        index=False,
    )

    violin_sample = build_violin_sample()
    violin_sample.to_csv(
        OUTPUT_DIR / "baseline_wis_violin_sample.csv",
        index=False,
    )

    mean_wis.to_csv(
        OUTPUT_DIR / "baseline_mean_wis.csv",
    )

    median_wis.to_csv(
        OUTPUT_DIR / "baseline_median_wis.csv",
    )

    p95_wis.to_csv(
        OUTPUT_DIR / "baseline_p95_wis.csv",
    )

    if results["mean_R2"].notna().any():
        mean_r2.to_csv(
            OUTPUT_DIR / "baseline_mean_r2.csv",
        )

    rank_table.to_csv(
        OUTPUT_DIR / "baseline_model_ranks.csv",
        index=False,
    )

    degradation.to_csv(
        OUTPUT_DIR / "baseline_horizon_degradation.csv",
        index=False,
    )

    print("\n")
    print("=" * 90)
    print("SAVED")
    print("=" * 90)
    for p in sorted(OUTPUT_DIR.glob("baseline_*.csv")):
        print(p)


if __name__ == "__main__":
    main()
