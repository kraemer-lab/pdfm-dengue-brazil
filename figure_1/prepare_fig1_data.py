
#!/usr/bin/env python3
"""
Prepare reproducible publication data for Figure 1 v5.

This version uses the actual Thucia database interface (read_db) for both
the PDFM embeddings and the cases/covariate source, matching the way the
analysis pipeline reads these files.

Fixed sources
-------------
/Users/jsb/repos/jsbrittain/Thucia/data/PDFM_analysis/BRA_full/embeddings.nc
/Users/jsb/repos/jsbrittain/Thucia/data/PDFM_analysis/BRA_full/cases_with_climate
/Users/jsb/repos/jsbrittain/Thucia/data/nightlight/nightlight.duckdb

PDFM representation
-------------------
The 330-dimensional embedding is divided into three modality-specific
blocks:

  feature0-feature127    Search Trends
  feature128-feature255  Maps & Busyness
  feature256-feature329  Weather & Air Quality

The visualization uses a 3 x 330 correlation matrix:
  Search Trends        -> log night-light
  Maps & Busyness      -> population
  Weather & Air Quality -> temperature

For each of the three modality blocks, the representative map in Panel D is
the embedding dimension with the largest absolute Pearson correlation with
its corresponding anchor. The selection is therefore deterministic and is
not hand-picked.

The anchor variables are constructed as follows:
  * log night-light: log(mean night-light) from nightlight.duckdb
  * population: municipality mean population over Jan 2010-Dec 2019
  * temperature: municipality mean of (minimum + maximum temperature)/2 over
    Jan 2010-Dec 2019

No command-line arguments are required.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from thucia.core.fs import read_db

HERE = Path(__file__).resolve().parent
THUCIA = HERE.parents[1]
BRA_FULL = THUCIA / "data" / "PDFM_analysis" / "BRA_full"
EMBEDDINGS_PATH = BRA_FULL / "embeddings.nc"
CASES_PATH = BRA_FULL / "cases_with_climate"
NIGHTLIGHT_PATH = THUCIA / "data" / "nightlight" / "nightlight.duckdb"

OUT_DIR = HERE / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Period("2010-01", freq="M")
TRAIN_END = pd.Period("2019-12", freq="M")
END = pd.Period("2025-08", freq="M")

PDFM_BLOCKS = {
    "Search Trends": list(range(0, 128)),
    "Maps & Busyness": list(range(128, 256)),
    "Weather & Air Quality": list(range(256, 330)),
}

ANCHORS = [
    ("Search Trends", "log_nightlight"),
    ("Maps & Busyness", "population"),
    ("Weather & Air Quality", "temperature"),
]


def resolve_column(columns, aliases, required=True):
    lower = {str(c).lower(): c for c in columns}
    for alias in aliases:
        if alias.lower() in lower:
            return lower[alias.lower()]

    normalized = [
        (c, str(c).lower().replace(" ", "_").replace("-", "_"))
        for c in columns
    ]
    for alias in aliases:
        a = alias.lower().replace(" ", "_").replace("-", "_")
        matches = sorted(
            c for c, norm in normalized
            if a == norm or a in norm
        )
        if matches:
            return matches[0]

    if required:
        raise ValueError(
            f"Could not resolve required column from aliases {aliases}. "
            f"Available columns: {list(columns)}"
        )
    return None


def load_cases():
    df = read_db(str(CASES_PATH)).df.copy()
    columns = list(df.columns)

    gid_col = resolve_column(
        columns, ["GID_2", "GID2", "gid_2"]
    )
    date_col = resolve_column(
        columns, ["Date", "date", "Month", "month"]
    )
    cases_col = resolve_column(
        columns, ["Cases", "cases", "Dengue_cases", "reported_cases"]
    )

    pop_col = resolve_column(
        columns, ["Population", "population", "pop_count", "worldpop"],
    )
    min_temp_col = resolve_column(
        columns,
        [
            "Minimum temperature",
            "minimum_temperature",
            "min_temperature",
            "tmin",
            "temperature_min",
            "temp_min",
            "min_temp",
        ],
    )
    max_temp_col = resolve_column(
        columns,
        [
            "Maximum temperature",
            "maximum_temperature",
            "max_temperature",
            "tmax",
            "temperature_max",
            "temp_max",
            "max_temp",
        ],
    )

    df = df.rename(
        columns={
            gid_col: "GID_2",
            date_col: "Date",
            cases_col: "Cases",
            pop_col: "Population",
            min_temp_col: "MinTemperature",
            max_temp_col: "MaxTemperature",
        }
    )

    df["Date"] = pd.PeriodIndex(df["Date"], freq="M")
    for col in [
        "Cases", "Population", "MinTemperature", "MaxTemperature"
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        df["GID_2"].notna()
        & (df["Date"] >= START)
        & (df["Date"] <= END)
    ].copy()

    national = (
        df.groupby("Date", as_index=False)["Cases"]
        .sum(min_count=1)
        .rename(columns={"Cases": "national_cases"})
    )
    national.to_csv(
        OUT_DIR / "figure1_national_cases.csv", index=False
    )

    burden = (
        df.groupby("GID_2", as_index=False)["Cases"]
        .sum(min_count=1)
        .rename(columns={"Cases": "cumulative_cases"})
    )
    burden.to_csv(
        OUT_DIR / "figure1_municipality_burden.csv", index=False
    )

    train = df[df["Date"] <= TRAIN_END].copy()
    anchors = (
        train.groupby("GID_2", as_index=False)
        .agg(
            population=("Population", "mean"),
            min_temperature=("MinTemperature", "mean"),
            max_temperature=("MaxTemperature", "mean"),
        )
    )
    anchors["temperature"] = (
        anchors["min_temperature"] + anchors["max_temperature"]
    ) / 2.0

    return anchors


def load_nightlight():
    if not NIGHTLIGHT_PATH.exists():
        raise FileNotFoundError(NIGHTLIGHT_PATH)

    nl = read_db(str(NIGHTLIGHT_PATH)).df.copy()
    gid_col = resolve_column(
        nl.columns, ["GID_2", "GID2", "gid_2"]
    )
    mean_col = resolve_column(
        nl.columns, ["mean", "Mean", "nightlight_mean"]
    )

    nl = nl[[gid_col, mean_col]].copy()
    nl = nl.rename(
        columns={gid_col: "GID_2", mean_col: "nightlight_mean"}
    )
    nl["nightlight_mean"] = pd.to_numeric(
        nl["nightlight_mean"], errors="coerce"
    )
    nl = nl.dropna(subset=["GID_2", "nightlight_mean"])
    nl = nl.drop_duplicates("GID_2")
    nl["log_nightlight"] = np.log(
        nl["nightlight_mean"].clip(lower=np.finfo(float).tiny)
    )
    return nl[["GID_2", "log_nightlight"]]


def load_embeddings():
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(EMBEDDINGS_PATH)

    emb = read_db(str(EMBEDDINGS_PATH)).df.copy()
    if "GID_2" not in emb.columns:
        raise ValueError("embeddings.nc does not contain GID_2")

    feature_cols = [f"feature{i}" for i in range(330)]
    missing = [c for c in feature_cols if c not in emb.columns]
    if missing:
        raise ValueError(
            f"embeddings.nc is missing {len(missing)} expected dimensions: "
            f"{missing[:10]}"
        )

    emb = emb[["GID_2"] + feature_cols].copy()
    return emb


def build_products():
    anchors = load_cases()
    nl = load_nightlight()
    emb = load_embeddings()

    merged = (
        emb.merge(anchors, on="GID_2", how="inner")
           .merge(nl, on="GID_2", how="inner")
    )

    feature_cols = [f"feature{i}" for i in range(330)]

    corr_rows = []
    for group, anchor in ANCHORS:
        for i in PDFM_BLOCKS[group]:
            feature = f"feature{i}"
            x = pd.to_numeric(merged[anchor], errors="coerce")
            y = pd.to_numeric(merged[feature], errors="coerce")
            mask = x.notna() & y.notna()

            r = np.nan
            if mask.sum() >= 10 and x[mask].nunique() > 1 and y[mask].nunique() > 1:
                r = float(x[mask].corr(y[mask]))

            corr_rows.append(
                {
                    "group": group,
                    "anchor": anchor,
                    "feature": feature,
                    "feature_index": i,
                    "pearson_r": r,
                    "n": int(mask.sum()),
                }
            )

    corr_long = pd.DataFrame(corr_rows)
    corr_long.to_csv(
        OUT_DIR / "figure1_pdfm_anchor_correlations.csv",
        index=False,
    )

    selected = (
        corr_long.loc[
            corr_long.groupby("group")["pearson_r"]
            .apply(lambda s: s.abs().idxmax())
            .values
        ]
        .sort_values("feature_index")
        .reset_index(drop=True)
    )
    selected.to_csv(
        OUT_DIR / "figure1_pdfm_selected_dimensions.csv",
        index=False,
    )

    map_rows = []
    for _, row in selected.iterrows():
        feature = row["feature"]
        part = merged[["GID_2", feature]].copy()
        part = part.rename(columns={feature: "embedding_value"})
        part["group"] = row["group"]
        part["anchor"] = row["anchor"]
        part["feature"] = feature
        part["pearson_r"] = row["pearson_r"]
        map_rows.append(part)

    maps = pd.concat(map_rows, ignore_index=True)
    maps.to_csv(
        OUT_DIR / "figure1_pdfm_selected_maps.csv",
        index=False,
    )

    # Matrix in publication order: three anchors x 330 embedding dimensions.
    matrix = corr_long.pivot(
        index="anchor",
        columns="feature_index",
        values="pearson_r",
    )
    matrix = matrix.reindex(
        index=["log_nightlight", "population", "temperature"],
        columns=list(range(330)),
    )
    matrix.to_csv(
        OUT_DIR / "figure1_pdfm_anchor_matrix.csv"
    )

    print("\nPDFM block definitions:")
    for group, dims in PDFM_BLOCKS.items():
        print(
            f"  {group}: feature{dims[0]}-feature{dims[-1]} "
            f"({len(dims)} dimensions)"
        )

    print("\nRepresentative dimensions:")
    print(
        selected[
            ["group", "anchor", "feature", "pearson_r", "n"]
        ].to_string(index=False)
    )


def main():
    print("=" * 100)
    print("FIGURE 1 V5 DATA PREPARATION")
    print("=" * 100)
    print(f"BRA_FULL: {BRA_FULL}")
    print(f"Embeddings: {EMBEDDINGS_PATH}")
    print(f"Cases/covariates: {CASES_PATH}")
    print(f"Night-light: {NIGHTLIGHT_PATH}")

    build_products()

    print(f"\nWrote Figure 1 v5 data to: {OUT_DIR}")


if __name__ == "__main__":
    main()
