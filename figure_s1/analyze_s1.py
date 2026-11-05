#!/usr/bin/env python3
"""
Analyze data for Figure 4.

This script is allowed to read the prepared WIS CSV tables only.
It does NOT access raw case/climate databases.

Outputs:
  data/figure4_violin_data.csv
  data/figure4_top20_regions.csv
"""
import logging
from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA_ROOT = (HERE.parent.parent / "data" / "PDFM_analysis").resolve()
OUT = HERE / "data"
OUT.mkdir(parents=True, exist_ok=True)

START_DATE = pd.Period("2020-01", freq="M")
END_DATE = pd.Period("2025-08", freq="M")
MODEL_ORDER = ["SARIMA", "TCN", "N-BEATS", "XGBoost", "TimesFM"]

EXCLUDED_GIDS = {
    "BRA.21.229_2",
    "BRA.21.231_2",
}

class PlotChoice(Enum):
    FULL = 1
    AGG = 2
    TOP20 = 3
    ADM1 = 4

MATCH = {
    PlotChoice.ADM1: {
        "suffix": "adm1",
        "path": DATA_ROOT / "BRA_adm1",
        "geo_col": "GID_1",
    },
    PlotChoice.AGG: {
        "suffix": "h12_agg",
        "path": DATA_ROOT / "BRA_full",
        "geo_col": "GID_1",
    },
    PlotChoice.FULL: {
        "suffix": "h12",
        "path": DATA_ROOT / "BRA_full",
        "geo_col": "GID_2",
    },
    PlotChoice.TOP20: {
        "suffix": "top20",
        "path": DATA_ROOT / "BRA_top20",
        "geo_col": "GID_2",
    },
}

MODEL_STEM = {
    "SARIMA": "sarima",
    "TCN": "tcn",
    "N-BEATS": "nbeats",
    "XGBoost": "xgboost",
    "TimesFM": "timesfm",
}

def normalise_gid(gid):
    gid = str(gid)
    return gid.rsplit("_", 1)[0] if "_" in gid else gid

def infer_gid1_from_gid2(gid2):
    gid = normalise_gid(gid2)
    parts = gid.split(".")
    if len(parts) != 3:
        raise ValueError(f"Unexpected GID_2 structure: {gid2}")
    return ".".join(parts[:2])

def read_wis(path):
    df = pd.read_csv(path)
    required = {"Date", "Cases", "WIS", "R2", "horizon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    df["Date"] = pd.PeriodIndex(df["Date"], freq="M")
    df = df[
        (df["Date"] >= START_DATE)
        & (df["Date"] <= END_DATE)
    ].copy()
    df = df[df["Cases"].notna()].copy()
    if "GID_2" in df.columns:
        df = df.loc[
            ~df["GID_2"].isin(EXCLUDED_GIDS)
        ].copy()
    df["R2"] = df["R2"].clip(lower=0, upper=1)
    return df

def read_model(model_name, filestem, data_path, geo_col):
    base = read_wis(data_path / f"{filestem}_wis.csv")
    pdfm = read_wis(data_path / f"{filestem}_pdfmrr_pinball_wis.csv")

    keys = [geo_col, "Date", "horizon"]
    merged = pd.merge(
        pdfm,
        base,
        on=keys,
        suffixes=(" PDFM", " Base"),
        validate="one_to_one",
    )

    merged.rename(columns={"Cases PDFM": "Cases"}, inplace=True)
    if "Cases Base" in merged.columns:
        merged.drop(columns=["Cases Base"], inplace=True)

    merged["Cases"] = np.expm1(merged["Cases"])
    merged["WIS Diff"] = merged["WIS PDFM"] - merged["WIS Base"]
    merged["R2 Diff"] = merged["R2 PDFM"] - merged["R2 Base"]
    merged["model"] = model_name

    if geo_col == "GID_2":
        merged["GID_2"] = merged["GID_2"].astype(str)
        merged["GID_1"] = merged["GID_2"].map(infer_gid1_from_gid2)

    return merged

def process_config(choice):
    cfg = MATCH[choice]
    parts = []
    for model in MODEL_ORDER:
        stem = f"{MODEL_STEM[model]}_{cfg['suffix']}"
        parts.append(
            read_model(
                model,
                stem,
                cfg["path"],
                cfg["geo_col"],
            )
        )
    df = pd.concat(parts, ignore_index=True)

    geo = cfg["geo_col"]
    out = (
        df[
            [
                geo,
                "Cases",
                "WIS Base",
                "WIS PDFM",
                "R2 Base",
                "R2 PDFM",
                "horizon",
                "model",
            ]
        ]
        .groupby(["model", "horizon", geo], observed=True)
        .agg(
            {
                "Cases": "mean",
                "WIS Base": "mean",
                "WIS PDFM": "mean",
                "R2 Base": "mean",
                "R2 PDFM": "mean",
            }
        )
        .reset_index()
    )
    out["configuration"] = {
        PlotChoice.FULL: "Multivariate",
        PlotChoice.TOP20: "Univariate",
        PlotChoice.ADM1: "Aggregate cases",
        PlotChoice.AGG: "Aggregate forecasts",
    }[choice]
    out["geo_col"] = geo
    return out

def main():
    frames = []
    for choice in [PlotChoice.FULL, PlotChoice.TOP20, PlotChoice.ADM1, PlotChoice.AGG]:
        print(f"Processing {choice.name}...")
        frames.append(process_config(choice))

    violin = pd.concat(frames, ignore_index=True)

    # Preserve the exact map logic from v6: all municipalities are shown faintly,
    # and the selected top20 municipalities are highlighted.
    full_regions = (
        read_wis(DATA_ROOT / "BRA_full" / "tcn_h12_wis.csv")[["GID_2"]]
        .drop_duplicates()
        .sort_values("GID_2")
    )
    top20 = (
        violin.loc[
            violin["configuration"] == "Univariate",
            "GID_2",
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
    )
    top20_set = set(top20)
    full_regions["selected"] = full_regions["GID_2"].astype(str).isin(top20_set).astype(int)

    violin.to_csv(OUT / "figure4_violin_data.csv", index=False)
    full_regions.to_csv(OUT / "figure4_top20_regions.csv", index=False)

    print("=" * 90)
    print("FIGURE 4 ANALYSIS")
    print("=" * 90)
    print(f"Rows: {len(violin):,}")
    print(f"Configurations: {violin['configuration'].unique().tolist()}")
    print(f"Top20 selected municipalities: {int(full_regions['selected'].sum())}")
    print(f"Wrote: {OUT / 'figure4_violin_data.csv'}")
    print(f"Wrote: {OUT / 'figure4_top20_regions.csv'}")

if __name__ == "__main__":
    main()
