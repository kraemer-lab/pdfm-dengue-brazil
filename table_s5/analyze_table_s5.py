#!/usr/bin/env python3
"""
Generate regional PDFM adjustment data for Supplementary Table S5.

Reads original Thucia forecast-evaluation outputs.

DeltaWIS = WIS_PDFM - WIS_base
    Negative values indicate improvement.

Log-relative skill = log1p(WIS_base) - log1p(WIS_PDFM)
    Positive values indicate improvement.

Both metrics are calculated for individual paired forecasts
before aggregation.

No model fitting or modification of data lakes.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from thucia.core.fs import read_db


HERE = Path(__file__).resolve().parent

DEFAULT_ROOT = (
    HERE.parent.parent / "data/PDFM_analysis/BRA_full"
).resolve()

DEFAULT_OUTPUT = HERE / "data"

MODELS = {
    "SARIMA": "sarima",
    "TCN": "tcn",
    "XGBoost": "xgboost",
    "N-BEATS": "nbeats",
    "TimesFM": "timesfm",
}

HORIZONS = (1, 3, 6, 12)

START = pd.Period("2020-01", freq="M")
END = pd.Period("2025-08", freq="M")

EXCLUDED_GIDS = {
    "BRA.21.229_2",
    "BRA.21.231_2",
}

REGION_STATES = {
    "North": (1, 3, 4, 14, 22, 23, 27),
    "Northeast": (2, 5, 6, 10, 15, 17, 18, 20, 26),
    "Central-West": (7, 9, 11, 12),
    "Southeast": (8, 13, 19, 25),
    "South": (16, 21, 24),
}

STATE_REGION = {
    f"BRA.{state}_1": region
    for region, states in REGION_STATES.items()
    for state in states
}

assert len(STATE_REGION) == 27


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def to_months(series):
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.dt.asfreq("M")

    return pd.Series(
        pd.PeriodIndex(series.astype(str), freq="M"),
        index=series.index,
    )


def load_geography(root):
    """Read and validate geographical assignments from Thucia."""

    lake = read_db(str(root / "cases_with_climate")).df

    required = {"GID_2", "GID_1", "Date", "pop_count"}

    if not required.issubset(lake.columns):
        raise ValueError(
            f"Missing lake columns: {required - set(lake.columns)}"
        )

    df = lake[list(required)].copy()
    df["Date"] = to_months(df["Date"])

    df = df.loc[
        df["Date"].between(START, END)
        & ~df["GID_2"].isin(EXCLUDED_GIDS)
    ].copy()

    if df.duplicated(["GID_2", "Date"]).any():
        raise ValueError("Duplicate municipality-month records")

    population = pd.to_numeric(
        df["pop_count"], errors="coerce"
    )

    if (
        not np.isfinite(population).all()
        or (population <= 0).any()
    ):
        raise ValueError("Invalid population estimates")

    if (
        len(df) != 5570 * 68
        or df["GID_2"].nunique() != 5570
        or df["Date"].nunique() != 68
    ):
        raise ValueError("Unexpected surveillance coverage")

    geography = (
        df[["GID_2", "GID_1"]]
        .drop_duplicates()
        .copy()
    )

    if geography["GID_2"].duplicated().any():
        raise ValueError("Inconsistent state assignments")

    geography["macroregion"] = (
        geography["GID_1"].map(STATE_REGION)
    )

    if geography["macroregion"].isna().any():
        raise ValueError("Unmapped Brazilian states")

    if geography["GID_1"].nunique() != 27:
        raise ValueError("Expected 27 states")

    return geography


def load_forecasts(path):
    """Read existing forecast-evaluation results."""

    df = pd.read_csv(
        path,
        usecols=["GID_2", "Date", "horizon", "WIS", "Cases"],
    )

    df["Date"] = to_months(df["Date"])

    df = df.loc[
        df["Date"].between(START, END)
        & ~df["GID_2"].isin(EXCLUDED_GIDS)
    ].copy()

    keys = ["GID_2", "Date", "horizon"]

    if df.duplicated(keys).any():
        raise ValueError(f"Duplicate forecast keys: {path}")

    if not df["horizon"].isin(HORIZONS).all():
        raise ValueError(f"Unexpected horizons: {path}")

    return df


def load_pairs(root, stem, geography, model):
    """Match baseline and PDFM-adjusted forecasts."""

    baseline = load_forecasts(
        root / f"{stem}_h12_wis.csv"
    )

    adjusted = load_forecasts(
        root / f"{stem}_h12_pdfmrr_pinball_wis.csv"
    )

    keys = ["GID_2", "Date", "horizon"]

    if len(baseline) != 5570 * 68 * 4:
        raise ValueError(f"{model}: incomplete baseline coverage")

    if len(adjusted) != 5568 * 68 * 4:
        raise ValueError(f"{model}: incomplete PDFM coverage")

    if adjusted["GID_2"].nunique() != 5568:
        raise ValueError(f"{model}: unexpected PDFM municipalities")

    paired = adjusted.merge(
        baseline,
        on=keys,
        how="left",
        suffixes=("_PDFM", "_base"),
        validate="one_to_one",
        indicator=True,
    )

    if not paired["_merge"].eq("both").all():
        raise ValueError(f"{model}: unmatched forecasts")

    cases_a = paired["Cases_PDFM"].to_numpy(dtype=float)
    cases_b = paired["Cases_base"].to_numpy(dtype=float)

    if (
        not np.isfinite(cases_a).all()
        or not np.isfinite(cases_b).all()
        or not np.allclose(
            cases_a, cases_b, rtol=0, atol=1e-8
        )
    ):
        raise ValueError(f"{model}: observed cases differ")

    base = paired["WIS_base"].to_numpy(dtype=float)
    pdfm = paired["WIS_PDFM"].to_numpy(dtype=float)

    valid = (
        np.isfinite(base)
        & np.isfinite(pdfm)
        & (base >= 0)
        & (pdfm >= 0)
    )

    invalid = int((~valid).sum())

    if model == "TimesFM":
        if (
            invalid != 198
            or not paired.loc[~valid, "horizon"].eq(12).all()
        ):
            raise ValueError(
                "Unexpected TimesFM nonfinite observations"
            )
    elif invalid:
        raise ValueError(
            f"{model}: {invalid} unexpected invalid pairs"
        )

    paired = paired.loc[valid].copy()

    paired = paired.merge(
        geography,
        on="GID_2",
        how="left",
        validate="many_to_one",
    )

    if paired["macroregion"].isna().any():
        raise ValueError(f"{model}: unmatched geography")

    paired["DeltaWIS"] = (
        paired["WIS_PDFM"] - paired["WIS_base"]
    )

    paired["log_relative_skill"] = (
        np.log1p(paired["WIS_base"])
        - np.log1p(paired["WIS_PDFM"])
    )

    paired["improved"] = paired["DeltaWIS"] < 0

    print(
        f"{model}: {len(paired):,} valid paired forecasts; "
        f"{invalid} excluded"
    )

    return paired, invalid


def summarise_model(paired, model):
    """Calculate regional summaries for all four horizons."""

    rows = []

    for (region, horizon), group in paired.groupby(
        ["macroregion", "horizon"],
        observed=True,
    ):

        municipality = group.groupby(
            "GID_2", observed=True
        )["DeltaWIS"].mean()

        rows.append({
            "model": model,
            "macroregion": region,
            "horizon": int(horizon),

            "n_municipalities": group["GID_2"].nunique(),
            "n_pairs": len(group),
            "n_months": group["Date"].nunique(),

            "mean_base_WIS": group["WIS_base"].mean(),
            "mean_PDFM_WIS": group["WIS_PDFM"].mean(),

            "mean_DeltaWIS": group["DeltaWIS"].mean(),

            "mean_log_relative_skill": (
                group["log_relative_skill"].mean()
            ),

            "pct_forecasts_improved": (
                100 * group["improved"].mean()
            ),

            "pct_municipalities_improved": (
                100 * (municipality < 0).mean()
            ),
        })

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_ROOT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    args = parser.parse_args()

    root = args.data_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()

    print("Loading Thucia geographical data...")
    geography = load_geography(root)

    results = []
    invalid_counts = {}
    source_files = {}

    for model, stem in MODELS.items():

        for suffix in (
            "_wis.csv",
            "_pdfmrr_pinball_wis.csv",
        ):
            path = root / f"{stem}_h12{suffix}"

            if not path.is_file():
                raise FileNotFoundError(path)

            source_files[path.name] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }

        paired, invalid = load_pairs(
            root, stem, geography, model
        )

        invalid_counts[model] = invalid

        results.extend(
            summarise_model(paired, model)
        )

    summary = pd.DataFrame(results)

    if len(summary) != 100:
        raise ValueError(
            f"Expected 100 regional summaries; got {len(summary)}"
        )

    expected = pd.MultiIndex.from_product(
        [
            MODELS.keys(),
            REGION_STATES.keys(),
            HORIZONS,
        ],
        names=["model", "macroregion", "horizon"],
    )

    actual = pd.MultiIndex.from_frame(
        summary[["model", "macroregion", "horizon"]]
    )

    if set(actual) != set(expected):
        raise ValueError("Incomplete model-region-horizon coverage")

    for (model, horizon), group in summary.groupby(
        ["model", "horizon"], observed=True
    ):
        if group["n_municipalities"].sum() != 5568:
            raise ValueError(
                f"{model} h{horizon}: incorrect municipality coverage"
            )

    summary = summary.sort_values(
        ["macroregion", "model", "horizon"]
    )

    output.mkdir(parents=True, exist_ok=True)

    result_path = output / "regional_pdfm_skill.csv"

    summary.to_csv(result_path, index=False)

    manifest = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),

        "data_root": str(root),
        "output_dir": str(output),

        "effective_cli_arguments": {
            "data_root": str(root),
            "output_dir": str(output),
        },

        "models": MODELS,
        "horizons": list(HORIZONS),

        "evaluation_start": str(START),
        "evaluation_end": str(END),

        "excluded_gids": sorted(EXCLUDED_GIDS),

        "state_to_macroregion": STATE_REGION,

        "expected_pdfm_municipalities": 5568,

        "invalid_pairs": invalid_counts,

        "delta_definition": "WIS_PDFM - WIS_base",

        "log_relative_skill_definition": (
            "log1p(WIS_base) - log1p(WIS_PDFM)"
        ),

        "aggregation": (
            "Mean of individual paired forecast-level metrics"
        ),

        "source_files": source_files,

        "output_file": result_path.name,
        "output_sha256": sha256(result_path),
    }

    (output / "analysis_manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nWrote {result_path}")


if __name__ == "__main__":
    main()
