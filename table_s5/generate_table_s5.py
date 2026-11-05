#!/usr/bin/env python3
"""
Generate Supplementary Table S5.

Each cell reports:

    Mean DeltaWIS (mean log-relative WIS skill)

DeltaWIS = WIS_PDFM - WIS_base
    Negative indicates improvement.

Log-relative skill = log1p(WIS_base) - log1p(WIS_PDFM)
    Positive indicates improvement.

Run:
    python generate_table_s5.py
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent

INPUT = HERE / "data/regional_pdfm_skill.csv"
OUTPUT = HERE / "table_s5.tsv"

REGIONS = (
    "North",
    "Northeast",
    "Central-West",
    "Southeast",
    "South",
)

MODELS = (
    "SARIMA",
    "TCN",
    "XGBoost",
    "N-BEATS",
    "TimesFM",
)

HORIZONS = (1, 3, 6, 12)


def sha256(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def main():

    data = pd.read_csv(INPUT)

    required = {
        "model",
        "macroregion",
        "horizon",
        "n_municipalities",
        "n_pairs",
        "mean_DeltaWIS",
        "mean_log_relative_skill",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    keys = ["macroregion", "model", "horizon"]

    if data.duplicated(keys).any():
        raise ValueError("Duplicate regional summaries")

    expected = pd.MultiIndex.from_product(
        [REGIONS, MODELS, HORIZONS],
        names=keys,
    )

    actual = pd.MultiIndex.from_frame(data[keys])

    if (
        len(data) != len(expected)
        or set(actual) != set(expected)
    ):
        raise ValueError(
            "Expected exactly 100 model-region-horizon combinations"
        )

    numeric = data[
        [
            "mean_DeltaWIS",
            "mean_log_relative_skill",
            "n_municipalities",
            "n_pairs",
        ]
    ].apply(pd.to_numeric, errors="raise")

    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("Nonfinite summary values")

    if (numeric["n_pairs"] <= 0).any():
        raise ValueError("Invalid paired observation counts")

    # Verify the PDFM geographical population.

    for model in MODELS:
        for horizon in HORIZONS:

            subset = data.loc[
                (data["model"] == model)
                & (data["horizon"] == horizon)
            ]

            total = int(
                subset["n_municipalities"].sum()
            )

            if total != 5568:
                raise ValueError(
                    f"{model}, h{horizon}: "
                    f"expected 5568 municipalities, got {total}"
                )

    rows = []

    for region in REGIONS:

        for i, model in enumerate(MODELS):

            subset = data.loc[
                (data["macroregion"] == region)
                & (data["model"] == model)
            ].set_index("horizon")

            n = int(
                subset.loc[1, "n_municipalities"]
            )

            row = {
                "Geography": region if i == 0 else "",
                "Model": model,
                "Municipalities": n if i == 0 else "",
            }

            for horizon in HORIZONS:

                record = subset.loc[horizon]

                if int(record["n_municipalities"]) != n:
                    raise ValueError(
                        f"Inconsistent coverage: {region}, {model}"
                    )

                delta = float(record["mean_DeltaWIS"])

                skill = float(
                    record["mean_log_relative_skill"]
                )

                label = (
                    f"{horizon} month"
                    + ("s" if horizon != 1 else "")
                )

                row[label] = (
                    f"{delta:+.3f} ({skill:+.3f})"
                )

            rows.append(row)

    table = pd.DataFrame(rows)

    table.to_csv(
        OUTPUT,
        sep="\t",
        index=False,
        encoding="utf-8",
    )

    manifest = {
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "script_sha256": sha256(
            Path(__file__).resolve()
        ),

        "input": str(INPUT),
        "input_sha256": sha256(INPUT),

        "output": str(OUTPUT),
        "output_sha256": sha256(OUTPUT),

        "models": list(MODELS),
        "regions": list(REGIONS),
        "horizons": list(HORIZONS),

        "cell_definition": (
            "Mean paired DeltaWIS "
            "(mean paired log-relative WIS skill)"
        ),

        "analysis_unit": (
            "Municipality-month forecast pair"
        ),

        "rounding": "Three decimal places",

        "notes": [
            "Negative DeltaWIS indicates PDFM improvement.",
            "Positive log-relative skill indicates PDFM improvement.",
            "No data-lake access in table generator.",
            "Nonfinite pairs excluded upstream without imputation.",
        ],
    }

    (HERE / "table_manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {OUTPUT} ({len(table)} rows)")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
