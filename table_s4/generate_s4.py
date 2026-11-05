#!/usr/bin/env python3
"""Generate Supplementary Table S9 from Figure 2F bootstrap results."""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "figure_2" / "data"

BLOCK_MONTHS = 3

MODELS = ["SARIMA", "TCN", "XGBoost", "N-BEATS", "TimesFM"]
HORIZONS = [1, 3, 6, 12]
STRATA = ["zero", "low", "medium", "high"]


def main():
    a = pd.read_csv(
        DATA / "fig2f_stratum_effects_bootstrap.csv"
    )
    b = pd.read_csv(
        DATA / "fig2f_high_vs_low_bootstrap.csv"
    )

    high = a[
        (a["block_months"] == 3)
        & (a["stratum"] == "high")
    ].copy()

    contrasts = b[
        (b["block_months"] == 3)
        & (b["contrast"] == "high_minus_low")
    ].copy()

    keys = ["model", "horizon"]

    assert len(high) == 20
    assert len(contrasts) == 20
    assert not high.duplicated(keys).any()
    assert not contrasts.duplicated(keys).any()

    merged = high.merge(
        contrasts,
        on=keys,
        validate="one_to_one",
        suffixes=("_high", "_contrast"),
    )

    merged["model"] = pd.Categorical(
        merged["model"],
        categories=MODELS,
        ordered=True,
    )

    merged = merged.sort_values(["model", "horizon"])

    def format_ci(lower, upper):
        return [
            f"[{lo:.3f}, {hi:.3f}]"
            for lo, hi in zip(lower, upper)
        ]

    table = pd.DataFrame({
        "Model": merged["model"].astype(str),
        "Horizon (months)": merged["horizon"].astype(int),
        "High-stratum mean ΔWIS":
            merged["mean_DeltaWIS_month_weighted"],
        "High-stratum 95% CI": format_ci(
            merged["CI95_lower_high"],
            merged["CI95_upper_high"],
        ),
        "High minus low ΔWIS":
            merged["mean_difference"],
        "High minus low 95% CI": format_ci(
            merged["CI95_lower_contrast"],
            merged["CI95_upper_contrast"],
        ),
    })

    for extension, separator in [
        ("csv", ","),
        ("tsv", "\t"),
    ]:
        output = ROOT / f"table_s4.{extension}"

        table.to_csv(
            output,
            sep=separator,
            index=False,
            float_format="%.3f",
        )

        print(f"Wrote: {output}")

    print(f"Generated {len(table)} rows.")

if __name__ == "__main__":
    main()
