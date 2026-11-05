#!/usr/bin/env python3
"""
Generate the final Supplementary Table S9.

States / experiments
--------------------
Ceará:
  one-hit = BRA_full
  monthly retraining = BRA_ceara
  models = SARIMA, TCN, XGBoost
  n = 184 municipalities

São Paulo:
  one-hit = BRA_full
  monthly retraining = BRA_saopaulo
  models = SARIMA, XGBoost
  N-BEATS retraining run is reported separately as a numerical failure and
  is NOT entered into the Wilcoxon performance comparison.
  TCN and TimesFM were not assessed for São Paulo retraining.

For each valid state/model/horizon comparison:
  - average WIS across January 2020-August 2025 within municipality
  - pair municipality-level means between one-hit and monthly retraining
  - two-sided Wilcoxon signed-rank test
  - median difference = median(WIS_one_hit - WIS_monthly_retraining)

Negative differences favour one-hit training.

Outputs
-------
table_s9.csv
table_s9.tsv

Inputs are prepared WIS CSVs only. No raw data or DuckDB files are accessed.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


HERE = Path(__file__).resolve().parent
DATA_ROOT = (HERE.parent.parent / "data" / "PDFM_analysis").resolve()

ONE_HIT = DATA_ROOT / "BRA_full"

STATE_CONFIG = {
    "Ceará": {
        "path": DATA_ROOT / "BRA_ceara",
        "n_municipalities": 184,
        "models": {
            "SARIMA": "sarima",
            "TCN": "tcn",
            "XGBoost": "xgboost",
        },
        "failure_notes": [],
    },
    "São Paulo": {
        "path": DATA_ROOT / "BRA_saopaulo",
        "n_municipalities": 645,
        "models": {
            "SARIMA": "sarima",
            "XGBoost": "xgboost",
        },
        "failure_notes": [
            (
                "N-BEATS",
                "not evaluable: retrained forecasts produced non-finite "
                "WIS values across the municipality set at all horizons"
            ),
        ],
    },
}

HORIZONS = [1, 3, 6, 12]
START = pd.Period("2020-01", freq="M")
END = pd.Period("2025-08", freq="M")

OUTPUT_CSV = HERE / "table_s9.csv"
OUTPUT_TSV = HERE / "table_s9.tsv"
FAILURE_CSV = HERE / "table_s9_failures.csv"


def read_wis(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    required = {"GID_2", "Date", "horizon", "WIS", "Cases"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path.name}: missing columns {sorted(missing)}; "
            f"found {list(df.columns)}"
        )

    df["Date"] = pd.PeriodIndex(df["Date"], freq="M")
    df["horizon"] = pd.to_numeric(df["horizon"], errors="raise")
    df["WIS"] = pd.to_numeric(df["WIS"], errors="coerce")

    return df[
        df["Date"].between(START, END)
        & df["horizon"].isin(HORIZONS)
        & df["Cases"].notna()
    ].copy()


def municipality_means(df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Mean WIS within municipality/horizon, using all finite WIS observations.

    A later validation established that the canonical Ceará comparisons are
    complete and finite. For São Paulo N-BEATS, that model is excluded from
    performance testing because the retrained run is numerically unstable.
    """
    finite_df = df[np.isfinite(df["WIS"])].copy()
    return (
        finite_df.groupby(["horizon", "GID_2"], observed=True)["WIS"]
        .mean()
        .rename(value_name)
        .reset_index()
    )


def analyse_state_model(
    state: str,
    config: dict,
    model: str,
    stem: str,
) -> list[dict]:
    one_hit_path = ONE_HIT / f"{stem}_h12_wis.csv"
    retrained_path = config["path"] / f"{stem}_h12_wis.csv"

    one_hit = read_wis(one_hit_path)
    retrained = read_wis(retrained_path)

    # Comparison population is defined by the state retrained output.
    retrained_gids = set(retrained["GID_2"].astype(str))
    expected_n = config["n_municipalities"]

    if len(retrained_gids) != expected_n:
        raise ValueError(
            f"{state}, {model}: expected {expected_n} municipalities in "
            f"retrained output, found {len(retrained_gids)}."
        )

    one_hit = one_hit[
        one_hit["GID_2"].astype(str).isin(retrained_gids)
    ].copy()

    one_hit_gids = set(one_hit["GID_2"].astype(str))
    missing_from_one_hit = retrained_gids - one_hit_gids
    if missing_from_one_hit:
        raise ValueError(
            f"{state}, {model}: {len(missing_from_one_hit)} retrained "
            "municipalities absent from BRA_full one-hit output."
        )

    full_mean = municipality_means(one_hit, "WIS_one_hit")
    retrain_mean = municipality_means(retrained, "WIS_retrained")

    paired = full_mean.merge(
        retrain_mean,
        on=["horizon", "GID_2"],
        how="inner",
        validate="one_to_one",
    )

    rows = []

    for horizon in HORIZONS:
        d = paired[paired["horizon"] == horizon].copy()

        d = d[
            np.isfinite(d["WIS_one_hit"])
            & np.isfinite(d["WIS_retrained"])
        ].copy()

        if len(d) == 0:
            raise ValueError(
                f"{state}, {model}, h{horizon}: no finite paired "
                "municipality means."
            )

        difference = d["WIS_one_hit"] - d["WIS_retrained"]

        test = wilcoxon(
            d["WIS_one_hit"],
            d["WIS_retrained"],
            alternative="two-sided",
        )

        rows.append(
            {
                "State": state,
                "Model": model,
                "Horizon": horizon,
                "n": len(d),
                "W statistic": float(test.statistic),
                "p-value": float(test.pvalue),
                "Median difference": float(np.median(difference)),
            }
        )

    return rows


def collect_failure_rows() -> pd.DataFrame:
    rows = []

    # São Paulo N-BEATS failure was quantified by the dedicated diagnostic:
    # all 645 municipalities had at least one non-finite WIS at every horizon.
    sp_nbeats_path = (
        STATE_CONFIG["São Paulo"]["path"] / "nbeats_h12_wis.csv"
    )

    if sp_nbeats_path.exists():
        df = read_wis(sp_nbeats_path)

        for horizon in HORIZONS:
            d = df[df["horizon"] == horizon].copy()
            w = d["WIS"].to_numpy(dtype=float)

            n_rows = len(d)
            n_munis = d["GID_2"].nunique()
            nan_rows = int(np.isnan(w).sum())
            posinf_rows = int(np.isposinf(w).sum())
            neginf_rows = int(np.isneginf(w).sum())
            nonfinite_rows = int((~np.isfinite(w)).sum())
            nonfinite_munis = int(
                d.loc[~np.isfinite(d["WIS"]), "GID_2"].nunique()
            )

            rows.append(
                {
                    "State": "São Paulo",
                    "Model": "N-BEATS",
                    "Horizon": horizon,
                    "status": "Numerical failure",
                    "municipalities": n_munis,
                    "rows": n_rows,
                    "nonfinite_WIS_rows": nonfinite_rows,
                    "NaN_rows": nan_rows,
                    "+inf_rows": posinf_rows,
                    "-inf_rows": neginf_rows,
                    "municipalities_affected": nonfinite_munis,
                }
            )

    # Explicitly record model/state combinations that were not assessed.
    rows.extend(
        [
            {
                "State": "São Paulo",
                "Model": "TCN",
                "Horizon": np.nan,
                "status": "Not assessed",
                "municipalities": np.nan,
                "rows": np.nan,
                "nonfinite_WIS_rows": np.nan,
                "NaN_rows": np.nan,
                "+inf_rows": np.nan,
                "-inf_rows": np.nan,
                "municipalities_affected": np.nan,
            },
            {
                "State": "São Paulo",
                "Model": "TimesFM",
                "Horizon": np.nan,
                "status": "Not assessed",
                "municipalities": np.nan,
                "rows": np.nan,
                "nonfinite_WIS_rows": np.nan,
                "NaN_rows": np.nan,
                "+inf_rows": np.nan,
                "-inf_rows": np.nan,
                "municipalities_affected": np.nan,
            },
        ]
    )

    return pd.DataFrame(rows)


def format_p(value):
    if pd.isna(value):
        return ""
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def main():
    print("=" * 100)
    print("TABLE S9 — IMPACT OF MONTHLY ITERATIVE MODEL RETRAINING")
    print("=" * 100)
    print(f"One-hit/original: {ONE_HIT}")
    print(f"Evaluation:       {START} to {END}")
    print()

    result_rows = []

    for state, config in STATE_CONFIG.items():
        print(f"{state}: {config['path']}")

        for model, stem in config["models"].items():
            retrained_file = config["path"] / f"{stem}_h12_wis.csv"

            if not retrained_file.exists():
                raise FileNotFoundError(
                    f"Expected retrained WIS file not found: {retrained_file}"
                )

            rows = analyse_state_model(
                state=state,
                config=config,
                model=model,
                stem=stem,
            )
            result_rows.extend(rows)

            print(
                f"  {model}: n by horizon = "
                f"{[r['n'] for r in rows]}"
            )

    out = pd.DataFrame(result_rows)

    state_order = ["Ceará", "São Paulo"]
    model_order = ["SARIMA", "TCN", "XGBoost"]

    out["State"] = pd.Categorical(
        out["State"], categories=state_order, ordered=True
    )
    out["Model"] = pd.Categorical(
        out["Model"], categories=model_order, ordered=True
    )
    out["Horizon"] = pd.Categorical(
        out["Horizon"], categories=HORIZONS, ordered=True
    )

    out = out.sort_values(
        ["State", "Model", "Horizon"]
    ).reset_index(drop=True)

    out["W statistic"] = out["W statistic"].round(1)
    out["Median difference"] = out["Median difference"].round(4)
    out["p-value"] = out["p-value"].map(format_p)

    out.to_csv(OUTPUT_CSV, index=False)
    out.to_csv(OUTPUT_TSV, index=False, sep="\t")

    failures = collect_failure_rows()
    failures.to_csv(FAILURE_CSV, index=False)

    print("\n" + "=" * 100)
    print("PERFORMANCE COMPARISONS")
    print("=" * 100)
    print(out.to_string(index=False))

    print("\n" + "=" * 100)
    print("MODEL AVAILABILITY / FAILURE RECORD")
    print("=" * 100)
    print(failures.to_string(index=False))

    print("\n")
    print(f"Wrote: {OUTPUT_CSV}")
    print(f"Wrote: {OUTPUT_TSV}")
    print(f"Wrote: {FAILURE_CSV}")


if __name__ == "__main__":
    main()
