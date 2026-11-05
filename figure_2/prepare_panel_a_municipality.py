#!/usr/bin/env python3
"""Prepare municipality-level Figure 2A and C inputs with actual predictive quantiles.

Selection is intentionally easy to refine: set EXAMPLE_GID below. If it is None,
the script selects the municipality with the largest cumulative reported cases
over the 2020-01 to 2025-08 evaluation period and prints the selected GID.

Outputs
-------
data/figure2_example_municipality_observed.csv
data/figure2_example_municipality_forecasts.csv
data/figure2_example_municipality_pdfm_forecasts.csv
"""
from pathlib import Path
import pandas as pd
from thucia.core.cases import read_db

HERE = Path(__file__).resolve().parent
BRA_FULL = (HERE.parent.parent / "data" / "PDFM_analysis" / "BRA_full").resolve()
OUT = HERE / "data"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = {
    "TCN": "tcn_h12",
    "XGBoost": "xgboost_h12",
    "TimesFM": "timesfm_h12",
}
EXAMPLES = [("TCN", 1), ("XGBoost", 6), ("TimesFM", 1)]
PDFM_MODEL = "XGBoost"
PDFM_HORIZON = 6
PDFM_FILESTEM = "xgboost_h12_pdfmrr_pinball"
START = pd.Period("2020-01", freq="M")
END = pd.Period("2025-08", freq="M")

# Set to a GID_2 string to force a municipality. Leave None for automatic selection.
EXAMPLE_GID = None


def to_period_month(series):
    if isinstance(series.dtype, pd.PeriodDtype):
        return series.astype("period[M]")
    return pd.to_datetime(series).dt.to_period("M")


def _choose_gid(cases, index=0):
    if EXAMPLE_GID is not None:
        if EXAMPLE_GID not in set(cases["GID_2"].astype(str)):
            raise ValueError(f"EXAMPLE_GID {EXAMPLE_GID!r} not found in case data")
        return EXAMPLE_GID
    d = cases[(cases["Date"] >= START) & (cases["Date"] <= END)].copy()
    totals = d.groupby("GID_2", observed=True)["Cases"].sum().sort_values(ascending=False)
    if totals.empty:
        raise ValueError("No municipality case data in evaluation period")
    return str(totals.index[index])


def choose_gid(cases):
    """
    Select an illustrative municipality for Panel C with sustained
    TCN 1-month PDFM improvement among relatively high-burden municipalities.

    A may use a separately specified EXAMPLE_GID; this function is intended
    only for identifying a good PDFM-improvement example.
    """
    if EXAMPLE_GID is not None:
        if EXAMPLE_GID not in set(cases["GID_2"].astype(str)):
            raise ValueError(
                f"EXAMPLE_GID {EXAMPLE_GID!r} not found in case data"
            )
        return EXAMPLE_GID

    # ------------------------------------------------------------------
    # Load paired TCN baseline and PDFM WIS
    # ------------------------------------------------------------------
    baseline_path = BRA_FULL / "tcn_h12_wis.csv"
    pdfm_path = BRA_FULL / "tcn_h12_pdfmrr_wis.csv"

    if not baseline_path.exists():
        raise FileNotFoundError(f"Missing baseline WIS: {baseline_path}")
    if not pdfm_path.exists():
        raise FileNotFoundError(f"Missing PDFM WIS: {pdfm_path}")

    base = pd.read_csv(baseline_path)
    pdfm = pd.read_csv(pdfm_path)

    base["Date"] = to_period_month(base["Date"])
    pdfm["Date"] = to_period_month(pdfm["Date"])

    # We want the 1-month TCN comparison for Panel C.
    if "horizon" in base.columns:
        base = base[base["horizon"].astype(int) == 1]
    if "horizon" in pdfm.columns:
        pdfm = pdfm[pdfm["horizon"].astype(int) == 1]

    base = base[
        (base["Date"] >= START)
        & (base["Date"] <= END)
    ].copy()

    pdfm = pdfm[
        (pdfm["Date"] >= START)
        & (pdfm["Date"] <= END)
    ].copy()

    def find_wis_column(df, label):
        # Be explicit about what we accept rather than guessing silently.
        candidates = [c for c in ["WIS", "wis", "score"] if c in df.columns]
        if len(candidates) != 1:
            raise ValueError(
                f"{label}: expected exactly one WIS column among "
                f"WIS/wis/score; found {candidates}. "
                f"Available columns: {list(df.columns)}"
            )
        return candidates[0]

    base_col = find_wis_column(base, "baseline WIS")
    pdfm_col = find_wis_column(pdfm, "PDFM WIS")

    base = base[
        ["GID_2", "Date", base_col]
    ].rename(columns={base_col: "baseline_wis"})

    pdfm = pdfm[
        ["GID_2", "Date", pdfm_col]
    ].rename(columns={pdfm_col: "pdfm_wis"})

    paired = base.merge(
        pdfm,
        on=["GID_2", "Date"],
        how="inner",
        validate="one_to_one",
    )

    if paired.empty:
        raise ValueError("No paired TCN baseline/PDFM WIS observations found")

    # Negative = PDFM improves WIS.
    paired["delta_wis"] = (
        paired["pdfm_wis"] - paired["baseline_wis"]
    )

    # ------------------------------------------------------------------
    # Observed burden/activity
    # ------------------------------------------------------------------
    eval_cases = cases[
        (cases["Date"] >= START)
        & (cases["Date"] <= END)
    ].copy()

    burden = (
        eval_cases.groupby("GID_2", observed=True)
        .agg(
            cumulative_cases=("Cases", "sum"),
            nonzero_months=("Cases", lambda x: (x > 0).sum()),
            peak_cases=("Cases", "max"),
        )
    )

    # ------------------------------------------------------------------
    # Summarise PDFM effect by municipality
    # ------------------------------------------------------------------
    summary = (
        paired.groupby("GID_2", observed=True)
        .agg(
            mean_delta_wis=("delta_wis", "mean"),
            median_delta_wis=("delta_wis", "median"),
            improved_months=("delta_wis", lambda x: (x < 0).sum()),
            worsened_months=("delta_wis", lambda x: (x > 0).sum()),
            n_months=("delta_wis", "count"),
        )
    )

    summary["fraction_improved"] = (
        summary["improved_months"] / summary["n_months"]
    )

    summary = summary.join(burden, how="inner")

    # ------------------------------------------------------------------
    # Restrict to relatively high-burden municipalities.
    #
    # This avoids selecting sparse municipalities where a few isolated
    # events dominate the mean WIS improvement.
    # ------------------------------------------------------------------
    burden_cutoff = summary["cumulative_cases"].quantile(0.75)

    eligible = summary[
        (summary["cumulative_cases"] >= burden_cutoff)
        & (summary["nonzero_months"] >= 36)
        & (summary["n_months"] >= 48)
        & (summary["fraction_improved"] >= 0.50)
        & (summary["mean_delta_wis"] < 0)
    ].copy()

    if eligible.empty:
        raise ValueError(
            "No municipalities satisfy the high-burden/sustained-improvement "
            "criteria."
        )

    # ------------------------------------------------------------------
    # Rank primarily by consistency of improvement, then magnitude.
    # This avoids selecting a municipality because of one extreme month.
    # ------------------------------------------------------------------
    eligible = eligible.sort_values(
        [
            "fraction_improved",
            "mean_delta_wis",
            "cumulative_cases",
        ],
        ascending=[False, True, False],
    )

    # Print candidates rather than hiding the selection.
    print("\nTop PDFM-improvement candidates for Panel C:")
    print(
        eligible[
            [
                "mean_delta_wis",
                "median_delta_wis",
                "fraction_improved",
                "improved_months",
                "worsened_months",
                "n_months",
                "cumulative_cases",
                "peak_cases",
                "nonzero_months",
            ]
        ]
        .head(15)
        .to_string()
    )

    selected = str(eligible.index[0])

    print(f"\nSelected Panel C municipality: {selected}")

    return selected


def main():
    cases = read_db(str(BRA_FULL / "cases_with_climate")).df.copy()
    required = {"GID_2", "Date", "Cases"}
    missing = required - set(cases.columns)
    if missing:
        raise ValueError(f"cases_with_climate missing columns: {sorted(missing)}")
    cases["GID_2"] = cases["GID_2"].astype(str)
    cases["Date"] = to_period_month(cases["Date"])

    gid = _choose_gid(cases, index=0)
    obs = cases[
        (cases["GID_2"] == gid)
        & (cases["Date"] >= START)
        & (cases["Date"] <= END)
    ][["GID_2", "Date", "Cases"]].copy()
    obs = obs.groupby(["GID_2", "Date"], as_index=False)["Cases"].sum()
    obs["Date"] = obs["Date"].dt.to_timestamp("M")
    obs.to_csv(OUT / "figure2_example_municipality_observed.csv", index=False)

    rows = []
    for model, horizon in EXAMPLES:
        path = BRA_FULL / f"{MODELS[model]}_cases_quantiles"
        q = read_db(str(path)).df.copy()
        required = {"GID_2", "Date", "horizon", "quantile", "prediction"}
        missing = required - set(q.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        q["GID_2"] = q["GID_2"].astype(str)
        q["Date"] = to_period_month(q["Date"])
        d = q[
            (q["GID_2"] == gid)
            & (q["horizon"].astype(int) == horizon)
            & (q["quantile"].isin([0.1, 0.5, 0.9]))
            & (q["Date"] >= START)
            & (q["Date"] <= END)
        ][["Date", "quantile", "prediction"]].copy()
        if d.empty:
            raise ValueError(f"No {model} horizon={horizon} forecasts for {gid}")
        wide = d.pivot_table(index="Date", columns="quantile", values="prediction", aggfunc="first")
        missing_q = [x for x in [0.1, 0.5, 0.9] if x not in wide.columns]
        if missing_q:
            raise ValueError(f"{model} horizon={horizon}: missing quantiles {missing_q}")
        wide = wide.rename(columns={0.1: "q10", 0.5: "q50", 0.9: "q90"}).reset_index()
        wide["Date"] = wide["Date"].dt.to_timestamp("M")
        wide["GID_2"] = gid
        wide["model"] = model
        wide["horizon"] = horizon
        rows.append(wide[["GID_2", "model", "horizon", "Date", "q10", "q50", "q90"]])

    fc = pd.concat(rows, ignore_index=True)
    fc.to_csv(OUT / "figure2_example_municipality_forecasts.csv", index=False)

    pdfm_path = BRA_FULL / f"{PDFM_FILESTEM}_cases_quantiles"
    q = read_db(str(pdfm_path)).df.copy()
    required = {"GID_2", "Date", "horizon", "quantile", "prediction"}
    missing = required - set(q.columns)
    if missing:
        raise ValueError(f"{pdfm_path.name} missing columns: {sorted(missing)}")
    q["GID_2"] = q["GID_2"].astype(str)
    q["Date"] = to_period_month(q["Date"])
    d = q[(q["GID_2"] == gid) & (q["horizon"].astype(int) == PDFM_HORIZON) & (q["quantile"].isin([0.1, 0.5, 0.9])) & (q["Date"] >= START) & (q["Date"] <= END)][["Date", "quantile", "prediction"]].copy()
    if d.empty:
        raise ValueError(f"No PDFM-adjusted {PDFM_MODEL} horizon={PDFM_HORIZON} forecasts for {gid}")
    wide = d.pivot_table(index="Date", columns="quantile", values="prediction", aggfunc="first")
    missing_q = [x for x in [0.1, 0.5, 0.9] if x not in wide.columns]
    if missing_q:
        raise ValueError(f"PDFM-adjusted {PDFM_MODEL} horizon={PDFM_HORIZON}: missing quantiles {missing_q}")
    wide = wide.rename(columns={0.1: "q10", 0.5: "q50", 0.9: "q90"}).reset_index()
    wide["Date"] = wide["Date"].dt.to_timestamp("M")
    wide["GID_2"] = gid
    wide["model"] = PDFM_MODEL
    wide["horizon"] = PDFM_HORIZON
    wide.to_csv(OUT / "figure2_example_municipality_pdfm_forecasts.csv", index=False)
    print(f"Selected municipality: {gid}")
    print(f"Wrote {OUT / 'figure2_example_municipality_observed.csv'} ({len(obs):,} rows)")
    print(f"Wrote {OUT / 'figure2_example_municipality_forecasts.csv'} ({len(fc):,} rows)")
    print(f"Wrote {OUT / 'figure2_example_municipality_pdfm_forecasts.csv'} ({len(wide):,} rows)")


if __name__ == "__main__":
    main()
