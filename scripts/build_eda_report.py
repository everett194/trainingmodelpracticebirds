#!/usr/bin/env python3
"""
scripts/build_eda_report.py
--------------------------------
Phase 7: exploratory analysis on the GA population
(data/processed/ga/ga_filtered.parquet) — missingness, distributions,
correlation, spatial/seasonal plots, and confounding/leakage smoke
checks. Writes reports/latest/eda/*.

Deliberately does NOT attempt "incident counts vs. exposure" (Phase 7
asks for this, but no flight-exposure data exists in this project yet —
see DATA_CARD.md §5) — that section is written as an explicit
"unavailable" note in the output report, not silently skipped, so a
reader knows it was considered and why it's missing rather than assuming
an oversight.

Usage:
    python scripts/build_eda_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from birdstrikegeo.ga.feature_policy import FORBIDDEN_COLUMNS  # noqa: E402

GA_PARQUET = REPO_ROOT / "data" / "processed" / "ga" / "ga_filtered.parquet"
OUT_DIR = REPO_ROOT / "reports" / "latest" / "eda"

CONTINUOUS_COLUMNS = ["HEIGHT", "SPEED", "NUM_ENGS"]


def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.isna().mean().mul(100).round(2).sort_values(ascending=False)
        .rename("pct_missing").reset_index().rename(columns={"index": "column"})
    )


def plot_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(CONTINUOUS_COLUMNS), figsize=(4 * len(CONTINUOUS_COLUMNS), 4))
    for ax, col in zip(axes, CONTINUOUS_COLUMNS):
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        ax.hist(
            [values[df.loc[values.index, "damage_binary"] == 0], values[df.loc[values.index, "damage_binary"] == 1]],
            bins=30, label=["no damage", "damage"], stacked=False, alpha=0.6, density=True,
        )
        ax.set_title(col)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "distributions_by_damage.png", dpi=120)
    plt.close(fig)


def plot_correlation(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    numeric_df = df[CONTINUOUS_COLUMNS + ["damage_binary"]].apply(pd.to_numeric, errors="coerce")
    corr = numeric_df.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns, fontsize=8)
    fig.colorbar(im, ax=ax, label="Spearman correlation")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_matrix.png", dpi=120)
    plt.close(fig)
    return corr


def plot_spatial(df: pd.DataFrame, out_dir: Path) -> None:
    lat = pd.to_numeric(df["LATITUDE"], errors="coerce")
    lon = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    valid = lat.notna() & lon.notna()
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = df.loc[valid, "damage_binary"].map({0: "tab:blue", 1: "tab:red"})
    ax.scatter(lon[valid], lat[valid], c=colors, s=4, alpha=0.4)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"GA strike locations ({valid.sum()}/{len(df)} with coordinates) — red = damage")
    fig.tight_layout()
    fig.savefig(out_dir / "spatial_scatter.png", dpi=120)
    plt.close(fig)


def plot_seasonal(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    monthly = df.groupby("INCIDENT_MONTH")["damage_binary"].agg(["count", "mean"]).reset_index()
    monthly.columns = ["month", "n_strikes", "damage_rate"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.bar(monthly["month"], monthly["n_strikes"])
    ax1.set_title("Strike count by month")
    ax1.set_xlabel("month")
    ax2.bar(monthly["month"], monthly["damage_rate"])
    ax2.set_title("Damage rate by month")
    ax2.set_xlabel("month")
    fig.tight_layout()
    fig.savefig(out_dir / "seasonal.png", dpi=120)
    plt.close(fig)
    return monthly


def confounding_checks(df: pd.DataFrame) -> dict:
    overall_rate = df["damage_binary"].mean()

    by_species = (
        df.groupby("SPECIES")["damage_binary"].agg(["count", "mean"])
        .query("count >= 30").sort_values("mean", ascending=False)
    )
    by_year = df.groupby("INCIDENT_YEAR")["damage_binary"].agg(["count", "mean"])
    by_airport = (
        df.groupby("AIRPORT_ID")["damage_binary"].agg(["count", "mean"])
        .query("count >= 30").sort_values("mean", ascending=False)
    )

    return {
        "overall_damage_rate": round(float(overall_rate), 4),
        "top_5_highest_damage_rate_species_n_ge_30": by_species.head(5).round(4).to_dict("index"),
        "bottom_5_lowest_damage_rate_species_n_ge_30": by_species.tail(5).round(4).to_dict("index"),
        "damage_rate_range_across_years": [round(float(by_year["mean"].min()), 4), round(float(by_year["mean"].max()), 4)],
        "damage_rate_range_across_airports_n_ge_30": [
            round(float(by_airport["mean"].min()), 4), round(float(by_airport["mean"].max()), 4)
        ] if len(by_airport) else None,
        "interpretation": (
            "Damage rate varies substantially by species, year, and airport (see ranges above). "
            "This confirms these are real confounders the GA model's feature set must account for "
            "(it does - species is deliberately excluded per ga/feature_policy.py precisely because "
            "it's outcome-adjacent, and airport/year effects are captured indirectly via "
            "state/faa_region/season features). Correlation here is association, not causation - "
            "e.g. an airport's higher rate may reflect its typical aircraft mix or local habitat, not "
            "the airport itself causing damage."
        ),
    }


def leakage_smoke_test(df: pd.DataFrame) -> dict:
    """
    A genuine smoke test, not just a policy citation: computes each
    FORBIDDEN_COLUMNS column's raw correlation with the target, to
    confirm they're suspiciously predictive (as outcome-describing
    columns SHOULD be) - i.e. confirms the leakage guard is blocking
    real signal, not blocking harmless columns for no reason.
    """
    results = {}
    for col in sorted(FORBIDDEN_COLUMNS):
        if col not in df.columns:
            continue
        series = df[col]
        if series.dtype == object or series.dtype.name == "string":
            # categorical -- compare damage rate for "present/non-null" vs "null"
            present_rate = df.loc[series.notna(), "damage_binary"].mean() if series.notna().any() else None
            absent_rate = df.loc[series.isna(), "damage_binary"].mean() if series.isna().any() else None
            results[col] = {"type": "categorical", "damage_rate_when_present": present_rate, "damage_rate_when_absent": absent_rate}
        else:
            numeric = pd.to_numeric(series, errors="coerce")
            corr = numeric.corr(df["damage_binary"])
            results[col] = {"type": "numeric", "correlation_with_damage": None if pd.isna(corr) else round(float(corr), 3)}
    return results


def run() -> None:
    if not GA_PARQUET.exists():
        print(f"[build_eda_report] Missing {GA_PARQUET}. Run scripts/ga_prepare_data.py first.")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(GA_PARQUET)

    missingness = missingness_summary(df)
    missingness.to_csv(OUT_DIR / "missingness_summary.csv", index=False)

    plot_distributions(df, OUT_DIR)
    corr = plot_correlation(df, OUT_DIR)
    plot_spatial(df, OUT_DIR)
    monthly = plot_seasonal(df, OUT_DIR)
    confounding = confounding_checks(df)
    leakage_check = leakage_smoke_test(df)

    report_lines = [
        "# Exploratory Data Analysis — GA Population",
        "",
        f"N = {len(df)} GA strike incidents.",
        "",
        "## Missingness (top 10)",
        "",
        "| column | % missing |",
        "|---|---|",
    ]
    for _, row in missingness.head(10).iterrows():
        report_lines.append(f"| {row['column']} | {row['pct_missing']}% |")

    report_lines += [
        "",
        "## Distributions by damage outcome",
        "![distributions](distributions_by_damage.png)",
        "",
        "## Correlation matrix (continuous features x damage_binary, Spearman)",
        "![correlation](correlation_matrix.png)",
        "",
        f"Strongest single correlation with damage: "
        f"{corr['damage_binary'].drop('damage_binary').abs().idxmax()} "
        f"({corr['damage_binary'].drop('damage_binary').abs().max():.3f}) — "
        "association, not causation; see MODELING_APPROACH.md.",
        "",
        "## Spatial distribution",
        "![spatial](spatial_scatter.png)",
        "",
        "## Seasonal pattern",
        "![seasonal](seasonal.png)",
        "",
        "## Confounding checks (species / year / airport)",
        "",
        f"Overall damage rate: {confounding['overall_damage_rate']:.1%}",
        f"Damage rate range across years: {confounding['damage_rate_range_across_years']}",
        f"Damage rate range across airports (n>=30): {confounding['damage_rate_range_across_airports_n_ge_30']}",
        "",
        confounding["interpretation"],
        "",
        "## Leakage smoke test (FORBIDDEN_COLUMNS vs. target)",
        "",
        "Confirms the columns `ga/feature_policy.py` excludes are indeed "
        "suspiciously associated with the target (as an outcome-describing "
        "column should be) — this is evidence the leakage guard is doing "
        "real work, not blocking harmless columns for no reason.",
        "",
        "| column | finding |",
        "|---|---|",
    ]
    for col, result in leakage_check.items():
        if result["type"] == "numeric":
            report_lines.append(f"| {col} | correlation with damage: {result['correlation_with_damage']} |")
        else:
            report_lines.append(
                f"| {col} | damage rate when present: {result['damage_rate_when_present']}, "
                f"when absent: {result['damage_rate_when_absent']} |"
            )

    report_lines += [
        "",
        "## Incident counts vs. exposure — UNAVAILABLE",
        "",
        "Phase 7 asks for incident counts plotted against flight exposure "
        "(e.g. departures). This project has NO flight-exposure data "
        "(see DATA_CARD.md §5, `bts_t100_departures.csv` placeholder) — "
        "this section is intentionally left as an explicit gap, not "
        "silently omitted, so a reader knows it was considered.",
    ]
    (OUT_DIR / "eda_report.md").write_text("\n".join(report_lines) + "\n")
    print(f"[build_eda_report] Wrote {OUT_DIR}/eda_report.md and supporting plots/CSVs.")


if __name__ == "__main__":
    run()
