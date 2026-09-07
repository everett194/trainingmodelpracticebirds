#!/usr/bin/env python3
"""
scripts/build_eastern_shore_risk_report.py
-----------------------------------------------
Exploratory bird-strike risk analysis, step 2 of 2 (run
scripts/train_severity_model.py first). Joins per-species national risk
scores against local Trektellen/FBBO activity, and writes the
scatterplot + markdown report to reports/latest/.

Local-activity source (auto-selected, in this priority order):
  1. Multi-year ANNUAL data (data/reference/trektellen*annual*.pdf,
     parsed via birdstrikegeo.data.ingest_trektellen_annual_pdf) --
     preferred when present, since pooling across more monitored years
     gives a less noisy per-hour rate. Run
     scripts/build_trektellen_annual_dataset.py first (this script also
     calls it automatically if the processed dataset is stale/missing),
     so dropping a NEW annual PDF into data/reference/ and re-running
     this script picks it up with no code changes.
  2. Single-SEASON monthly data (configs/eastern_shore_risk.yaml's
     trektellen_csv, e.g. the 2025 year-totals export) -- used only when
     no annual PDF is found, preserving this script's original behavior.

Usage:
    python scripts/build_eastern_shore_risk_report.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from birdstrikegeo.data.ingest_trektellen_annual_pdf import discover_annual_pdfs  # noqa: E402
from birdstrikegeo.hazard.report import build_markdown_report, plot_local_risk_scatter  # noqa: E402
from birdstrikegeo.hazard.species_risk import merge_local_risk  # noqa: E402
from birdstrikegeo.hazard.trektellen_season import (  # noqa: E402
    aggregate_season_local_activity,
    effort_normalize_monthly,
    expand_composite_species,
    load_monthly_effort_hours,
    load_season_totals,
)

CONFIG_PATH = REPO_ROOT / "configs" / "eastern_shore_risk.yaml"
REFERENCE_DIR = REPO_ROOT / "data" / "reference"
ANNUAL_PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "trektellen"


def _load_annual_local_activity() -> tuple[pd.DataFrame, str] | None:
    """
    Returns (local_activity_df, description) from the multi-year annual
    dataset, or None if no annual-totals PDF is present in
    data/reference/ -- in which case the caller falls back to the
    single-season monthly path.
    """
    pdf_paths = discover_annual_pdfs(REFERENCE_DIR)
    if not pdf_paths:
        return None

    activity_path = ANNUAL_PROCESSED_DIR / "annual_local_activity.csv"
    long_path = ANNUAL_PROCESSED_DIR / "annual_long.parquet"
    newest_pdf_mtime = max(p.stat().st_mtime for p in pdf_paths)
    if not activity_path.exists() or activity_path.stat().st_mtime < newest_pdf_mtime:
        print("[build_eastern_shore_risk_report] Annual dataset missing or older than the source PDF(s) "
              "-- regenerating via scripts/build_trektellen_annual_dataset.py.")
        subprocess.run([sys.executable, str(REPO_ROOT / "scripts" / "build_trektellen_annual_dataset.py")], check=True)

    activity = pd.read_csv(activity_path)
    long_df = pd.read_parquet(long_path)
    years_monitored = sorted(long_df.loc[long_df["monitored"], "year"].unique())
    total_hours = long_df.loc[long_df["monitored"], ["year", "observation_hours"]].drop_duplicates()["observation_hours"].sum()
    sources = pdf_paths[0].name if len(pdf_paths) == 1 else f"{len(pdf_paths)} source files"
    description = (
        f"{years_monitored[0]}-{years_monitored[-1]} pooled ({len(years_monitored)} monitored years, "
        f"{total_hours:.0f} total observation hours, source: {sources})"
    )
    return activity, description


def run() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    paths = config["paths"]

    species_risk_path = REPO_ROOT / paths["reports_dir"] / "national_species_risk.csv"
    if not species_risk_path.exists():
        print(f"[build_eastern_shore_risk_report] Missing {species_risk_path}. Run scripts/train_severity_model.py first.")
        sys.exit(1)
    species_risk = pd.read_csv(species_risk_path)

    annual = _load_annual_local_activity()
    if annual is not None:
        local_activity, basis_description = annual
        activity_label = "FBBO " + basis_description.split(" pooled")[0]
        additional_years_note = (
            "This already pools multiple monitored years (see basis above); further station-years "
            "will continue to refine it, but the single-season noise concern from earlier versions "
            "of this report no longer applies at the same scale."
        )
        print(f"[build_eastern_shore_risk_report] Using multi-year annual local activity: {basis_description} "
              f"({len(local_activity)} species)")
    else:
        trektellen_csv = REPO_ROOT / paths["trektellen_csv"]
        if not trektellen_csv.exists():
            print(f"[build_eastern_shore_risk_report] Missing {trektellen_csv}, and no annual-totals PDF found "
                  f"in {REFERENCE_DIR} either.")
            sys.exit(1)

        month_cols = config["trektellen"]["month_columns"]
        season_totals = load_season_totals(trektellen_csv)
        effort_hours = load_monthly_effort_hours(trektellen_csv)
        long_df = effort_normalize_monthly(season_totals, effort_hours, month_cols)
        long_df = expand_composite_species(long_df)
        local_activity = aggregate_season_local_activity(long_df)
        basis_description = f"{config['trektellen']['season_year']} season only (single station-season)"
        activity_label = f"FBBO {config['trektellen']['season_year']}"
        additional_years_note = "More station-years are expected to refine this analysis."
        print(f"[build_eastern_shore_risk_report] No annual-totals PDF found; using single-season monthly "
              f"local activity: {len(local_activity)} distinct local species (after composite-name expansion), "
              f"monitored hours={effort_hours[effort_hours > 0].sum():.1f}")

    merged = merge_local_risk(species_risk, local_activity)
    n_unmatched = int((~merged["matched_faa_species"]).sum())
    print(f"[build_eastern_shore_risk_report] {n_unmatched} / {len(merged)} local species had no FAA risk-score match")

    reports_dir = REPO_ROOT / paths["reports_dir"]
    scatter_path = reports_dir / "eastern_shore_risk_scatter.png"
    plot_local_risk_scatter(merged, scatter_path, activity_label=activity_label)

    metrics = pd.read_json(reports_dir / "severity_model_metrics.json", typ="series").to_dict()
    context = {
        "model_metrics": metrics,
        "national_top_species": species_risk.sort_values("risk_score", ascending=False).to_dict("records"),
        "local_top_species": merged[merged["matched_faa_species"]]
            .sort_values("local_risk_contribution", ascending=False).to_dict("records"),
        "scatter_image_relpath": scatter_path.name,
        "n_unmatched_local_species": n_unmatched,
        "local_activity_basis_description": basis_description,
        "additional_station_years_note": additional_years_note,
    }
    report_path = reports_dir / "eastern_shore_birdstrike_risk_analysis.md"
    build_markdown_report(context, report_path)
    print(f"[build_eastern_shore_risk_report] Wrote {report_path} and {scatter_path}")


if __name__ == "__main__":
    run()
