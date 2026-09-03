#!/usr/bin/env python3
"""
scripts/build_eastern_shore_risk_report.py
-----------------------------------------------
Exploratory bird-strike risk analysis, step 2 of 2 (run
scripts/train_severity_model.py first). Loads the FBBO 2025 Trektellen
export, effort-normalizes it, joins it against the national per-species
risk scores that script produced, and writes the scatterplot + markdown
report to reports/latest/.

Usage:
    python scripts/build_eastern_shore_risk_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

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


def run() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    paths = config["paths"]

    species_risk_path = REPO_ROOT / paths["reports_dir"] / "national_species_risk.csv"
    if not species_risk_path.exists():
        print(f"[build_eastern_shore_risk_report] Missing {species_risk_path}. Run scripts/train_severity_model.py first.")
        sys.exit(1)
    species_risk = pd.read_csv(species_risk_path)

    trektellen_csv = REPO_ROOT / paths["trektellen_csv"]
    if not trektellen_csv.exists():
        print(f"[build_eastern_shore_risk_report] Missing {trektellen_csv}.")
        sys.exit(1)

    month_cols = config["trektellen"]["month_columns"]
    season_totals = load_season_totals(trektellen_csv)
    effort_hours = load_monthly_effort_hours(trektellen_csv)
    long_df = effort_normalize_monthly(season_totals, effort_hours, month_cols)
    long_df = expand_composite_species(long_df)
    local_activity = aggregate_season_local_activity(long_df)
    print(f"[build_eastern_shore_risk_report] {len(local_activity)} distinct local species "
          f"(after composite-name expansion), monitored hours={effort_hours[effort_hours > 0].sum():.1f}")

    merged = merge_local_risk(species_risk, local_activity)
    n_unmatched = int((~merged["matched_faa_species"]).sum())
    print(f"[build_eastern_shore_risk_report] {n_unmatched} / {len(merged)} local species had no FAA risk-score match")

    reports_dir = REPO_ROOT / paths["reports_dir"]
    scatter_path = reports_dir / "eastern_shore_risk_scatter.png"
    plot_local_risk_scatter(merged, scatter_path)

    metrics = pd.read_json(reports_dir / "severity_model_metrics.json", typ="series").to_dict()
    context = {
        "model_metrics": metrics,
        "national_top_species": species_risk.sort_values("risk_score", ascending=False).to_dict("records"),
        "local_top_species": merged[merged["matched_faa_species"]]
            .sort_values("local_risk_contribution", ascending=False).to_dict("records"),
        "scatter_image_relpath": scatter_path.name,
        "n_unmatched_local_species": n_unmatched,
    }
    report_path = reports_dir / "eastern_shore_birdstrike_risk_analysis.md"
    build_markdown_report(context, report_path)
    print(f"[build_eastern_shore_risk_report] Wrote {report_path} and {scatter_path}")


if __name__ == "__main__":
    run()
