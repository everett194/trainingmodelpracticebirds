#!/usr/bin/env python3
"""
scripts/build_join_quality_report.py
-----------------------------------------
Phase 5: builds the spatial-temporal join between every GA strike
incident and the one real bird-activity source this project has
(Trektellen FBBO, 2025 season), with full provenance per incident, and
writes a join-quality report showing what percentage of strikes can be
matched at different spatial/temporal tolerances.

Usage:
    python scripts/build_join_quality_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from birdstrikegeo.hazard.spatiotemporal_join import (  # noqa: E402
    build_join_quality_report,
    join_incidents_to_trektellen,
)
from birdstrikegeo.hazard.trektellen_season import (  # noqa: E402
    effort_normalize_monthly,
    load_monthly_effort_hours,
    load_season_totals,
)

CONFIG_PATH = REPO_ROOT / "configs" / "eastern_shore_risk.yaml"

# FBBO (Foreman's Branch Bird Observatory) is publicly described as
# "3 miles northeast of Chestertown, MD" -- no exact surveyed coordinate
# was found during this project's research. This is a documented
# APPROXIMATION (see DATA_AUDIT_REPORT.md section 8 "Assumptions that
# cannot yet be verified"), not a confirmed precise point. Update this
# if/when an authoritative coordinate is obtained.
FBBO_LAT_APPROX = 39.248
FBBO_LON_APPROX = -76.026
FBBO_COORDINATE_PRECISION = "approximate -- derived from a public description (3mi NE of Chestertown, MD), not a surveyed point"


def run() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    paths = config["paths"]
    trektellen_cfg = config["trektellen"]

    ga_path = REPO_ROOT / paths["ga_filtered_parquet"]
    trektellen_csv = REPO_ROOT / paths["trektellen_csv"]
    if not ga_path.exists() or not trektellen_csv.exists():
        print(f"[build_join_quality_report] Missing {ga_path} or {trektellen_csv}. "
              f"Run scripts/ga_prepare_data.py first.")
        sys.exit(1)

    incidents = pd.read_parquet(ga_path)
    n_incidents_total = len(incidents)

    season_totals = load_season_totals(trektellen_csv)
    month_cols = trektellen_cfg["month_columns"]
    effort_hours = load_monthly_effort_hours(trektellen_csv)
    long_df = effort_normalize_monthly(season_totals, effort_hours, month_cols)

    joined = join_incidents_to_trektellen(
        incidents,
        trektellen_long_df=long_df,
        trektellen_station_lat=FBBO_LAT_APPROX,
        trektellen_station_lon=FBBO_LON_APPROX,
        trektellen_season_year=trektellen_cfg["season_year"],
    )

    report = build_join_quality_report(joined, n_incidents_total=n_incidents_total)
    report["fbbo_coordinate_precision"] = FBBO_COORDINATE_PRECISION

    reports_dir = REPO_ROOT / paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "join_quality_report.json").write_text(json.dumps(report, indent=2))

    lines = [
        "# Join Quality Report — GA Strikes × Trektellen (FBBO, 2025)",
        "",
        f"**{report['n_incidents_total']}** GA strike incidents total, "
        f"**{report['n_dropped_missing_coordinates']}** dropped for missing coordinates before joining.",
        "",
        f"FBBO coordinate: {FBBO_LAT_APPROX}, {FBBO_LON_APPROX} ({FBBO_COORDINATE_PRECISION}).",
        "",
        f"**{report['pct_fully_observed']}%** of all incidents got a fully observed "
        "(species matched AND that calendar month was monitored) local bird-activity value.",
        "",
        "## Match rate by spatial tolerance (distance from incident to FBBO)",
        "",
        "| Tolerance | Matched | % of all incidents |",
        "|---|---|---|",
    ]
    for k, v in report["match_rate_by_spatial_tolerance"].items():
        lines.append(f"| {k} | {v['n_matched']} | {v['pct_of_all_incidents']}% |")
    lines += [
        "",
        "## Match rate by temporal tolerance (years between incident and 2025 season)",
        "",
        "| Tolerance | Matched | % of all incidents |",
        "|---|---|---|",
    ]
    for k, v in report["match_rate_by_temporal_tolerance"].items():
        lines.append(f"| {k} | {v['n_matched']} | {v['pct_of_all_incidents']}% |")
    lines += ["", f"**Caveat:** {report['caveat']}"]

    (reports_dir / "join_quality_report.md").write_text("\n".join(lines) + "\n")
    print(f"[build_join_quality_report] Wrote {reports_dir / 'join_quality_report.md'} "
          f"and .json — {report['pct_fully_observed']}% fully observed.")


if __name__ == "__main__":
    run()
