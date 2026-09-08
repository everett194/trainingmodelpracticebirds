#!/usr/bin/env python3
"""
scripts/build_consolidated_database.py
--------------------------------------------
Consolidates every real, structured data source in this project into
ONE SQLite database (data/birdstrikegeo.db) -- plain files scattered
across data/raw, data/reference, data/processed, and reports/latest,
in four different formats (xlsx, PDF-derived, parquet, CSV), unified
into queryable tables with a self-documenting metadata table.

Why SQLite specifically: zero new dependencies (Python's stdlib sqlite3
+ pandas' built-in to_sql/read_sql), one portable file any tool or AI
agent can open with a single line of code or the `sqlite3` CLI, and no
server/setup required. See DATABASE.md for the full schema and example
queries.

This is a BUILD step, not a source of truth -- data/birdstrikegeo.db is
gitignored and fully regeneratable by re-running this script. It never
invents data: any source file that isn't present is skipped with a
clear message, never silently faked.

Usage:
    python scripts/build_consolidated_database.py
"""

from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from birdstrikegeo.data.ingest_trektellen_annual_pdf import (  # noqa: E402
    discover_annual_pdfs,
    parse_and_merge_annual_pdfs,
)
from birdstrikegeo.data.ingest_trektellen_crosstabs_xlsx import parse_crosstabs_xlsx, read_index  # noqa: E402

DB_PATH = REPO_ROOT / "data" / "birdstrikegeo.db"

_table_log: list[dict] = []


def _write_table(conn: sqlite3.Connection, name: str, df: pd.DataFrame, description: str, source: str) -> None:
    df.to_sql(name, conn, if_exists="replace", index=False)
    _table_log.append(
        {
            "table_name": name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": ", ".join(df.columns),
            "source": source,
            "description": description,
        }
    )
    print(f"[build_consolidated_database]   wrote '{name}': {len(df)} rows x {len(df.columns)} cols")


def _add_faa_ga(conn: sqlite3.Connection) -> None:
    path = REPO_ROOT / "data" / "processed" / "ga" / "ga_filtered.parquet"
    if not path.exists():
        print(f"[build_consolidated_database] SKIP faa_strikes_ga: {path} not found "
              f"(run scripts/ga_prepare_data.py first).")
        return
    df = pd.read_parquet(path)
    _write_table(
        conn, "faa_strikes_ga", df,
        description=(
            "One row per reported wildlife strike, confirmed business/private fixed-wing GA "
            "population only (29,522 rows as of the model receipt). Raw FAA columns plus "
            "damage_binary/damage_target_status/damage_level_inconsistent. This is the PRIMARY "
            "strike-level fact table -- see LEGACY_SYSTEMS.md for why the all-aircraft-population "
            "damage pipeline is not included here."
        ),
        source=str(path.relative_to(REPO_ROOT)),
    )


def _add_airports(conn: sqlite3.Connection) -> None:
    path = REPO_ROOT / "data" / "raw" / "airports.csv"
    if not path.exists():
        print(f"[build_consolidated_database] SKIP airports: {path} not found.")
        return
    df = pd.read_csv(path)
    _write_table(
        conn, "airports", df,
        description=(
            "OurAirports (public domain) worldwide airport point locations, RAW column names "
            "(ident, icao_code, iata_code, latitude_deg, longitude_deg, ...) -- NOT yet renamed to "
            "AIRPORT_SCHEMA; see DATA_DOWNLOAD_GUIDE.md for the exact mapping needed before this is "
            "usable by birdstrikegeo.data.ingest_airports.load_airports()."
        ),
        source=str(path.relative_to(REPO_ROOT)),
    )


def _add_trektellen_crosstabs(conn: sqlite3.Connection) -> None:
    xlsx_candidates = sorted((REPO_ROOT / "data" / "reference").glob("trektellen_fbbo_yeartotals_all_presets*.xlsx"))
    if not xlsx_candidates:
        print("[build_consolidated_database] SKIP trektellen_counts: no all-presets crosstabs xlsx found.")
        return
    xlsx_path = xlsx_candidates[0]

    counts = parse_crosstabs_xlsx(xlsx_path)
    _write_table(
        conn, "trektellen_counts", counts,
        description=(
            "AUTHORITATIVE Trektellen fact table (FBBO, site 3460): species x year (2017-2026) x "
            "period preset (37 presets: 12 months, 'All months', 2 named seasons, 11 rolling 3-month "
            "windows, 12 rolling 2-month windows) x metric (combined / newly_ringed / retraps). "
            "93,450 rows. Supersedes both trektellen_career_summary and "
            "trektellen_single_season_2025_legacy in resolution -- prefer this table for any new "
            "analysis. Years with count=0 in a monitored period ARE real zeros here (this workbook "
            "only includes years/periods the station actually reported); if a (species, year, period) "
            "combination is absent entirely, that means it was never reported, not that it was zero."
        ),
        source=str(xlsx_path.relative_to(REPO_ROOT)) if xlsx_path.is_relative_to(REPO_ROOT) else xlsx_path.name,
    )

    index_df = read_index(xlsx_path)
    _write_table(
        conn, "trektellen_period_index", index_df,
        description="One row per period preset (see trektellen_counts.period_code) with its combined "
                     "species-recorded and total counts across all years -- used to validate "
                     "trektellen_counts against (see tests/test_trektellen_crosstabs_xlsx.py).",
        source=str(xlsx_path.relative_to(REPO_ROOT)) if xlsx_path.is_relative_to(REPO_ROOT) else xlsx_path.name,
    )


def _add_trektellen_annual_summary(conn: sqlite3.Connection) -> None:
    reference_dir = REPO_ROOT / "data" / "reference"
    pdf_paths = discover_annual_pdfs(reference_dir)
    if not pdf_paths:
        print(f"[build_consolidated_database] SKIP trektellen_career_summary: no annual PDF found in {reference_dir}.")
        return
    long_df = parse_and_merge_annual_pdfs(pdf_paths)
    # Career-level fields (year_max_count/year_max_year, and the PDF's
    # own career newly_ringed/retraps totals) aren't in trektellen_counts
    # at all -- reparse the PDF's species-level summary rows directly
    # for those, rather than duplicating trektellen_counts' per-year
    # combined counts (which the xlsx table already covers, more richly).
    from birdstrikegeo.data.ingest_trektellen_annual_pdf import parse_annual_pdf

    summary_frames = []
    for pdf_path in pdf_paths:
        parsed = parse_annual_pdf(pdf_path)
        summary_frames.append(
            parsed["species_rows"][["species", "total", "average", "newly_ringed", "retraps",
                                     "year_max_count", "year_max_year", "source_file"]]
        )
    summary = pd.concat(summary_frames, ignore_index=True).drop_duplicates(subset=["species", "source_file"])

    _write_table(
        conn, "trektellen_career_summary", summary,
        description=(
            "Per-species CAREER totals (not broken out by year) from the annual banding-totals PDF(s): "
            "total, average, career newly_ringed, career retraps, and the single best year_max_count "
            "(with its year). These fields are NOT present in trektellen_counts and are the reason "
            "this table is kept alongside it, even though trektellen_counts' per-year combined counts "
            "supersede this table's own year columns."
        ),
        source=", ".join(p.name for p in pdf_paths),
    )
    _ = long_df  # per-year long form already superseded by trektellen_counts; kept unused intentionally


def _add_trektellen_legacy_season(conn: sqlite3.Connection) -> None:
    path = REPO_ROOT / "data" / "raw" / "trektellen_2025_year_totals.csv"
    if not path.exists():
        print(f"[build_consolidated_database] SKIP trektellen_single_season_2025_legacy: {path} not found.")
        return
    df = pd.read_csv(path)
    _write_table(
        conn, "trektellen_single_season_2025_legacy", df,
        description=(
            "LEGACY, lowest-resolution Trektellen source (single 2025 season, monthly, combined metric "
            "only) -- kept for provenance. Superseded by trektellen_counts (which includes 2025 at "
            "monthly grain across all 3 metrics, plus 9 other years) -- do not use this table for new "
            "analysis, see trektellen_counts instead."
        ),
        source=str(path.relative_to(REPO_ROOT)),
    )


def _add_reports(conn: sqlite3.Connection) -> None:
    reports_dir = REPO_ROOT / "reports" / "latest"

    species_risk_path = reports_dir / "national_species_risk.csv"
    if species_risk_path.exists():
        _write_table(
            conn, "national_species_risk", pd.read_csv(species_risk_path),
            description="FAA-wide (all GA strikes) per-species severity risk score "
                         "(mean_predicted_severity x log1p(n_strikes)) from the hazard severity model.",
            source=str(species_risk_path.relative_to(REPO_ROOT)),
        )

    model_comparison_path = reports_dir / "model_comparison.csv"
    if model_comparison_path.exists():
        _write_table(
            conn, "legacy_damage_model_comparison", pd.read_csv(model_comparison_path),
            description="SAMPLE-DATA-ONLY baseline comparison for the legacy PyTorch damage model "
                         "(dummy/logistic_regression/histogram_gradient_boosting/neural_net) -- see "
                         "LEGACY_SYSTEMS.md. Not real-world performance.",
            source=str(model_comparison_path.relative_to(REPO_ROOT)),
        )

    ga_summary_path = reports_dir / "ga_population_summary.json"
    if ga_summary_path.exists():
        data = json.loads(ga_summary_path.read_text())
        population_counts = data.pop("population_counts", {})
        target_report = data.pop("target_report", {})

        pop_df = pd.DataFrame(
            [{"population_name": name, **fields} for name, fields in population_counts.items()]
        )
        _write_table(
            conn, "ga_population_definitions", pop_df,
            description="GA population-filter definitions and counts (confirmed_business_or_private, "
                         "..._fixed_wing [the principal/training population], light_fixed_wing_ga, "
                         "light_piston_ga) -- see model_receipt.md.",
            source=str(ga_summary_path.relative_to(REPO_ROOT)),
        )

        summary_flat = {**data, **{f"target_{k}": v for k, v in target_report.items()}}
        _write_table(
            conn, "ga_population_summary", pd.DataFrame([summary_flat]),
            description="Top-level GA-population/target-construction summary: source file/hash, "
                         "principal population count, and damage-target class balance "
                         "(target_n_positive/target_n_negative/target_positive_class_rate).",
            source=str(ga_summary_path.relative_to(REPO_ROOT)),
        )

    severity_metrics_path = reports_dir / "severity_model_metrics.json"
    if severity_metrics_path.exists():
        data = json.loads(severity_metrics_path.read_text())
        _write_table(
            conn, "severity_model_metrics", pd.DataFrame([data]),
            description="Validation/test RMSE and R2 for the hazard severity regressor.",
            source=str(severity_metrics_path.relative_to(REPO_ROOT)),
        )


def run() -> None:
    print(f"[build_consolidated_database] Building {DB_PATH} ...")
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()  # full rebuild, not an incremental merge -- always reflects current source files exactly

    conn = sqlite3.connect(DB_PATH)
    try:
        _add_faa_ga(conn)
        _add_airports(conn)
        _add_trektellen_crosstabs(conn)
        _add_trektellen_annual_summary(conn)
        _add_trektellen_legacy_season(conn)
        _add_reports(conn)

        metadata_df = pd.DataFrame(_table_log)
        metadata_df["generated_at"] = datetime.now(timezone.utc).isoformat()
        metadata_df.to_sql("_table_metadata", conn, if_exists="replace", index=False)

        # Indexes on the columns most future joins/filters will use.
        cur = conn.cursor()
        for table, col in [
            ("faa_strikes_ga", "SPECIES"), ("faa_strikes_ga", "AIRPORT_ID"), ("faa_strikes_ga", "INCIDENT_YEAR"),
            ("trektellen_counts", "species"), ("trektellen_counts", "year"), ("trektellen_counts", "period_code"),
            ("national_species_risk", "species"), ("airports", "ident"),
        ]:
            try:
                cur.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON "{table}"("{col}")')
            except sqlite3.OperationalError:
                pass  # table wasn't written this run (source file missing) -- nothing to index
        conn.commit()
    finally:
        conn.close()

    n_tables = len(_table_log)
    total_rows = sum(t["row_count"] for t in _table_log)
    print(f"[build_consolidated_database] Done: {n_tables} tables, {total_rows} total rows, "
          f"{DB_PATH.stat().st_size / 1e6:.1f} MB -> {DB_PATH}")
    print("[build_consolidated_database] See DATABASE.md for schema and example queries, or run:")
    print(f"    sqlite3 {DB_PATH.relative_to(REPO_ROOT)} \".tables\"")


if __name__ == "__main__":
    run()
