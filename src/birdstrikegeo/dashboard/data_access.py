"""
birdstrikegeo.dashboard.data_access
----------------------------------------
Thin, deliberately untested-by-design glue: loads real rows from
data/birdstrikegeo.db, configs/eastern_shore_airports.yaml, and
data/raw/operations/*.csv into the plain-dict shapes
birdstrikegeo.dashboard's pure query/evidence/exposure/activity modules
expect. All the actual logic those modules implement IS unit-tested
directly on synthetic rows - this module's only job is the SQL/CSV/YAML
plumbing, which is exercised end-to-end by running the app itself.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import yaml

from birdstrikegeo.hazard.trektellen_season import (
    effort_normalize_monthly,
    expand_composite_species,
    load_monthly_effort_hours,
    load_season_totals,
)


def load_eastern_shore_airport_roster(config_path: str | Path) -> list[dict]:
    data = yaml.safe_load(Path(config_path).read_text())
    return data["airports"]


def _strike_row_query(where_clause: str = "") -> str:
    return f"""
        SELECT AIRPORT_ID as airport_id, AIRPORT as airport_name, INCIDENT_YEAR as incident_year,
               INCIDENT_MONTH as incident_month, INCIDENT_DATE as incident_date, SPECIES as species,
               INDICATED_DAMAGE as indicated_damage, DAMAGE_LEVEL as damage_level,
               PHASE_OF_FLIGHT as phase_of_flight, HEIGHT as height, AC_MASS as ac_mass, TYPE_ENG as type_eng
        FROM faa_strikes_ga
        {where_clause}
    """


def load_ga_strike_rows(db_path: str | Path, airport_ids: list[str] | None = None) -> list[dict]:
    """
    Returns every faa_strikes_ga row (already the confirmed business/
    private fixed-wing GA population), normalized to
    dashboard.query's expected row shape. airport_ids=None loads the
    whole national population (used for the national-context fallback);
    pass the Eastern Shore roster's FAA LIDs to scope to the region.
    """
    where = ""
    params: tuple = ()
    if airport_ids is not None:
        placeholders = ", ".join("?" for _ in airport_ids)
        where = f"WHERE AIRPORT_ID IN ({placeholders})"
        params = tuple(airport_ids)

    conn = sqlite3.connect(db_path)
    try:
        df = pd.read_sql(_strike_row_query(where), conn, params=params)
    finally:
        conn.close()

    df["indicated_damage"] = df["indicated_damage"].apply(
        lambda v: int(v) if v is not None and str(v) in ("0", "1") else None
    )
    return df.to_dict("records")


def load_trektellen_effort_rows(csv_path: str | Path, month_columns: list[str]) -> list[dict]:
    """
    One row per (species, month) with count/observation_hours, expanded
    so a composite Trektellen species label ("Alder/Willow Flycatcher")
    becomes one row per plain component species - see
    birdstrikegeo.hazard.trektellen_season/species_normalize.
    """
    season_totals = load_season_totals(csv_path)
    effort_hours = load_monthly_effort_hours(csv_path)
    long_df = effort_normalize_monthly(season_totals, effort_hours, month_columns)
    long_df = expand_composite_species(long_df)
    return long_df.to_dict("records")


def load_ops_rows(operations_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """
    Returns (atads_monthly_rows, faa_5010_annual_rows). An absent or
    header-only CSV simply yields an empty list - dashboard.exposure
    already treats "no matching rows" as "data unavailable," so there's
    no special-casing needed here beyond not crashing on a missing file.
    """
    operations_dir = Path(operations_dir)
    atads_path = operations_dir / "atads_monthly.csv"
    annual_path = operations_dir / "faa_5010_annual.csv"

    atads_rows = pd.read_csv(atads_path).to_dict("records") if atads_path.exists() else []
    annual_rows = pd.read_csv(annual_path).to_dict("records") if annual_path.exists() else []
    return atads_rows, annual_rows
