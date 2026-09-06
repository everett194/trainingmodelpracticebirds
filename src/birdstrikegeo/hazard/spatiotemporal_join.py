"""
birdstrikegeo.hazard.spatiotemporal_join
--------------------------------------------
Phase 5: for each FAA strike incident, attaches the nearest available
Trektellen bird-activity estimate for that incident's species and month
— WITH the join's provenance recorded explicitly on every row, per
project requirement: never silently assign an external bird estimate to
an incident.

This is deliberately built on top of the existing, tested
`hazard.trektellen_season` functions rather than re-deriving Trektellen
parsing — it adds the INCIDENT side of the join (which
`hazard.species_risk.merge_local_risk` does not do: that module joins
species-to-species, at the season-pooled grain, not incident-to-source
with per-row distance/time tracking).

Today there is exactly ONE real bird-activity source (Trektellen, FBBO,
2025 season) and ONE real strike-incident source (the FAA GA
population), so this join is necessarily single-source. The schema is
written so a second source (e.g. eBird Status & Trends, see
geo/ebird_adapter.py) could be added as another row per incident later,
without changing this module's shape — see `build_join_quality_report`
for exactly what "match rate" would need re-running once that happens.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from birdstrikegeo.geo.crs import geodesic_distance_km

_MONTH_NUMBER_TO_COLUMN = {
    1: "jan", 2: "feb", 3: "mar", 4: "apr", 5: "may", 6: "jun",
    7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov", 12: "dec",
}


@dataclass(frozen=True)
class BirdContextJoinRecord:
    """One incident's join result — every field here is the provenance
    this project requires before any bird estimate reaches a model or
    report: source, resolution, join method, distance, time gap, and
    whether the value was actually observed vs. simply absent."""

    record_id: object
    source_dataset: str
    spatial_resolution: str
    temporal_resolution: str
    join_method: str
    distance_km: float
    time_diff_years: float
    species_matched: bool
    month_monitored: bool
    value_type: str  # "observed" | "unavailable"
    local_activity_count_per_hour: float | None


def _build_species_month_lookup(trektellen_long_df: pd.DataFrame) -> dict[tuple[str, str], float | None]:
    """
    (species_upper, month_column) -> count_per_hour (or None if that
    month had zero observation hours). Built ONCE per report build, so
    join_incidents_to_trektellen does an O(1) dict lookup per incident
    instead of a pandas boolean-mask filter per row (the latter is
    correct but scales badly across ~30k GA incidents against even a
    ~1,000-row long-format Trektellen table).
    """
    lookup: dict[tuple[str, str], float | None] = {}
    for _, row in trektellen_long_df.iterrows():
        key = (str(row["species"]).strip().upper(), row["month"])
        lookup[key] = float(row["count_per_hour"]) if pd.notna(row["count_per_hour"]) else None
    return lookup


def join_incident_to_trektellen(
    incident_lat: float,
    incident_lon: float,
    incident_year: int,
    incident_month: int,
    incident_species: str | None,
    record_id,
    species_month_lookup: dict[tuple[str, str], float | None],
    trektellen_station_lat: float,
    trektellen_station_lon: float,
    trektellen_season_year: int,
) -> BirdContextJoinRecord:
    """
    species_month_lookup: from _build_species_month_lookup() — a
    (species_upper, month_column) -> count_per_hour|None dict, built
    once from the long-format Trektellen table (see
    hazard.trektellen_season.effort_normalize_monthly).
    """
    distance_km = geodesic_distance_km(incident_lon, incident_lat, trektellen_station_lon, trektellen_station_lat)
    time_diff_years = abs(incident_year - trektellen_season_year)
    month_col = _MONTH_NUMBER_TO_COLUMN.get(incident_month)

    species_matched = False
    month_monitored = False
    count_per_hour = None

    if incident_species and month_col:
        key = (str(incident_species).strip().upper(), month_col)
        if key in species_month_lookup:
            species_matched = True
            count_per_hour = species_month_lookup[key]
            month_monitored = count_per_hour is not None

    return BirdContextJoinRecord(
        record_id=record_id,
        source_dataset="trektellen_fbbo_2025",
        spatial_resolution="single_station_point",
        temporal_resolution="monthly",
        join_method="nearest_available_station_same_species_same_calendar_month",
        distance_km=round(distance_km, 1),
        time_diff_years=time_diff_years,
        species_matched=species_matched,
        month_monitored=month_monitored,
        value_type="observed" if (species_matched and month_monitored) else "unavailable",
        local_activity_count_per_hour=count_per_hour,
    )


def join_incidents_to_trektellen(
    incidents_df: pd.DataFrame,
    trektellen_long_df: pd.DataFrame,
    trektellen_station_lat: float,
    trektellen_station_lon: float,
    trektellen_season_year: int,
    *,
    id_col: str = "INDEX_NR",
    lat_col: str = "LATITUDE",
    lon_col: str = "LONGITUDE",
    year_col: str = "INCIDENT_YEAR",
    month_col: str = "INCIDENT_MONTH",
    species_col: str = "SPECIES",
) -> pd.DataFrame:
    """
    Applies join_incident_to_trektellen across every row of incidents_df,
    using a precomputed species/month lookup (see
    _build_species_month_lookup) so this scales to the full GA
    population (~30k rows) without a per-row DataFrame filter. Rows with
    missing lat/lon are excluded entirely (cannot compute a spatial join
    without coordinates) rather than guessed; rows with missing
    year/month/species still get a distance/time_diff computed, just
    with species_matched=False.
    """
    lookup = _build_species_month_lookup(trektellen_long_df)
    records = []
    for _, row in incidents_df.iterrows():
        lat, lon = row.get(lat_col), row.get(lon_col)
        if pd.isna(lat) or pd.isna(lon):
            continue
        record = join_incident_to_trektellen(
            incident_lat=float(lat),
            incident_lon=float(lon),
            incident_year=int(row[year_col]) if pd.notna(row.get(year_col)) else trektellen_season_year,
            incident_month=int(row[month_col]) if pd.notna(row.get(month_col)) else 0,
            incident_species=row.get(species_col),
            record_id=row.get(id_col),
            species_month_lookup=lookup,
            trektellen_station_lat=trektellen_station_lat,
            trektellen_station_lon=trektellen_station_lon,
            trektellen_season_year=trektellen_season_year,
        )
        records.append(record.__dict__)
    return pd.DataFrame(records)


def build_join_quality_report(
    joined_df: pd.DataFrame,
    n_incidents_total: int,
    distance_tolerances_km: tuple[float, ...] = (50, 100, 250, 500, 1000, 2500),
    time_tolerances_years: tuple[int, ...] = (0, 1, 5, 10, 20),
) -> dict:
    """
    Per project requirement: "an initial join-quality report showing
    what percentage of strikes can be matched at different spatial and
    temporal tolerances." n_incidents_total is passed separately (not
    just len(joined_df)) so rows dropped for missing coordinates are
    visible in the report, not silently absent from the denominator.
    """
    n_joined = len(joined_df)
    n_dropped_no_coordinates = n_incidents_total - n_joined

    observed = joined_df[joined_df["value_type"] == "observed"]

    by_distance = {
        f"within_{d}km": {
            "n_matched": int((observed["distance_km"] <= d).sum()),
            "pct_of_all_incidents": round(100 * (observed["distance_km"] <= d).sum() / n_incidents_total, 2),
        }
        for d in distance_tolerances_km
    }
    by_time = {
        f"within_{t}yr": {
            "n_matched": int((observed["time_diff_years"] <= t).sum()),
            "pct_of_all_incidents": round(100 * (observed["time_diff_years"] <= t).sum() / n_incidents_total, 2),
        }
        for t in time_tolerances_years
    }

    return {
        "n_incidents_total": n_incidents_total,
        "n_dropped_missing_coordinates": n_dropped_no_coordinates,
        "n_species_matched_any_month": int(joined_df["species_matched"].sum()),
        "n_fully_observed_species_and_month": int(len(observed)),
        "pct_fully_observed": round(100 * len(observed) / n_incidents_total, 2),
        "match_rate_by_spatial_tolerance": by_distance,
        "match_rate_by_temporal_tolerance": by_time,
        "caveat": (
            "Only ONE bird-activity source (Trektellen FBBO, single station, 2025 "
            "season) exists in this repository today, so match rate is fundamentally "
            "capped by that station's distance to each incident's airport -- this "
            "report measures 'how often is FBBO a usable proxy', not 'how often is "
            "any bird-activity data available'. See BIRD_DATA_SOURCE_COMPARISON.md "
            "for why eBird Status & Trends would substantially raise this rate once integrated."
        ),
    }
