from __future__ import annotations

import pandas as pd

from birdstrikegeo.hazard.spatiotemporal_join import (
    build_join_quality_report,
    join_incident_to_trektellen,
    join_incidents_to_trektellen,
)

FBBO_LAT, FBBO_LON = 39.248, -76.026


def _long_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "species": ["Northern Cardinal", "Northern Cardinal", "Song Sparrow"],
            "month": ["mar", "apr", "mar"],
            "count": [5.0, 10.0, 0.0],
            "observation_hours": [100.0, 100.0, 0.0],
            "count_per_hour": [0.05, 0.10, pd.NA],
        }
    )


def test_species_and_month_match_gives_observed_value():
    long_df = _long_df()
    lookup = {("NORTHERN CARDINAL", "mar"): 0.05, ("NORTHERN CARDINAL", "apr"): 0.10, ("SONG SPARROW", "mar"): None}
    record = join_incident_to_trektellen(
        incident_lat=FBBO_LAT, incident_lon=FBBO_LON, incident_year=2025, incident_month=3,
        incident_species="Northern Cardinal", record_id=1, species_month_lookup=lookup,
        trektellen_station_lat=FBBO_LAT, trektellen_station_lon=FBBO_LON, trektellen_season_year=2025,
    )
    assert record.species_matched is True
    assert record.month_monitored is True
    assert record.value_type == "observed"
    assert record.local_activity_count_per_hour == 0.05
    assert record.distance_km < 0.001  # same point as the station


def test_unmonitored_month_gives_unavailable_but_species_matched():
    lookup = {("SONG SPARROW", "mar"): None}
    record = join_incident_to_trektellen(
        incident_lat=FBBO_LAT, incident_lon=FBBO_LON, incident_year=2025, incident_month=3,
        incident_species="Song Sparrow", record_id=2, species_month_lookup=lookup,
        trektellen_station_lat=FBBO_LAT, trektellen_station_lon=FBBO_LON, trektellen_season_year=2025,
    )
    assert record.species_matched is True
    assert record.month_monitored is False
    assert record.value_type == "unavailable"
    assert record.local_activity_count_per_hour is None


def test_unmatched_species_gives_unavailable():
    lookup = {("NORTHERN CARDINAL", "mar"): 0.05}
    record = join_incident_to_trektellen(
        incident_lat=FBBO_LAT, incident_lon=FBBO_LON, incident_year=2025, incident_month=3,
        incident_species="Bald Eagle", record_id=3, species_month_lookup=lookup,
        trektellen_station_lat=FBBO_LAT, trektellen_station_lon=FBBO_LON, trektellen_season_year=2025,
    )
    assert record.species_matched is False
    assert record.value_type == "unavailable"


def test_time_diff_years_computed_correctly():
    record = join_incident_to_trektellen(
        incident_lat=FBBO_LAT, incident_lon=FBBO_LON, incident_year=2015, incident_month=3,
        incident_species=None, record_id=4, species_month_lookup={},
        trektellen_station_lat=FBBO_LAT, trektellen_station_lon=FBBO_LON, trektellen_season_year=2025,
    )
    assert record.time_diff_years == 10


def test_join_incidents_drops_rows_missing_coordinates():
    incidents = pd.DataFrame(
        {
            "INDEX_NR": [1, 2],
            "LATITUDE": [FBBO_LAT, None],
            "LONGITUDE": [FBBO_LON, -76.0],
            "INCIDENT_YEAR": [2025, 2025],
            "INCIDENT_MONTH": [3, 3],
            "SPECIES": ["Northern Cardinal", "Northern Cardinal"],
        }
    )
    joined = join_incidents_to_trektellen(
        incidents, _long_df(), trektellen_station_lat=FBBO_LAT, trektellen_station_lon=FBBO_LON,
        trektellen_season_year=2025,
    )
    assert len(joined) == 1  # the row with a null latitude is excluded, not guessed


def test_build_join_quality_report_accounts_for_dropped_rows():
    incidents = pd.DataFrame(
        {
            "INDEX_NR": [1, 2, 3],
            "LATITUDE": [FBBO_LAT, None, FBBO_LAT],
            "LONGITUDE": [FBBO_LON, -76.0, FBBO_LON],
            "INCIDENT_YEAR": [2025, 2025, 2015],
            "INCIDENT_MONTH": [3, 3, 4],
            "SPECIES": ["Northern Cardinal", "Northern Cardinal", "Northern Cardinal"],
        }
    )
    joined = join_incidents_to_trektellen(
        incidents, _long_df(), trektellen_station_lat=FBBO_LAT, trektellen_station_lon=FBBO_LON,
        trektellen_season_year=2025,
    )
    report = build_join_quality_report(joined, n_incidents_total=3)
    assert report["n_incidents_total"] == 3
    assert report["n_dropped_missing_coordinates"] == 1
    assert report["match_rate_by_spatial_tolerance"]["within_50km"]["n_matched"] == 2  # both remaining rows at distance 0
    assert report["match_rate_by_temporal_tolerance"]["within_1yr"]["n_matched"] == 1  # only the 2025 incident
