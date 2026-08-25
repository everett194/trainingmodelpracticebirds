"""
inference/calculate_activity.py
------------------------------------
Task B inference: given an airport, a query timestamp, and optional
radius/lookback overrides, loads Trektellen sites/counts and airports,
builds the activity features for that one query, and computes the
activity_index. This is what the Flask activity explorer (Section 21)
calls.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from birdstrikegeo.data.ingest_airports import load_airports
from birdstrikegeo.data.ingest_trektellen import load_counts, load_sites
from birdstrikegeo.features.build_activity_features import build_activity_features, prepare_counts_with_timestamps
from birdstrikegeo.models.activity_index import compute_activity_index


def calculate_activity_for_airport(
    airport_id: str,
    query_timestamp: str,
    airports_path: str | Path,
    sites_path: str | Path,
    counts_path: str | Path,
    radii_km: tuple[int, ...] = (50, 100, 250),
    max_coverage_radius_km: float = 250.0,
) -> dict:
    airports = load_airports(airports_path)
    airport_rows = airports[airports["airport_id"] == airport_id]
    if len(airport_rows) == 0:
        raise ValueError(f"Unknown airport_id: {airport_id}")
    airport = airport_rows.iloc[0]

    sites = load_sites(sites_path)
    counts, _validation_report = load_counts(counts_path, sites)
    counts_with_ts = prepare_counts_with_timestamps(counts, sites)

    query_ts = pd.Timestamp(query_timestamp)
    query_ts = query_ts.tz_localize("UTC") if query_ts.tzinfo is None else query_ts

    features = build_activity_features(
        query_lon=airport["longitude"], query_lat=airport["latitude"], query_timestamp_utc=query_ts,
        sites=sites, counts_with_ts=counts_with_ts, radii_km=radii_km, max_coverage_radius_km=max_coverage_radius_km,
    )
    index_result = compute_activity_index(features)

    nearest = None
    if features.get("trektellen_available") and features.get("distance_to_nearest_trektellen_site_km") is not None:
        from birdstrikegeo.geo.spatial_join import nearest_site

        nearest = nearest_site(airport["longitude"], airport["latitude"], sites)

    return {
        "airport_id": airport_id,
        "airport_name": airport.get("airport_name"),
        "query_timestamp_utc": str(query_ts),
        "features": features,
        "nearest_trektellen_site": nearest,
        **index_result,
    }
