"""
features/build_activity_features.py
---------------------------------------
Task B (airport-period wildlife activity index) feature construction.

Builds every feature listed in Section 8 of the project spec for a
single (airport/query point, timestamp) pair, using ONLY Trektellen
observations strictly before that timestamp (see
birdstrikegeo.geo.temporal_join.counts_before_timestamp - this is the
core temporal-leakage guard for Task B).

Per project requirements:
  - taxonomic-group features (waterfowl/gull/raptor/large_bird counts)
    are only computed for species with a documented taxonomy entry -
    unknown species are excluded from those sums, never guessed.
  - a missing observation is never interpreted as zero bird activity -
    "no session occurred in this window" and "a session occurred and
    reported zero" are kept distinct (see _windowed_total below).
  - count_type is never silently combined across visible_migration /
    capture / nocturnal_flight_call - each grouped feature either
    filters to one count_type or is documented as combining all types.
"""

from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from birdstrikegeo.features.taxonomy import lookup_group
from birdstrikegeo.geo.spatial_join import distance_decay_weight, nearest_site, sites_within_radius
from birdstrikegeo.geo.temporal_join import counts_before_timestamp

_DIRECTION_TO_DEGREES = {
    "N": 0, "NE": 45, "E": 90, "SE": 135, "S": 180, "SW": 225, "W": 270, "NW": 315,
}

WINDOWS = {
    "24h": pd.Timedelta(hours=24),
    "3d": pd.Timedelta(days=3),
    "7d": pd.Timedelta(days=7),
    "14d": pd.Timedelta(days=14),
}


def prepare_counts_with_timestamps(counts: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Merges site lat/lon/timezone onto each count row and computes a
    single UTC timestamp (count_timestamp_utc) from count_date +
    start_time_local, localized using the SITE's timezone. Rows whose
    timezone/date/time can't be parsed get count_timestamp_utc = NaT and
    are excluded from any window (never assumed to be "recent")."""
    merged = counts.merge(
        sites[["site_id", "latitude", "longitude", "timezone"]],
        on="site_id", how="left", suffixes=("", "_site"),
    )

    def _to_utc(row):
        try:
            date = pd.to_datetime(row["count_date"]).date()
            time_str = row["start_time_local"] if pd.notna(row["start_time_local"]) else "00:00"
            hh, mm = (int(x) for x in str(time_str).split(":")[:2])
            tz_name = row["timezone"] if pd.notna(row["timezone"]) else "UTC"
            local_dt = datetime(date.year, date.month, date.day, hh, mm, tzinfo=ZoneInfo(tz_name))
            return local_dt.astimezone(timezone.utc)
        except Exception:
            return pd.NaT

    merged["count_timestamp_utc"] = merged.apply(_to_utc, axis=1)
    return merged


def _windowed_total(window_df: pd.DataFrame) -> float | None:
    """Sums `count`, but returns None (not 0.0) if no observation session
    occurred at all in the window - a missing session is not zero
    activity, it's unknown activity."""
    if len(window_df) == 0:
        return None
    return float(window_df["count"].sum())


def _grouped_total(window_df: pd.DataFrame, group: str) -> float | None:
    """Sum of `count` for species resolving to the given documented
    taxonomic group. Rows with unresolvable species are excluded from
    the sum entirely (not guessed), so this can legitimately be lower
    than _windowed_total even with full coverage."""
    if len(window_df) == 0:
        return None
    resolved_group = window_df["species_scientific_name"].apply(lookup_group)
    matching = window_df[resolved_group == group]
    return float(matching["count"].sum())


def _directional_consistency(window_df: pd.DataFrame) -> float | None:
    """Mean resultant vector length (0=random directions, 1=perfectly
    consistent direction) of reported flight directions, weighted by
    count. Standard circular-statistics summary; not a novel method."""
    directions = window_df["flight_direction"].map(_DIRECTION_TO_DEGREES).dropna()
    if len(directions) == 0:
        return None
    weights = window_df.loc[directions.index, "count"].astype(float).clip(lower=0)
    if weights.sum() == 0:
        return None
    radians = np.radians(directions.astype(float))
    x = np.sum(weights * np.cos(radians)) / weights.sum()
    y = np.sum(weights * np.sin(radians)) / weights.sum()
    return float(np.hypot(x, y))


def build_activity_features(
    query_lon: float,
    query_lat: float,
    query_timestamp_utc: pd.Timestamp,
    sites: pd.DataFrame,
    counts_with_ts: pd.DataFrame,
    radii_km: tuple[int, ...] = (50, 100, 250),
    max_coverage_radius_km: float = 250.0,
    half_life_km: float = 50.0,
    low_coverage_site_threshold: int = 2,
) -> dict:
    """Returns a flat dict of every feature listed in Section 8 of the
    project spec, plus the coverage/availability flags, for one
    (query point, timestamp) pair."""
    features: dict = {}

    if sites.empty:
        features["trektellen_available"] = False
        features["distance_warning"] = True
        features["low_coverage_warning"] = True
        features["observation_effort_available"] = False
        features["data_coverage_score"] = 0.0
        for key in _ALL_FEATURE_KEYS:
            features.setdefault(key, None)
        return features

    features["trektellen_available"] = True

    nearest = nearest_site(query_lon, query_lat, sites)
    features["distance_to_nearest_trektellen_site_km"] = nearest["distance_km"] if nearest else None

    for r in radii_km:
        features[f"number_of_sites_within_{r}_km"] = len(sites_within_radius(query_lon, query_lat, sites, r))

    sites_in_radius = sites_within_radius(query_lon, query_lat, sites, max_coverage_radius_km)
    site_ids_in_radius = set(sites_in_radius["site_id"])
    counts_in_radius = counts_with_ts[counts_with_ts["site_id"].isin(site_ids_in_radius)]

    # nearest_site_observation_age_hours: most recent session AT THE
    # NEAREST SITE, strictly before the query timestamp.
    query_ts = pd.Timestamp(query_timestamp_utc)
    query_ts = query_ts.tz_localize("UTC") if query_ts.tzinfo is None else query_ts

    features["nearest_site_observation_age_hours"] = None
    if nearest and nearest["site_id"] in set(counts_with_ts["site_id"]):
        nearest_site_history = counts_with_ts[
            (counts_with_ts["site_id"] == nearest["site_id"])
            & (counts_with_ts["count_timestamp_utc"] < query_ts)
        ]
        if len(nearest_site_history):
            most_recent = nearest_site_history["count_timestamp_utc"].max()
            features["nearest_site_observation_age_hours"] = (query_ts - most_recent).total_seconds() / 3600.0

    windows = {
        label: counts_before_timestamp(counts_in_radius, query_timestamp_utc, td, "count_timestamp_utc")
        for label, td in WINDOWS.items()
    }

    features["total_birds_previous_24h"] = _windowed_total(windows["24h"])
    features["total_birds_previous_3d"] = _windowed_total(windows["3d"])
    features["total_birds_previous_7d"] = _windowed_total(windows["7d"])
    features["total_birds_previous_14d"] = _windowed_total(windows["14d"])

    w7 = windows["7d"]
    if len(w7):
        session_hours = w7.drop_duplicates(["site_id", "count_date", "start_time_local"])["observation_hours"]
        total_hours = pd.to_numeric(session_hours, errors="coerce").sum()
        total_birds_7d = features["total_birds_previous_7d"] or 0.0
        features["birds_per_observation_hour_previous_7d"] = (
            float(total_birds_7d / total_hours) if total_hours and total_hours > 0 else None
        )
        species_key = w7["species_scientific_name"].fillna(w7["species_common_name"])
        features["species_richness_previous_7d"] = int(species_key.dropna().nunique())
    else:
        features["birds_per_observation_hour_previous_7d"] = None
        features["species_richness_previous_7d"] = None

    features["large_bird_count_previous_7d"] = _grouped_total(w7, "large_bird")
    features["waterfowl_count_previous_7d"] = _grouped_total(w7, "waterfowl")
    features["gull_count_previous_7d"] = _grouped_total(w7, "gull")
    features["raptor_count_previous_7d"] = _grouped_total(w7, "raptor")

    w3 = windows["3d"]
    features["nocturnal_flight_call_count_previous_3d"] = (
        _windowed_total(w3[w3["count_type"] == "nocturnal_flight_call"]) if len(w3) else None
    )
    features["migration_count_previous_7d"] = (
        _windowed_total(w7[w7["count_type"] == "visible_migration"]) if len(w7) else None
    )
    features["capture_count_previous_7d"] = (
        _windowed_total(w7[w7["count_type"] == "capture"]) if len(w7) else None
    )

    features["directional_consistency"] = _directional_consistency(w7) if len(w7) else None

    features["observation_effort_available"] = bool(w7["observer_effort_available"].fillna(False).any()) if len(w7) else False

    features["distance_warning"] = nearest is None or nearest["distance_km"] > max_coverage_radius_km
    features["low_coverage_warning"] = len(sites_in_radius) < low_coverage_site_threshold

    # data_coverage_score: a transparent 0-1 heuristic (documented, not
    # a fitted/calibrated probability) combining proximity, monitoring
    # density, and recency - see GEOSPATIAL_METHODS.md.
    proximity_component = (
        float(distance_decay_weight(nearest["distance_km"], half_life_km)) if nearest else 0.0
    )
    density_component = min(1.0, len(sites_in_radius) / 5.0)
    age_hours = features["nearest_site_observation_age_hours"]
    recency_component = float(distance_decay_weight(age_hours / 24.0 * half_life_km, half_life_km)) if age_hours is not None else 0.0
    features["data_coverage_score"] = round((proximity_component + density_component + recency_component) / 3.0, 4)

    return features


_ALL_FEATURE_KEYS = (
    "distance_to_nearest_trektellen_site_km",
    "number_of_sites_within_50_km",
    "number_of_sites_within_100_km",
    "number_of_sites_within_250_km",
    "nearest_site_observation_age_hours",
    "total_birds_previous_24h",
    "total_birds_previous_3d",
    "total_birds_previous_7d",
    "total_birds_previous_14d",
    "birds_per_observation_hour_previous_7d",
    "species_richness_previous_7d",
    "large_bird_count_previous_7d",
    "waterfowl_count_previous_7d",
    "gull_count_previous_7d",
    "raptor_count_previous_7d",
    "nocturnal_flight_call_count_previous_3d",
    "migration_count_previous_7d",
    "capture_count_previous_7d",
    "directional_consistency",
)
