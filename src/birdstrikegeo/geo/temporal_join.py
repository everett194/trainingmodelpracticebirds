"""
geo/temporal_join.py
-----------------------
Strict "no future data" temporal joins, used for both:

  - Trektellen count lookback windows (counts_before_timestamp): a
    strike on a given date must NEVER be joined against bird counts
    observed after it.
  - Weather nearest-prior-hour join (nearest_prior_weather): the default
    NEVER uses a weather observation recorded after the query timestamp.
    A symmetric research window (allow_future=True) is available but
    must be explicitly opted into - it exists for exploratory research
    only, not for the damage/activity pipelines' default behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from birdstrikegeo.geo.spatial_join import distances_to_sites


def counts_before_timestamp(
    counts: pd.DataFrame,
    query_timestamp: pd.Timestamp,
    lookback: pd.Timedelta,
    timestamp_column: str = "count_timestamp_utc",
) -> pd.DataFrame:
    """
    Returns only rows with query_timestamp - lookback <= timestamp <
    query_timestamp (STRICTLY before the query, matching Section 8's
    "must never use later bird observations"). Rows with a null
    timestamp are excluded, not assumed to be in-window.
    """
    ts = pd.to_datetime(counts[timestamp_column], errors="coerce", utc=True)
    query_timestamp = pd.Timestamp(query_timestamp)
    if query_timestamp.tzinfo is None:
        query_timestamp = query_timestamp.tz_localize("UTC")
    window_start = query_timestamp - lookback
    in_window = (ts >= window_start) & (ts < query_timestamp)
    return counts.loc[in_window].copy()


@dataclass
class WeatherJoinResult:
    row: pd.Series | None
    weather_available: bool
    weather_station_distance_km: float | None
    weather_observation_age_minutes: float | None


def nearest_prior_weather(
    weather: pd.DataFrame,
    query_lon: float,
    query_lat: float,
    query_timestamp: pd.Timestamp,
    max_distance_km: float = 100.0,
    max_age_hours: float = 3.0,
    allow_future: bool = False,
) -> WeatherJoinResult:
    """
    Finds the nearest weather station within max_distance_km, then the
    most recent observation at or before query_timestamp (or, if
    allow_future=True, the single closest observation in time in either
    direction - an explicit, opt-in symmetric research window, never the
    default for damage/activity predictions).

    Returns weather_available=False (not a fabricated/interpolated row)
    if no station is within range or no qualifying observation exists.
    """
    if weather.empty:
        return WeatherJoinResult(None, False, None, None)

    query_timestamp = pd.Timestamp(query_timestamp)
    if query_timestamp.tzinfo is None:
        query_timestamp = query_timestamp.tz_localize("UTC")

    stations = weather[["station_id", "latitude", "longitude"]].drop_duplicates("station_id").reset_index(drop=True)
    dists = distances_to_sites(query_lon, query_lat, stations)
    stations = stations.assign(distance_km=dists)
    nearby = stations[stations["distance_km"] <= max_distance_km].sort_values("distance_km")

    if nearby.empty:
        return WeatherJoinResult(None, False, None, None)

    for _, station in nearby.iterrows():
        station_obs = weather[weather["station_id"] == station["station_id"]].copy()
        station_obs["timestamp_utc"] = pd.to_datetime(station_obs["timestamp_utc"], errors="coerce", utc=True)

        if allow_future:
            candidates = station_obs.assign(
                _age_minutes=(station_obs["timestamp_utc"] - query_timestamp).abs().dt.total_seconds() / 60.0
            )
        else:
            prior = station_obs[station_obs["timestamp_utc"] <= query_timestamp]
            candidates = prior.assign(
                _age_minutes=(query_timestamp - prior["timestamp_utc"]).dt.total_seconds() / 60.0
            )

        candidates = candidates[candidates["_age_minutes"] <= max_age_hours * 60.0]
        if candidates.empty:
            continue

        best = candidates.sort_values("_age_minutes").iloc[0]
        return WeatherJoinResult(
            row=best.drop("_age_minutes"),
            weather_available=True,
            weather_station_distance_km=float(station["distance_km"]),
            weather_observation_age_minutes=float(best["_age_minutes"]),
        )

    return WeatherJoinResult(None, False, None, None)
