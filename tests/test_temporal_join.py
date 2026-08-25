from datetime import datetime, timezone

import pandas as pd
import pytest

from birdstrikegeo.features.build_activity_features import _windowed_total, prepare_counts_with_timestamps
from birdstrikegeo.geo.temporal_join import counts_before_timestamp, nearest_prior_weather


def test_counts_before_timestamp_excludes_future_rows():
    counts = pd.DataFrame(
        {
            "count_timestamp_utc": pd.to_datetime(
                ["2024-05-01T08:00Z", "2024-05-03T08:00Z", "2024-05-10T08:00Z"]
            ),
            "count": [1, 2, 3],
        }
    )
    query = pd.Timestamp("2024-05-05T00:00Z")
    window = counts_before_timestamp(counts, query, pd.Timedelta(days=7))
    # The 2024-05-10 row is AFTER the query timestamp and must never appear,
    # even though it's within a 7-day window of the query date.
    assert (window["count_timestamp_utc"] < query).all()
    assert 3 not in window["count"].values


def test_counts_before_timestamp_respects_lookback_window():
    counts = pd.DataFrame(
        {
            "count_timestamp_utc": pd.to_datetime(["2024-04-01T08:00Z", "2024-05-04T08:00Z"]),
            "count": [1, 2],
        }
    )
    query = pd.Timestamp("2024-05-05T00:00Z")
    window = counts_before_timestamp(counts, query, pd.Timedelta(days=7))
    assert list(window["count"]) == [2]  # 2024-04-01 is more than 7 days before the query


def test_counts_before_timestamp_excludes_rows_at_exactly_the_query_time():
    counts = pd.DataFrame({"count_timestamp_utc": pd.to_datetime(["2024-05-05T00:00Z"]), "count": [5]})
    query = pd.Timestamp("2024-05-05T00:00Z")
    window = counts_before_timestamp(counts, query, pd.Timedelta(days=7))
    assert len(window) == 0  # strictly before, not "at or before"


def test_counts_before_timestamp_excludes_null_timestamps():
    counts = pd.DataFrame({"count_timestamp_utc": [pd.NaT, pd.Timestamp("2024-05-04T08:00Z")], "count": [99, 1]})
    query = pd.Timestamp("2024-05-05T00:00Z")
    window = counts_before_timestamp(counts, query, pd.Timedelta(days=7))
    assert 99 not in window["count"].values


def test_windowed_total_distinguishes_no_session_from_a_zero_count_session():
    empty_window = pd.DataFrame({"count": []})
    assert _windowed_total(empty_window) is None  # no session happened - unknown, not zero

    zero_count_session = pd.DataFrame({"count": [0]})
    assert _windowed_total(zero_count_session) == 0.0  # a session happened and reported zero


def test_prepare_counts_with_timestamps_converts_local_time_to_utc():
    sites = pd.DataFrame(
        {"site_id": ["S1"], "latitude": [40.7], "longitude": [-74.0], "timezone": ["America/New_York"]}
    )
    counts = pd.DataFrame(
        {"site_id": ["S1"], "count_date": ["2024-07-01"], "start_time_local": ["08:00"], "count": [3]}
    )
    result = prepare_counts_with_timestamps(counts, sites)
    ts = result["count_timestamp_utc"].iloc[0]
    # 08:00 EDT (UTC-4 in July) -> 12:00 UTC
    assert ts.tzinfo is not None
    assert ts.astimezone(timezone.utc).hour == 12


def test_prepare_counts_with_timestamps_handles_unparseable_rows_gracefully():
    sites = pd.DataFrame({"site_id": ["S1"], "latitude": [40.7], "longitude": [-74.0], "timezone": ["Not/AZone"]})
    counts = pd.DataFrame(
        {"site_id": ["S1"], "count_date": ["2024-07-01"], "start_time_local": ["08:00"], "count": [3]}
    )
    result = prepare_counts_with_timestamps(counts, sites)
    assert pd.isna(result["count_timestamp_utc"].iloc[0])


def test_nearest_prior_weather_never_uses_future_observation_by_default():
    weather = pd.DataFrame(
        {
            "station_id": ["W1", "W1"],
            "latitude": [40.0, 40.0],
            "longitude": [-100.0, -100.0],
            "timestamp_utc": [
                datetime(2024, 5, 1, 6, tzinfo=timezone.utc),
                datetime(2024, 5, 1, 14, tzinfo=timezone.utc),  # AFTER the query time below
            ],
        }
    )
    query_time = datetime(2024, 5, 1, 10, tzinfo=timezone.utc)
    result = nearest_prior_weather(weather, -100.0, 40.0, query_time, max_distance_km=50, max_age_hours=24)
    assert result.weather_available
    assert result.row["timestamp_utc"] == datetime(2024, 5, 1, 6, tzinfo=timezone.utc)


def test_nearest_prior_weather_respects_max_age_hours():
    weather = pd.DataFrame(
        {
            "station_id": ["W1"],
            "latitude": [40.0],
            "longitude": [-100.0],
            "timestamp_utc": [datetime(2024, 5, 1, 0, tzinfo=timezone.utc)],
        }
    )
    query_time = datetime(2024, 5, 1, 10, tzinfo=timezone.utc)  # 10 hours later
    result = nearest_prior_weather(weather, -100.0, 40.0, query_time, max_distance_km=50, max_age_hours=3)
    assert not result.weather_available


def test_nearest_prior_weather_respects_max_distance_km():
    weather = pd.DataFrame(
        {
            "station_id": ["FAR"],
            "latitude": [10.0],
            "longitude": [10.0],
            "timestamp_utc": [datetime(2024, 5, 1, 9, tzinfo=timezone.utc)],
        }
    )
    query_time = datetime(2024, 5, 1, 10, tzinfo=timezone.utc)
    result = nearest_prior_weather(weather, -100.0, 40.0, query_time, max_distance_km=50, max_age_hours=24)
    assert not result.weather_available


def test_nearest_prior_weather_allow_future_opts_into_symmetric_window():
    weather = pd.DataFrame(
        {
            "station_id": ["W1"],
            "latitude": [40.0],
            "longitude": [-100.0],
            "timestamp_utc": [datetime(2024, 5, 1, 11, tzinfo=timezone.utc)],  # 1 hour AFTER query
        }
    )
    query_time = datetime(2024, 5, 1, 10, tzinfo=timezone.utc)
    default_result = nearest_prior_weather(weather, -100.0, 40.0, query_time, max_distance_km=50, max_age_hours=3)
    assert not default_result.weather_available  # future-only observation, default must reject it

    symmetric_result = nearest_prior_weather(
        weather, -100.0, 40.0, query_time, max_distance_km=50, max_age_hours=3, allow_future=True
    )
    assert symmetric_result.weather_available  # explicitly opted in
