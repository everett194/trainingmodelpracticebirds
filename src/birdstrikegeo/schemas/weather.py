"""
schemas/weather.py
--------------------
Canonical hourly-weather schema (e.g. for NOAA ISD/METAR-derived data)
and the fields recorded when weather observations are joined to a strike
or activity-index query. See `birdstrikegeo.geo.temporal_join` for the
nearest-station, nearest-prior-hour join logic.
"""

WEATHER_SCHEMA: dict[str, str] = {
    "station_id": "str",
    "timestamp_utc": "datetime64[ns, UTC]",
    "latitude": "float",
    "longitude": "float",
    "temperature_c": "float",
    "dewpoint_c": "float",
    "relative_humidity_pct": "float",
    "wind_speed_knots": "float",
    "wind_direction_deg": "float",
    "wind_gust_knots": "float",
    "visibility_km": "float",
    "precipitation_mm": "float",
    "pressure_hpa": "float",
    "cloud_ceiling_ft": "float",
    "present_weather_code": "str",
    "quality_flag": "str",
}

# Fields recorded on the JOINED record (e.g. a FAA strike row, or an
# activity-index query result) describing how the weather match was
# made, not the weather itself. Always populate these - never silently
# leave a weather join unlabeled.
WEATHER_JOIN_METADATA_SCHEMA: dict[str, str] = {
    "weather_station_distance_km": "float",
    "weather_observation_age_minutes": "float",
    "weather_available": "bool",
}
