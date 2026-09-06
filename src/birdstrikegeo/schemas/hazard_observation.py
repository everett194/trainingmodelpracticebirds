"""
schemas/hazard_observation.py
--------------------------------
The standardized SECONDARY-output observation schema: one row per
(airport, date), NOT one row per strike incident. This is deliberately
separate from `schemas.faa.FAA_SCHEMA` (one row per reported strike,
used by the primary conditional-damage model) — see
DATA_AUDIT_REPORT.md section 9 and README.md for why aggregating strike
incidents up to airport-date would be the wrong unit for a damage model,
and why aggregating bird/weather/exposure context down to a single
strike-instant would be false precision.

Every column here is grouped by concept (geographic / time / bird /
weather / aviation-exposure / target), matching the project's Phase 2
brief. No column is fabricated: any field this project cannot currently
populate for a given row is left null with its own `*_available`
companion flag (the same "distance_warning / low_coverage_warning /
data_coverage_score" pattern already used by
`birdstrikegeo.models.activity_index` and
`birdstrikegeo.features.build_activity_features`), never defaulted to
zero or a guessed value.

This module defines the schema (column names, dtypes, grouping,
required vs optional) — it does not parse or ingest any file. See
`birdstrikegeo.hazard.spatiotemporal_join` for the code that actually
builds rows conforming to this schema, and
`birdstrikegeo.schemas.validation` for the generic checker that
validates a DataFrame against it.
"""

from __future__ import annotations

# Stable identifiers. Every row must have these, and (airport_id, date)
# together form the natural primary key for this unit of observation.
IDENTIFIER_SCHEMA: dict[str, str] = {
    "airport_id": "str",
    "date": "datetime64[ns]",  # calendar date, local to the airport
    "observation_id": "str",  # f"{airport_id}_{date:%Y-%m-%d}" — precomputed, never re-derived ad hoc
}

# Geographic variables. Latitude/longitude are the ONLY authoritative
# coordinates (see GEOSPATIAL_METHODS.md) — h3_cell is a derived index,
# never a substitute for storing the original WGS84 point.
GEOGRAPHIC_SCHEMA: dict[str, str] = {
    "latitude": "float",  # WGS84 (EPSG:4326)
    "longitude": "float",  # WGS84 (EPSG:4326)
    "elevation_ft": "float",
    "h3_cell_r5": "str",  # H3 index at resolution 5 (~8.5 km edge) — see geo/h3_grid.py
    "h3_cell_r7": "str",  # H3 index at resolution 7 (~1.2 km edge) — finer, for dense-data regions
    "distance_to_water_km": "float",
    "distance_to_wetland_km": "float",
    "distance_to_coastline_km": "float",
    "distance_to_landfill_km": "float",
    "distance_to_migration_corridor_km": "float",
    "habitat_data_available": "bool",  # False means "we have no layer", never "no habitat nearby"
}

# Time variables.
TIME_SCHEMA: dict[str, str] = {
    "timestamp_utc": "datetime64[ns, UTC]",  # midnight local, expressed in UTC, for this airport-date row
    "timestamp_local": "datetime64[ns]",
    "hour_local": "float",  # null at daily grain; populated only if/when an hourly row is ever built
    "month": "int",
    "day_of_year": "int",
    "season": "str",  # {"winter","spring","summer","fall"} — Northern Hemisphere convention, documented
    "is_migration_season": "bool",
    "daylight_category": "str",  # {"dawn","day","dusk","night"}
    "hours_from_sunrise": "float",
    "hours_from_sunset": "float",
}

# Bird variables. `bird_data_resolution` is the single most important
# honesty field in this schema — see module docstring and
# BIRD_DATA_SOURCE_COMPARISON.md. It must always be set explicitly, never
# left to be inferred from which columns happen to be non-null.
BIRD_SCHEMA: dict[str, str] = {
    "species_or_group": "str",  # null at the aggregate-activity grain; set only for a species-specific row
    "estimated_abundance": "float",
    "relative_abundance_index": "float",  # 0-1, only if a modeled source provides one (e.g. eBird S&T)
    "seasonal_presence": "str",  # {"resident","breeding","wintering","migrant_only","absent","unknown"}
    "migration_intensity": "float",
    "observation_effort_hours": "float",  # null (not 0) when effort is unknown, per DATA_CARD.md Trektellen notes
    "distance_from_bird_observation_to_airport_km": "float",
    "bird_data_source": "str",  # e.g. "trektellen_fbbo_2025", "ebird_status_trends_v2026"
    "bird_data_resolution": "str",  # e.g. "monthly_single_station", "weekly_3km_modeled" — REQUIRED whenever any bird_* field is non-null
    "bird_data_available": "bool",
}

# Weather variables. Mirrors schemas/weather.py's existing station-based
# fields; repeated here (not imported) because this schema's grain is
# airport-date, not station-hour — a daily summary, not a raw reading.
WEATHER_SCHEMA: dict[str, str] = {
    "temperature_c_mean": "float",
    "precipitation_mm_total": "float",
    "visibility_km_min": "float",
    "wind_speed_knots_mean": "float",
    "wind_direction_deg_mean": "float",
    "cloud_ceiling_ft_min": "float",
    "severe_weather_flag": "bool",
    "weather_station_distance_km": "float",
    "weather_available": "bool",
}

# Aviation-exposure variables. THIS GROUP IS THE ONE MOST LIKELY TO BE
# ENTIRELY NULL for the foreseeable future — see DATA_CARD.md §5
# (BTS T-100, not yet integrated). Its presence in the schema now,
# fully null, is deliberate: it documents the exact shape of data this
# project would need to ever build a true strike-probability or
# expected-count model, without pretending that data exists today.
AVIATION_EXPOSURE_SCHEMA: dict[str, str] = {
    "departures": "float",
    "arrivals": "float",
    "total_movements": "float",
    "predominant_aircraft_class": "str",
    "exposure_data_available": "bool",  # MUST be False until a real exposure source is wired in
}

# Target variables. Kept as SEPARATE columns, never combined — a query
# for "the damage rate" must never be confused with "the strike count"
# or "the activity index". Every target also carries its own basis flag
# so a consumer can tell a real fitted value from an absent one, exactly
# like MODEL_CARD.md already requires for activity_index.
TARGET_SCHEMA: dict[str, str] = {
    "strike_count": "float",  # count of REPORTED strikes at this airport-date — NOT strike probability (see README)
    "strike_count_basis": "str",  # e.g. "faa_reported_count" — never invented
    "damage_rate_given_strike": "float",  # mean damage outcome among reported strikes this airport-date, if any occurred
    "injury_flag": "bool",
    "species_involved": "str",  # comma-joined species observed in strikes this airport-date, if any
    "relative_hazard_index": "float",  # 0-100, see hazard/report.py — NOT a probability, see README "Scientific limitations"
    "relative_hazard_index_basis": "str",
    "activity_index": "float",  # 0-1, legacy formula (models/activity_index.py) — see LEGACY_SYSTEMS.md
}

# Every group merged, for convenience (e.g. building a full empty
# DataFrame or iterating all field names at once). Order matters only
# for documentation/CSV column order — no code should rely on dict order
# for correctness.
HAZARD_OBSERVATION_SCHEMA: dict[str, str] = {
    **IDENTIFIER_SCHEMA,
    **GEOGRAPHIC_SCHEMA,
    **TIME_SCHEMA,
    **BIRD_SCHEMA,
    **WEATHER_SCHEMA,
    **AVIATION_EXPOSURE_SCHEMA,
    **TARGET_SCHEMA,
}

# The minimum a row must have to be usable at all. Every other column in
# HAZARD_OBSERVATION_SCHEMA may legitimately be null (with its companion
# *_available flag explaining why) — these may not.
REQUIRED_FIELDS: tuple[str, ...] = ("airport_id", "date", "observation_id", "latitude", "longitude")

# Columns whose non-null-ness must always be accompanied by a specific
# other column being set — enforced by schemas.validation, not just
# documented here. Each entry: (value_column, required_companion_column).
REQUIRES_COMPANION: tuple[tuple[str, str], ...] = (
    ("estimated_abundance", "bird_data_resolution"),
    ("relative_abundance_index", "bird_data_resolution"),
    ("migration_intensity", "bird_data_resolution"),
)
