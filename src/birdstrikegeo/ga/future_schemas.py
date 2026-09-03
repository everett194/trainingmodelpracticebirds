"""
birdstrikegeo.ga.future_schemas
----------------------------------
Documented, UNIMPLEMENTED interfaces for the future occurrence-risk
model (a separate model from the conditional-damage model in this
module). Nothing here is wired into training or inference yet - it
exists so the eventual airport x date occurrence dataset has an agreed
shape to build toward, without delaying the current GBT milestone.

The occurrence model will need a genuine non-strike denominator (see
README "Scientific limitations") - each row below is one
AIRPORT x DATE, whether or not a strike happened that day, which is
exactly what the FAA strike-only export cannot provide by itself.
"""

from __future__ import annotations

# One row per (airport, calendar date) - the future occurrence model's
# unit of analysis. Real (non-strike) days must be included, sourced from
# GA movement/exposure data (e.g. a future FAA/BTS-derived GA flight
# count), not just days on which a strike happened to be reported.
AIRPORT_DAY_SCHEMA: dict[str, str] = {
    "airport_id": "str",
    "date": "date",
    "strike_count": "int",  # 0 on non-strike days - REQUIRES a real non-strike source, not FAA strikes alone
    "ga_departures_estimate": "float | None",  # future GA movement/exposure denominator
    "trektellen_activity_index": "float | None",  # birdstrikegeo.models.activity_index output, if in range/coverage
    "trektellen_distance_km": "float | None",
    "trektellen_coverage_available": "bool",
    "weather_summary": "dict | None",  # birdstrikegeo.data.ingest_weather-derived, forecast-style fields only
    "habitat_features": "dict | None",  # future geospatial/visual habitat layer (water proximity, land cover, etc.)
}

# Deliberately not implemented in this milestone - see README.md
# "Scientific limitations" and DATA_CARD.md "BTS T-100 departures".
NOT_YET_IMPLEMENTED = (
    "GA movement/exposure denominator (non-strike days)",
    "Trektellen integration for GA airports (most have no nearby site - see DATA_CARD.md)",
    "Weather integration beyond forecast-style sky/precipitation categories already in operational_core",
    "Geospatial/visual habitat features",
    "Airport x date occurrence model training/evaluation code",
)
