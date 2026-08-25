"""
schemas/airports.py
--------------------
Canonical airport schema, plus optional environmental-context fields that
are only ever populated when real spatial layers (wetlands, water,
land cover, etc.) are supplied by the user - never fabricated. See
`birdstrikegeo.geo.airport_layers` and Section 13 of the project spec in
README.md.
"""

AIRPORT_SCHEMA: dict[str, str] = {
    "airport_id": "str",
    "icao": "str",
    "iata": "str",
    "faa_lid": "str",
    "airport_name": "str",
    "latitude": "float",
    "longitude": "float",
    "elevation_ft": "float",
    "airport_type": "str",
    "runway_count": "float",
    "longest_runway_ft": "float",
    "state": "str",
    "country": "str",
}

# Derived environmental features. These are only computed when the
# corresponding real spatial layer is available (see
# birdstrikegeo.geo.airport_layers.compute_environmental_features).
# Each has a companion *_available boolean so downstream code and the
# UI can distinguish "no habitat nearby" from "we don't have data."
ENVIRONMENTAL_FEATURE_SCHEMA: dict[str, str] = {
    "distance_to_water_km": "float",
    "water_area_within_5km": "float",
    "wetland_area_within_10km": "float",
    "agricultural_area_within_10km": "float",
    "urban_area_within_10km": "float",
    "landfill_distance_km": "float",
    "habitat_data_available": "bool",
}

# Optional environmental layers this project knows how to consume IF the
# user supplies them locally. None of these are bundled, scraped, or
# fabricated for real airports.
OPTIONAL_ENVIRONMENTAL_LAYERS: tuple[str, ...] = (
    "wetlands",
    "surface_water",
    "coastline",
    "land_cover",
    "protected_areas",
    "agricultural_land",
    "landfills",
    "airport_boundaries",
)
