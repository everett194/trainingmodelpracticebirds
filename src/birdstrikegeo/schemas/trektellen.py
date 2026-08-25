"""
schemas/trektellen.py
----------------------
Canonical schemas for Trektellen-style bird-migration monitoring data:
one schema for monitoring *sites*, one for individual *count sessions*
(one row per site x count-session x species, in canonical long format).

Trektellen (https://www.trektellen.org/) is a Dutch-led birdwatching
network. This project never scrapes it - see DATA_DOWNLOAD_GUIDE.md and
`birdstrikegeo.data.ingest_trektellen` for the required, user-authorized
export + provenance workflow.
"""

TREKTELLEN_SITE_SCHEMA: dict[str, str] = {
    "site_id": "str",
    "site_name": "str",
    "country": "str",
    "latitude": "float",
    "longitude": "float",
    "count_type": "str",  # e.g. "visible_migration", "capture", "nocturnal_flight_call"
    "timezone": "str",  # IANA timezone name, e.g. "Europe/Amsterdam"
    "site_active_from": "datetime64[ns]",
    "site_active_to": "datetime64[ns]",
}

TREKTELLEN_COUNT_SCHEMA: dict[str, str] = {
    "count_id": "str",
    "site_id": "str",
    "count_date": "datetime64[ns]",
    "start_time_local": "str",
    "end_time_local": "str",
    "observation_hours": "float",
    "species_common_name": "str",
    "species_scientific_name": "str",
    "species_code": "str",
    "count": "int",
    "flight_direction": "str",
    "count_type": "str",
    "weather_notes": "str",
    "observer_effort_available": "bool",
}

# Count types are NOT interchangeable measurement processes - see
# GEOSPATIAL_METHODS.md and LIMITATIONS in MODEL_CARD.md. Feature code
# must stratify or explicitly choose a count_type rather than summing
# across them silently.
KNOWN_COUNT_TYPES: tuple[str, ...] = (
    "visible_migration",
    "capture",
    "nocturnal_flight_call",
    "other",
)

# Required provenance fields for any Trektellen export used by this
# project. See birdstrikegeo.data.ingest_trektellen.load_provenance() and
# configs/data_sources.yaml.
TREKTELLEN_PROVENANCE_FIELDS: tuple[str, ...] = (
    "source_name",
    "source_url",
    "download_date",
    "access_method",
    "permission_or_license",
    "contact_or_citation",
    "geographic_scope",
    "temporal_scope",
    "notes",
)
