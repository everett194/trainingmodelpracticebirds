"""
schemas/faa.py
---------------
The canonical (conceptual) schema for FAA Wildlife Strike Database
records, after column-alias resolution (see
`birdstrikegeo.data.column_aliases`).

This module does NOT parse files - it only documents field names, types,
and which fields are forbidden as model inputs because they describe the
outcome of the strike rather than information known beforehand (see
`FORBIDDEN_LEAKAGE_COLUMNS` and `birdstrikegeo.features.leakage`).
"""

from dataclasses import dataclass, field

# Conceptual field name -> expected pandas/py dtype, used to validate a
# DataFrame after ingestion. "str" covers free text and categorical
# fields; numeric/date types are explicit.
FAA_SCHEMA: dict[str, str] = {
    "record_id": "str",
    "incident_date": "datetime64[ns]",
    "incident_time_local": "str",  # HH:MM, kept as string - see ingest_faa.py for parsing notes
    "airport_id": "str",
    "airport_name": "str",
    "state": "str",
    "latitude": "float",
    "longitude": "float",
    "phase_of_flight": "str",
    "height_agl_ft": "float",
    "speed_ias_knots": "float",
    "aircraft_make": "str",
    "aircraft_model": "str",
    "aircraft_type": "str",
    "aircraft_mass_class": "str",
    "engine_type": "str",
    "engine_count": "float",
    "sky_condition": "str",
    "precipitation": "str",
    "species_common_name": "str",
    "species_scientific_name": "str",
    "wildlife_size": "str",
    "number_seen": "str",  # FAA reports this as a banded category as often as a count
    "number_struck": "str",
    "damage_flag": "str",
    "damage_level": "str",
    "effect_on_flight": "str",
    "parts_struck": "str",
    "parts_damaged": "str",
    "repair_cost": "float",
    "other_cost": "float",
    "injuries": "float",
    "fatalities": "float",
    "narrative": "str",
}

# Fields that describe the OUTCOME of the strike, not information known
# before/at the time of the strike. These must never reach a feature
# matrix used to predict damage - see birdstrikegeo.features.leakage,
# which raises an error if any of these appear in model inputs.
FORBIDDEN_LEAKAGE_COLUMNS: tuple[str, ...] = (
    "damage_flag",
    "damage_level",
    "effect_on_flight",
    "parts_damaged",
    "repair_cost",
    "other_cost",
    "injuries",
    "fatalities",
    "narrative",
)

# The single field damage_flag/damage_level are read FROM to construct
# the supervised target (see birdstrikegeo.features.build_damage_features
# .make_damage_target). Once the target column is built, the source
# columns are dropped from the feature matrix - they are terminal, not
# reusable as inputs.
TARGET_SOURCE_COLUMNS: tuple[str, ...] = ("damage_flag", "damage_level")

# Columns that are safe to use as model inputs (schema columns minus the
# forbidden/target-source ones). Downstream code should still run
# birdstrikegeo.features.leakage.assert_no_leakage() rather than trusting
# this constant alone.
ALLOWED_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    c for c in FAA_SCHEMA if c not in FORBIDDEN_LEAKAGE_COLUMNS
)


@dataclass(frozen=True)
class FaaRecord:
    """
    A single, type-checked FAA strike record. Constructed after ingestion
    and alias resolution - see birdstrikegeo.data.ingest_faa.
    """

    record_id: str
    incident_date: "object"  # pandas.Timestamp at runtime
    airport_id: str | None = None
    airport_name: str | None = None
    state: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    phase_of_flight: str | None = None
    species_common_name: str | None = None
    species_scientific_name: str | None = None
    wildlife_size: str | None = None
    damage: int | None = None  # 1, 0, or None (unknown/ambiguous) - see target construction
    extra: dict = field(default_factory=dict)
