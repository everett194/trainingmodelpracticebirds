"""
data/column_aliases.py
------------------------
Alias-based column resolution for FAA Wildlife Strike Database exports.

The FAA database has been exported under slightly different column
naming conventions over the years (and users may rename columns when
saving from Excel). Rather than guessing positionally or fuzzy-matching,
we maintain an explicit table of known aliases per canonical field.

Design rule: if a raw column matches aliases for MORE THAN ONE canonical
field, that is an error, not a "best guess" - see resolve_columns().
Silent guessing between ambiguous columns is exactly what this module is
built to avoid (see project requirements, Section 6).
"""

from __future__ import annotations

from birdstrikegeo.schemas.faa import FAA_SCHEMA

# canonical_name -> tuple of raw column names (case/whitespace-insensitive)
# known to represent that field across different FAA export vintages.
# Extend this table, rather than the ingestion code, when a new export
# variant is encountered.
FAA_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "record_id": ("INDEX_NR", "RECORD_ID", "INDEX NR", "ID"),
    "incident_date": ("INCIDENT_DATE", "INCIDENT DATE", "DATE"),
    "incident_time_local": ("TIME", "TIME_OF_DAY", "INCIDENT_TIME", "LOCAL_TIME"),
    "airport_id": ("AIRPORT_ID", "AIRPORTID", "AIRPORT CODE"),
    "airport_name": ("AIRPORT", "AIRPORT_NAME", "AIRPORT NAME"),
    "state": ("STATE",),
    "latitude": ("LATITUDE", "LAT"),
    "longitude": ("LONGITUDE", "LON", "LONG"),
    "phase_of_flight": ("PHASE_OF_FLIGHT", "PHASE OF FLIGHT", "FLT_PHASE", "PHASE"),
    "height_agl_ft": ("HEIGHT", "HEIGHT_AGL", "HEIGHT (FT)"),
    "speed_ias_knots": ("SPEED", "IAS", "SPEED_IAS", "SPEED (KTS)"),
    "aircraft_make": ("AC_MAKE", "AIRCRAFT_MAKE", "MAKE"),
    "aircraft_model": ("AC_MODEL", "AIRCRAFT_MODEL", "MODEL"),
    "aircraft_type": ("AC_CLASS", "AIRCRAFT_TYPE", "AC_TYPE"),
    "aircraft_mass_class": ("AC_MASS", "AIRCRAFT_MASS", "MASS_CLASS"),
    "engine_type": ("ENG_TYPE", "ENGINE_TYPE"),
    "engine_count": ("NUM_ENGS", "ENGINE_COUNT", "NR_ENGINES"),
    "sky_condition": ("SKY", "SKY_CONDITION"),
    "precipitation": ("PRECIPITATION", "PRECIP"),
    "species_common_name": ("SPECIES", "SPECIES_NAME", "SPECIES_COMMON_NAME"),
    "species_scientific_name": ("SPECIES_ID", "SCIENTIFIC_NAME", "SPECIES_SCIENTIFIC_NAME"),
    "wildlife_size": ("SIZE", "WILDLIFE_SIZE"),
    "number_seen": ("NUM_SEEN", "NUMBER_SEEN"),
    "number_struck": ("NUM_STRUCK", "NUMBER_STRUCK"),
    "damage_flag": ("DAMAGE_IND", "DAMAGE_FLAG", "INDICATED_DAMAGE"),
    "damage_level": ("DAM_LEVEL", "DAMAGE_LEVEL"),
    "effect_on_flight": ("EFFECT", "EFFECT_ON_FLIGHT", "EFFECT_FLT"),
    "parts_struck": ("STR_PARTS", "PARTS_STRUCK"),
    "parts_damaged": ("DAM_PARTS", "PARTS_DAMAGED"),
    "repair_cost": ("COST_REPAIRS", "REPAIR_COST"),
    "other_cost": ("COST_OTHER", "OTHER_COST"),
    "injuries": ("NR_INJURIES", "INJURIES"),
    "fatalities": ("NR_FATALITIES", "FATALITIES"),
    "narrative": ("REMARKS", "NARRATIVE", "REMARKS_TEXT"),
}


def _normalize(name: str) -> str:
    return " ".join(str(name).strip().upper().replace("_", " ").split())


def resolve_columns(raw_columns: list[str]) -> dict[str, str]:
    """
    Given the raw column names present in an ingested file, return a
    mapping {canonical_name: raw_column_name} for every canonical field
    that could be confidently resolved.

    Raises ValueError if a raw column's normalized name matches aliases
    registered under more than one canonical field (a genuinely ambiguous
    export we should not silently guess about), or if two raw columns
    both resolve to the same canonical field.
    """
    normalized_to_raw = {_normalize(c): c for c in raw_columns}

    # Build normalized_alias -> set of canonical fields it could mean.
    alias_to_canonicals: dict[str, set[str]] = {}
    for canonical, aliases in FAA_COLUMN_ALIASES.items():
        for alias in (canonical, *aliases):
            alias_to_canonicals.setdefault(_normalize(alias), set()).add(canonical)

    resolved: dict[str, str] = {}
    for normalized_raw, raw_col in normalized_to_raw.items():
        canonicals = alias_to_canonicals.get(normalized_raw)
        if not canonicals:
            continue  # unrecognized column - left out, not guessed. Preserved in the audit table instead.
        if len(canonicals) > 1:
            raise ValueError(
                f"Ambiguous FAA column '{raw_col}': could map to any of "
                f"{sorted(canonicals)}. Refusing to guess - update "
                f"FAA_COLUMN_ALIASES to disambiguate."
            )
        canonical = next(iter(canonicals))
        if canonical in resolved:
            raise ValueError(
                f"Both '{resolved[canonical]}' and '{raw_col}' resolve to the "
                f"same canonical field '{canonical}'. Refusing to guess which "
                f"one is correct - rename one column or update FAA_COLUMN_ALIASES."
            )
        resolved[canonical] = raw_col

    return resolved


def missing_required_columns(resolved: dict[str, str]) -> list[str]:
    """Canonical FAA_SCHEMA fields that resolve_columns() could not find."""
    return [c for c in FAA_SCHEMA if c not in resolved]
