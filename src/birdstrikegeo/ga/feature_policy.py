"""
birdstrikegeo.ga.feature_policy
---------------------------------
The single source of truth for which raw FAA columns may reach the GA
conditional-damage model's feature matrix, and which may never. Every
feature-building function in birdstrikegeo.ga.features must build its
output columns from ALLOWED_* below and must call assert_ga_no_leakage()
on the final column list before returning - tests/test_ga_feature_policy.py
enforces this.

Two feature modes (see project requirements):
  - operational_core: everything plausibly known before/during a strike
    encounter, including the actual phase/height/speed of THIS encounter.
  - reduced_preflight: only what would be known before an encounter even
    starts (aircraft + airport/region + calendar + forecast-style
    weather) - no actual encounter phase/height/speed.

FORBIDDEN_COLUMNS is deliberately broader than
birdstrikegeo.schemas.faa.FORBIDDEN_LEAKAGE_COLUMNS: it also excludes
fields that describe the ACTUAL strike (species, size, counts struck/seen)
which are outcome-adjacent - unknowable before a strike happens, not just
"outcome" in the damage-severity sense. See README/MODEL_CARD for why.
"""

from __future__ import annotations

# Raw FAA columns that describe what happened AS A RESULT of the strike,
# or that are pure reporting/administrative metadata. Never allowed as
# model inputs, in either feature mode.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        # Outcome / damage description
        "INDICATED_DAMAGE", "DAMAGE_LEVEL", "damage_binary", "damage_target_status",
        "damage_level_inconsistent",
        "STR_RAD", "DAM_RAD", "STR_WINDSHLD", "DAM_WINDSHLD", "STR_NOSE", "DAM_NOSE",
        "STR_ENG1", "DAM_ENG1", "ING_ENG1", "STR_ENG2", "DAM_ENG2", "ING_ENG2",
        "STR_ENG3", "DAM_ENG3", "ING_ENG3", "STR_ENG4", "DAM_ENG4", "ING_ENG4",
        "INGESTED_OTHER", "STR_PROP", "DAM_PROP", "STR_WING_ROT", "DAM_WING_ROT",
        "STR_FUSE", "DAM_FUSE", "STR_LG", "DAM_LG", "STR_TAIL", "DAM_TAIL",
        "STR_LGHTS", "DAM_LGHTS", "STR_OTHER", "DAM_OTHER", "OTHER_SPECIFY",
        "EFFECT", "EFFECT_OTHER",
        # Cost / operational consequence
        "COST_REPAIRS", "COST_OTHER", "COST_REPAIRS_INFL_ADJ", "COST_OTHER_INFL_ADJ", "AOS",
        "NR_INJURIES", "NR_FATALITIES",
        # The actual struck wildlife - unavailable before a strike happens.
        # A future model may use ADVANCE Trektellen-derived local
        # abundance/composition estimates, but never the realized species.
        "SPECIES_ID", "SPECIES", "OUT_OF_RANGE_SPECIES", "SIZE", "BIRD_BAND_NUMBER",
        "NUM_SEEN", "NUM_STRUCK", "REMAINS_COLLECTED", "REMAINS_SENT",
        # Reporting / administrative metadata, never predictive
        "REMARKS", "COMMENTS", "REPORTED_NAME", "REPORTED_TITLE", "REPORTED_DATE",
        "SOURCE", "PERSON", "LUPDATE", "TRANSFER", "IMAGE", "INDEX_NR", "REG", "FLT",
        "OPID", "ENROUTE_STATE", "LOCATION", "RUNWAY", "DISTANCE",
    }
)

# Columns used for population filtering / splitting, not as model inputs
# in either mode (kept off the feature matrix so the model can't just
# memorize "this is the filter"); also excluded to avoid an accidental
# near-duplicate of aircraft_mass_class.
NON_FEATURE_CONTROL_COLUMNS: frozenset[str] = frozenset(
    {"OPERATOR", "AC_CLASS", "INCIDENT_DATE", "INCIDENT_YEAR", "LATITUDE", "LONGITUDE"}
)

# Raw column -> engineered feature name, for encounter-specific inputs
# (only ever allowed in operational_core mode).
ENCOUNTER_ONLY_RAW_COLUMNS: frozenset[str] = frozenset(
    {"PHASE_OF_FLIGHT", "HEIGHT", "SPEED", "TIME_OF_DAY", "WARNED"}
)

# Raw column -> engineered feature name, allowed in BOTH modes (known
# before an encounter: aircraft identity, airport/region, calendar,
# forecast-style sky/precipitation).
PREFLIGHT_KNOWN_RAW_COLUMNS: frozenset[str] = frozenset(
    {
        "AC_MASS", "TYPE_ENG", "NUM_ENGS", "ENG_1_POS", "ENG_2_POS", "ENG_3_POS", "ENG_4_POS",
        "AIRCRAFT", "AIRPORT_ID", "STATE", "FAAREGION", "SKY", "PRECIPITATION",
        "INCIDENT_MONTH",
    }
)

OPERATIONAL_CORE_RAW_COLUMNS: frozenset[str] = PREFLIGHT_KNOWN_RAW_COLUMNS | ENCOUNTER_ONLY_RAW_COLUMNS
REDUCED_PREFLIGHT_RAW_COLUMNS: frozenset[str] = PREFLIGHT_KNOWN_RAW_COLUMNS


class LeakageError(ValueError):
    """Raised when a forbidden or encounter-only-in-wrong-mode column reaches a GA feature matrix."""


def assert_ga_no_leakage(columns, mode: str) -> None:
    """
    columns: iterable of RAW column names a feature-building function is
    about to read from (call this BEFORE deriving engineered feature
    names, since engineered names won't match FORBIDDEN_COLUMNS by
    construction and that would defeat the check).
    mode: "operational_core" | "reduced_preflight".
    """
    if mode not in ("operational_core", "reduced_preflight"):
        raise ValueError(f"Unknown feature mode: {mode}")

    columns = set(columns)
    forbidden_hit = columns & FORBIDDEN_COLUMNS
    if forbidden_hit:
        raise LeakageError(
            f"Forbidden (outcome/administrative) columns requested as GA model inputs: "
            f"{sorted(forbidden_hit)}"
        )

    allowed = OPERATIONAL_CORE_RAW_COLUMNS if mode == "operational_core" else REDUCED_PREFLIGHT_RAW_COLUMNS
    disallowed_for_mode = (columns - allowed) - NON_FEATURE_CONTROL_COLUMNS
    if disallowed_for_mode:
        raise LeakageError(
            f"Columns not permitted in '{mode}' feature mode: {sorted(disallowed_for_mode)}. "
            f"Encounter-specific fields (phase/height/speed/light/warned) are only allowed in "
            f"'operational_core' mode."
        )
