"""
birdstrikegeo.ga.presets
---------------------------
Named GA damage-prediction scenarios for scripts/predict_damage.py. Every
field name matches birdstrikegeo.ga.features.build_operational_core_features
output columns exactly, and every value is a category or numeric range
actually observed in the confirmed business/private fixed-wing GA training
population - see reports/latest/ga_population_summary.json for the source
value counts. FAA's phase-of-flight taxonomy has no literal "cruise" state;
"EN ROUTE" is used for the cruise-phase presets.

precipitation=None represents "not reported" (missing), NOT "no
precipitation" - this export rarely records an explicit none/no-precip
value, so a "clear" scenario is honestly represented as unreported
precipitation with a clear sky, not fabricated as a seen-but-absent
category. See MODEL_CARD-equivalent notes in the receipt.
"""

from __future__ import annotations

PRESETS: dict[str, dict] = {
    "c172_day_clear_takeoff": {
        "description": "Cessna 172, daytime, clear, takeoff roll",
        "aircraft_type": "C-172", "aircraft_mass_class": "1", "engine_type": "A", "engine_count": 1.0,
        "engine_position_primary": "7", "engine_position_secondary": "NONE",
        "state": "TX", "faa_region": "ASW", "sky_condition": "NO CLOUD", "precipitation": None,
        "month": 6, "phase_of_flight": "TAKE-OFF RUN", "height_agl_ft": 0.0, "speed_ias_knots": 65.0,
        "light_condition": "DAY", "bird_warned": "NO",
    },
    "c172_night_clear_approach": {
        "description": "Cessna 172, nighttime, clear, approach",
        "aircraft_type": "C-172", "aircraft_mass_class": "1", "engine_type": "A", "engine_count": 1.0,
        "engine_position_primary": "7", "engine_position_secondary": "NONE",
        "state": "TX", "faa_region": "ASW", "sky_condition": "NO CLOUD", "precipitation": None,
        "month": 10, "phase_of_flight": "APPROACH", "height_agl_ft": 500.0, "speed_ias_knots": 75.0,
        "light_condition": "NIGHT", "bird_warned": "NO",
    },
    "pa28_day_clear_approach": {
        "description": "Piper PA-28, daytime, clear, approach",
        "aircraft_type": "PA-28", "aircraft_mass_class": "1", "engine_type": "A", "engine_count": 1.0,
        "engine_position_primary": "7", "engine_position_secondary": "NONE",
        "state": "CA", "faa_region": "AWP", "sky_condition": "NO CLOUD", "precipitation": None,
        "month": 6, "phase_of_flight": "APPROACH", "height_agl_ft": 500.0, "speed_ias_knots": 80.0,
        "light_condition": "DAY", "bird_warned": "NO",
    },
    "cirrus_day_clear_cruise": {
        "description": "Cirrus SR20/22, daytime, clear, en route (cruise)",
        "aircraft_type": "CIRRUS SR 20/22", "aircraft_mass_class": "1", "engine_type": "A", "engine_count": 1.0,
        "engine_position_primary": "7", "engine_position_secondary": "NONE",
        "state": "FL", "faa_region": "ASO", "sky_condition": "NO CLOUD", "precipitation": None,
        "month": 6, "phase_of_flight": "EN ROUTE", "height_agl_ft": 6500.0, "speed_ias_knots": 170.0,
        "light_condition": "DAY", "bird_warned": "UNKNOWN",
    },
    "king_air_day_clear_approach": {
        "description": "Beechcraft King Air 200, daytime, clear, approach",
        "aircraft_type": "BE-200 KING", "aircraft_mass_class": "2", "engine_type": "C", "engine_count": 2.0,
        "engine_position_primary": "4", "engine_position_secondary": "4",
        "state": "CO", "faa_region": "ANM", "sky_condition": "NO CLOUD", "precipitation": None,
        "month": 6, "phase_of_flight": "APPROACH", "height_agl_ft": 1000.0, "speed_ias_knots": 140.0,
        "light_condition": "DAY", "bird_warned": "NO",
    },
    "business_jet_day_clear_approach": {
        "description": "Bombardier Challenger 300 (business jet), daytime, clear, approach",
        "aircraft_type": "CL-300", "aircraft_mass_class": "3", "engine_type": "D", "engine_count": 2.0,
        "engine_position_primary": "5", "engine_position_secondary": "5",
        "state": "NJ", "faa_region": "AEA", "sky_condition": "NO CLOUD", "precipitation": None,
        "month": 6, "phase_of_flight": "APPROACH", "height_agl_ft": 1500.0, "speed_ias_knots": 160.0,
        "light_condition": "DAY", "bird_warned": "NO",
    },
}

REQUIRED_FIELDS = [
    "aircraft_type", "aircraft_mass_class", "engine_type", "engine_count", "engine_position_primary",
    "engine_position_secondary", "state", "faa_region", "sky_condition", "precipitation", "month",
    "phase_of_flight", "height_agl_ft", "speed_ias_knots", "light_condition", "bird_warned",
]

NUMERIC_FIELDS = {"engine_count", "month", "height_agl_ft", "speed_ias_knots"}


class ScenarioValidationError(ValueError):
    pass


def validate_scenario(scenario: dict, seen_categories: dict, numeric_ranges: dict) -> None:
    """
    seen_categories: {column: set of raw string values observed in the
    TRAINING split} (built at training time, saved in the model
    artifact). numeric_ranges: {column: (min, max)} observed in training.
    Raises ScenarioValidationError on the first problem found, with a
    clear, specific message.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in scenario]
    if missing:
        raise ScenarioValidationError(f"Missing required field(s): {missing}")

    unknown = [f for f in scenario if f not in REQUIRED_FIELDS]
    if unknown:
        raise ScenarioValidationError(f"Unrecognized field(s): {unknown}")

    for field in REQUIRED_FIELDS:
        value = scenario[field]
        if field in NUMERIC_FIELDS:
            if value is None:
                raise ScenarioValidationError(f"'{field}' is required and cannot be null.")
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise ScenarioValidationError(f"'{field}' must be numeric, got {scenario[field]!r}.")
            lo, hi = numeric_ranges.get(field, (float("-inf"), float("inf")))
            if not (lo <= value <= hi):
                raise ScenarioValidationError(
                    f"'{field}'={value} is outside the training data's observed range [{lo}, {hi}]."
                )
        else:
            if value is None:
                continue  # missing/not-reported is a valid, seen state for several categorical fields
            allowed = seen_categories.get(field)
            if allowed is not None and value not in allowed:
                raise ScenarioValidationError(
                    f"'{field}'={value!r} was never seen in training data. Known values: {sorted(allowed)}"
                )


def list_presets() -> list[str]:
    return sorted(PRESETS)


def get_preset(name: str) -> dict:
    if name not in PRESETS:
        raise ScenarioValidationError(f"Unknown preset '{name}'. Available: {list_presets()}")
    return dict(PRESETS[name])
