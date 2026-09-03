"""
birdstrikegeo.ga.features
---------------------------
Builds the two GA feature frames (operational_core / reduced_preflight)
from the population-filtered, target-labeled dataframe produced by
birdstrikegeo.ga.data, and the chronological + geographic splits used to
evaluate them.

Every column this module reads is checked against
birdstrikegeo.ga.feature_policy.assert_ga_no_leakage() before any
engineered feature is derived from it - see that module for what's
forbidden and why.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from birdstrikegeo.ga.feature_policy import (
    ENCOUNTER_ONLY_RAW_COLUMNS,
    PREFLIGHT_KNOWN_RAW_COLUMNS,
    assert_ga_no_leakage,
)

# Standard meteorological (Northern-hemisphere) seasons, from month alone.
# An approximation - the FGN (foreign) FAAREGION rows may lie in the
# southern hemisphere, a documented limitation (see MODEL_CARD-equivalent
# receipt notes), not silently corrected for.
_MONTH_TO_SEASON = {
    12: "WINTER", 1: "WINTER", 2: "WINTER",
    3: "SPRING", 4: "SPRING", 5: "SPRING",
    6: "SUMMER", 7: "SUMMER", 8: "SUMMER",
    9: "FALL", 10: "FALL", 11: "FALL",
}


@dataclass
class FeatureSet:
    mode: str
    features: pd.DataFrame  # engineered feature columns only
    numeric_columns: list[str]
    categorical_columns: list[str]


def _cyclic(series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
    radians = 2 * np.pi * series.astype(float) / period
    return np.sin(radians), np.cos(radians)


def _shared_preflight_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["aircraft_type"] = df.get("AIRCRAFT")
    out["aircraft_mass_class"] = df.get("AC_MASS")
    out["engine_type"] = df.get("TYPE_ENG")
    out["engine_count"] = pd.to_numeric(df.get("NUM_ENGS"), errors="coerce")
    out["engine_position_primary"] = df.get("ENG_1_POS")
    out["engine_position_secondary"] = df.get("ENG_2_POS").astype("string").fillna("NONE") if "ENG_2_POS" in df else "NONE"
    out["state"] = df.get("STATE")
    out["faa_region"] = df.get("FAAREGION")
    out["sky_condition"] = df.get("SKY")
    out["precipitation"] = df.get("PRECIPITATION")

    month = pd.to_numeric(df.get("INCIDENT_MONTH"), errors="coerce")
    out["month"] = month
    sin, cos = _cyclic(month.fillna(6), 12)
    out["month_sin"] = sin
    out["month_cos"] = cos
    out["season"] = month.map(_MONTH_TO_SEASON).astype("object")
    out.loc[month.isna(), "season"] = pd.NA
    return out


_SHARED_NUMERIC = ["engine_count", "month", "month_sin", "month_cos"]
_SHARED_CATEGORICAL = [
    "aircraft_type", "aircraft_mass_class", "engine_type", "engine_position_primary",
    "engine_position_secondary", "state", "faa_region", "sky_condition", "precipitation", "season",
]

_ENCOUNTER_NUMERIC = ["height_agl_ft", "speed_ias_knots"]
_ENCOUNTER_CATEGORICAL = ["phase_of_flight", "light_condition", "bird_warned"]


def build_operational_core_features(df: pd.DataFrame) -> FeatureSet:
    """
    Encounter-conditional feature set: everything plausibly known before
    or during the modeled encounter, including the actual phase, height,
    speed, light condition, and warning status of THIS encounter. Only
    valid for the "given a strike occurs during this described encounter"
    prediction moment - never for a pure preflight forecast (use
    build_reduced_preflight_features for that).
    """
    used_raw = PREFLIGHT_KNOWN_RAW_COLUMNS | ENCOUNTER_ONLY_RAW_COLUMNS
    assert_ga_no_leakage(used_raw, mode="operational_core")

    features = _shared_preflight_features(df)
    features["phase_of_flight"] = df.get("PHASE_OF_FLIGHT")
    features["height_agl_ft"] = pd.to_numeric(df.get("HEIGHT"), errors="coerce")
    features["speed_ias_knots"] = pd.to_numeric(df.get("SPEED"), errors="coerce")
    features["light_condition"] = df.get("TIME_OF_DAY")
    # WARNED reflects whether the PILOT reported having been warned -
    # this is real information available at/before the encounter, but is
    # self-reported and may correlate with reporting thoroughness rather
    # than true warning system performance. Documented caution, not
    # excluded - see receipt "known limitations".
    features["bird_warned"] = df.get("WARNED")

    return FeatureSet(
        mode="operational_core", features=features,
        numeric_columns=_SHARED_NUMERIC + _ENCOUNTER_NUMERIC,
        categorical_columns=_SHARED_CATEGORICAL + _ENCOUNTER_CATEGORICAL,
    )


def build_reduced_preflight_features(df: pd.DataFrame) -> FeatureSet:
    """
    Preflight-only feature set: aircraft characteristics, airport/region,
    calendar, and forecast-style sky/precipitation - excludes the actual
    encounter's phase, height, speed, light condition, and warning status,
    none of which would be known before an encounter starts.
    """
    assert_ga_no_leakage(PREFLIGHT_KNOWN_RAW_COLUMNS, mode="reduced_preflight")
    features = _shared_preflight_features(df)
    return FeatureSet(
        mode="reduced_preflight", features=features,
        numeric_columns=list(_SHARED_NUMERIC), categorical_columns=list(_SHARED_CATEGORICAL),
    )


FEATURE_BUILDERS = {
    "operational_core": build_operational_core_features,
    "reduced_preflight": build_reduced_preflight_features,
}


@dataclass
class YearSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end_year: int
    validation_end_year: int


def chronological_split_by_year(df: pd.DataFrame, year_column: str, train_end_year: int, validation_end_year: int) -> YearSplit:
    """
    Splits by explicit calendar-year boundaries (inclusive) rather than a
    row-count quantile, so the split lines up with human-readable
    "through year X" reporting. Rows with a missing year are dropped (a
    GA record always has INCIDENT_YEAR in this export - see
    ga_population_summary.json for the count if that ever changes).
    """
    year = pd.to_numeric(df[year_column], errors="coerce")
    working = df[year.notna()].copy()
    year = year[year.notna()]

    train = working[year <= train_end_year]
    validation = working[(year > train_end_year) & (year <= validation_end_year)]
    test = working[year > validation_end_year]

    return YearSplit(train=train, validation=validation, test=test,
                      train_end_year=train_end_year, validation_end_year=validation_end_year)
