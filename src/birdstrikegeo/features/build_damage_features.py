"""
features/build_damage_features.py
------------------------------------
Task A (damage prediction) feature/target construction.

Two separate, deliberately explicit steps:

1. make_damage_target(): reads ONLY damage_flag/damage_level to build a
   binary target. Rows where the outcome isn't explicit get target=NA
   and are excluded from supervised training (never silently dropped -
   the caller is expected to report them, see
   birdstrikegeo.data.quality_report).

2. build_feature_frame(): builds the model-input feature matrix from
   everything EXCEPT the forbidden/outcome-describing columns, adds
   time/season features, and asserts no leakage before returning.

Categorical encoding (one-hot / rare-category bucketing) is intentionally
NOT done here - it belongs to birdstrikegeo.training.train_damage, which
fits it on the training split only (see splits.py). This module only
ever produces a clean, un-encoded feature frame.
"""

from __future__ import annotations

import pandas as pd

from birdstrikegeo.features.leakage import assert_no_leakage
from birdstrikegeo.geo.feature_engineering import build_time_features
from birdstrikegeo.schemas.faa import ALLOWED_FEATURE_COLUMNS, FORBIDDEN_LEAKAGE_COLUMNS

# record_id is an identifier, not a predictive feature; narrative is
# forbidden already. airport_name is dropped in favor of airport_id
# (avoids duplicating essentially the same categorical information).
NON_FEATURE_IDENTIFIER_COLUMNS = ("record_id", "airport_name")

DEFAULT_TARGET_DEFINITION = {
    "positive_values": ["Y", "YES", "TRUE", "1", "DAMAGE"],
    "negative_values": ["N", "NO", "FALSE", "0", "NONE", "N/A - NO DAMAGE"],
}


def make_damage_target(df: pd.DataFrame, target_definition: dict | None = None) -> pd.DataFrame:
    """
    Adds two columns to a copy of df:
      - "damage": pandas nullable Int64, 1 / 0 / <NA>
      - "damage_target_status": "explicit_positive" | "explicit_negative"
        | "missing_or_ambiguous" (for quality reporting)

    Only damage_flag is consulted for the yes/no decision; damage_level
    being non-empty is treated as corroborating evidence but a blank
    damage_flag with a non-blank damage_level (a contradictory record) is
    treated as ambiguous, not guessed.
    """
    target_definition = target_definition or DEFAULT_TARGET_DEFINITION
    positive = {str(v).strip().upper() for v in target_definition["positive_values"]}
    negative = {str(v).strip().upper() for v in target_definition["negative_values"]}

    df = df.copy()
    flag = df["damage_flag"].astype("string").str.strip().str.upper()

    is_positive = flag.isin(positive)
    is_negative = flag.isin(negative)
    is_contradictory = is_positive & is_negative  # only possible if positive/negative sets overlap - defensive

    status = pd.Series("missing_or_ambiguous", index=df.index, dtype="object")
    status[is_positive & ~is_contradictory] = "explicit_positive"
    status[is_negative & ~is_contradictory] = "explicit_negative"

    damage = pd.Series(pd.NA, index=df.index, dtype="Int64")
    damage[status == "explicit_positive"] = 1
    damage[status == "explicit_negative"] = 0

    df["damage"] = damage
    df["damage_target_status"] = status
    return df


def select_allowed_feature_columns(df: pd.DataFrame) -> list[str]:
    """Conceptual FAA columns allowed as model inputs (schema columns
    minus forbidden/outcome columns minus non-feature identifiers),
    restricted to whatever columns are actually present in df."""
    allowed = [c for c in ALLOWED_FEATURE_COLUMNS if c not in NON_FEATURE_IDENTIFIER_COLUMNS]
    return [c for c in allowed if c in df.columns]


def build_feature_frame(df_with_target: pd.DataFrame) -> pd.DataFrame:
    """
    df_with_target: output of make_damage_target(). Returns a feature
    frame (no target column) with time/season features added, restricted
    to eligible rows would be the CALLER's job (see splits.py /
    train_damage.py) - this function operates on whatever rows it's given.

    Raises LeakageError if a forbidden column somehow ends up in the
    output (defensive check - should be structurally impossible given
    select_allowed_feature_columns, but this is the last line of defense
    tests/test_leakage.py checks against).
    """
    feature_cols = select_allowed_feature_columns(df_with_target)
    features = df_with_target[feature_cols].copy()

    if "incident_date" in features.columns:
        # incident_time_local is a separate HH:MM string field; combine
        # for time-of-day features where possible, else fall back to the
        # date alone (hour_local will be 0 / midnight in that case - the
        # missing time-of-day is NOT distinguishable from true midnight
        # with this fallback, a known limitation documented in MODEL_CARD.md).
        combined = pd.to_datetime(features["incident_date"], errors="coerce")
        if "incident_time_local" in features.columns:
            time_str = features["incident_time_local"].astype("string").fillna("00:00")
            parsed_time = pd.to_timedelta(time_str.str.strip().replace("", "00:00") + ":00", errors="coerce")
            combined = combined + parsed_time.fillna(pd.Timedelta(0))
        features["_incident_timestamp"] = combined
        features = build_time_features(features, "_incident_timestamp")
        features = features.drop(columns=["_incident_timestamp"])

    assert_no_leakage(features.columns, extra_forbidden=FORBIDDEN_LEAKAGE_COLUMNS)
    return features


def fit_rare_categories(train_df: pd.DataFrame, categorical_columns: list[str], threshold: int) -> dict[str, set]:
    """
    Fit-on-train-only step: for each categorical column, records which
    values appear at least `threshold` times in the TRAINING data.
    Anything else (in train, validation, or test) gets bucketed into
    "rare_other" by apply_rare_categories() - this keeps the one-hot
    feature space bounded without ever looking at validation/test data
    to decide what counts as "rare".
    """
    frequent_values: dict[str, set] = {}
    for col in categorical_columns:
        counts = train_df[col].astype("string").value_counts()
        frequent_values[col] = set(counts[counts >= threshold].index)
    return frequent_values


def apply_rare_categories(df: pd.DataFrame, frequent_values: dict[str, set]) -> pd.DataFrame:
    df = df.copy()
    for col, allowed in frequent_values.items():
        if col not in df.columns:
            continue
        as_string = df[col].astype("string")
        df[col] = as_string.where(as_string.isin(allowed) | as_string.isna(), "rare_other")
    return df
