"""
birdstrikegeo.hazard.severity_target
--------------------------------------
Builds an ordinal damage-severity regression target (0-3) on top of the
damage_binary/damage_target_status/damage_level_inconsistent columns
already produced by birdstrikegeo.ga.data.build_damage_binary_target().

Same "don't guess ambiguous evidence" rule as that module: a row only
gets a severity score when INDICATED_DAMAGE and DAMAGE_LEVEL agree.
Anything flagged damage_level_inconsistent, or with unresolved
damage_target_status, is left <NA> rather than defaulted to a value.
"""

from __future__ import annotations

import pandas as pd

# DAMAGE_LEVEL -> ordinal severity, for explicit_positive/consistent rows
# only. "N"/"" map to 0 via the explicit_negative branch, not this table.
_LEVEL_TO_SEVERITY: dict[str, float] = {
    "M": 1.0,
    "M?": 1.0,
    "S": 2.0,
    "D": 3.0,
}


def build_damage_severity_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `damage_severity_score` (nullable Float64: 0.0-3.0 or <NA>) to a
    copy of df. Requires damage_target_status and
    damage_level_inconsistent columns already present.
    """
    df = df.copy()
    status = df["damage_target_status"]
    inconsistent = df["damage_level_inconsistent"]
    level = df["DAMAGE_LEVEL"].astype("string").fillna("")

    consistent = ~inconsistent
    is_negative = (status == "explicit_negative") & consistent
    is_positive = (status == "explicit_positive") & consistent

    severity = pd.Series(pd.NA, index=df.index, dtype="Float64")
    severity[is_negative] = 0.0
    severity[is_positive] = level[is_positive].map(_LEVEL_TO_SEVERITY)

    df["damage_severity_score"] = severity
    return df
