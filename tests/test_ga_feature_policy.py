"""
tests/test_ga_feature_policy.py
-----------------------------------
Enforces birdstrikegeo.ga.feature_policy: a forbidden (outcome or
administrative) column must never pass assert_ga_no_leakage(), and an
encounter-only column must never pass it under reduced_preflight mode.
Also checks the real feature-building functions actually stay inside
policy, so a future edit that adds a forbidden column to
build_operational_core_features/build_reduced_preflight_features fails
loudly here rather than silently shipping.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from birdstrikegeo.ga.feature_policy import (  # noqa: E402
    ENCOUNTER_ONLY_RAW_COLUMNS,
    FORBIDDEN_COLUMNS,
    LeakageError,
    OPERATIONAL_CORE_RAW_COLUMNS,
    REDUCED_PREFLIGHT_RAW_COLUMNS,
    assert_ga_no_leakage,
)
from birdstrikegeo.ga.features import build_operational_core_features, build_reduced_preflight_features  # noqa: E402


@pytest.mark.parametrize("forbidden_col", sorted(FORBIDDEN_COLUMNS))
def test_forbidden_column_always_rejected(forbidden_col):
    with pytest.raises(LeakageError):
        assert_ga_no_leakage(OPERATIONAL_CORE_RAW_COLUMNS | {forbidden_col}, mode="operational_core")
    with pytest.raises(LeakageError):
        assert_ga_no_leakage(REDUCED_PREFLIGHT_RAW_COLUMNS | {forbidden_col}, mode="reduced_preflight")


@pytest.mark.parametrize("encounter_col", sorted(ENCOUNTER_ONLY_RAW_COLUMNS))
def test_encounter_only_column_rejected_in_reduced_preflight_mode(encounter_col):
    with pytest.raises(LeakageError):
        assert_ga_no_leakage(REDUCED_PREFLIGHT_RAW_COLUMNS | {encounter_col}, mode="reduced_preflight")


def test_operational_core_mode_accepts_its_own_columns():
    assert_ga_no_leakage(OPERATIONAL_CORE_RAW_COLUMNS, mode="operational_core")


def test_reduced_preflight_mode_accepts_its_own_columns():
    assert_ga_no_leakage(REDUCED_PREFLIGHT_RAW_COLUMNS, mode="reduced_preflight")


def _fake_ga_dataframe(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AC_MASS": ["1"] * n, "TYPE_ENG": ["A"] * n, "NUM_ENGS": ["1"] * n,
            "ENG_1_POS": ["7"] * n, "ENG_2_POS": [None] * n, "AIRCRAFT": ["C-172"] * n,
            "STATE": ["TX"] * n, "FAAREGION": ["ASW"] * n, "SKY": ["NO CLOUD"] * n,
            "PRECIPITATION": [None] * n, "INCIDENT_MONTH": ["6"] * n,
            "PHASE_OF_FLIGHT": ["APPROACH"] * n, "HEIGHT": ["500"] * n, "SPEED": ["80"] * n,
            "TIME_OF_DAY": ["DAY"] * n, "WARNED": ["NO"] * n,
            # forbidden columns present in the raw frame, as they would be
            # in a real ingest - these must NOT leak into either feature set.
            "INDICATED_DAMAGE": ["1"] * n, "DAMAGE_LEVEL": ["S"] * n, "SPECIES": ["MOURNING DOVE"] * n,
        }
    )


def test_operational_core_feature_frame_excludes_forbidden_columns():
    features = build_operational_core_features(_fake_ga_dataframe()).features
    assert not (set(features.columns) & FORBIDDEN_COLUMNS)
    assert "damage_binary" not in features.columns
    assert "SPECIES" not in features.columns


def test_reduced_preflight_feature_frame_excludes_encounter_columns():
    fs = build_reduced_preflight_features(_fake_ga_dataframe())
    assert "phase_of_flight" not in fs.features.columns
    assert "height_agl_ft" not in fs.features.columns
    assert "speed_ias_knots" not in fs.features.columns
    assert "light_condition" not in fs.features.columns
    assert "bird_warned" not in fs.features.columns
