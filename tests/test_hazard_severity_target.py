import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from birdstrikegeo.hazard.severity_target import build_damage_severity_target  # noqa: E402


def _base_df(rows: list[dict]) -> pd.DataFrame:
    # Minimal frame shaped like the output of
    # birdstrikegeo.ga.data.build_damage_binary_target: one row per
    # strike, with damage_target_status / damage_level_inconsistent /
    # DAMAGE_LEVEL already populated.
    return pd.DataFrame(rows)


def test_explicit_negative_consistent_row_gets_zero_severity():
    df = _base_df([
        {"damage_target_status": "explicit_negative", "damage_level_inconsistent": False, "DAMAGE_LEVEL": "N"},
    ])
    result = build_damage_severity_target(df)
    assert result["damage_severity_score"].tolist() == [0.0]


def test_explicit_positive_minor_level_gets_severity_one():
    df = _base_df([
        {"damage_target_status": "explicit_positive", "damage_level_inconsistent": False, "DAMAGE_LEVEL": "M"},
        {"damage_target_status": "explicit_positive", "damage_level_inconsistent": False, "DAMAGE_LEVEL": "M?"},
    ])
    result = build_damage_severity_target(df)
    assert result["damage_severity_score"].tolist() == [1.0, 1.0]


def test_explicit_positive_substantial_level_gets_severity_two():
    df = _base_df([
        {"damage_target_status": "explicit_positive", "damage_level_inconsistent": False, "DAMAGE_LEVEL": "S"},
    ])
    result = build_damage_severity_target(df)
    assert result["damage_severity_score"].tolist() == [2.0]


def test_explicit_positive_destroyed_level_gets_severity_three():
    df = _base_df([
        {"damage_target_status": "explicit_positive", "damage_level_inconsistent": False, "DAMAGE_LEVEL": "D"},
    ])
    result = build_damage_severity_target(df)
    assert result["damage_severity_score"].tolist() == [3.0]


def test_missing_or_ambiguous_status_is_excluded_not_guessed():
    df = _base_df([
        {"damage_target_status": "missing_or_ambiguous", "damage_level_inconsistent": False, "DAMAGE_LEVEL": ""},
    ])
    result = build_damage_severity_target(df)
    assert result["damage_severity_score"].isna().tolist() == [True]


def test_inconsistent_row_is_excluded_even_if_status_looks_decisive():
    df = _base_df([
        {"damage_target_status": "explicit_positive", "damage_level_inconsistent": True, "DAMAGE_LEVEL": ""},
        {"damage_target_status": "explicit_negative", "damage_level_inconsistent": True, "DAMAGE_LEVEL": "D"},
    ])
    result = build_damage_severity_target(df)
    assert result["damage_severity_score"].isna().tolist() == [True, True]
