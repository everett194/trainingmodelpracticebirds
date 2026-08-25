import pandas as pd

from birdstrikegeo.features.build_damage_features import make_damage_target


def _base_df(damage_flag_values, damage_level_values=None):
    n = len(damage_flag_values)
    return pd.DataFrame(
        {
            "damage_flag": damage_flag_values,
            "damage_level": damage_level_values or [""] * n,
        }
    )


def test_explicit_positive_values_map_to_1():
    df = _base_df(["Y", "YES", "true", "1"])
    result = make_damage_target(df)
    assert result["damage"].tolist() == [1, 1, 1, 1]
    assert (result["damage_target_status"] == "explicit_positive").all()


def test_explicit_negative_values_map_to_0():
    df = _base_df(["N", "no", "FALSE", "0", "None"])
    result = make_damage_target(df)
    assert result["damage"].tolist() == [0, 0, 0, 0, 0]
    assert (result["damage_target_status"] == "explicit_negative").all()


def test_ambiguous_or_blank_values_are_excluded_not_guessed():
    df = _base_df(["", "UNK", "N/A", None])
    result = make_damage_target(df)
    assert result["damage"].isna().all()
    assert (result["damage_target_status"] == "missing_or_ambiguous").all()


def test_unrecognized_value_is_missing_not_guessed():
    df = _base_df(["MAYBE"])
    result = make_damage_target(df)
    assert result["damage"].isna().all()


def test_eligible_rows_exclude_only_missing_target():
    df = _base_df(["Y", "N", "", "UNK"])
    result = make_damage_target(df)
    eligible = result[result["damage"].notna()]
    excluded = result[result["damage"].isna()]
    assert len(eligible) == 2
    assert len(excluded) == 2
