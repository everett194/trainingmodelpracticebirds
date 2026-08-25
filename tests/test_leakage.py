import pandas as pd
import pytest

from birdstrikegeo.features.build_damage_features import build_feature_frame, make_damage_target
from birdstrikegeo.features.leakage import LeakageError, assert_no_leakage
from birdstrikegeo.schemas.faa import FORBIDDEN_LEAKAGE_COLUMNS


def test_assert_no_leakage_passes_for_clean_columns():
    assert_no_leakage(["temperature_c", "hour_local", "species_common_name"])


@pytest.mark.parametrize("forbidden_col", FORBIDDEN_LEAKAGE_COLUMNS)
def test_assert_no_leakage_blocks_each_forbidden_column(forbidden_col):
    with pytest.raises(LeakageError):
        assert_no_leakage(["temperature_c", forbidden_col])


def test_build_feature_frame_never_contains_forbidden_columns():
    df = pd.DataFrame(
        {
            "record_id": ["A1", "A2"],
            "incident_date": ["2024-01-01", "2024-01-02"],
            "incident_time_local": ["08:00", "14:30"],
            "latitude": [40.0, 41.0],
            "longitude": [-75.0, -74.0],
            "damage_flag": ["Y", "N"],
            "damage_level": ["M", ""],
            "effect_on_flight": ["None", "None"],
            "parts_damaged": ["", ""],
            "repair_cost": [0.0, 0.0],
            "other_cost": [0.0, 0.0],
            "injuries": [0, 0],
            "fatalities": [0, 0],
            "narrative": ["text", "text"],
        }
    )
    with_target = make_damage_target(df)
    features = build_feature_frame(with_target)

    for forbidden in FORBIDDEN_LEAKAGE_COLUMNS:
        assert forbidden not in features.columns
    assert "damage" not in features.columns  # target must be added separately by the caller, not baked in here
