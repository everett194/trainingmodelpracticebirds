import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from birdstrikegeo.hazard.features import build_severity_features  # noqa: E402


def _raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AIRCRAFT": ["CESSNA 172"],
            "AC_MASS": ["1"],
            "TYPE_ENG": ["A"],
            "NUM_ENGS": ["1"],
            "ENG_1_POS": ["1"],
            "STATE": ["MD"],
            "FAAREGION": ["AEA"],
            "SKY": ["NO CLOUD"],
            "PRECIPITATION": ["NONE"],
            "INCIDENT_MONTH": ["8"],
            "PHASE_OF_FLIGHT": ["APPROACH"],
            "HEIGHT": ["100"],
            "SPEED": ["90"],
            "TIME_OF_DAY": ["DAY"],
            "WARNED": ["N"],
            "SPECIES": ["MOURNING DOVE"],
        }
    )


def test_severity_features_includes_species_as_a_predictor():
    result = build_severity_features(_raw_df())
    assert result.features["species"].tolist() == ["MOURNING DOVE"]
    assert "species" in result.categorical_columns


def test_severity_features_still_includes_operational_core_toggle_variables():
    result = build_severity_features(_raw_df())
    assert result.features["engine_type"].tolist() == ["A"]
    assert result.features["phase_of_flight"].tolist() == ["APPROACH"]
    assert "sky_condition" in result.categorical_columns
