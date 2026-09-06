from __future__ import annotations

import pandas as pd

from birdstrikegeo.schemas.hazard_observation import HAZARD_OBSERVATION_SCHEMA, REQUIRED_FIELDS, REQUIRES_COMPANION
from birdstrikegeo.schemas.validation import validate_dataframe


def _clean_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "airport_id": ["KJFK", "KLAX"],
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "observation_id": ["KJFK_2026-01-01", "KLAX_2026-01-02"],
            "latitude": [40.64, 33.94],
            "longitude": [-73.78, -118.41],
            "estimated_abundance": [12.0, None],
            "bird_data_resolution": ["monthly_single_station", None],
        }
    )


def test_clean_dataframe_is_clean():
    df = _clean_df()
    result = validate_dataframe(
        df,
        HAZARD_OBSERVATION_SCHEMA,
        schema_name="hazard_observation",
        required_fields=REQUIRED_FIELDS,
        lat_col="latitude",
        lon_col="longitude",
        timestamp_columns=("date",),
        duplicate_key_columns=("airport_id", "date"),
        requires_companion=REQUIRES_COMPANION,
    )
    assert result.is_clean
    assert result.missing_columns == []
    assert result.invalid_coordinate_rows == 0
    assert result.duplicate_key_count == 0
    assert result.companion_violations["estimated_abundance"] == 0


def test_missing_required_column_detected():
    df = _clean_df().drop(columns=["latitude"])
    result = validate_dataframe(
        df, HAZARD_OBSERVATION_SCHEMA, schema_name="hazard_observation", required_fields=REQUIRED_FIELDS
    )
    assert "latitude" in result.missing_columns
    assert not result.is_clean


def test_invalid_coordinates_detected():
    df = _clean_df()
    df.loc[0, "latitude"] = 200.0  # out of range
    result = validate_dataframe(
        df, HAZARD_OBSERVATION_SCHEMA, schema_name="hazard_observation",
        lat_col="latitude", lon_col="longitude",
    )
    assert result.invalid_coordinate_rows == 1
    assert not result.is_clean


def test_invalid_timestamp_detected():
    df = _clean_df()
    df["date"] = df["date"].astype("object")
    df.loc[0, "date"] = "not-a-date"
    result = validate_dataframe(
        df, HAZARD_OBSERVATION_SCHEMA, schema_name="hazard_observation", timestamp_columns=("date",)
    )
    assert result.invalid_timestamp_columns["date"] == 1
    assert not result.is_clean


def test_null_timestamp_is_missingness_not_invalidity():
    df = _clean_df()
    df["date"] = df["date"].astype("object")
    df.loc[0, "date"] = None
    result = validate_dataframe(
        df, HAZARD_OBSERVATION_SCHEMA, schema_name="hazard_observation", timestamp_columns=("date",)
    )
    assert result.invalid_timestamp_columns["date"] == 0
    assert result.missingness_by_column["date"] > 0


def test_duplicate_key_detected():
    df = pd.concat([_clean_df(), _clean_df().iloc[[0]]], ignore_index=True)
    result = validate_dataframe(
        df, HAZARD_OBSERVATION_SCHEMA, schema_name="hazard_observation",
        duplicate_key_columns=("airport_id", "date"),
    )
    assert result.duplicate_key_count == 2  # both copies of the duplicated row
    assert not result.is_clean


def test_companion_violation_detected():
    df = _clean_df()
    df.loc[1, "estimated_abundance"] = 5.0  # non-null, but bird_data_resolution stays null for row 1
    result = validate_dataframe(
        df, HAZARD_OBSERVATION_SCHEMA, schema_name="hazard_observation", requires_companion=REQUIRES_COMPANION
    )
    assert result.companion_violations["estimated_abundance"] == 1
    assert not result.is_clean


def test_missingness_reported_for_every_present_column():
    df = _clean_df()
    result = validate_dataframe(df, HAZARD_OBSERVATION_SCHEMA, schema_name="hazard_observation")
    assert result.missingness_by_column["estimated_abundance"] == 0.5
    assert result.missingness_by_column["airport_id"] == 0.0


def test_to_dict_is_json_serializable_shape():
    df = _clean_df()
    result = validate_dataframe(df, HAZARD_OBSERVATION_SCHEMA, schema_name="hazard_observation")
    d = result.to_dict()
    assert d["schema_name"] == "hazard_observation"
    assert d["n_rows"] == 2
    assert isinstance(d["missingness_by_column"], dict)
