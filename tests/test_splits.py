import pandas as pd
import pytest

from birdstrikegeo.training.splits import (
    chronological_split,
    holdout_by_airport,
    holdout_by_region,
    holdout_by_trektellen_coverage,
    holdout_latest_year,
)


def _dated_df(n=100):
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.DataFrame({"incident_date": dates, "value": range(n)})


def test_chronological_split_respects_fractions_approximately():
    df = _dated_df(100)
    result = chronological_split(df, "incident_date", train_fraction=0.70, validation_fraction=0.15)
    assert len(result.train) == 70
    assert len(result.validation) == 15
    assert len(result.test) == 15


def test_chronological_split_train_is_strictly_before_test():
    df = _dated_df(100)
    result = chronological_split(df, "incident_date", train_fraction=0.70, validation_fraction=0.15)
    assert result.train["incident_date"].max() < result.test["incident_date"].min()
    assert result.train["incident_date"].max() < result.validation["incident_date"].min()


def test_chronological_split_works_regardless_of_input_row_order():
    df = _dated_df(50).sample(frac=1, random_state=1).reset_index(drop=True)
    result = chronological_split(df, "incident_date", train_fraction=0.6, validation_fraction=0.2)
    assert result.train["incident_date"].max() < result.validation["incident_date"].min()


def test_chronological_split_rejects_invalid_fractions():
    df = _dated_df(10)
    with pytest.raises(ValueError):
        chronological_split(df, "incident_date", train_fraction=0.9, validation_fraction=0.2)


def test_holdout_by_airport_keeps_all_rows_for_held_out_airport_together():
    df = pd.DataFrame({"airport_id": ["A", "A", "B", "C"], "value": [1, 2, 3, 4]})
    train, held_out = holdout_by_airport(df, "airport_id", ["A"])
    assert set(held_out["airport_id"]) == {"A"}
    assert set(train["airport_id"]) == {"B", "C"}


def test_holdout_latest_year_isolates_most_recent_calendar_year():
    df = pd.DataFrame({"incident_date": ["2021-01-01", "2022-06-01", "2022-12-01"]})
    train, held_out, latest_year = holdout_latest_year(df, "incident_date")
    assert latest_year == 2022
    assert len(held_out) == 2
    assert len(train) == 1


def test_holdout_by_region():
    df = pd.DataFrame({"region": ["conus", "conus", "europe"], "value": [1, 2, 3]})
    train, held_out = holdout_by_region(df, "region", ["europe"])
    assert list(held_out["region"]) == ["europe"]
    assert list(train["region"]) == ["conus", "conus"]


def test_holdout_by_trektellen_coverage():
    df = pd.DataFrame({"trektellen_available": [True, False, True], "value": [1, 2, 3]})
    covered, uncovered = holdout_by_trektellen_coverage(df)
    assert len(covered) == 2
    assert len(uncovered) == 1
