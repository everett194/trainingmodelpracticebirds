import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from birdstrikegeo.hazard.trektellen_season import (  # noqa: E402
    aggregate_season_local_activity,
    effort_normalize_monthly,
    expand_composite_species,
    load_monthly_effort_hours,
    load_season_totals,
    parse_observation_hours,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _write_fixture(tmp_path: Path) -> Path:
    csv_path = tmp_path / "trektellen_season_fixture.csv"
    csv_path.write_text(
        "row_type,rank,species,mar,apr,total,presence_percent,presence_days,"
        "first_date,last_date,maximum_date\n"
        "species,1,Mourning Dove,4,10,14,50,2,4 Mar,10 Apr,10 Apr\n"
        "species,2,Alder/Willow Flycatcher,0,159,159,10,5,23 Apr,23 Apr,23 Apr\n"
        "totals,,Totals,4,169,173,,7,,,\n"
        "observation_hours,,Observation hours,10:30,0:00,10:30,,,,,\n"
    )
    return csv_path


def test_parse_observation_hours_converts_hhmm_to_float_hours():
    assert parse_observation_hours("126:55") == pytest.approx(126 + 55 / 60)


def test_parse_observation_hours_handles_zero():
    assert parse_observation_hours("0:00") == 0.0


def test_load_season_totals_excludes_non_species_rows(tmp_path):
    df = load_season_totals(_write_fixture(tmp_path))
    assert set(df["species"]) == {"Mourning Dove", "Alder/Willow Flycatcher"}
    assert df["mar"].tolist() == [4, 0]


def test_load_monthly_effort_hours_reads_observation_hours_row(tmp_path):
    hours = load_monthly_effort_hours(_write_fixture(tmp_path))
    assert hours["mar"] == pytest.approx(10.5)
    assert hours["apr"] == 0.0


def test_effort_normalize_monthly_divides_count_by_hours(tmp_path):
    species_df = load_season_totals(_write_fixture(tmp_path))
    hours = load_monthly_effort_hours(_write_fixture(tmp_path))
    long_df = effort_normalize_monthly(species_df, hours, month_cols=["mar", "apr"])

    dove_mar = long_df[(long_df["species"] == "Mourning Dove") & (long_df["month"] == "mar")].iloc[0]
    assert dove_mar["count"] == 4
    assert dove_mar["observation_hours"] == pytest.approx(10.5)
    assert dove_mar["count_per_hour"] == pytest.approx(4 / 10.5)


def test_effort_normalize_monthly_leaves_unmonitored_month_as_missing_not_zero(tmp_path):
    species_df = load_season_totals(_write_fixture(tmp_path))
    hours = load_monthly_effort_hours(_write_fixture(tmp_path))
    long_df = effort_normalize_monthly(species_df, hours, month_cols=["mar", "apr"])

    flycatcher_apr = long_df[
        (long_df["species"] == "Alder/Willow Flycatcher") & (long_df["month"] == "apr")
    ].iloc[0]
    assert flycatcher_apr["count"] == 159
    assert flycatcher_apr["observation_hours"] == 0.0
    assert pd.isna(flycatcher_apr["count_per_hour"])


def test_expand_composite_species_duplicates_row_per_component_name(tmp_path):
    species_df = load_season_totals(_write_fixture(tmp_path))
    hours = load_monthly_effort_hours(_write_fixture(tmp_path))
    long_df = effort_normalize_monthly(species_df, hours, month_cols=["mar", "apr"])

    expanded = expand_composite_species(long_df)

    dove_rows = expanded[expanded["species"] == "Mourning Dove"]
    assert len(dove_rows) == 2  # unchanged, one per month

    alder_mar = expanded[(expanded["species"] == "Alder Flycatcher") & (expanded["month"] == "mar")].iloc[0]
    willow_mar = expanded[(expanded["species"] == "Willow Flycatcher") & (expanded["month"] == "mar")].iloc[0]
    assert alder_mar["count"] == 0
    assert willow_mar["count"] == 0
    assert "Alder/Willow Flycatcher" not in set(expanded["species"])


def test_aggregate_season_local_activity_pools_count_over_hours_excluding_unmonitored_months(tmp_path):
    species_df = load_season_totals(_write_fixture(tmp_path))
    hours = load_monthly_effort_hours(_write_fixture(tmp_path))
    long_df = effort_normalize_monthly(species_df, hours, month_cols=["mar", "apr"])

    activity = aggregate_season_local_activity(long_df)

    # apr has 0 observation hours, so only mar (count=4, hours=10.5)
    # contributes to Mourning Dove's pooled rate - apr's count=10 is
    # excluded rather than treated as a real (undivided-by-effort) rate.
    dove = activity[activity["species"] == "Mourning Dove"].iloc[0]
    assert dove["total_count"] == 4
    assert dove["total_hours"] == pytest.approx(10.5)
    assert dove["local_activity"] == pytest.approx(4 / 10.5)
