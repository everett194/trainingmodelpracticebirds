from pathlib import Path

import pandas as pd
import pytest

from birdstrikegeo.data.ingest_trektellen import (
    _is_wide_format,
    convert_wide_to_long,
    load_counts,
    load_provenance,
    load_sites,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _sites_df():
    return pd.DataFrame(
        {
            "site_id": ["S1", "S2"],
            "site_name": ["Site One", "Site Two"],
            "country": ["US-SAMPLE", "US-SAMPLE"],
            "latitude": [40.0, 41.0],
            "longitude": [-75.0, -74.0],
            "count_type": ["visible_migration", "visible_migration"],
            "timezone": ["America/New_York", "America/New_York"],
        }
    )


def test_long_format_is_detected_and_passed_through():
    long_df = pd.DataFrame(
        {
            "site_id": ["S1"],
            "count_date": ["2024-05-01"],
            "species_common_name": ["Mallard"],
            "count": [3],
        }
    )
    assert not _is_wide_format(long_df)


def test_wide_format_is_detected():
    wide_df = pd.DataFrame(
        {
            "site_id": ["S1"],
            "count_date": ["2024-05-01"],
            "Mallard": [3],
            "Herring Gull": [0],
        }
    )
    assert _is_wide_format(wide_df)


def test_convert_wide_to_long_produces_one_row_per_species_with_nonzero_count():
    wide_df = pd.DataFrame(
        {
            "site_id": ["S1", "S1"],
            "count_date": ["2024-05-01", "2024-05-02"],
            "Mallard": [3, 0],
            "Herring Gull": [0, 5],
        }
    )
    long_df = convert_wide_to_long(wide_df)
    assert set(long_df["species_common_name"]) == {"Mallard", "Herring Gull"}
    assert len(long_df) == 2  # zero-count cells dropped
    assert long_df["species_scientific_name"].isna().all()  # not guessable from a wide export


def test_load_counts_flags_unresolved_site_ids():
    sites = _sites_df()
    counts = pd.DataFrame(
        {
            "site_id": ["S1", "UNKNOWN_SITE"],
            "count_date": ["2024-05-01", "2024-05-01"],
            "start_time_local": ["08:00", "08:00"],
            "end_time_local": ["09:00", "09:00"],
            "species_common_name": ["Mallard", "Mallard"],
            "species_scientific_name": ["Anas platyrhynchos", "Anas platyrhynchos"],
            "count": [3, 2],
            "count_type": ["visible_migration", "visible_migration"],
            "observation_hours": [1.0, 1.0],
        }
    )
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        counts.to_csv(f.name, index=False)
        path = f.name

    _, report = load_counts(path, sites)
    assert report.n_unresolved_site_ids == 1


def test_load_counts_flags_negative_counts_and_bad_time_ranges():
    sites = _sites_df()
    counts = pd.DataFrame(
        {
            "site_id": ["S1", "S1"],
            "count_date": ["2024-05-01", "2024-05-01"],
            "start_time_local": ["09:00", "10:00"],
            "end_time_local": ["08:00", "11:00"],  # first row: end before start
            "species_common_name": ["Mallard", "Mallard"],
            "species_scientific_name": ["Anas platyrhynchos", "Anas platyrhynchos"],
            "count": [-1, 4],
            "count_type": ["visible_migration", "visible_migration"],
            "observation_hours": [1.0, 1.0],
        }
    )
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        counts.to_csv(f.name, index=False)
        path = f.name

    _, report = load_counts(path, sites)
    assert report.n_negative_counts == 1
    assert report.n_invalid_time_ranges == 1


def test_load_counts_separates_count_type_rather_than_combining():
    sites = _sites_df()
    counts = pd.DataFrame(
        {
            "site_id": ["S1", "S1"],
            "count_date": ["2024-05-01", "2024-05-01"],
            "start_time_local": ["08:00", "20:00"],
            "end_time_local": ["09:00", "22:00"],
            "species_common_name": ["Mallard", "Mallard"],
            "species_scientific_name": ["Anas platyrhynchos", "Anas platyrhynchos"],
            "count": [3, 7],
            "count_type": ["visible_migration", "nocturnal_flight_call"],
            "observation_hours": [1.0, 2.0],
        }
    )
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        counts.to_csv(f.name, index=False)
        path = f.name

    df, report = load_counts(path, sites)
    assert set(df["count_type"]) == {"visible_migration", "nocturnal_flight_call"}
    assert report.count_type_counts["visible_migration"] == 1
    assert report.count_type_counts["nocturnal_flight_call"] == 1


def test_load_provenance_requires_all_fields(tmp_path):
    incomplete = tmp_path / "provenance.json"
    incomplete.write_text('{"source_name": "test"}')
    with pytest.raises(ValueError, match="missing required fields"):
        load_provenance(incomplete)


def test_load_provenance_accepts_complete_record(tmp_path):
    import json

    complete = {
        "source_name": "test", "source_url": "", "download_date": "2024-01-01",
        "access_method": "manual_export", "permission_or_license": "test",
        "contact_or_citation": "", "geographic_scope": "", "temporal_scope": "", "notes": "",
    }
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(complete))
    loaded = load_provenance(path)
    assert loaded["source_name"] == "test"


def test_load_sites_from_sample_data():
    sample_path = Path(__file__).parent.parent / "data" / "sample" / "trektellen_sites_sample.csv"
    if not sample_path.exists():
        pytest.skip("Sample data not generated yet - run scripts/generate_sample_data.py")
    sites = load_sites(sample_path)
    assert len(sites) == 40
    assert sites["latitude"].between(-90, 90).all()
    assert sites["longitude"].between(-180, 180).all()
