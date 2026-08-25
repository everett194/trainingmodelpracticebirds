from pathlib import Path

import pandas as pd
import pytest

from birdstrikegeo.data import column_aliases
from birdstrikegeo.data.column_aliases import resolve_columns
from birdstrikegeo.data.ingest_faa import ingest_faa

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_resolve_columns_handles_alternate_naming_convention():
    raw_columns = ["INDEX NR", "INCIDENT DATE", "AIRPORT CODE", "SPECIES_NAME", "SIZE"]
    resolved = resolve_columns(raw_columns)
    assert resolved["record_id"] == "INDEX NR"
    assert resolved["incident_date"] == "INCIDENT DATE"
    assert resolved["airport_id"] == "AIRPORT CODE"
    assert resolved["species_common_name"] == "SPECIES_NAME"
    assert resolved["wildlife_size"] == "SIZE"


def test_resolve_columns_ignores_unrecognized_columns_rather_than_guessing():
    resolved = resolve_columns(["INDEX_NR", "SOME_TOTALLY_UNKNOWN_FIELD"])
    assert "record_id" in resolved
    assert "SOME_TOTALLY_UNKNOWN_FIELD" not in resolved.values()


def test_resolve_columns_raises_when_two_raw_columns_map_to_same_canonical_field():
    # Both "AIRPORT_ID" and "AIRPORT CODE" are registered aliases for the
    # same canonical field (airport_id). If a file somehow has both, we
    # must refuse to guess which one is authoritative.
    with pytest.raises(ValueError, match="resolve to the same canonical field"):
        resolve_columns(["AIRPORT_ID", "AIRPORT CODE"])


def test_resolve_columns_raises_when_one_alias_is_ambiguous_between_fields(monkeypatch):
    # Construct a deliberately ambiguous alias table to exercise the
    # "one column could mean two different things" guard, independent of
    # whether today's real alias table happens to contain such a case.
    ambiguous_aliases = {
        "airport_id": ("SHARED_COL",),
        "airport_name": ("SHARED_COL",),
    }
    monkeypatch.setattr(column_aliases, "FAA_COLUMN_ALIASES", ambiguous_aliases)
    with pytest.raises(ValueError, match="Ambiguous FAA column"):
        resolve_columns(["SHARED_COL"])


def test_ingest_faa_reads_alternate_naming_convention_csv():
    result = ingest_faa(FIXTURES_DIR / "faa_alias_variant.csv")
    assert result.n_rows_read == 2
    assert list(result.data["record_id"]) == ["ALIAS-001", "ALIAS-002"]
    assert result.data["incident_date"].dt.year.tolist() == [2021, 2021]
    assert result.data["airport_id"].tolist() == ["SAMPLE01", "SAMPLE02"]
    # Fields not present in this smaller fixture are left null, not guessed.
    assert result.data["engine_type"].isna().all()
    assert "engine_type" in result.missing_columns


def test_ingest_faa_audit_preserves_every_original_column():
    result = ingest_faa(FIXTURES_DIR / "faa_alias_variant.csv")
    assert "DAM_LEVEL" in result.audit.columns
    assert "AIRPORT CODE" in result.audit.columns
    assert len(result.audit) == result.n_rows_read


def test_ingest_faa_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        ingest_faa(FIXTURES_DIR / "does_not_exist.csv")


def test_ingest_faa_reads_sample_dataset_end_to_end():
    sample_path = Path(__file__).parent.parent / "data" / "sample" / "faa_strikes_sample.csv"
    if not sample_path.exists():
        pytest.skip("Sample data not generated yet - run scripts/generate_sample_data.py")
    result = ingest_faa(sample_path)
    assert result.n_rows_read == 500
    assert not result.missing_columns
    assert isinstance(result.data["incident_date"].dtype, pd.core.dtypes.dtypes.DatetimeTZDtype) or \
        pd.api.types.is_datetime64_any_dtype(result.data["incident_date"])
