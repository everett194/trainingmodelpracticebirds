from __future__ import annotations

import pytest

from birdstrikegeo.geo.ebird_adapter import (
    EbirdAbundanceRequest,
    EbirdNotConfiguredError,
    access_key,
    fetch_abundance_raster,
    is_configured,
    offline_placeholder_record,
)


@pytest.fixture(autouse=True)
def _clear_ebird_key(monkeypatch):
    monkeypatch.delenv("EBIRD_ST_ACCESS_KEY", raising=False)


def test_is_configured_false_when_no_key_set():
    assert access_key() is None
    assert is_configured() is False


def test_is_configured_true_when_key_set(monkeypatch):
    monkeypatch.setenv("EBIRD_ST_ACCESS_KEY", "fake-key-for-test")
    assert is_configured() is True


def test_fetch_raises_not_configured_without_a_key(tmp_path):
    request = EbirdAbundanceRequest(species_code="amecro", week=20, bounding_box=(-75, 40, -73, 41))
    with pytest.raises(EbirdNotConfiguredError):
        fetch_abundance_raster(request, tmp_path)


def test_fetch_raises_not_implemented_with_a_key_set(tmp_path, monkeypatch):
    monkeypatch.setenv("EBIRD_ST_ACCESS_KEY", "fake-key-for-test")
    request = EbirdAbundanceRequest(species_code="amecro", week=20, bounding_box=(-75, 40, -73, 41))
    with pytest.raises(NotImplementedError):
        fetch_abundance_raster(request, tmp_path)


def test_offline_placeholder_record_has_expected_shape_and_is_marked_unavailable():
    request = EbirdAbundanceRequest(species_code="amecro", week=20, bounding_box=(-75, 40, -73, 41))
    record = offline_placeholder_record(request)
    assert record["bird_data_available"] is False
    assert record["bird_data_source"] == "ebird_status_trends"
    assert record["species_or_group"] == "amecro"
    assert record["estimated_abundance"] is None
