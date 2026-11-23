import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from birdstrikegeo.dashboard.exposure import compute_strike_rate  # noqa: E402


def test_no_ops_data_at_all_reports_missing_not_a_rate():
    result = compute_strike_rate(5, "KESN", months=[6, 7], year_range=(2020, 2020),
                                  atads_monthly_rows=[], faa_5010_annual_rows=[])
    assert result["available"] is False
    assert "KESN" in result["missing_reason"]


def test_region_scope_always_unavailable():
    result = compute_strike_rate(5, None, months=None, year_range=(2020, 2020),
                                  atads_monthly_rows=[{"airport_id": "KESN", "year": 2020, "month": 6, "ga_ops": 100}],
                                  faa_5010_annual_rows=[])
    assert result["available"] is False


def test_matching_atads_monthly_row_computes_a_rate():
    atads = [{"airport_id": "KESN", "year": 2020, "month": 6, "ga_ops": 500}]
    result = compute_strike_rate(5, "KESN", months=[6], year_range=(2020, 2020),
                                  atads_monthly_rows=atads, faa_5010_annual_rows=[])
    assert result["available"] is True
    assert result["source"] == "atads_monthly"
    assert result["n_ops"] == 500
    assert result["rate_per_10k_ops"] == 100.0


def test_seasonal_request_never_falls_back_to_annual_5010():
    annual = [{"airport_id": "KESN", "year": 2020, "ga_ops_estimate": 10000}]
    result = compute_strike_rate(5, "KESN", months=[6, 7], year_range=(2020, 2020),
                                  atads_monthly_rows=[], faa_5010_annual_rows=annual)
    assert result["available"] is False
    assert "seasonal" in result["missing_reason"]


def test_full_year_request_can_use_annual_5010_when_no_monthly_data():
    annual = [{"airport_id": "KESN", "year": 2020, "ga_ops_estimate": 20000}]
    result = compute_strike_rate(10, "KESN", months=None, year_range=(2020, 2020),
                                  atads_monthly_rows=[], faa_5010_annual_rows=annual)
    assert result["available"] is True
    assert result["source"] == "faa_5010_annual"
    assert result["rate_per_10k_ops"] == 5.0


def test_wrong_airport_does_not_match():
    atads = [{"airport_id": "KSBY", "year": 2020, "month": 6, "ga_ops": 500}]
    result = compute_strike_rate(5, "KESN", months=[6], year_range=(2020, 2020),
                                  atads_monthly_rows=atads, faa_5010_annual_rows=[])
    assert result["available"] is False
