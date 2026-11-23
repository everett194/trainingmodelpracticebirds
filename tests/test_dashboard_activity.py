import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from birdstrikegeo.dashboard.activity import monthly_activity_for_taxon  # noqa: E402


def _effort_row(species, month, count, hours):
    return {"species": species, "month": month, "count": count, "observation_hours": hours}


def test_unmonitored_month_is_distinct_from_zero_capture_month():
    rows = [
        _effort_row("Barn Swallow", "jun", count=0.0, hours=0.0),   # not monitored at all
        _effort_row("Barn Swallow", "jul", count=0.0, hours=100.0),  # monitored, genuinely zero captures
        _effort_row("Barn Swallow", "aug", count=5.0, hours=100.0),
    ]
    result = monthly_activity_for_taxon(rows, "Barn Swallow", grouped=False, months=["jun", "jul", "aug"])
    by_month = {r["month"]: r for r in result}

    assert by_month["jun"]["status"] == "not_monitored"
    assert by_month["jun"]["count"] is None

    assert by_month["jul"]["status"] == "monitored"
    assert by_month["jul"]["count"] == 0.0
    assert by_month["jul"]["count_per_hour"] == 0.0

    assert by_month["aug"]["status"] == "monitored"
    assert by_month["aug"]["count"] == 5.0


def test_banding_undetectable_taxon_shown_as_not_sampled_even_when_monitored():
    rows = [_effort_row("Canada Goose", "aug", count=0.0, hours=120.0)]
    result = monthly_activity_for_taxon(rows, "Canada Goose", grouped=False, months=["aug"])
    assert result[0]["status"] == "not_sampled_by_method"
    assert result[0]["count"] is None


def test_banding_undetectable_taxon_in_unmonitored_month_reports_not_monitored_first():
    rows = [_effort_row("Canada Goose", "jun", count=0.0, hours=0.0)]
    result = monthly_activity_for_taxon(rows, "Canada Goose", grouped=False, months=["jun"])
    assert result[0]["status"] == "not_monitored"


def test_month_with_no_row_at_all_treated_as_not_monitored():
    result = monthly_activity_for_taxon([], "Barn Swallow", grouped=False, months=["jun"])
    assert result[0]["status"] == "not_monitored"
    assert result[0]["observation_hours"] == 0.0


def test_grouped_matching_by_taxonomy_group():
    rows = [_effort_row("Canada Goose", "aug", count=3.0, hours=100.0),
            _effort_row("Mallard", "aug", count=2.0, hours=100.0)]
    result = monthly_activity_for_taxon(rows, "waterfowl", grouped=True, months=["aug"])
    # waterfowl isn't banding-detectable, so even grouped it shows not_sampled_by_method
    assert result[0]["status"] == "not_sampled_by_method"


def test_matching_is_case_insensitive():
    rows = [_effort_row("Barn Swallow", "aug", count=4.0, hours=50.0)]
    result = monthly_activity_for_taxon(rows, "barn swallow", grouped=False, months=["aug"])
    assert result[0]["status"] == "monitored"
    assert result[0]["count"] == 4.0
