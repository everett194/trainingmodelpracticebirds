import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from birdstrikegeo.dashboard.evidence import (  # noqa: E402
    LOCAL_STRIKES,
    NATIONAL_FAA_CONTEXT,
    REGIONAL_STRIKES,
    assemble_evidence,
)
from birdstrikegeo.dashboard.query import COHORT_CONFIRMED_BIZ_PVT_FIXED_WING  # noqa: E402


def _row(airport_id, year=2020, species="Canada goose"):
    return {"airport_id": airport_id, "incident_year": year, "incident_month": 6,
            "species": species, "indicated_damage": 0, "ac_mass": "2", "type_eng": "A"}


def test_region_scope_is_the_primary_tier_with_no_fallback():
    region_rows = [_row("KESN"), _row("KSBY")]
    result = assemble_evidence(region_rows, national_rows=[], airport=None,
                                months=None, year_range=None, cohort=COHORT_CONFIRMED_BIZ_PVT_FIXED_WING)
    assert result["primary"]["tier"] == REGIONAL_STRIKES
    assert result["primary"]["n"] == 2
    assert result["regional"] is None
    assert result["national"] is None


def test_airport_with_zero_reports_triggers_regional_and_national_fallback():
    region_rows = [_row("KSBY"), _row("KSBY")]
    national_rows = [_row("KSBY"), _row("KOTHER")]

    result = assemble_evidence(
        region_rows, national_rows, airport="KESN",  # KESN has no rows at all
        months=None, year_range=None, cohort=COHORT_CONFIRMED_BIZ_PVT_FIXED_WING,
    )

    assert result["primary"]["tier"] == LOCAL_STRIKES
    assert result["primary"]["n"] == 0
    assert result["regional"]["tier"] == REGIONAL_STRIKES
    assert result["regional"]["n"] == 2
    assert result["national"]["tier"] == NATIONAL_FAA_CONTEXT
    assert result["national"]["n"] == 2


def test_airport_with_three_reports_still_below_threshold_triggers_fallback():
    region_rows = [_row("KESN"), _row("KESN"), _row("KESN"), _row("KSBY")]
    result = assemble_evidence(
        region_rows, national_rows=region_rows, airport="KESN",
        months=None, year_range=None, cohort=COHORT_CONFIRMED_BIZ_PVT_FIXED_WING, fallback_threshold=10,
    )
    assert result["primary"]["n"] == 3
    assert result["regional"] is not None
    assert result["regional"]["n"] == 4


def test_airport_at_or_above_threshold_has_no_fallback():
    region_rows = [_row("KESN") for _ in range(10)]
    result = assemble_evidence(
        region_rows, national_rows=[], airport="KESN",
        months=None, year_range=None, cohort=COHORT_CONFIRMED_BIZ_PVT_FIXED_WING, fallback_threshold=10,
    )
    assert result["primary"]["n"] == 10
    assert result["regional"] is None
    assert result["national"] is None
