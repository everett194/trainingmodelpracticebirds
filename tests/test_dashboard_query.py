import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from birdstrikegeo.dashboard.query import (  # noqa: E402
    COHORT_CONFIRMED_BIZ_PVT_FIXED_WING,
    COHORT_LIGHT_PISTON_GA,
    filter_by_cohort,
    filter_by_scope,
    query_taxon_rows,
    recommend_default_cohort,
    species_id_share,
)


def _row(airport_id="KESN", year=2020, month=6, species="Canada goose", damage=0, ac_mass="2", type_eng="A"):
    return {
        "airport_id": airport_id, "incident_year": year, "incident_month": month,
        "species": species, "indicated_damage": damage, "ac_mass": ac_mass, "type_eng": type_eng,
    }


def test_unknown_species_row_becomes_its_own_visible_taxon():
    rows = [_row(species="Unknown bird - small"), _row(species="Unknown bird - small")]
    result = query_taxon_rows(rows)
    assert len(result) == 1
    assert result[0]["taxon"] == "Unknown bird - small"
    assert result[0]["is_unknown"] is True
    assert result[0]["n_reports"] == 2


def test_unknown_damage_outcome_excluded_from_denominator_but_kept_in_report_count():
    rows = [_row(damage=1), _row(damage=0), _row(damage=None)]
    result = query_taxon_rows(rows)
    row = result[0]
    assert row["n_reports"] == 3
    assert row["n_known_outcome"] == 2
    assert row["n_unknown_outcome"] == 1
    assert row["n_damaging"] == 1
    assert row["damage_share"] == 0.5


def test_group_level_report_is_never_attributed_to_a_specific_species():
    rows = [_row(species="Geese"), _row(species="Canada goose")]
    result = query_taxon_rows(rows, grouped=False)
    taxa = {r["taxon"] for r in result}
    # ungrouped view: "Geese" and "Canada goose" stay distinct taxa, never merged
    assert taxa == {"Geese", "Canada goose"}
    for row in result:
        assert row["n_reports"] == 1


def test_grouped_view_rolls_both_up_into_the_same_group_without_losing_original_counts():
    rows = [_row(species="Geese"), _row(species="Canada goose"), _row(species="Canada goose")]
    result = query_taxon_rows(rows, grouped=True)
    assert len(result) == 1
    assert result[0]["n_reports"] == 3
    assert result[0]["group"] == "waterfowl"


def test_small_sample_threshold_below_20_known_outcome():
    rows = [_row(damage=0, species="Rock pigeon") for _ in range(15)]
    result = query_taxon_rows(rows)
    assert result[0]["small_sample"] is True
    assert result[0]["insufficient_evidence"] is False


def test_insufficient_evidence_threshold_below_10_known_outcome_shows_no_damage_share():
    rows = [_row(damage=0, species="Rock pigeon") for _ in range(5)]
    result = query_taxon_rows(rows)
    row = result[0]
    assert row["insufficient_evidence"] is True
    # still has a damage_share (it's real, just unreliable) - "insufficient evidence"
    # governs ranking/sorting eligibility in the UI layer, not whether the number exists.
    assert row["damage_share"] == 0.0


def test_mammals_excluded_by_default_and_included_on_toggle():
    rows = [_row(species="White-tailed deer"), _row(species="Canada goose")]
    birds_only = query_taxon_rows(rows, include_mammals=False)
    assert all(not r["is_mammal"] for r in birds_only)
    assert len(birds_only) == 1

    with_mammals = query_taxon_rows(rows, include_mammals=True)
    assert len(with_mammals) == 2


def test_filter_by_scope_airport_months_year_range():
    rows = [_row(airport_id="KESN", year=2020, month=3), _row(airport_id="KSBY", year=2021, month=8)]
    scoped = filter_by_scope(rows, airports=["KESN"])
    assert len(scoped) == 1
    assert scoped[0]["airport_id"] == "KESN"

    scoped = filter_by_scope(rows, months=[3])
    assert len(scoped) == 1

    scoped = filter_by_scope(rows, year_range=(2021, 2021))
    assert len(scoped) == 1
    assert scoped[0]["airport_id"] == "KSBY"


def test_cohort_filter_light_piston_requires_light_mass_and_piston_engine():
    rows = [
        _row(ac_mass="2", type_eng="A"),   # light piston
        _row(ac_mass="2", type_eng="B"),   # light, not piston
        _row(ac_mass="4", type_eng="A"),   # not light
    ]
    piston = filter_by_cohort(rows, COHORT_LIGHT_PISTON_GA)
    assert len(piston) == 1

    all_confirmed = filter_by_cohort(rows, COHORT_CONFIRMED_BIZ_PVT_FIXED_WING)
    assert len(all_confirmed) == 3


def test_recommend_default_cohort_falls_back_when_light_piston_too_thin():
    rows = [_row(ac_mass="2", type_eng="A") for _ in range(5)]
    assert recommend_default_cohort(rows, minimum_light_piston_n=30) == COHORT_CONFIRMED_BIZ_PVT_FIXED_WING


def test_recommend_default_cohort_uses_light_piston_when_sufficient():
    rows = [_row(ac_mass="2", type_eng="A") for _ in range(35)]
    assert recommend_default_cohort(rows, minimum_light_piston_n=30) == COHORT_LIGHT_PISTON_GA


def test_species_id_share():
    rows = [_row(species="Canada goose"), _row(species="Unknown bird")]
    assert species_id_share(rows) == 0.5


def test_species_id_share_empty_is_none():
    assert species_id_share([]) is None
