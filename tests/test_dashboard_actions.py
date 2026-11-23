import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from birdstrikegeo.dashboard.actions import generate_actions  # noqa: E402
from birdstrikegeo.dashboard.citations import CATEGORY_ASSESSMENT, CATEGORY_MONITORING  # noqa: E402
from birdstrikegeo.dashboard.query import query_taxon_rows  # noqa: E402


def _row(year=2020, month=6, species="Canada goose", damage=0):
    return {"airport_id": "KESN", "incident_year": year, "incident_month": month,
            "species": species, "indicated_damage": damage, "ac_mass": "2", "type_eng": "A"}


def test_no_actions_for_a_quiet_scope():
    rows = [_row(month=1, species="Rock pigeon", damage=0)]
    taxon_rows = query_taxon_rows(rows)
    actions = generate_actions(taxon_rows, rows)
    assert actions == []


def test_seasonal_cluster_triggers_monitoring_action():
    rows = [_row(month=9, species="Rock pigeon"), _row(month=9, species="Rock pigeon"),
            _row(month=10, species="Rock pigeon"), _row(month=1, species="Rock pigeon")]
    taxon_rows = query_taxon_rows(rows)
    actions = generate_actions(taxon_rows, rows)
    assert any("Sep-Oct" in a["finding"] or "Sep" in a["finding"] for a in actions)
    assert actions[0]["category"] == CATEGORY_MONITORING
    assert actions[0]["citations"]


def test_high_unknown_species_share_triggers_reporting_action():
    rows = [_row(species="Unknown bird") for _ in range(4)] + [_row(species="Rock pigeon")]
    taxon_rows = query_taxon_rows(rows)
    actions = generate_actions(taxon_rows, rows)
    assert any("Smithsonian" in str(a["citations"]) or "identification" in a["recommendation"] for a in actions)


def test_damaging_flocking_bird_triggers_habitat_action():
    rows = [_row(species="Canada goose", damage=1)]
    taxon_rows = query_taxon_rows(rows)
    actions = generate_actions(taxon_rows, rows)
    assert any("habitat" in a["recommendation"] for a in actions)


def test_any_damaging_strike_triggers_assessment_action():
    rows = [_row(species="Rock pigeon", damage=1)]
    taxon_rows = query_taxon_rows(rows)
    actions = generate_actions(taxon_rows, rows)
    assert any(a["category"] == CATEGORY_ASSESSMENT for a in actions)


def test_recurring_multiyear_pattern_without_damage_triggers_assessment_action():
    rows = [_row(year=2019, species="Rock pigeon", damage=0), _row(year=2022, species="Rock pigeon", damage=0)]
    taxon_rows = query_taxon_rows(rows)
    actions = generate_actions(taxon_rows, rows)
    assert any(a["category"] == CATEGORY_ASSESSMENT for a in actions)


def test_max_items_caps_the_list():
    rows = ([_row(month=9, species="Rock pigeon", damage=1) for _ in range(3)]
            + [_row(month=10, species="Rock pigeon", damage=1) for _ in range(3)]
            + [_row(species="Unknown bird") for _ in range(5)]
            + [_row(species="Canada goose", damage=1)]
            + [_row(year=2019, species="Rock pigeon", damage=1), _row(year=2022, species="Rock pigeon", damage=1)])
    taxon_rows = query_taxon_rows(rows)
    actions = generate_actions(taxon_rows, rows, max_items=2)
    assert len(actions) <= 2


def test_every_action_names_the_triggering_finding():
    rows = [_row(species="Rock pigeon", damage=1)]
    taxon_rows = query_taxon_rows(rows)
    actions = generate_actions(taxon_rows, rows)
    for action in actions:
        assert action["finding"]
        assert action["recommendation"]
