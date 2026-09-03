import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from birdstrikegeo.hazard.species_normalize import expand_species_name  # noqa: E402


def test_plain_name_returns_itself_unchanged():
    assert expand_species_name("Northern Bobwhite") == ["Northern Bobwhite"]


def test_strips_trailing_parenthetical_subspecies_tag():
    assert expand_species_name("Northern Flicker (Yellow-shafted)") == ["Northern Flicker"]


def test_splits_slash_pair_sharing_common_suffix():
    assert expand_species_name("Alder/Willow Flycatcher") == ["Alder Flycatcher", "Willow Flycatcher"]


def test_splits_slash_pair_with_hyphenated_first_part():
    assert expand_species_name("Blue-winged/Golden-winged Warbler") == [
        "Blue-winged Warbler",
        "Golden-winged Warbler",
    ]


def test_splits_slash_pair_of_two_already_complete_names():
    assert expand_species_name("Grey-cheeked Thrush / Bicknell's Thrush") == [
        "Grey-cheeked Thrush",
        "Bicknell's Thrush",
    ]


def test_strips_surrounding_whitespace():
    assert expand_species_name("  Mourning Dove  ") == ["Mourning Dove"]
