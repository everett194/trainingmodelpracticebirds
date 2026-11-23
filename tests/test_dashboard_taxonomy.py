import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from birdstrikegeo.dashboard.taxonomy import classify, is_covered  # noqa: E402


def test_classify_known_species_returns_documented_group():
    info = classify("Canada goose")
    assert info.group == "waterfowl"
    assert info.is_mammal is False
    assert info.banding_detectable is False


def test_classify_is_case_insensitive_for_trektellen_style_names():
    assert classify("Canada Goose").group == classify("Canada goose").group


def test_classify_mammal_species():
    info = classify("White-tailed deer")
    assert info.is_mammal is True
    assert info.group == "deer"


def test_classify_unknown_bird_is_flagged_unknown_not_a_mammal():
    info = classify("Unknown bird - large")
    assert info.is_unknown is True
    assert info.is_mammal is False


def test_classify_unresolved_species_falls_back_without_guessing():
    info = classify("Species Not In Any Table")
    assert info.group == "unknown"
    assert info.is_unknown is False  # "unknown" here means "FAA said unknown," not "we don't have an entry"
    assert info.banding_detectable is None


def test_classify_blank_species_falls_back():
    assert classify(None).group == "unknown"
    assert classify("").group == "unknown"


def test_is_covered():
    assert is_covered("Canada goose") is True
    assert is_covered("Not A Real Species") is False
    assert is_covered(None) is False


def test_small_songbird_is_banding_detectable():
    assert classify("Barn swallow").banding_detectable is True


def test_group_level_faa_category_classified_same_as_specific_species_in_its_group():
    assert classify("Geese").group == classify("Canada goose").group
