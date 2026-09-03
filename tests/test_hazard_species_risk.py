import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from birdstrikegeo.hazard.species_risk import aggregate_species_risk, merge_local_risk  # noqa: E402


def test_aggregate_species_risk_computes_frequency_and_mean_severity():
    faa = pd.DataFrame(
        {
            "SPECIES": ["Mourning Dove", "Mourning Dove", "Canada Goose"],
            "predicted_severity": [1.0, 3.0, 2.0],
        }
    )
    result = aggregate_species_risk(faa)
    dove = result[result["species"] == "Mourning Dove"].iloc[0]
    assert dove["n_strikes"] == 2
    assert dove["mean_predicted_severity"] == pytest.approx(2.0)


def test_aggregate_species_risk_score_rewards_both_frequency_and_severity():
    faa = pd.DataFrame(
        {
            "SPECIES": ["Mourning Dove", "Mourning Dove", "Canada Goose"],
            "predicted_severity": [2.0, 2.0, 2.0],
        }
    )
    result = aggregate_species_risk(faa)
    dove = result[result["species"] == "Mourning Dove"].iloc[0]
    goose = result[result["species"] == "Canada Goose"].iloc[0]
    # Same mean severity, but Mourning Dove has 2 strikes vs Canada Goose's 1 -
    # risk_score must be strictly higher for the more frequent species.
    assert dove["risk_score"] > goose["risk_score"]
    assert dove["risk_score"] == pytest.approx(2.0 * math.log1p(2))


def test_merge_local_risk_multiplies_national_score_by_local_activity():
    species_risk = pd.DataFrame(
        {"species": ["Mourning Dove"], "risk_score": [4.0], "n_strikes": [10], "mean_predicted_severity": [1.5]}
    )
    local = pd.DataFrame({"species": ["Mourning Dove"], "local_activity": [2.0]})

    merged = merge_local_risk(species_risk, local)
    row = merged.iloc[0]
    assert row["local_risk_contribution"] == pytest.approx(8.0)
    assert row["matched_faa_species"] is True or row["matched_faa_species"] == True  # noqa: E712


def test_merge_local_risk_matches_species_case_insensitively():
    # Real-world mismatch: FAA exports capitalize only the first word
    # ("Mourning dove") while Trektellen title-cases both ("Mourning Dove").
    species_risk = pd.DataFrame(
        {"species": ["Mourning dove"], "risk_score": [4.0], "n_strikes": [10], "mean_predicted_severity": [1.5]}
    )
    local = pd.DataFrame({"species": ["Mourning Dove"], "local_activity": [2.0]})

    merged = merge_local_risk(species_risk, local)
    row = merged.iloc[0]
    assert row["matched_faa_species"] == True  # noqa: E712
    assert row["local_risk_contribution"] == pytest.approx(8.0)
    # Display name stays the local (Trektellen) casing, not the FAA casing.
    assert row["species"] == "Mourning Dove"


def test_merge_local_risk_flags_species_with_no_faa_match_instead_of_dropping():
    species_risk = pd.DataFrame(
        {"species": ["Mourning Dove"], "risk_score": [4.0], "n_strikes": [10], "mean_predicted_severity": [1.5]}
    )
    local = pd.DataFrame({"species": ["Some Obscure Warbler"], "local_activity": [1.0]})

    merged = merge_local_risk(species_risk, local)
    row = merged.iloc[0]
    assert row["matched_faa_species"] == False  # noqa: E712
    assert pd.isna(row["risk_score"])
    assert pd.isna(row["local_risk_contribution"])
