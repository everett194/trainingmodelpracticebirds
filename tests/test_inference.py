from pathlib import Path

import pytest

from birdstrikegeo.inference.predict_damage import PREDICTION_SEMANTICS, predict_damage
from birdstrikegeo.models.checkpoints import load_checkpoint

CHECKPOINT_PATH = Path(__file__).parent.parent / "models" / "damage" / "damage_net_sample.pt"


def _sample_record():
    return {
        "latitude": 40.0, "longitude": -100.0, "incident_date": "2024-05-15", "incident_time_local": "07:30",
        "phase_of_flight": "Approach", "height_agl_ft": 200, "speed_ias_knots": 130,
        "aircraft_make": "Boeing", "aircraft_model": "Model-500", "aircraft_type": "Airplane",
        "aircraft_mass_class": "2", "engine_type": "Turbofan", "engine_count": 2,
        "sky_condition": "No Cloud", "precipitation": "None", "species_common_name": "Canada Goose",
        "species_scientific_name": "Branta canadensis", "wildlife_size": "Large",
        "number_seen": "1", "number_struck": "1", "state": "SS", "airport_id": "SAMPLE01",
        "parts_struck": "Windshield",
    }


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Run scripts/train_models.py --mode sample first.")
def test_checkpoint_loads_successfully():
    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    assert checkpoint["model"] is not None
    assert 0.0 <= checkpoint["threshold"] <= 1.0
    assert checkpoint["model_version"]


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Run scripts/train_models.py --mode sample first.")
def test_predict_damage_returns_valid_probability():
    result = predict_damage(_sample_record(), str(CHECKPOINT_PATH))
    assert 0.0 <= result["probability_of_damage_given_reported_strike"] <= 1.0
    assert result["classification"] in ("ELEVATED", "LOWER")


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Run scripts/train_models.py --mode sample first.")
def test_predict_damage_always_states_conditional_semantics():
    result = predict_damage(_sample_record(), str(CHECKPOINT_PATH))
    assert result["prediction_semantics"] == PREDICTION_SEMANTICS
    assert "conditional on a reported wildlife strike" in result["prediction_semantics"]
    assert "NOT the probability that a strike will occur" in result["prediction_semantics"]


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Run scripts/train_models.py --mode sample first.")
def test_predict_damage_handles_missing_fields_gracefully():
    partial_record = {"latitude": 40.0, "phase_of_flight": "Approach"}
    result = predict_damage(partial_record, str(CHECKPOINT_PATH))
    assert 0.0 <= result["probability_of_damage_given_reported_strike"] <= 1.0


@pytest.mark.skipif(not CHECKPOINT_PATH.exists(), reason="Run scripts/train_models.py --mode sample first.")
def test_predict_damage_is_deterministic_for_the_same_input():
    r1 = predict_damage(_sample_record(), str(CHECKPOINT_PATH))
    r2 = predict_damage(_sample_record(), str(CHECKPOINT_PATH))
    assert r1["probability_of_damage_given_reported_strike"] == r2["probability_of_damage_given_reported_strike"]
