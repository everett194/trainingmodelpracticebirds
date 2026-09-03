import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from birdstrikegeo.hazard.modeling import predict_severity, train_severity_regressor  # noqa: E402


def _synthetic_data(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    # species "HEAVY" always causes severity 3, species "LIGHT" always 0 -
    # an easy, deterministic signal a GBT should recover with low error.
    rng = np.random.default_rng(seed)
    species = rng.choice(["HEAVY", "LIGHT"], size=n)
    severity = np.where(species == "HEAVY", 3.0, 0.0)
    df = pd.DataFrame({"species": species, "engine_count": rng.integers(1, 3, size=n).astype(float)})
    return df, pd.Series(severity)


def test_train_severity_regressor_recovers_a_clean_species_signal():
    train_df, y_train = _synthetic_data(200, seed=1)
    val_df, y_val = _synthetic_data(50, seed=2)

    model, metrics = train_severity_regressor(
        train_df, y_train, val_df, y_val,
        cat_features=["species"],
        config={"iterations": 50, "depth": 4, "learning_rate": 0.2, "early_stopping_rounds": 20},
        random_seed=42,
    )

    assert metrics["val_rmse"] < 0.5
    assert "best_iteration" in metrics


def test_predict_severity_returns_one_prediction_per_row():
    train_df, y_train = _synthetic_data(200, seed=1)
    val_df, y_val = _synthetic_data(50, seed=2)
    model, _ = train_severity_regressor(
        train_df, y_train, val_df, y_val,
        cat_features=["species"],
        config={"iterations": 50, "depth": 4, "learning_rate": 0.2, "early_stopping_rounds": 20},
        random_seed=42,
    )

    predictions = predict_severity(model, val_df, cat_features=["species"])

    assert len(predictions) == len(val_df)
    heavy_mean = predictions[val_df["species"] == "HEAVY"].mean()
    light_mean = predictions[val_df["species"] == "LIGHT"].mean()
    assert heavy_mean > light_mean
