import numpy as np
import pandas as pd
import pytest

from birdstrikegeo.features.build_damage_features import apply_rare_categories, fit_rare_categories
from birdstrikegeo.training.train_damage import build_preprocessor, prepare_datasets


def _toy_split(n=60, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "height_agl_ft": rng.uniform(0, 1000, n),
            "speed_ias_knots": rng.uniform(80, 200, n),
            "engine_count": rng.choice([2, 4], n),
            "year": 2023,
            "month": rng.integers(1, 13, n),
            "day_of_year": rng.integers(1, 366, n),
            "hour_local": rng.integers(0, 24, n),
            "month_sin": 0.0, "month_cos": 0.0, "day_of_year_sin": 0.0, "day_of_year_cos": 0.0,
            "hour_sin": 0.0, "hour_cos": 0.0,
            "state": rng.choice(["SS", "TT", "UU"], n),
            "phase_of_flight": rng.choice(["Approach", "Climb"], n),
            "species_common_name": rng.choice(["Mallard", "Herring Gull", "RareSpecies"], n, p=[0.5, 0.45, 0.05]),
            "damage": rng.integers(0, 2, n),
        }
    )
    return df


def test_fit_rare_categories_uses_only_training_frequency():
    train = pd.DataFrame({"species": ["A"] * 10 + ["B"] * 3 + ["C"] * 1})
    frequent = fit_rare_categories(train, ["species"], threshold=5)
    assert frequent["species"] == {"A"}


def test_apply_rare_categories_buckets_infrequent_values():
    train = pd.DataFrame({"species": ["A"] * 10 + ["B"] * 1})
    frequent = fit_rare_categories(train, ["species"], threshold=5)
    other = pd.DataFrame({"species": ["A", "B", "NEVER_SEEN"]})
    result = apply_rare_categories(other, frequent)
    assert list(result["species"]) == ["A", "rare_other", "rare_other"]


def test_apply_rare_categories_leaves_nulls_alone():
    train = pd.DataFrame({"species": ["A"] * 10})
    frequent = fit_rare_categories(train, ["species"], threshold=5)
    other = pd.DataFrame({"species": ["A", None]})
    result = apply_rare_categories(other, frequent)
    assert result["species"].iloc[0] == "A"
    assert pd.isna(result["species"].iloc[1])


def test_preprocessor_is_fit_on_train_only():
    train = _toy_split(60, seed=1)
    val = _toy_split(20, seed=2)
    test = _toy_split(20, seed=3)
    # Introduce a category in val/test that never appears in train.
    val.loc[0, "species_common_name"] = "NeverInTrain"

    prepared = prepare_datasets(train, val, test, rare_category_threshold=3)
    # Should not raise, and should not silently create a feature column
    # for "NeverInTrain" (OneHotEncoder(handle_unknown="ignore") + our own
    # rare-category bucketing based only on train stats).
    assert prepared.X_val.shape[1] == prepared.X_train.shape[1] == prepared.X_test.shape[1]


def test_preprocessor_handles_unseen_categories_at_transform_time_without_error():
    train = _toy_split(60, seed=1)
    unseen = _toy_split(5, seed=99)
    unseen["state"] = "COMPLETELY_NEW_STATE"

    preprocessor, frequent_values = build_preprocessor(train, rare_category_threshold=3)
    from birdstrikegeo.training.train_damage import _apply_rare_and_stringify

    train_ready = _apply_rare_and_stringify(train, frequent_values)
    preprocessor.fit(train_ready)

    unseen_ready = _apply_rare_and_stringify(unseen, frequent_values)
    transformed = preprocessor.transform(unseen_ready)  # must not raise
    assert transformed.shape[0] == 5


def test_prepared_data_has_no_nan_after_imputation():
    train = _toy_split(60, seed=1)
    train.loc[0, "height_agl_ft"] = np.nan
    val = _toy_split(20, seed=2)
    test = _toy_split(20, seed=3)

    prepared = prepare_datasets(train, val, test, rare_category_threshold=3)
    assert not np.isnan(prepared.X_train).any()
