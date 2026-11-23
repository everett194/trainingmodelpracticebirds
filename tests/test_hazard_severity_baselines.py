import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from birdstrikegeo.hazard.severity_baselines import (  # noqa: E402
    build_severity_baselines,
    evaluate_regressor,
    fit_and_evaluate_baselines,
)


def _synthetic_dense(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    # A clean linear signal: y depends only on the first column, the
    # rest is noise - any sensible regressor should beat the dummy mean.
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = 2.0 * X[:, 0] + rng.normal(scale=0.01, size=n)
    return X, y


def test_build_severity_baselines_caps_pls_components_below_feature_count():
    baselines = build_severity_baselines(random_seed=42, n_features=3)

    assert baselines["pls"].n_components <= 2


def test_evaluate_regressor_reports_rmse_and_r2():
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.0, 1.0, 2.0, 3.0])

    metrics = evaluate_regressor(y_true, y_pred)

    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0


def test_fit_and_evaluate_baselines_beats_dummy_on_a_clean_linear_signal():
    X_train, y_train = _synthetic_dense(300, seed=1)
    X_val, y_val = _synthetic_dense(80, seed=2)
    X_test, y_test = _synthetic_dense(80, seed=3)

    rows = fit_and_evaluate_baselines(X_train, y_train, X_val, y_val, X_test, y_test, random_seed=42)
    by_model = {row["model"]: row for row in rows}

    assert set(by_model) == {"dummy_mean", "ridge", "knn", "random_forest", "pls"}
    assert by_model["ridge"]["test_rmse"] < by_model["dummy_mean"]["test_rmse"]
    assert by_model["ridge"]["test_r2"] > 0.9
