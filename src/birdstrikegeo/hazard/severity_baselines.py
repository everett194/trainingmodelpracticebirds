"""
birdstrikegeo.hazard.severity_baselines
-------------------------------------------
Closes the "no baseline comparison" gap flagged in MODELING_APPROACH.md
for the severity regressor (birdstrikegeo.hazard.modeling): a dummy
floor plus four classical regressors, fit on the same dense
(one-hot + scaled, train-only-fit) feature matrix already used for the
GA damage model's non-tree baselines (birdstrikegeo.ga.modeling.prepare_dense).

Single fixed-config fit per model, no hyperparameter search - same
"exploratory risk-ranking analysis" rigor level as the CatBoost severity
model itself (see hazard.modeling's docstring), not a tuned production
comparison.
"""

from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor


def build_severity_baselines(random_seed: int, n_features: int) -> dict:
    """
    n_features: width of the dense feature matrix these will be fit on -
    PLS needs n_components <= n_features, so it's capped here rather
    than left to fail at fit time on a small/synthetic feature set.
    """
    pls_components = max(1, min(10, n_features - 1))
    return {
        "dummy_mean": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0, random_state=random_seed),
        "knn": KNeighborsRegressor(n_neighbors=15),
        "random_forest": RandomForestRegressor(
            n_estimators=200, random_state=random_seed, n_jobs=-1,
        ),
        "pls": PLSRegression(n_components=pls_components),
    }


def _predict(model, X) -> np.ndarray:
    # PLSRegression.predict returns shape (n, 1); every other model here
    # returns shape (n,) already.
    return np.asarray(model.predict(X)).ravel()


def evaluate_regressor(y_true, y_pred) -> dict:
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def fit_and_evaluate_baselines(
    X_train: np.ndarray, y_train, X_val: np.ndarray, y_val, X_test: np.ndarray, y_test,
    random_seed: int,
) -> list[dict]:
    """
    Fits every baseline on (X_train, y_train) only, then reports
    val/test RMSE + R^2 for each - one row per model, in the same shape
    as reports/latest/model_comparison.csv (Task A) so both comparison
    tables read the same way.
    """
    baselines = build_severity_baselines(random_seed, n_features=X_train.shape[1])

    rows = []
    for name, model in baselines.items():
        model.fit(X_train, y_train)
        val_metrics = evaluate_regressor(y_val, _predict(model, X_val))
        test_metrics = evaluate_regressor(y_test, _predict(model, X_test))
        rows.append({
            "model": name,
            "val_rmse": val_metrics["rmse"], "val_r2": val_metrics["r2"],
            "test_rmse": test_metrics["rmse"], "test_r2": test_metrics["r2"],
        })
    return rows
