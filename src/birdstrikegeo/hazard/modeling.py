"""
birdstrikegeo.hazard.modeling
----------------------------------
CatBoost regressor for the hazard.severity_target ordinal target
(0-3). Single fixed-config fit, no hyperparameter search - this is an
exploratory risk-ranking analysis (see reports/), not a tuned production
model like birdstrikegeo.ga's calibrated classifier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_squared_error, r2_score


def train_severity_regressor(
    train_df: pd.DataFrame, y_train, val_df: pd.DataFrame, y_val,
    cat_features: list[str], config: dict, random_seed: int,
) -> tuple[CatBoostRegressor, dict]:
    train_pool = Pool(train_df, y_train, cat_features=cat_features)
    val_pool = Pool(val_df, y_val, cat_features=cat_features)

    model = CatBoostRegressor(
        iterations=config["iterations"],
        depth=config["depth"],
        learning_rate=config["learning_rate"],
        loss_function="RMSE",
        eval_metric="RMSE",
        early_stopping_rounds=config["early_stopping_rounds"],
        random_seed=random_seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    val_pred = model.predict(val_pool)
    metrics = {
        "val_rmse": float(mean_squared_error(y_val, val_pred) ** 0.5),
        "val_r2": float(r2_score(y_val, val_pred)),
        "best_iteration": int(model.get_best_iteration()),
    }
    return model, metrics


def predict_severity(model: CatBoostRegressor, df: pd.DataFrame, cat_features: list[str]) -> np.ndarray:
    pool = Pool(df, cat_features=cat_features)
    return model.predict(pool)
