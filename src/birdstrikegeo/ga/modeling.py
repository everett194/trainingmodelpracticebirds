"""
birdstrikegeo.ga.modeling
----------------------------
Preprocessing + model training for the GA conditional-damage milestone.

Two preprocessing tracks, because the two model families want different
input shapes:
  - "dense": impute + one-hot + standardize, fit on TRAIN ONLY. Feeds the
    dummy baseline, logistic regression, and the reused DamageNet
    PyTorch network (birdstrikegeo.models.neural_net).
  - "tree": raw category strings (missing -> explicit "missing_value"
    token) + numeric columns with NaN left as NaN (CatBoost's native
    missing-value handling). Feeds CatBoostClassifier, the principal
    gradient-boosted-tree model.

CatBoost was chosen over the existing
birdstrikegeo.models.baselines.HistGradientBoostingClassifier because
that implementation only sees data AFTER one-hot encoding (bloats
dimensionality for a tree model, throws away CatBoost-style native
categorical splits) - see the milestone report for the full comparison.
The dummy and logistic-regression baselines are unchanged/reused.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from birdstrikegeo.features.build_damage_features import apply_rare_categories, fit_rare_categories


# ---------------------------------------------------------------- dense --
@dataclass
class DensePrepared:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    preprocessor: ColumnTransformer
    frequent_categories: dict
    feature_names_out: list[str]


def _stringify_categoricals(df: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype("string")
    return df


def prepare_dense(train_features, val_features, test_features, numeric_columns, categorical_columns,
                   rare_category_threshold: int) -> DensePrepared:
    train_str = _stringify_categoricals(train_features, categorical_columns)
    val_str = _stringify_categoricals(val_features, categorical_columns)
    test_str = _stringify_categoricals(test_features, categorical_columns)

    frequent = fit_rare_categories(train_str, categorical_columns, rare_category_threshold)
    train_bucketed = apply_rare_categories(train_str, frequent)
    val_bucketed = apply_rare_categories(val_str, frequent)
    test_bucketed = apply_rare_categories(test_str, frequent)
    for df in (train_bucketed, val_bucketed, test_bucketed):
        for col in categorical_columns:
            df[col] = df[col].fillna("missing_value")

    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer(
        [("numeric", numeric_pipeline, numeric_columns), ("categorical", categorical_pipeline, categorical_columns)]
    )

    X_train = preprocessor.fit_transform(train_bucketed)
    X_val = preprocessor.transform(val_bucketed)
    X_test = preprocessor.transform(test_bucketed)

    return DensePrepared(
        X_train=X_train, X_val=X_val, X_test=X_test, preprocessor=preprocessor,
        frequent_categories=frequent, feature_names_out=list(preprocessor.get_feature_names_out()),
    )


def build_dummy_and_logistic(random_seed: int) -> dict:
    return {
        "dummy": DummyClassifier(strategy="stratified", random_state=random_seed),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_seed),
    }


# ----------------------------------------------------------------- tree --
def prepare_tree(train_features, val_features, test_features, numeric_columns, categorical_columns):
    """
    Returns (train_df, val_df, test_df, cat_feature_names) ready for a
    CatBoost Pool: categorical columns are strings with an explicit
    "missing_value" token; numeric columns keep NaN as NaN (CatBoost's
    native handling, not imputed away).
    """
    def _prep(df):
        df = df.copy()
        for col in categorical_columns:
            df[col] = df[col].astype("string").fillna("missing_value")
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[numeric_columns + categorical_columns]

    return _prep(train_features), _prep(val_features), _prep(test_features), list(categorical_columns)


@dataclass
class CatBoostSearchResult:
    best_model: CatBoostClassifier
    best_params: dict
    search_table: list[dict] = field(default_factory=list)
    train_val_metrics: dict = field(default_factory=dict)


def train_catboost_with_search(
    train_df: pd.DataFrame, y_train, val_df: pd.DataFrame, y_val,
    cat_features: list[str], base_config: dict, search_grid: dict, random_seed: int,
) -> CatBoostSearchResult:
    """
    Trains one baseline CatBoost model at base_config, then a small
    validation-selected search over depth x learning_rate x l2_leaf_reg
    (class weighting is fixed at "Balanced" throughout, not searched -
    keeps the grid modest per project requirements). Selection metric:
    validation log loss (never the test split).
    """
    train_pool = Pool(train_df, y_train, cat_features=cat_features)
    val_pool = Pool(val_df, y_val, cat_features=cat_features)

    def _fit(depth, learning_rate, l2_leaf_reg):
        model = CatBoostClassifier(
            iterations=base_config["iterations"],
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            loss_function=base_config["loss_function"],
            eval_metric="Logloss",
            early_stopping_rounds=base_config["early_stopping_rounds"],
            auto_class_weights="Balanced",
            random_seed=random_seed,
            verbose=False,
            allow_writing_files=False,
        )
        t0 = time.time()
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        elapsed = time.time() - t0
        val_proba = model.predict_proba(val_pool)[:, 1]
        return model, {
            "depth": depth, "learning_rate": learning_rate, "l2_leaf_reg": l2_leaf_reg,
            "best_iteration": model.get_best_iteration(),
            "val_logloss": log_loss(y_val, val_proba),
            "val_roc_auc": roc_auc_score(y_val, val_proba),
            "seconds": round(elapsed, 1),
        }

    search_table = []
    best_model, best_row = None, None
    for depth, lr, l2 in product(search_grid["depth"], search_grid["learning_rate"], search_grid["l2_leaf_reg"]):
        model, row = _fit(depth, lr, l2)
        search_table.append(row)
        if best_row is None or row["val_logloss"] < best_row["val_logloss"]:
            best_model, best_row = model, row

    train_proba = best_model.predict_proba(train_pool)[:, 1]
    val_proba = best_model.predict_proba(val_pool)[:, 1]
    train_val_metrics = {
        "train_logloss": log_loss(y_train, train_proba), "train_roc_auc": roc_auc_score(y_train, train_proba),
        "val_logloss": best_row["val_logloss"], "val_roc_auc": best_row["val_roc_auc"],
    }

    return CatBoostSearchResult(
        best_model=best_model,
        best_params={"depth": best_row["depth"], "learning_rate": best_row["learning_rate"],
                     "l2_leaf_reg": best_row["l2_leaf_reg"], "best_iteration": best_row["best_iteration"]},
        search_table=search_table, train_val_metrics=train_val_metrics,
    )


def predict_proba_catboost(model: CatBoostClassifier, df: pd.DataFrame, cat_features: list[str]) -> np.ndarray:
    pool = Pool(df, cat_features=cat_features)
    return model.predict_proba(pool)[:, 1]


def predict_proba_sklearn(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]
