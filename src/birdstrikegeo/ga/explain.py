"""
birdstrikegeo.ga.explain
---------------------------
Lightweight explainability for the principal CatBoost model: global
feature importance (CatBoost's own PredictionValuesChange - no SHAP
library needed, CatBoost computes fast native SHAP values itself),
permutation importance on the validation split, and one local
(per-preset) explanation.

Explanations are read directly off the fitted model - nothing here is
generated or paraphrased by an LLM. Associations reported are NOT
necessarily causal - every consumer of this module's output should
carry that caveat (see WARNING_NOT_CAUSAL).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import Pool
from sklearn.metrics import roc_auc_score

WARNING_NOT_CAUSAL = (
    "Feature importances and SHAP contributions describe ASSOCIATIONS the "
    "model found in reported FAA strike data, not causal effects. Reporting "
    "bias, confounding (e.g. airport, aircraft type, and season all "
    "correlate with each other), and voluntary-reporting patterns can all "
    "produce a strong association without a causal relationship."
)


def global_feature_importance(model, feature_names: list[str]) -> list[dict]:
    importances = model.get_feature_importance()
    rows = sorted(
        ({"feature": name, "importance": float(value)} for name, value in zip(feature_names, importances)),
        key=lambda r: r["importance"], reverse=True,
    )
    return rows


def permutation_importance(model, df: pd.DataFrame, y_true, cat_features: list[str],
                            n_repeats: int = 5, random_seed: int = 42) -> list[dict]:
    """
    Manual permutation importance: shuffles one column at a time on the
    VALIDATION frame and measures the drop in ROC-AUC vs. the unshuffled
    baseline, averaged over n_repeats shuffles. Works directly against
    CatBoost's Pool interface rather than sklearn's wrapper, since the
    model consumes native categorical columns rather than a dense matrix.
    """
    rng = np.random.default_rng(random_seed)
    baseline_proba = model.predict_proba(Pool(df, cat_features=cat_features))[:, 1]
    baseline_auc = roc_auc_score(y_true, baseline_proba)

    rows = []
    for col in df.columns:
        drops = []
        for _ in range(n_repeats):
            shuffled = df.copy()
            shuffled[col] = shuffled[col].sample(frac=1.0, random_state=rng.integers(0, 1_000_000)).to_numpy()
            proba = model.predict_proba(Pool(shuffled, cat_features=cat_features))[:, 1]
            drops.append(baseline_auc - roc_auc_score(y_true, proba))
        rows.append({"feature": col, "mean_roc_auc_drop": float(np.mean(drops)), "std_roc_auc_drop": float(np.std(drops))})

    rows.sort(key=lambda r: r["mean_roc_auc_drop"], reverse=True)
    return rows


def local_shap_explanation(model, row_df: pd.DataFrame, cat_features: list[str]) -> list[dict]:
    """
    row_df: a single-row dataframe in the model's tree-native feature
    format. Returns each feature's SHAP contribution to THIS prediction,
    most-influential first, using CatBoost's built-in fast SHAP
    implementation (get_feature_importance(type="ShapValues")).
    """
    pool = Pool(row_df, cat_features=cat_features)
    shap_values = model.get_feature_importance(pool, type="ShapValues")[0]  # [n_features + 1] (last = base value)
    base_value = float(shap_values[-1])
    contributions = shap_values[:-1]

    rows = [
        {"feature": col, "value": row_df.iloc[0][col], "shap_contribution": float(contrib)}
        for col, contrib in zip(row_df.columns, contributions)
    ]
    rows.sort(key=lambda r: abs(r["shap_contribution"]), reverse=True)
    return {"base_value": base_value, "contributions": rows}
