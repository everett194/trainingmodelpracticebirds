"""
evaluation/explainability.py
---------------------------------
Feature-importance reporting for Task A. Everything here answers "what
is this model's prediction associated with", never "what causes damage" -
see FEATURE_IMPORTANCE_DISCLAIMER, which every report using this module
must display.

NOTE on Trektellen coverage: the project spec asks explainability to
include Trektellen coverage variables so sparse-monitoring behavior is
visible. Task A (damage prediction) is documented elsewhere (Section 2)
as trainable from FAA data ALONE, so Trektellen features are not part of
its default feature set - see MODEL_CARD.md "Integration limitations".
If a caller enriches the damage feature matrix with Trektellen coverage
columns (e.g. trektellen_available, data_coverage_score) before calling
grouped_feature_importance(), they will be grouped and reported exactly
like any other feature - no special-casing is needed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

FEATURE_IMPORTANCE_DISCLAIMER = (
    "Feature importance shows predictive association, not causation. A high-"
    "importance feature is one the model relies on to make predictions; it is "
    "not evidence that changing that feature would change real-world outcomes."
)


def logistic_regression_coefficients(model: LogisticRegression, feature_names: list[str]) -> pd.DataFrame:
    coefs = model.coef_.ravel()
    return pd.DataFrame({"feature": feature_names, "coefficient": coefs}).sort_values(
        "coefficient", key=abs, ascending=False
    ).reset_index(drop=True)


def permutation_importance_report(model, X, y, feature_names: list[str], n_repeats: int = 10,
                                   random_seed: int = 42, scoring: str = "roc_auc") -> pd.DataFrame:
    result = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_seed, scoring=scoring)
    return pd.DataFrame(
        {"feature": feature_names, "importance_mean": result.importances_mean, "importance_std": result.importances_std}
    ).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def group_onehot_columns(encoded_feature_names: list[str], conceptual_columns: list[str]) -> dict[str, list[str]]:
    """
    Maps a fitted ColumnTransformer/OneHotEncoder's expanded output
    column names (e.g. "phase_of_flight_Approach", "phase_of_flight_Climb")
    back to the conceptual column they came from ("phase_of_flight"), so
    importance can be reported per real-world feature rather than per
    one-hot dummy. Longest-prefix match, since conceptual column names
    can themselves contain underscores.
    """
    groups: dict[str, list[str]] = {c: [] for c in conceptual_columns}
    sorted_conceptual = sorted(conceptual_columns, key=len, reverse=True)
    for encoded_name in encoded_feature_names:
        matched = next((c for c in sorted_conceptual if encoded_name.startswith(c)), None)
        if matched:
            groups[matched].append(encoded_name)
        else:
            groups.setdefault("_unmatched", []).append(encoded_name)
    return {k: v for k, v in groups.items() if v}


def grouped_feature_importance(per_column_importance: pd.DataFrame, column_groups: dict[str, list[str]],
                                value_col: str = "importance_mean") -> pd.DataFrame:
    """Sums (for permutation importance) or sums-of-abs (for coefficients)
    the per-dummy-column values back up to one row per conceptual feature."""
    rows = []
    lookup = per_column_importance.set_index("feature")[value_col]
    for conceptual, encoded_cols in column_groups.items():
        values = lookup.reindex(encoded_cols).dropna()
        rows.append({"feature": conceptual, value_col: float(values.abs().sum()) if len(values) else 0.0})
    return pd.DataFrame(rows).sort_values(value_col, ascending=False).reset_index(drop=True)


def try_shap_explanation(model, X_sample, feature_names: list[str], max_samples: int = 100):
    """
    Optional SHAP explanation - only attempted if the `shap` package is
    installed (see requirements-dev.txt) AND runs without error. Returns
    None (not an exception) on any failure, since SHAP is explicitly
    optional per project requirements ("only if it works reliably with
    the installed dependencies").
    """
    try:
        import shap
    except ImportError:
        return None

    try:
        background = X_sample[: min(max_samples, len(X_sample))]
        explainer = shap.Explainer(model.predict, background)
        shap_values = explainer(background)
        mean_abs = np.abs(shap_values.values).mean(axis=0)
        return pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values(
            "mean_abs_shap", ascending=False
        ).reset_index(drop=True)
    except Exception:
        return None
