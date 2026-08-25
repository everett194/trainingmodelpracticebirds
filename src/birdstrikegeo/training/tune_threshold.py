"""
training/tune_threshold.py
------------------------------
Chooses the classification decision threshold by scanning candidate
thresholds against the VALIDATION split only (never the test split -
tuning on test would leak test-set information into a "final" number).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score


def tune_threshold(y_true, y_proba, objective: str = "f1", n_candidates: int = 199) -> dict:
    """
    objective: "f1" | "balanced_accuracy" | "youden_j"
    Returns {threshold, objective, score_at_threshold}.
    """
    candidates = np.linspace(0.01, 0.99, n_candidates)
    scores = []

    for t in candidates:
        y_pred = (np.asarray(y_proba) >= t).astype(int)
        if objective == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif objective == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, y_pred)
        elif objective == "youden_j":
            recall = recall_score(y_true, y_pred, zero_division=0)
            specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
            score = recall + specificity - 1
        else:
            raise ValueError(f"Unknown threshold objective: {objective}")
        scores.append(score)

    best_idx = int(np.argmax(scores))
    return {
        "threshold": float(candidates[best_idx]),
        "objective": objective,
        "score_at_threshold": float(scores[best_idx]),
    }
