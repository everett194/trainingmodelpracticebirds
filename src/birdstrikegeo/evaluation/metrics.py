"""
evaluation/metrics.py
-------------------------
Every metric listed in Section 18 of the project spec, computed from
true labels + predicted probabilities + a chosen threshold (tuned on
validation data - see training/tune_threshold.py, never on the metrics'
own test split).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_proba, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    n_positive = int(y_true.sum())
    n_total = len(y_true)

    metrics = {
        "sample_count": n_total,
        "positive_class_prevalence": n_positive / n_total if n_total else float("nan"),
        "threshold_used": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier_score": brier_score_loss(y_true, y_proba),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    # ROC-AUC / average precision are undefined with only one class present
    # (e.g. a tiny sample-data test split) - report as null rather than
    # crashing or silently emitting a misleading number.
    if len(set(y_true.tolist())) < 2:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None
        metrics["_warning"] = "Only one class present in y_true - ROC-AUC/average precision are undefined."
    else:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["average_precision"] = average_precision_score(y_true, y_proba)

    return metrics


def model_comparison_table(model_metrics: dict[str, dict]) -> list[dict]:
    """model_metrics: {model_name: compute_metrics() result}. Returns a
    list of flat rows suitable for a single comparison table/CSV."""
    rows = []
    for name, m in model_metrics.items():
        rows.append(
            {
                "model": name,
                "sample_count": m["sample_count"],
                "prevalence": round(m["positive_class_prevalence"], 4),
                "accuracy": round(m["accuracy"], 4),
                "balanced_accuracy": round(m["balanced_accuracy"], 4),
                "precision": round(m["precision"], 4),
                "recall": round(m["recall"], 4),
                "f1": round(m["f1"], 4),
                "roc_auc": round(m["roc_auc"], 4) if m["roc_auc"] is not None else None,
                "average_precision": round(m["average_precision"], 4) if m["average_precision"] is not None else None,
                "brier_score": round(m["brier_score"], 4),
            }
        )
    return rows
