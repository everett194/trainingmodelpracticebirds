"""
evaluation/calibration.py
-----------------------------
Thin wrapper around sklearn's calibration_curve, returning plain lists
(JSON-serializable) rather than numpy arrays, for the quality report /
plots.
"""

from __future__ import annotations

from sklearn.calibration import calibration_curve


def compute_calibration_curve(y_true, y_proba, n_bins: int = 10) -> dict:
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="uniform")
    return {"prob_true": prob_true.tolist(), "prob_pred": prob_pred.tolist(), "n_bins": n_bins}
