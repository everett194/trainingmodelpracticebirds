"""
birdstrikegeo.ga.evaluation
------------------------------
Extra evaluation beyond birdstrikegeo.evaluation.metrics.compute_metrics
(reused as-is - it already covers accuracy/precision/recall/f1/balanced
accuracy/specificity/brier/confusion-matrix/ROC-AUC/average-precision):
Efron's pseudo-R^2, bootstrap confidence intervals, predicted-vs-observed
risk deciles, and validation-selected probability calibration.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def efron_pseudo_r2(y_true, y_proba, y_train_mean: float) -> float:
    """
    Efron pseudo-R^2 for probability predictions:
        1 - sum((y - p)^2) / sum((y - mean(y_train))^2)
    NOT the same as ordinary least-squares R^2 - it is a heuristic
    "variance explained" analogue for probability forecasts, included
    for educational comparison only. Do not treat it as the primary
    measure of model quality (use ROC-AUC / PR-AUC / log loss / Brier /
    calibration for that).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    residual_ss = np.sum((y_true - y_proba) ** 2)
    total_ss = np.sum((y_true - y_train_mean) ** 2)
    return 1 - residual_ss / total_ss if total_ss > 0 else float("nan")


def bootstrap_metric_ci(y_true, y_proba, metric: str, n_resamples: int = 500,
                         confidence_level: float = 0.95, random_seed: int = 42) -> dict:
    """
    metric: "roc_auc" | "pr_auc" | "brier". Resamples (y_true, y_proba)
    pairs WITH replacement n_resamples times and reports the empirical
    percentile interval - a standard nonparametric bootstrap CI.
    """
    metric_fns = {
        "roc_auc": roc_auc_score, "pr_auc": average_precision_score, "brier": brier_score_loss,
    }
    if metric not in metric_fns:
        raise ValueError(f"Unknown bootstrap metric: {metric}")
    fn = metric_fns[metric]

    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    rng = np.random.default_rng(random_seed)

    values = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        y_sample, p_sample = y_true[idx], y_proba[idx]
        if len(set(y_sample.tolist())) < 2:
            continue  # undefined for AUC-style metrics on a single-class resample
        values.append(fn(y_sample, p_sample))

    alpha = 1 - confidence_level
    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return {"metric": metric, "point_estimate": float(fn(y_true, y_proba)), "ci_low": lower, "ci_high": upper,
            "confidence_level": confidence_level, "n_resamples_used": len(values)}


def risk_decile_table(y_true, y_proba, n_bins: int = 10) -> list[dict]:
    """
    Sorts by predicted probability into n_bins equal-COUNT groups (deciles
    by default) and reports predicted-mean vs. observed (actual) damage
    rate per bin - a plain-language calibration check alongside the
    smoother sklearn calibration_curve used for the plot.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_proba = np.asarray(y_proba, dtype=float)
    order = np.argsort(y_proba)
    y_true_sorted, y_proba_sorted = y_true[order], y_proba[order]

    rows = []
    bin_edges = np.linspace(0, len(y_true), n_bins + 1).astype(int)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if hi <= lo:
            continue
        rows.append({
            "risk_decile": i + 1,
            "n": int(hi - lo),
            "predicted_mean_probability": float(y_proba_sorted[lo:hi].mean()),
            "observed_damage_rate": float(y_true_sorted[lo:hi].mean()),
        })
    return rows


def fit_platt_calibrator(y_val, val_proba) -> LogisticRegression:
    """Classic Platt scaling: 1-D logistic regression of the label on the raw predicted probability."""
    calibrator = LogisticRegression()
    calibrator.fit(np.asarray(val_proba).reshape(-1, 1), np.asarray(y_val))
    return calibrator


def apply_platt(calibrator: LogisticRegression, proba) -> np.ndarray:
    return calibrator.predict_proba(np.asarray(proba).reshape(-1, 1))[:, 1]


def fit_isotonic_calibrator(y_val, val_proba) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.asarray(val_proba), np.asarray(y_val))
    return calibrator


def apply_isotonic(calibrator: IsotonicRegression, proba) -> np.ndarray:
    return calibrator.predict(np.asarray(proba))


def select_calibration(y_val, val_proba, isotonic_min_validation_rows: int) -> dict:
    """
    Fits Platt (always) and isotonic (only if there's enough validation
    data to trust a nonparametric fit) calibrators on VALIDATION DATA
    ONLY, and picks whichever - including "raw" (no calibration) - gives
    the best validation Brier score. Returns the fitted calibrator object
    (or None for "raw") plus a comparison table.
    """
    candidates = {"raw": (None, np.asarray(val_proba))}

    platt = fit_platt_calibrator(y_val, val_proba)
    candidates["platt"] = (platt, apply_platt(platt, val_proba))

    if len(val_proba) >= isotonic_min_validation_rows:
        isotonic = fit_isotonic_calibrator(y_val, val_proba)
        candidates["isotonic"] = (isotonic, apply_isotonic(isotonic, val_proba))

    comparison = {name: brier_score_loss(y_val, proba) for name, (_, proba) in candidates.items()}
    best_name = min(comparison, key=comparison.get)
    best_calibrator, _ = candidates[best_name]

    return {
        "selected_method": best_name, "calibrator": best_calibrator,
        "validation_brier_by_method": comparison,
    }


def apply_calibration(method: str, calibrator, proba) -> np.ndarray:
    if method == "raw":
        return np.asarray(proba)
    if method == "platt":
        return apply_platt(calibrator, proba)
    if method == "isotonic":
        return apply_isotonic(calibrator, proba)
    raise ValueError(f"Unknown calibration method: {method}")
