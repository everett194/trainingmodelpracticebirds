"""
evaluation/bootstrap.py
---------------------------
Bootstrap confidence intervals for a metric (e.g. ROC-AUC), computed by
resampling the test set with replacement many times. If the test sample
is too small for the resulting interval to be meaningful, this
explicitly reports "confidence intervals are unavailable" rather than
returning a number that looks precise but isn't - per project
requirements, an honest "we can't tell" beats a misleading number.
"""

from __future__ import annotations

import numpy as np

MIN_SAMPLE_SIZE_FOR_BOOTSTRAP = 30
MIN_POSITIVE_CASES_FOR_BOOTSTRAP = 10


def bootstrap_confidence_interval(
    y_true, y_proba, metric_fn, n_resamples: int = 1000, confidence: float = 0.95, random_seed: int = 42,
) -> dict:
    """
    metric_fn: callable(y_true_sample, y_proba_sample) -> float, e.g.
    sklearn.metrics.roc_auc_score.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    n_positive = int(y_true.sum())

    if n < MIN_SAMPLE_SIZE_FOR_BOOTSTRAP or n_positive < MIN_POSITIVE_CASES_FOR_BOOTSTRAP:
        return {
            "available": False,
            "reason": (
                f"Test sample too small for a meaningful bootstrap interval "
                f"(n={n}, positive_cases={n_positive}; need at least "
                f"{MIN_SAMPLE_SIZE_FOR_BOOTSTRAP} rows and "
                f"{MIN_POSITIVE_CASES_FOR_BOOTSTRAP} positive cases). "
                f"Reporting no interval rather than a misleading one."
            ),
        }

    rng = np.random.default_rng(random_seed)
    estimates = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        sample_true, sample_proba = y_true[idx], y_proba[idx]
        if len(set(sample_true.tolist())) < 2:
            continue  # skip degenerate resamples where the metric is undefined
        estimates.append(metric_fn(sample_true, sample_proba))

    if len(estimates) < n_resamples * 0.5:
        return {
            "available": False,
            "reason": "Too many degenerate resamples (single-class) to produce a stable interval.",
        }

    alpha = 1 - confidence
    lower = float(np.percentile(estimates, 100 * alpha / 2))
    upper = float(np.percentile(estimates, 100 * (1 - alpha / 2)))
    point_estimate = float(np.mean(estimates))

    return {
        "available": True,
        "point_estimate": point_estimate,
        "confidence": confidence,
        "lower": lower,
        "upper": upper,
        "n_resamples_used": len(estimates),
    }
