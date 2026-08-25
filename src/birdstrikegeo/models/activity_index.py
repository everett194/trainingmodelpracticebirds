"""
models/activity_index.py
----------------------------
Task B: turns the raw features from
birdstrikegeo.features.build_activity_features into a single, bounded
`activity_index` value in [0, 1].

Deliberately NOT a trained model. Per project requirements, we do not
train a strike-vs-no-strike classifier without a defensible non-strike
exposure denominator (see README.md "Scientific limitations" and
configs/data_sources.yaml's bts_t100_departures entry for the future
path). Until that exists, this is a transparent, documented formula over
the windowed bird-count features - an index for relative comparison
across time/airports, not a calibrated probability of anything.
"""

from __future__ import annotations

import numpy as np


def compute_activity_index(features: dict) -> dict:
    """
    features: the dict returned by build_activity_features().

    Returns {activity_index, activity_index_basis, coverage_caveat}.
    activity_index is None (not 0.0) when trektellen_available is False
    or no 7-day observation exists at all - "no data" is never presented
    as "no activity".
    """
    if not features.get("trektellen_available", False):
        return {
            "activity_index": None,
            "activity_index_basis": "no_trektellen_coverage",
            "coverage_caveat": "No Trektellen monitoring sites are configured for this query.",
        }

    total_7d = features.get("total_birds_previous_7d")
    hours_normalized_rate = features.get("birds_per_observation_hour_previous_7d")

    if total_7d is None or hours_normalized_rate is None:
        return {
            "activity_index": None,
            "activity_index_basis": "no_recent_observations",
            "coverage_caveat": (
                "No Trektellen observation session occurred in the previous 7 days "
                "within the configured radius - absence of data, not zero activity."
            ),
        }

    # A simple, log-compressed, effort-normalized rate: raw bird counts
    # scale hugely with observer effort and site density, so we normalize
    # by observation-hours (already done via
    # birds_per_observation_hour_previous_7d) and then log-compress so a
    # handful of exceptionally large counts don't dominate the index.
    # tanh(log1p(rate) / 4) maps [0, inf) -> [0, 1) smoothly; 4 is a
    # documented, hand-picked compression constant (chosen so a rate of
    # ~50 birds/observation-hour, a busy but plausible session, lands
    # near 0.8) - NOT a fitted parameter.
    raw_index = float(np.tanh(np.log1p(max(hours_normalized_rate, 0.0)) / 4.0))

    # Down-weight the index by data_coverage_score so sparse/old/distant
    # monitoring produces a visibly lower-confidence index rather than
    # pretending distant, stale data is as good as nearby, fresh data.
    coverage_score = features.get("data_coverage_score", 0.0) or 0.0
    adjusted_index = round(raw_index * (0.5 + 0.5 * coverage_score), 4)  # coverage can dampen by at most 50%

    caveat = None
    if features.get("distance_warning"):
        caveat = "Nearest monitoring site is farther than the configured coverage radius."
    elif features.get("low_coverage_warning"):
        caveat = "Fewer than the recommended number of monitoring sites are within range."

    return {
        "activity_index": adjusted_index,
        "activity_index_basis": "effort_normalized_7d_rate_with_coverage_adjustment",
        "coverage_caveat": caveat,
    }
