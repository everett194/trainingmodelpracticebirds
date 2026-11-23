"""
birdstrikegeo.dashboard.stats
----------------------------------
Wilson score interval for a binomial proportion - chosen over the plain
normal-approximation interval because it stays inside [0, 1] and
behaves sensibly at the small sample sizes (n < 20, often n < 10) this
dashboard's regional/airport-level damage-share numbers actually have.
"""

from __future__ import annotations

import math

_Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson_interval(successes: int, n: int, z: float = _Z_95) -> tuple[float, float] | None:
    """
    Returns (ci_low, ci_high) for successes/n, or None if n == 0 (there
    is no interval to report, not a 0%-100% placeholder).
    """
    if n <= 0:
        return None
    if successes < 0 or successes > n:
        raise ValueError(f"successes={successes} must be between 0 and n={n}")

    phat = successes / n
    denom = 1 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    margin = (z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))) / denom

    return (max(0.0, center - margin), min(1.0, center + margin))
