"""
birdstrikegeo.dashboard.exposure
-------------------------------------
Strike rate per 10,000 GA operations - computed ONLY when an operations
file exists whose airport, period, and population actually match the
strike numerator. Per the project brief, there is no interpolation or
estimation fallback: if the match doesn't exist, the UI says exactly
what's missing instead of ever showing a rate.

Two expected input sources (see data/raw/operations/README.md for
where to get them and scripts/*.py for how they're loaded into the
plain-dict rows this module expects):
  - ATADS monthly: {"airport_id", "year", "month", "ga_ops"} - itinerant
    + local GA operations for a towered field, for one specific month.
    Can support a seasonal (multi-month) rate, since the months sum.
  - FAA 5010 annual: {"airport_id", "year", "ga_ops_estimate"} -
    self-reported ANNUAL estimate only. Never used for a sub-year
    (seasonal) rate - there is no way to apportion an annual estimate
    across months without inventing a seasonality assumption this
    project will not make up.
"""

from __future__ import annotations

_FULL_YEAR_MONTHS = frozenset(range(1, 13))


def _is_full_year_request(months: list[int] | None) -> bool:
    return months is None or set(months) == _FULL_YEAR_MONTHS


def compute_strike_rate(
    n_strikes: int, airport_id: str | None, months: list[int] | None, year_range: tuple[int, int] | None,
    atads_monthly_rows: list[dict], faa_5010_annual_rows: list[dict],
) -> dict:
    """
    Returns either:
        {"available": True, "rate_per_10k_ops": float, "n_ops": int, "source": "atads_monthly"|"faa_5010_annual"}
    or:
        {"available": False, "missing_reason": str}

    airport_id=None (region/multi-airport scope) is always unavailable -
    an ops-based rate is only ever computed for one named airport, since
    summing operations across unrelated airports to match a multi-
    airport strike count would not be a real exposure denominator.
    """
    if airport_id is None:
        return {"available": False, "missing_reason": "No single airport selected - an operations-based rate "
                                                        "requires one airport, not a region total."}
    if year_range is None:
        return {"available": False, "missing_reason": "No year range selected."}

    year_lo, year_hi = year_range
    years = range(year_lo, year_hi + 1)
    month_set = set(months) if months is not None else _FULL_YEAR_MONTHS

    matching_monthly = [
        r for r in atads_monthly_rows
        if r["airport_id"] == airport_id and r["year"] in years and r["month"] in month_set
    ]
    if matching_monthly:
        n_ops = sum(r["ga_ops"] for r in matching_monthly)
        if n_ops > 0:
            return {"available": True, "rate_per_10k_ops": n_strikes / n_ops * 10_000,
                    "n_ops": n_ops, "source": "atads_monthly"}

    if _is_full_year_request(months):
        matching_annual = [r for r in faa_5010_annual_rows if r["airport_id"] == airport_id and r["year"] in years]
        if matching_annual:
            n_ops = sum(r["ga_ops_estimate"] for r in matching_annual)
            if n_ops > 0:
                return {"available": True, "rate_per_10k_ops": n_strikes / n_ops * 10_000,
                        "n_ops": n_ops, "source": "faa_5010_annual"}
        missing_what = "No ATADS monthly or FAA 5010 annual operations data found"
    else:
        missing_what = ("No ATADS monthly operations data found for this airport/period "
                         "(FAA 5010 annual estimates cannot be used for a seasonal rate)")

    return {"available": False,
            "missing_reason": f"{missing_what} for {airport_id}, {year_lo}-{year_hi}. "
                               f"See data/raw/operations/README.md to add it."}
