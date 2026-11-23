"""
birdstrikegeo.dashboard.actions
-------------------------------------
Rule-based action panel: every item is generated from a check against
the actual filtered data (never boilerplate), names the finding that
triggered it, and - where the project brief asked for one - cites a
live-verified source from birdstrikegeo.dashboard.citations. An item
whose rule doesn't fire is simply absent, not shown empty.

Most Eastern Shore MD GA airports are not Part 139-certificated, so
every recommendation is voluntary best practice framing, never a
compliance requirement - see citations.py's module docstring.
"""

from __future__ import annotations

from collections import Counter

from birdstrikegeo.dashboard.citations import CATEGORY_ASSESSMENT, CATEGORY_MONITORING, CITATIONS

UNKNOWN_SPECIES_SHARE_THRESHOLD = 0.25
SEASONAL_CLUSTER_MIN_REPORTS = 3
SEASONAL_CLUSTER_RATIO = 2.0
FLOCKING_BIRD_GROUPS = {"waterfowl", "vultures"}

_MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


def _cite(*keys: str) -> list[dict]:
    return [CITATIONS[k] for k in keys]


def _seasonal_cluster_action(scoped_rows: list[dict]) -> dict | None:
    months_present = [r["incident_month"] for r in scoped_rows if r.get("incident_month")]
    if len(months_present) < 2:
        return None
    counts = Counter(months_present)

    # Any 2-consecutive-calendar-month window (non-wrapping) with >= 3 reports.
    for m in range(1, 12):
        window_n = counts.get(m, 0) + counts.get(m + 1, 0)
        if window_n >= SEASONAL_CLUSTER_MIN_REPORTS:
            label = f"{_MONTH_NAMES[m]}-{_MONTH_NAMES[m + 1]}"
            return {
                "category": CATEGORY_MONITORING,
                "finding": f"{window_n} reports fall in the {label} window (n={window_n}), "
                           f"out of {len(months_present)} reports with a known month.",
                "recommendation": f"Prioritize wildlife observations and runway/approach checks during {label}.",
                "citations": _cite("ac_150_5200_33c"),
            }

    nonzero = [n for n in counts.values() if n > 0]
    mean_n = sum(nonzero) / len(nonzero) if nonzero else 0
    if mean_n > 0:
        for month, n in counts.items():
            if n >= SEASONAL_CLUSTER_RATIO * mean_n and n >= 2:
                return {
                    "category": CATEGORY_MONITORING,
                    "finding": f"{_MONTH_NAMES[month]} has {n} reports, "
                               f"{n / mean_n:.1f}x this airport's average month ({mean_n:.1f}).",
                    "recommendation": f"Prioritize wildlife observations and runway/approach checks in {_MONTH_NAMES[month]}.",
                    "citations": _cite("ac_150_5200_33c"),
                }
    return None


def _unknown_species_action(taxon_rows: list[dict]) -> dict | None:
    total = sum(r["n_reports"] for r in taxon_rows)
    if total == 0:
        return None
    n_unknown = sum(r["n_reports"] for r in taxon_rows if r["is_unknown"])
    share = n_unknown / total
    if share >= UNKNOWN_SPECIES_SHARE_THRESHOLD:
        return {
            "category": CATEGORY_MONITORING,
            "finding": f"{n_unknown} of {total} reports in scope ({share:.0%}) have no species identification.",
            "recommendation": "Improve strike-reporting detail at the time of the report, and send any "
                               "recovered remains to the Smithsonian Feather Identification Lab.",
            "citations": _cite("ac_150_5200_32c", "smithsonian_feather_lab"),
        }
    return None


def _flocking_bird_damage_action(taxon_rows: list[dict]) -> dict | None:
    flocking = [r for r in taxon_rows if r["group"] in FLOCKING_BIRD_GROUPS and r["n_damaging"] > 0]
    if not flocking:
        return None
    names = ", ".join(sorted({r["taxon"] for r in flocking}))
    total_damaging = sum(r["n_damaging"] for r in flocking)
    return {
        "category": CATEGORY_MONITORING,
        "finding": f"{total_damaging} damaging report(s) involve large flocking birds in scope: {names}.",
        "recommendation": "Review on- and near-field habitat attractants (standing water, mowing height, "
                           "food sources) that draw waterfowl or vultures.",
        "citations": _cite("ac_150_5200_33c"),
    }


def _assessment_action(taxon_rows: list[dict], scoped_rows: list[dict]) -> dict | None:
    total_damaging = sum(r["n_damaging"] for r in taxon_rows)
    years_with_reports = {r["incident_year"] for r in scoped_rows if r.get("incident_year") is not None}
    recurring = len(years_with_reports) >= 2 and len(scoped_rows) >= 2

    if total_damaging == 0 and not recurring:
        return None

    if total_damaging > 0:
        finding = f"{total_damaging} damaging report(s) in the current filter."
    else:
        finding = f"Reports recur across {len(years_with_reports)} different years in the current filter."

    return {
        "category": CATEGORY_ASSESSMENT,
        "finding": finding,
        "recommendation": "Consider requesting a USDA Wildlife Services site visit, or a formal Wildlife "
                           "Hazard Assessment, to evaluate on-field risk.",
        "citations": _cite("usda_aphis_wildlife_services", "ac_150_5200_38"),
    }


def _fbbo_only_action(fbbo_signals: list[dict] | None) -> dict | None:
    """
    fbbo_signals: optional pre-computed list of {"taxon": str, "month": str,
    "ratio": float} entries describing a Trektellen monthly peak (>= 2x
    that taxon's own monthly mean) for a taxon with few or no matching
    LOCAL strike reports in scope. Phrased as a monitoring observation
    about the station, never as an airport hazard finding - FBBO's
    counts describe bird activity at the banding station, not strikes
    or risk at any airport.
    """
    if not fbbo_signals:
        return None
    signal = fbbo_signals[0]
    return {
        "category": CATEGORY_MONITORING,
        "finding": f"FBBO monitoring shows a {signal['ratio']:.1f}x seasonal peak for {signal['taxon']} "
                   f"in {signal['month']}, with few matching local strike reports.",
        "recommendation": "Monitoring station shows a peak; check conditions at the airport during this window.",
        "citations": [],
    }


def generate_actions(
    taxon_rows: list[dict], scoped_rows: list[dict], fbbo_signals: list[dict] | None = None, max_items: int = 5,
) -> list[dict]:
    """
    taxon_rows: output of dashboard.query.query_taxon_rows for the
    current filter. scoped_rows: the raw rows underlying that same
    filter (for month/year-level checks the aggregated taxon_rows
    don't carry).
    """
    candidates = [
        _seasonal_cluster_action(scoped_rows),
        _unknown_species_action(taxon_rows),
        _flocking_bird_damage_action(taxon_rows),
        _assessment_action(taxon_rows, scoped_rows),
        _fbbo_only_action(fbbo_signals),
    ]
    return [a for a in candidates if a is not None][:max_items]
