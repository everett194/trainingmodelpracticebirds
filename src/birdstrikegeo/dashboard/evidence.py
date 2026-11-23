"""
birdstrikegeo.dashboard.evidence
-------------------------------------
Assembles the evidence-tier structure the dashboard renders: a primary
(local-airport or region) tier, plus an automatic regional + national
fallback when a single-airport selection is too thin to read anything
from. Every number carries its own n and tier label - never presented
bare, per DISPLAY_DESIGN.md's "required display elements" rule.
"""

from __future__ import annotations

from birdstrikegeo.dashboard.query import filter_by_cohort, filter_by_scope, query_taxon_rows

LOCAL_STRIKES = "LOCAL STRIKES"
REGIONAL_STRIKES = "REGIONAL STRIKES"
NATIONAL_FAA_CONTEXT = "NATIONAL FAA CONTEXT"
LOCAL_MONITORING = "LOCAL MONITORING"
MODEL_OUTPUT = "MODEL OUTPUT"

REGION_SCOPE_LABEL = "Region (all Eastern Shore MD airports)"
NATIONAL_SCOPE_LABEL = "National GA context"

DEFAULT_FALLBACK_THRESHOLD = 10


def _tier(rows: list[dict], airports: list[str] | None, months, year_range, cohort, grouped, include_mammals,
          tier_name: str, scope_label: str) -> dict:
    universe = filter_by_scope(filter_by_cohort(rows, cohort), airports=airports, months=months, year_range=year_range)
    taxon_rows = query_taxon_rows(
        rows, airports=airports, months=months, year_range=year_range, cohort=cohort,
        grouped=grouped, include_mammals=include_mammals,
    )
    return {"tier": tier_name, "n": len(universe), "taxon_rows": taxon_rows, "scope_label": scope_label}


def assemble_evidence(
    region_rows: list[dict], national_rows: list[dict], *, airport: str | None,
    months: list[int] | None, year_range: tuple[int, int] | None, cohort: str,
    grouped: bool = False, include_mammals: bool = False,
    fallback_threshold: int = DEFAULT_FALLBACK_THRESHOLD,
) -> dict:
    """
    airport=None means "Region" is selected (the default) - in that case
    the primary tier already IS the regional one, and no fallback is
    computed (there is nowhere narrower to have fallen back from).
    airport="<FAA LID>" means a single-airport selection; if that
    airport's own n is below `fallback_threshold`, regional and national
    rows are added below it, each carrying its own n and tier label.
    """
    is_region_scope = airport is None
    primary_airports = None if is_region_scope else [airport]
    primary_tier_name = REGIONAL_STRIKES if is_region_scope else LOCAL_STRIKES
    primary_scope_label = REGION_SCOPE_LABEL if is_region_scope else airport

    primary = _tier(region_rows, primary_airports, months, year_range, cohort, grouped, include_mammals,
                     primary_tier_name, primary_scope_label)

    result = {"primary": primary, "regional": None, "national": None}

    if (not is_region_scope) and primary["n"] < fallback_threshold:
        result["regional"] = _tier(region_rows, None, months, year_range, cohort, grouped, include_mammals,
                                    REGIONAL_STRIKES, REGION_SCOPE_LABEL)
        result["national"] = _tier(national_rows, None, months, year_range, cohort, grouped, include_mammals,
                                    NATIONAL_FAA_CONTEXT, NATIONAL_SCOPE_LABEL)

    return result
