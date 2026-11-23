"""
birdstrikegeo.dashboard.activity
-------------------------------------
FBBO (Trektellen) local monitoring activity for a taxon, by month -
the piece of BirdStrikeNet that has to actively prevent the mistake the
project brief flagged: FBBO is a mist-net BANDING station, so it
structurally cannot catch geese, vultures, gulls, raptors, herons, or
deer, and a "0 captures" reading for one of those taxa must never be
shown or used as if it meant "no local activity." See
birdstrikegeo.dashboard.taxonomy's banding_detectable flag, which this
module checks before reporting anything.

Three, and only three, states per (taxon, month):
  - "not_monitored": the station logged 0 observation hours that month.
  - "not_sampled_by_method": banding_detectable is False (or unresolved
    for an unknown taxon) - this method cannot see this taxon at all,
    regardless of how many hours were logged.
  - "monitored": real effort-normalized capture data, however small.
A month can be BOTH not_monitored and not_sampled_by_method (e.g. a
goose in an unmonitored month); not_monitored is checked first since
it's the more fundamental reason no count is being reported.

Expects rows already produced by birdstrikegeo.hazard.trektellen_season
(effort_normalize_monthly, optionally expand_composite_species): one
row per (species, month) with count, observation_hours, count_per_hour.
"""

from __future__ import annotations

from birdstrikegeo.dashboard.taxonomy import classify, group_banding_detectable

METHOD_LABEL = "mist-net banding"


def _taxon_matches(row_species: str, taxon_key: str, grouped: bool) -> bool:
    if grouped:
        return classify(row_species).group == taxon_key
    return row_species.strip().upper() == taxon_key.strip().upper()


def _taxon_banding_detectable(taxon_key: str, grouped: bool) -> bool | None:
    """
    Determined from the TAXON being queried, never from whether any
    Trektellen row happens to match it - a species/group mist-net
    banding structurally cannot catch (geese, vultures, gulls, raptors,
    herons, deer) legitimately has ZERO rows in a real export, and that
    absence must never be read as "not monitored."
    """
    return group_banding_detectable(taxon_key) if grouped else classify(taxon_key).banding_detectable


def monthly_activity_for_taxon(
    effort_rows: list[dict], taxon_key: str, grouped: bool, months: list[str],
) -> list[dict]:
    """
    One row per month in `months` (Trektellen's 3-letter lowercase
    month codes, e.g. "mar".."nov"), each:
        {"month": str, "status": "not_monitored"|"not_sampled_by_method"|"monitored",
         "count": float|None, "observation_hours": float, "count_per_hour": float|None}
    `count`/`count_per_hour` are None whenever status != "monitored" -
    never a numeric 0 standing in for "we don't know."
    """
    # observation_hours is a per-station-month fact, identical across
    # every species row for that month - looked up independently of
    # whether THIS taxon has a matching row, so "no row for this
    # species" is never confused with "the station wasn't monitoring."
    month_hours: dict[str, float] = {r["month"]: float(r["observation_hours"]) for r in effort_rows}

    matching = [r for r in effort_rows if _taxon_matches(r["species"], taxon_key, grouped)]
    by_month: dict[str, dict] = {r["month"]: r for r in matching}

    banding_detectable = _taxon_banding_detectable(taxon_key, grouped)

    results = []
    for month in months:
        hours = month_hours.get(month, 0.0)

        if hours <= 0:
            status = "not_monitored"
        elif banding_detectable is not True:
            # False (documented as undetectable) and None (species
            # unresolved, e.g. "Unknown bird") are both treated as "not
            # sampled" - only a species documented as banding_detectable
            # is allowed to claim a confident monitored count.
            status = "not_sampled_by_method"
        else:
            status = "monitored"

        row = by_month.get(month)
        # A monitored month with no row for a detectable species is a
        # genuine zero capture, not missing data - same convention as
        # hazard.trektellen_season (only a missing SESSION is unknown).
        count = (float(row["count"]) if row is not None else 0.0) if status == "monitored" else None
        count_per_hour = count / hours if (status == "monitored" and hours > 0) else None
        results.append({
            "month": month, "status": status, "count": count,
            "observation_hours": hours, "count_per_hour": count_per_hour,
        })
    return results


def aggregate_activity_for_taxon(
    effort_rows: list[dict], taxon_key: str, grouped: bool, months: list[str],
) -> dict:
    """
    Pools monthly_activity_for_taxon's per-month rows across `months`
    into one summary: {"status": ..., "rate_per_hour": float|None,
    "total_count": float|None, "total_hours": float, "n_months_monitored": int}.
    status is "not_sampled_by_method" if any month in scope hit that
    reason (it's the more informative, method-level fact - it would
    still apply in a month that also happens to be unmonitored),
    "not_monitored" if no month was ever monitored AND the method
    limitation never came up, else "monitored" (even if only some
    months were) - the per-month detail for a partial read is still in
    monthly_activity_for_taxon's output, this is just the pooled
    headline number.
    """
    monthly = monthly_activity_for_taxon(effort_rows, taxon_key, grouped, months)
    monitored = [m for m in monthly if m["status"] == "monitored"]

    if not monitored:
        statuses = {m["status"] for m in monthly}
        status = "not_sampled_by_method" if "not_sampled_by_method" in statuses else "not_monitored"
        return {"status": status, "rate_per_hour": None, "total_count": None,
                "total_hours": 0.0, "n_months_monitored": 0}

    total_count = sum(m["count"] for m in monitored)
    total_hours = sum(m["observation_hours"] for m in monitored)
    return {
        "status": "monitored",
        "rate_per_hour": total_count / total_hours if total_hours > 0 else None,
        "total_count": total_count, "total_hours": total_hours,
        "n_months_monitored": len(monitored),
    }


def station_metadata(station_id: str, years: list[int], total_observation_hours: float) -> dict:
    return {
        "station_id": station_id, "years": years, "method": METHOD_LABEL,
        "total_observation_hours": total_observation_hours,
    }
