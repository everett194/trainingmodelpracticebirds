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

from birdstrikegeo.dashboard.taxonomy import classify

METHOD_LABEL = "mist-net banding"


def _taxon_matches(row_species: str, taxon_key: str, grouped: bool) -> bool:
    if grouped:
        return classify(row_species).group == taxon_key
    return row_species.strip().upper() == taxon_key.strip().upper()


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
    matching = [r for r in effort_rows if _taxon_matches(r["species"], taxon_key, grouped)]
    by_month: dict[str, dict] = {r["month"]: r for r in matching}

    banding_detectable = classify(matching[0]["species"]).banding_detectable if matching else None

    results = []
    for month in months:
        row = by_month.get(month)
        hours = float(row["observation_hours"]) if row is not None else 0.0

        if hours <= 0:
            status = "not_monitored"
        elif banding_detectable is False:
            status = "not_sampled_by_method"
        else:
            status = "monitored"

        count = float(row["count"]) if (status == "monitored" and row is not None) else None
        count_per_hour = count / hours if (status == "monitored" and hours > 0) else None
        results.append({
            "month": month, "status": status, "count": count,
            "observation_hours": hours, "count_per_hour": count_per_hour,
        })
    return results


def station_metadata(station_id: str, years: list[int], total_observation_hours: float) -> dict:
    return {
        "station_id": station_id, "years": years, "method": METHOD_LABEL,
        "total_observation_hours": total_observation_hours,
    }
