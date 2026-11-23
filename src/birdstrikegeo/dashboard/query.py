"""
birdstrikegeo.dashboard.query
------------------------------------
The one pure-Python query function BirdStrikeNet's dashboard is built
on: (airports | region, months, year_range, cohort) -> per-taxon rows.

Takes plain dicts, not a database connection or a DataFrame, so it's
testable with a handful of synthetic rows and has no Flask/pandas/
sqlite3 dependency of its own - birdstrikegeo.dashboard.data_access is
the (thin, untested-by-design) glue that loads real rows from
data/birdstrikegeo.db into this shape.

Expected row shape (one dict per FAA strike report, already restricted
to the confirmed business/private fixed-wing GA population - i.e. what
`faa_strikes_ga` already is):
    {
        "airport_id": str,
        "incident_year": int,
        "incident_month": int | None,   # 1-12
        "species": str,                 # FAA's own SPECIES string, as reported
        "indicated_damage": int | None, # 1 damaging, 0 not, None unknown outcome
        "ac_mass": str | None,          # "1".."5", for cohort narrowing
        "type_eng": str | None,         # "A" = reciprocating/piston, for cohort narrowing
    }
"""

from __future__ import annotations

from birdstrikegeo.dashboard.stats import wilson_interval
from birdstrikegeo.dashboard.taxonomy import GROUP_LABELS, classify

# Small-sample / insufficient-evidence thresholds on KNOWN-OUTCOME n,
# per the project brief - never on raw report count, since a taxon can
# have many reports but few with a resolved damage outcome.
SMALL_SAMPLE_THRESHOLD = 20
INSUFFICIENT_EVIDENCE_THRESHOLD = 10

COHORT_CONFIRMED_BIZ_PVT_FIXED_WING = "confirmed_business_or_private_fixed_wing"
COHORT_LIGHT_FIXED_WING_GA = "light_fixed_wing_ga"
COHORT_LIGHT_PISTON_GA = "light_piston_ga"

COHORT_LABELS: dict[str, str] = {
    COHORT_CONFIRMED_BIZ_PVT_FIXED_WING: "All confirmed business/private fixed-wing GA",
    COHORT_LIGHT_FIXED_WING_GA: "Light fixed-wing GA (<= 5,700 kg)",
    COHORT_LIGHT_PISTON_GA: "Light piston fixed-wing GA",
}


def filter_by_cohort(rows: list[dict], cohort: str) -> list[dict]:
    """
    Narrows an already-confirmed-business/private-fixed-wing row set to
    a stricter cohort. Mirrors birdstrikegeo.ga.data.POPULATION_DEFINITIONS'
    predicates exactly, operating on this module's plain-dict rows
    instead of a pandas DataFrame.
    """
    if cohort == COHORT_CONFIRMED_BIZ_PVT_FIXED_WING:
        return list(rows)
    if cohort == COHORT_LIGHT_FIXED_WING_GA:
        return [r for r in rows if r.get("ac_mass") in ("1", "2")]
    if cohort == COHORT_LIGHT_PISTON_GA:
        return [r for r in rows if r.get("ac_mass") in ("1", "2") and r.get("type_eng") == "A"]
    raise ValueError(f"Unknown cohort: {cohort}")


def recommend_default_cohort(rows: list[dict], minimum_light_piston_n: int = 30) -> str:
    """
    COHORT_LIGHT_PISTON_GA if it has at least `minimum_light_piston_n`
    reports in this row set, else the broader confirmed-biz/pvt cohort -
    the exact rule from the project brief's Step 0 inventory.
    """
    n_light_piston = len(filter_by_cohort(rows, COHORT_LIGHT_PISTON_GA))
    return COHORT_LIGHT_PISTON_GA if n_light_piston >= minimum_light_piston_n else COHORT_CONFIRMED_BIZ_PVT_FIXED_WING


def filter_by_scope(
    rows: list[dict], airports: list[str] | None = None,
    months: list[int] | None = None, year_range: tuple[int, int] | None = None,
) -> list[dict]:
    """airports=None means every airport in `rows` (i.e. "region")."""
    out = rows
    if airports is not None:
        airport_set = set(airports)
        out = [r for r in out if r.get("airport_id") in airport_set]
    if months is not None:
        month_set = set(months)
        out = [r for r in out if r.get("incident_month") in month_set]
    if year_range is not None:
        lo, hi = year_range
        out = [r for r in out if r.get("incident_year") is not None and lo <= r["incident_year"] <= hi]
    return out


def species_id_share(rows: list[dict]) -> float | None:
    """Share of reports whose species is resolved (not an "Unknown bird..." FAA category)."""
    if not rows:
        return None
    n_identified = sum(1 for r in rows if not classify(r.get("species")).is_unknown)
    return n_identified / len(rows)


def _taxon_key(species: str | None, grouped: bool) -> tuple[str, str]:
    """Returns (key, display_label) for a row - the species string itself
    (ungrouped, FAA's own resolution) or its taxonomy group (grouped)."""
    info = classify(species)
    if not grouped:
        return (species or "(blank)", species or "(blank)")
    return (info.group, GROUP_LABELS.get(info.group, info.group))


def _build_taxon_row(key: str, label: str, group_rows: list[dict], grouped: bool) -> dict:
    n_reports = len(group_rows)
    known = [r for r in group_rows if r.get("indicated_damage") in (0, 1)]
    damaging = [r for r in known if r["indicated_damage"] == 1]
    unknown_outcome = n_reports - len(known)

    n_known = len(known)
    n_damaging = len(damaging)
    ci = wilson_interval(n_damaging, n_known)
    damage_share = (n_damaging / n_known) if n_known > 0 else None

    # is_mammal/is_unknown/banding_detectable are consistent across every
    # species sharing a taxonomy group (see taxonomy.py's table), so the
    # first row's classification represents the whole bucket - grouped
    # or not.
    sample_info = classify(group_rows[0].get("species"))

    return {
        "taxon": label,
        "taxon_key": key,
        "group": sample_info.group,
        "group_label": sample_info.group_label,
        "is_mammal": sample_info.is_mammal,
        "is_unknown": sample_info.is_unknown,
        "banding_detectable": sample_info.banding_detectable,
        "n_reports": n_reports,
        "n_known_outcome": n_known,
        "n_damaging": n_damaging,
        "n_unknown_outcome": unknown_outcome,
        "damage_share": damage_share,
        "damage_ci_low": ci[0] if ci else None,
        "damage_ci_high": ci[1] if ci else None,
        "small_sample": n_known < SMALL_SAMPLE_THRESHOLD,
        "insufficient_evidence": n_known < INSUFFICIENT_EVIDENCE_THRESHOLD,
    }


def query_taxon_rows(
    rows: list[dict], *, airports: list[str] | None = None, months: list[int] | None = None,
    year_range: tuple[int, int] | None = None, cohort: str = COHORT_CONFIRMED_BIZ_PVT_FIXED_WING,
    grouped: bool = False, include_mammals: bool = False,
) -> list[dict]:
    """
    The one query function BirdStrikeNet's dashboard is built on. Never
    drops a low-n taxon from the result - callers sort/rank, but every
    taxon that had any report in scope gets a row, with
    insufficient_evidence set rather than silently excluded.
    """
    scoped = filter_by_scope(filter_by_cohort(rows, cohort), airports=airports, months=months, year_range=year_range)

    buckets: dict[str, list[dict]] = {}
    labels: dict[str, str] = {}
    for row in scoped:
        key, label = _taxon_key(row.get("species"), grouped)
        buckets.setdefault(key, []).append(row)
        labels[key] = label

    taxon_rows = [_build_taxon_row(key, labels[key], group_rows, grouped) for key, group_rows in buckets.items()]
    if not include_mammals:
        taxon_rows = [r for r in taxon_rows if not r["is_mammal"]]

    taxon_rows.sort(key=lambda r: r["n_reports"], reverse=True)
    return taxon_rows
