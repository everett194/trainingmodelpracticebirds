"""
birdstrikegeo.hazard.trektellen_season
------------------------------------------
Loads a Trektellen "year totals" export: one row per species with
monthly sums for a single station-season, plus a `totals` row and an
`observation_hours` row (HH:MM per month) - NOT the per-session long/wide
format birdstrikegeo.data.ingest_trektellen expects. See the inspection
notes in reports/ for why this is a distinct, coarser grain.

Station effort varies a lot month to month (some months have zero
observation hours, not zero birds), so raw counts are not comparable
across months without normalizing by observation_hours - see
effort_normalize_monthly().
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from birdstrikegeo.hazard.species_normalize import expand_species_name


def parse_observation_hours(hhmm: str) -> float:
    """"HH:MM" (hours can exceed 24, e.g. a season total) -> float hours."""
    hours_str, minutes_str = hhmm.split(":")
    return int(hours_str) + int(minutes_str) / 60


_NON_MONTH_COLUMNS = {
    "row_type", "rank", "species", "total", "newly_ringed", "retraps", "maximum_count",
    "maximum_date", "maximum_url", "presence_percent", "presence_days", "first_date",
    "first_url", "last_date", "last_url",
}


def _month_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in _NON_MONTH_COLUMNS]


def load_season_totals(csv_path: str | Path) -> pd.DataFrame:
    """Species-only rows (excludes the `totals`/`observation_hours` summary rows)."""
    df = pd.read_csv(csv_path)
    species_df = df[df["row_type"] == "species"].reset_index(drop=True)

    numeric_cols = [
        c
        for c in (
            "rank", "total", "newly_ringed", "retraps", "maximum_count", "presence_percent",
            "presence_days", *_month_columns(species_df),
        )
        if c in species_df.columns
    ]
    for col in numeric_cols:
        species_df[col] = pd.to_numeric(species_df[col])

    return species_df


def load_monthly_effort_hours(csv_path: str | Path) -> pd.Series:
    """Month-column-name -> float observation hours, from the `observation_hours` row."""
    df = pd.read_csv(csv_path)
    row = df[df["row_type"] == "observation_hours"].iloc[0]
    month_cols = _month_columns(df)

    return pd.Series({month: parse_observation_hours(row[month]) for month in month_cols})


def effort_normalize_monthly(
    species_df: pd.DataFrame, effort_hours: pd.Series, month_cols: list[str]
) -> pd.DataFrame:
    """
    Long-format: one row per (species, month) with count, observation_hours,
    and count_per_hour. count_per_hour is <NA> (not 0) when the month had
    zero observation hours - that's missing effort, not absence of birds.
    """
    rows = []
    for _, species_row in species_df.iterrows():
        for month in month_cols:
            count = float(species_row[month])
            hours = float(effort_hours[month])
            count_per_hour = count / hours if hours > 0 else pd.NA
            rows.append(
                {
                    "species": species_row["species"],
                    "month": month,
                    "count": count,
                    "observation_hours": hours,
                    "count_per_hour": count_per_hour,
                }
            )
    return pd.DataFrame(rows)


def expand_composite_species(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces each row whose `species` is a parenthetical-tagged or
    slash-paired composite name (see species_normalize) with one row per
    plain component name, duplicating that row's counts unchanged - a
    known approximation (the composite's total is not split between
    components, since Trektellen doesn't record which one), not a
    resolution of which individual bird was which species.
    """
    expanded_rows = []
    for _, row in long_df.iterrows():
        for name in expand_species_name(row["species"]):
            expanded_row = row.copy()
            expanded_row["species"] = name
            expanded_rows.append(expanded_row)
    return pd.DataFrame(expanded_rows).reset_index(drop=True)


def aggregate_season_local_activity(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per species: total_count and total_hours pooled across
    MONITORED months only (observation_hours > 0), and local_activity =
    total_count / total_hours - a season-pooled per-hour encounter rate.
    Unmonitored months are excluded from both sums rather than treated
    as a real (zero-effort) data point.
    """
    monitored = long_df[long_df["observation_hours"] > 0]
    grouped = monitored.groupby("species").agg(
        total_count=("count", "sum"), total_hours=("observation_hours", "sum")
    ).reset_index()
    grouped["local_activity"] = grouped["total_count"] / grouped["total_hours"]
    return grouped
