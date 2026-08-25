"""
data/validate.py
------------------
Generic, reusable validation helpers shared across FAA, Trektellen,
airport, and weather ingestion. Each function returns a boolean mask (or
a small result object) rather than raising - callers decide whether an
invalid row should be excluded, flagged, or reported, and are expected to
report it via birdstrikegeo.data.quality_report rather than silently
dropping it.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def valid_coordinates_mask(lat: pd.Series, lon: pd.Series) -> pd.Series:
    """True where (lat, lon) is a plausible, non-null coordinate pair."""
    lat_num = pd.to_numeric(lat, errors="coerce")
    lon_num = pd.to_numeric(lon, errors="coerce")
    return lat_num.notna() & lon_num.notna() & lat_num.between(-90, 90) & lon_num.between(-180, 180)


def valid_timestamp_mask(ts: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(ts, errors="coerce")
    return parsed.notna()


def nonnegative_mask(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.notna() & (numeric >= 0)


def positive_mask(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.notna() & (numeric > 0)


@dataclass
class DuplicateCheckResult:
    n_duplicates: int
    duplicate_keys: list


def find_duplicate_sessions(df: pd.DataFrame, key_columns: list[str]) -> DuplicateCheckResult:
    """
    Detects duplicate Trektellen count sessions (e.g. same site_id +
    count_date + start_time_local + species reported twice), which would
    otherwise silently double-count bird activity.
    """
    dupes = df.duplicated(subset=key_columns, keep=False)
    dup_keys = df.loc[dupes, key_columns].drop_duplicates().to_dict("records")
    return DuplicateCheckResult(n_duplicates=int(dupes.sum()), duplicate_keys=dup_keys)


def end_not_before_start(start_time: pd.Series, end_time: pd.Series) -> pd.Series:
    """
    True where end_time >= start_time on the same calendar day (both
    given as "HH:MM" strings). Sessions crossing midnight are flagged
    False here deliberately - Trektellen visible-migration/capture
    sessions are same-day by convention, and a same-day-only check is a
    conservative validator, not a guess about what happened after
    midnight.
    """
    start = pd.to_timedelta(start_time.astype("string") + ":00", errors="coerce")
    end = pd.to_timedelta(end_time.astype("string") + ":00", errors="coerce")
    return (end >= start) & start.notna() & end.notna()


def unresolved_ids(records: pd.Series, known_ids: set) -> pd.Series:
    """IDs referenced in `records` (e.g. airport_id on a strike record)
    that don't appear in `known_ids` (e.g. the loaded airports table)."""
    referenced = records.dropna().unique()
    return pd.Series([r for r in referenced if r not in known_ids])
