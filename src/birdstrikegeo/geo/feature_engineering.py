"""
geo/feature_engineering.py
-----------------------------
Time/season feature engineering shared by both tasks (damage prediction
and activity index). Lives under geo/ because dawn/dusk/night are
computed from date AND coordinates (sunrise/sunset), not from a fixed
clock range - see Section 14 of the project spec.

All timestamps are expected in UTC on input; local-time-of-day features
(hour_local, is_dawn, etc.) require a timezone or lat/lon to derive one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from astral import LocationInfo
    from astral.sun import sun

    _ASTRAL_AVAILABLE = True
except ImportError:  # pragma: no cover - astral is a declared dependency, but degrade gracefully
    _ASTRAL_AVAILABLE = False

MIGRATION_MONTHS_DEFAULT = {3, 4, 5, 8, 9, 10}  # spring + fall, Northern Hemisphere default


def add_cyclical_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Adds year/month/day_of_year/hour plus their sine/cosine encodings.
    Cyclical (sin/cos) encoding avoids the discontinuity of e.g. month=12
    being "far" from month=1 in a plain numeric encoding.
    """
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col], errors="coerce")

    df["year"] = ts.dt.year
    df["month"] = ts.dt.month
    df["day_of_year"] = ts.dt.dayofyear
    df["hour_local"] = ts.dt.hour

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["day_of_year_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_local"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_local"] / 24)

    df["season"] = df["month"].map(_month_to_season)
    return df


def _month_to_season(month) -> str | float:
    if pd.isna(month):
        return np.nan
    month = int(month)
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    return "fall"


def add_migration_season_flag(df: pd.DataFrame, month_col: str = "month",
                               migration_months: set[int] | None = None) -> pd.DataFrame:
    df = df.copy()
    months = migration_months or MIGRATION_MONTHS_DEFAULT
    df["is_migration_season"] = df[month_col].isin(months)
    return df


def _sun_times_for_row(date, lat, lon):
    """Returns (dawn, sunrise, sunset, dusk) as UTC datetimes, or (None,)*4
    if astral is unavailable or inputs are invalid."""
    if not _ASTRAL_AVAILABLE or pd.isna(lat) or pd.isna(lon) or pd.isna(date):
        return None, None, None, None
    try:
        loc = LocationInfo(latitude=float(lat), longitude=float(lon))
        s = sun(loc.observer, date=pd.Timestamp(date).date())
        return s["dawn"], s["sunrise"], s["sunset"], s["dusk"]
    except Exception:
        # Astral can raise for polar day/night edge cases; treat as unknown
        # rather than crashing the whole feature-build.
        return None, None, None, None


def add_dawn_dusk_night_features(df: pd.DataFrame, timestamp_col: str,
                                  lat_col: str = "latitude", lon_col: str = "longitude") -> pd.DataFrame:
    """
    Computes is_dawn / is_dusk / is_night from actual sunrise/sunset at
    each row's date and coordinates (via astral), rather than an
    arbitrary fixed clock range (e.g. "6-8am is dawn everywhere").

    Rows with missing coordinates or where astral cannot compute sun
    times (e.g. polar regions, missing dependency) get is_dawn/is_dusk/
    is_night = pandas NA rather than a guessed value.
    """
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col], errors="coerce", utc=True)

    is_dawn, is_dusk, is_night = [], [], []
    for i in range(len(df)):
        t = ts.iloc[i]
        lat = df[lat_col].iloc[i] if lat_col in df.columns else None
        lon = df[lon_col].iloc[i] if lon_col in df.columns else None
        dawn, sunrise, sunset, dusk = _sun_times_for_row(t, lat, lon)

        if dawn is None or t is pd.NaT:
            is_dawn.append(pd.NA)
            is_dusk.append(pd.NA)
            is_night.append(pd.NA)
            continue

        is_dawn.append(bool(dawn <= t < sunrise))
        is_dusk.append(bool(sunset <= t < dusk))
        is_night.append(bool(t < dawn or t >= dusk))

    df["is_dawn"] = pd.array(is_dawn, dtype="boolean")
    df["is_dusk"] = pd.array(is_dusk, dtype="boolean")
    df["is_night"] = pd.array(is_night, dtype="boolean")
    return df


def build_time_features(df: pd.DataFrame, timestamp_col: str,
                         lat_col: str = "latitude", lon_col: str = "longitude") -> pd.DataFrame:
    """Convenience wrapper applying all time/season feature functions in order."""
    df = add_cyclical_time_features(df, timestamp_col)
    df = add_migration_season_flag(df)
    df = add_dawn_dusk_night_features(df, timestamp_col, lat_col, lon_col)
    return df
