"""
data/ingest_weather.py
--------------------------
Loads hourly station weather (Parquet preferred, CSV also accepted).
Optional data source - see configs/data_sources.yaml. If unavailable,
downstream joins record weather_available=false rather than fabricating
values (see birdstrikegeo.geo.temporal_join).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from birdstrikegeo.schemas.weather import WEATHER_SCHEMA


def load_weather(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weather file not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported weather file format: {path.suffix} (expected .parquet or .csv)")

    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    for col in WEATHER_SCHEMA:
        if col not in df.columns:
            df[col] = pd.NA

    return df
