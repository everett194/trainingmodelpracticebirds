"""
data/ingest_faa.py
--------------------
Loads a raw FAA Wildlife Strike Database export (CSV or XLSX), resolves
its columns to the canonical schema via alias matching, and returns:

  - a canonical DataFrame (only recognized columns, canonically named)
  - an "audit" DataFrame that preserves every ORIGINAL column exactly as
    read, indexed the same way, so nothing from the source file is ever
    silently lost even if a column wasn't recognized.

This module does not build the supervised target or drop leakage
columns - that happens in birdstrikegeo.features.build_damage_features,
which is a deliberately separate step so "load the data" and "decide
what's allowed as a model input" are never accidentally conflated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from birdstrikegeo.data.column_aliases import missing_required_columns, resolve_columns
from birdstrikegeo.schemas.faa import FAA_SCHEMA


@dataclass
class FaaIngestResult:
    data: pd.DataFrame  # canonical columns only
    audit: pd.DataFrame  # every original column, unmodified
    resolved_columns: dict[str, str]  # canonical -> original column name
    missing_columns: list[str]  # canonical fields not found in the source
    source_path: Path
    n_rows_read: int


def _read_raw(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=True)
    raise ValueError(f"Unsupported FAA file format: {path.suffix} (expected .csv or .xlsx)")


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "incident_date" in df.columns:
        df["incident_date"] = pd.to_datetime(df["incident_date"], errors="coerce")
    for numeric_col in ("latitude", "longitude", "height_agl_ft", "speed_ias_knots", "engine_count"):
        if numeric_col in df.columns:
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")
    return df


def ingest_faa(path: str | Path) -> FaaIngestResult:
    """
    Reads a raw FAA export and resolves it against FAA_SCHEMA. Raises
    ValueError (via resolve_columns) if the file has genuinely ambiguous
    columns that could map to more than one canonical field - this is a
    hard stop, not a warning, because guessing wrong here would corrupt
    every downstream model.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"FAA source file not found: {path}")

    raw = _read_raw(path)
    resolved = resolve_columns(list(raw.columns))
    missing = missing_required_columns(resolved)

    canonical = pd.DataFrame(index=raw.index)
    for canonical_name, raw_col in resolved.items():
        canonical[canonical_name] = raw[raw_col]

    # Ensure every FAA_SCHEMA column exists, even if unresolved, so
    # downstream code can rely on a stable column set. Unresolved columns
    # are entirely null, and are also listed in `missing_columns` so
    # nothing pretends to know something it doesn't.
    for canonical_name in FAA_SCHEMA:
        if canonical_name not in canonical.columns:
            canonical[canonical_name] = pd.NA

    canonical = canonical[list(FAA_SCHEMA.keys())]
    canonical = _coerce_types(canonical)

    return FaaIngestResult(
        data=canonical,
        audit=raw.copy(),
        resolved_columns=resolved,
        missing_columns=missing,
        source_path=path,
        n_rows_read=len(raw),
    )
