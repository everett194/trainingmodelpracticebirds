"""
data/ingest_trektellen.py
----------------------------
Loads Trektellen monitoring-site metadata and count-session observations.

This project NEVER scrapes Trektellen. It only accepts:
  - a user-authorized CSV/XLSX export
  - a documented API response (once the user supplies an authorized
    endpoint - not implemented here, see ESRI_DISCUSSION_NOTES.md)
  - manually downloaded data with recorded permission/provenance

load_provenance() REQUIRES the provenance sidecar described in
birdstrikegeo.schemas.trektellen.TREKTELLEN_PROVENANCE_FIELDS to exist
before counts are loaded - see DATA_DOWNLOAD_GUIDE.md.

Counts may arrive in either canonical LONG format (one row per
site x count-session x species) or WIDE format (one row per
site x count-session, one column per species) - convert_wide_to_long()
detects and normalizes either into the canonical long format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from birdstrikegeo.data.validate import (
    end_not_before_start,
    find_duplicate_sessions,
    nonnegative_mask,
    valid_coordinates_mask,
)
from birdstrikegeo.schemas.trektellen import (
    KNOWN_COUNT_TYPES,
    TREKTELLEN_COUNT_SCHEMA,
    TREKTELLEN_PROVENANCE_FIELDS,
)

# Metadata columns present on every long-format row, everything else in a
# wide export is assumed to be a per-species count column.
_LONG_FORMAT_REQUIRED_COLUMNS = {"species_common_name", "count"}
_WIDE_FORMAT_METADATA_COLUMNS = {
    "count_id", "site_id", "count_date", "start_time_local", "end_time_local",
    "observation_hours", "flight_direction", "count_type", "weather_notes",
    "observer_effort_available",
}


@dataclass
class TrektellenValidationReport:
    n_rows: int
    n_invalid_coordinates: int
    n_negative_counts: int
    n_invalid_time_ranges: int
    n_nonpositive_effort: int
    n_unresolved_site_ids: int
    duplicate_sessions: int
    count_type_counts: dict


def load_provenance(path: str | Path) -> dict:
    """
    Loads and validates the required provenance sidecar (a small JSON
    file) documenting how a Trektellen export was obtained and what its
    permitted use is. Raises ValueError if any required field is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trektellen provenance file not found: {path}. A provenance "
            f"record is REQUIRED before loading real Trektellen data - see "
            f"DATA_DOWNLOAD_GUIDE.md and birdstrikegeo.schemas.trektellen."
            f"TREKTELLEN_PROVENANCE_FIELDS."
        )
    provenance = json.loads(path.read_text())
    missing = [f for f in TREKTELLEN_PROVENANCE_FIELDS if f not in provenance]
    if missing:
        raise ValueError(f"Trektellen provenance file {path} is missing required fields: {missing}")
    return provenance


def load_sites(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trektellen sites file not found: {path}")
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    return df


def _is_wide_format(df: pd.DataFrame) -> bool:
    columns = set(df.columns)
    if _LONG_FORMAT_REQUIRED_COLUMNS.issubset(columns):
        return False
    # Wide format: metadata columns plus one column per species, no single
    # "species_common_name"/"count" pair.
    return bool(columns - _WIDE_FORMAT_METADATA_COLUMNS)


def convert_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts a wide export (one column per species) into canonical long
    format (one row per site x session x species). Species columns are
    whatever isn't a recognized metadata column; their header becomes
    species_common_name, and species_scientific_name/species_code are
    left null (unresolvable from a wide export without a species lookup
    table - left explicitly missing rather than guessed).
    """
    metadata_cols = [c for c in df.columns if c in _WIDE_FORMAT_METADATA_COLUMNS]
    species_cols = [c for c in df.columns if c not in _WIDE_FORMAT_METADATA_COLUMNS]

    long_df = df.melt(id_vars=metadata_cols, value_vars=species_cols,
                       var_name="species_common_name", value_name="count")
    long_df["count"] = pd.to_numeric(long_df["count"], errors="coerce").fillna(0)
    long_df = long_df[long_df["count"] > 0].reset_index(drop=True)  # wide exports use 0/blank for "not seen"
    long_df["species_scientific_name"] = pd.NA
    long_df["species_code"] = pd.NA
    if "count_id" not in long_df.columns:
        long_df["count_id"] = [f"WIDE{i:06d}" for i in range(len(long_df))]
    return long_df


def load_counts(path: str | Path, sites: pd.DataFrame) -> tuple[pd.DataFrame, TrektellenValidationReport]:
    """
    Loads a Trektellen count export (long or wide), converts to canonical
    long format if needed, and runs the validation checks described in
    the project spec. Returns (canonical_long_df, validation_report).
    Nothing is dropped automatically - callers decide what to do with
    invalid rows, using the report to know what's there.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trektellen counts file not found: {path}")
    raw = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)

    df = convert_wide_to_long(raw) if _is_wide_format(raw) else raw.copy()

    for col in TREKTELLEN_COUNT_SCHEMA:
        if col not in df.columns:
            df[col] = pd.NA

    known_site_ids = set(sites["site_id"].astype(str))
    referenced_site_ids = df["site_id"].astype(str)
    unresolved = referenced_site_ids[~referenced_site_ids.isin(known_site_ids)]

    site_coords = sites.set_index("site_id")[["latitude", "longitude"]]
    joined_coords = df.join(site_coords, on="site_id", rsuffix="_site")
    valid_coords = valid_coordinates_mask(joined_coords.get("latitude", pd.Series(dtype=float)),
                                           joined_coords.get("longitude", pd.Series(dtype=float)))

    valid_counts = nonnegative_mask(df["count"])
    valid_time_range = end_not_before_start(df["start_time_local"].astype("string"),
                                             df["end_time_local"].astype("string"))
    effort = pd.to_numeric(df["observation_hours"], errors="coerce")
    nonpositive_effort = effort.notna() & (effort <= 0)

    dup_check = find_duplicate_sessions(df, ["site_id", "count_date", "start_time_local", "species_scientific_name"])

    report = TrektellenValidationReport(
        n_rows=len(df),
        n_invalid_coordinates=int((~valid_coords).sum()),
        n_negative_counts=int((~valid_counts).sum()),
        n_invalid_time_ranges=int((~valid_time_range & df["start_time_local"].notna() & df["end_time_local"].notna()).sum()),
        n_nonpositive_effort=int(nonpositive_effort.sum()),
        n_unresolved_site_ids=int(unresolved.nunique()),
        duplicate_sessions=dup_check.n_duplicates,
        count_type_counts=df["count_type"].value_counts(dropna=False).to_dict(),
    )

    unknown_count_types = set(df["count_type"].dropna().unique()) - set(KNOWN_COUNT_TYPES)
    if unknown_count_types:
        # Not fatal - retained as-is (count_type is never silently
        # coerced), but surfaced so it shows up in the quality report.
        report.count_type_counts["_unrecognized_types"] = sorted(unknown_count_types)

    return df, report
