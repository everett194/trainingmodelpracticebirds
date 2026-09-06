"""
schemas/validation.py
------------------------
Generic schema-validation layer (Phase 2 deliverable): given any of this
project's canonical schemas (a plain `{column_name: dtype_string}` dict,
as already used throughout `birdstrikegeo.schemas.*`) and a DataFrame
that's supposed to conform to it, checks:

  - every required column is present
  - declared dtypes are at least coercible (never silently re-casts the
    caller's DataFrame — this is a report, not a mutator)
  - latitude/longitude columns (if given) are in valid coordinate ranges
  - timestamp columns (if given) parse to valid datetimes
  - duplicate rows by a caller-given key
  - missingness per column
  - "companion column" rules (e.g. schemas.hazard_observation's
    REQUIRES_COMPANION: a value column may only be non-null if its
    companion provenance/resolution column is also non-null)

This module composes the row-level primitives already in
`birdstrikegeo.data.validate` (valid_coordinates_mask,
valid_timestamp_mask, find_duplicate_sessions) rather than duplicating
them — this is the schema-level layer on top of those, usable by ANY
schema in this project (FAA, Trektellen, airports, weather, and the new
hazard_observation schema), not just the ones data/validate.py's
original callers were written for.

This module never raises on its own — like data/validate.py, it reports
findings via SchemaValidationResult and leaves the decision of what to
do (exclude rows, warn, fail a build) to the caller, via
`birdstrikegeo.data.quality_report`-style reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from birdstrikegeo.data.validate import find_duplicate_sessions, valid_coordinates_mask, valid_timestamp_mask


@dataclass
class SchemaValidationResult:
    schema_name: str
    n_rows: int
    missing_columns: list[str] = field(default_factory=list)
    unexpected_dtype_columns: dict[str, str] = field(default_factory=dict)  # column -> actual dtype found
    invalid_coordinate_rows: int = 0
    invalid_timestamp_columns: dict[str, int] = field(default_factory=dict)  # column -> count of unparseable values
    duplicate_key_count: int = 0
    duplicate_keys_sample: list = field(default_factory=list)
    missingness_by_column: dict[str, float] = field(default_factory=dict)  # column -> fraction missing, 0-1
    companion_violations: dict[str, int] = field(default_factory=dict)  # value_column -> count missing its companion

    @property
    def is_clean(self) -> bool:
        """
        True only if every structural check passed. Missingness itself is
        NOT a failure condition here (real data is allowed to be
        incomplete, per this project's whole philosophy of surfacing
        missingness rather than pretending it away) — only structural
        problems (missing columns, bad coordinates/timestamps,
        duplicates, or a value present without its required provenance
        companion) count against is_clean.
        """
        return not (
            self.missing_columns
            or self.invalid_coordinate_rows
            or any(self.invalid_timestamp_columns.values())
            or self.duplicate_key_count
            or any(self.companion_violations.values())
        )

    def to_dict(self) -> dict:
        return {
            "schema_name": self.schema_name,
            "n_rows": self.n_rows,
            "is_clean": self.is_clean,
            "missing_columns": self.missing_columns,
            "unexpected_dtype_columns": self.unexpected_dtype_columns,
            "invalid_coordinate_rows": self.invalid_coordinate_rows,
            "invalid_timestamp_columns": self.invalid_timestamp_columns,
            "duplicate_key_count": self.duplicate_key_count,
            "duplicate_keys_sample": self.duplicate_keys_sample[:20],
            "missingness_by_column": self.missingness_by_column,
            "companion_violations": self.companion_violations,
        }


# Pandas dtype-kind groups that are considered compatible with each
# schema dtype string. Deliberately permissive on numeric width/signedness
# (e.g. "float" accepts int64 too, since an all-integer column with no
# missing values is still a valid float column) — this validator checks
# structural conformance, not maximally strict typing.
_COMPATIBLE_KINDS: dict[str, tuple[str, ...]] = {
    "str": ("O", "U", "S", "string"),
    "float": ("f", "i", "u"),
    "int": ("i", "u"),
    "bool": ("b",),
}


def _dtype_is_compatible(schema_dtype: str, actual_dtype) -> bool:
    if schema_dtype.startswith("datetime64"):
        return actual_dtype.kind in ("M",) or str(actual_dtype) == "object"
    base = schema_dtype.split("[")[0]
    allowed_kinds = _COMPATIBLE_KINDS.get(base)
    if allowed_kinds is None:
        return True  # unknown declared type -- don't fail on something this validator doesn't understand
    kind = getattr(actual_dtype, "kind", None)
    return kind in allowed_kinds or str(actual_dtype) == "string"


def validate_dataframe(
    df: pd.DataFrame,
    schema: dict[str, str],
    *,
    schema_name: str,
    required_fields: tuple[str, ...] = (),
    lat_col: str | None = None,
    lon_col: str | None = None,
    timestamp_columns: tuple[str, ...] = (),
    duplicate_key_columns: tuple[str, ...] = (),
    requires_companion: tuple[tuple[str, str], ...] = (),
) -> SchemaValidationResult:
    """
    Validates `df` against `schema` (a {column: dtype_string} dict, e.g.
    birdstrikegeo.schemas.hazard_observation.HAZARD_OBSERVATION_SCHEMA).

    All parameters besides `df`/`schema`/`schema_name` are optional so
    this same function works for a schema with no natural coordinate
    columns, no timestamp columns, etc. — callers pass only what applies.
    """
    result = SchemaValidationResult(schema_name=schema_name, n_rows=len(df))

    result.missing_columns = [c for c in required_fields if c not in df.columns]

    for col, declared_dtype in schema.items():
        if col not in df.columns:
            continue
        if not _dtype_is_compatible(declared_dtype, df[col].dtype):
            result.unexpected_dtype_columns[col] = str(df[col].dtype)

    if lat_col and lon_col and lat_col in df.columns and lon_col in df.columns:
        valid_mask = valid_coordinates_mask(df[lat_col], df[lon_col])
        result.invalid_coordinate_rows = int((~valid_mask).sum())

    for ts_col in timestamp_columns:
        if ts_col not in df.columns:
            continue
        valid_mask = valid_timestamp_mask(df[ts_col])
        # Only genuinely-present-but-unparseable values count as
        # "invalid" -- a null timestamp is a missingness question,
        # reported separately below, not a structural validity failure.
        present = df[ts_col].notna()
        result.invalid_timestamp_columns[ts_col] = int((present & ~valid_mask).sum())

    if duplicate_key_columns and all(c in df.columns for c in duplicate_key_columns):
        dup = find_duplicate_sessions(df, list(duplicate_key_columns))
        result.duplicate_key_count = dup.n_duplicates
        result.duplicate_keys_sample = dup.duplicate_keys

    result.missingness_by_column = {
        c: round(float(df[c].isna().mean()), 4) for c in schema if c in df.columns
    }

    for value_col, companion_col in requires_companion:
        if value_col not in df.columns or companion_col not in df.columns:
            continue
        violating = df[value_col].notna() & df[companion_col].isna()
        result.companion_violations[value_col] = int(violating.sum())

    return result
