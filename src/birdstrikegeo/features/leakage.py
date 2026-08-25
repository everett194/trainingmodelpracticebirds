"""
features/leakage.py
---------------------
A single, deliberately simple guard against target leakage: any column
name that appears in FORBIDDEN_COLUMNS must never reach a model's
feature matrix, because it describes the strike's OUTCOME rather than
information available beforehand.

Every training/inference code path that builds a feature matrix must
call assert_no_leakage() on the final column list immediately before
fitting or predicting. tests/test_leakage.py enforces that this
actually happens and actually fails loudly.
"""

from __future__ import annotations

from birdstrikegeo.schemas.faa import FORBIDDEN_LEAKAGE_COLUMNS, TARGET_SOURCE_COLUMNS

# Columns that must never appear in a feature matrix passed to a model.
# Includes both the FAA outcome fields AND the raw columns the target
# itself was constructed from (once damage/damage_source columns exist,
# the ORIGINAL damage_flag/damage_level text columns are still off-limits
# as inputs - only the resulting binary target may be used, and only as
# a label, never as a feature).
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(FORBIDDEN_LEAKAGE_COLUMNS) | frozenset(TARGET_SOURCE_COLUMNS)


class LeakageError(ValueError):
    """Raised when a forbidden (outcome-describing) column is found in a feature matrix."""


def assert_no_leakage(columns, extra_forbidden: tuple[str, ...] = ()) -> None:
    """
    columns: iterable of column names about to be used as model inputs.
    extra_forbidden: additional column names to forbid (e.g. the target
    column itself, or a Trektellen equivalent), so callers aren't forced
    to only rely on the FAA-specific default list.

    Raises LeakageError listing every offending column found, so a
    developer can fix all of them at once rather than one at a time.
    """
    forbidden = FORBIDDEN_COLUMNS | frozenset(extra_forbidden)
    offending = sorted(set(columns) & forbidden)
    if offending:
        raise LeakageError(
            f"Forbidden (outcome-describing) columns found in feature matrix: "
            f"{offending}. These describe the result of a strike, not information "
            f"available beforehand, and must never be used as model inputs."
        )
