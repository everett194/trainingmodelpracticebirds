"""
training/splits.py
---------------------
Chronological train/validation/test splitting, plus optional holdout
strategies (by airport, by year, by geographic region, by Trektellen
coverage) used for extra robustness evaluation.

Chronological splitting means the model is always evaluated on data that
came AFTER anything it was trained or tuned on - a much more honest
estimate of real-world performance than a random shuffle-split, which
would let the model "see the future" relative to some of its test
examples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ChronologicalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end_date: pd.Timestamp
    validation_end_date: pd.Timestamp


def chronological_split(
    df: pd.DataFrame,
    date_column: str,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> ChronologicalSplit:
    """
    Sorts df by date_column (ascending) and splits by ROW COUNT quantile
    (not by fixed calendar dates), so the split fractions hold even when
    incident volume is uneven across years. The resulting date boundaries
    are then reported back to the caller for transparency/logging.

    Works identically for the sample dataset (few years, small N) and
    real FAA-scale data (decades, huge N) - the same code path is
    exercised either way, per project requirements.
    """
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("train_fraction and validation_fraction must each be in (0, 1)")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be < 1 (test gets the remainder)")

    working = df.copy()
    working["_sort_date"] = pd.to_datetime(working[date_column], errors="coerce")
    working = working[working["_sort_date"].notna()].sort_values("_sort_date").reset_index(drop=True)

    n = len(working)
    train_end_idx = int(np.floor(n * train_fraction))
    val_end_idx = int(np.floor(n * (train_fraction + validation_fraction)))

    train = working.iloc[:train_end_idx]
    validation = working.iloc[train_end_idx:val_end_idx]
    test = working.iloc[val_end_idx:]

    train_end_date = train["_sort_date"].max() if len(train) else pd.NaT
    validation_end_date = validation["_sort_date"].max() if len(validation) else pd.NaT

    return ChronologicalSplit(
        train=train.drop(columns=["_sort_date"]),
        validation=validation.drop(columns=["_sort_date"]),
        test=test.drop(columns=["_sort_date"]),
        train_end_date=train_end_date,
        validation_end_date=validation_end_date,
    )


def holdout_by_airport(df: pd.DataFrame, airport_column: str, held_out_airport_ids: list[str]):
    """Returns (train_df, held_out_df) - all rows for the given airport
    IDs go entirely into the held-out set, never split across sets, so
    the model is evaluated on airports it has genuinely never seen."""
    is_held_out = df[airport_column].isin(held_out_airport_ids)
    return df[~is_held_out], df[is_held_out]


def holdout_latest_year(df: pd.DataFrame, date_column: str):
    """Returns (train_df, held_out_df) where held_out_df is every row in
    the single most recent calendar year present in the data."""
    dates = pd.to_datetime(df[date_column], errors="coerce")
    latest_year = dates.dt.year.max()
    is_latest = dates.dt.year == latest_year
    return df[~is_latest], df[is_latest], latest_year


def holdout_by_region(df: pd.DataFrame, region_column: str, held_out_regions: list[str]):
    is_held_out = df[region_column].isin(held_out_regions)
    return df[~is_held_out], df[is_held_out]


def holdout_by_trektellen_coverage(df: pd.DataFrame, coverage_column: str = "trektellen_available"):
    """Splits by whether Trektellen coverage was available for the row,
    to evaluate whether the model behaves differently under sparse
    monitoring (see 'low_coverage_warning' in geo/spatial_join.py)."""
    covered = df[coverage_column].astype(bool)
    return df[covered], df[~covered]
