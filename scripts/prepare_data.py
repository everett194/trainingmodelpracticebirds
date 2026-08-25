#!/usr/bin/env python3
"""
scripts/prepare_data.py
--------------------------
Task A (damage prediction) data-preparation pipeline:
FAA ingestion -> column-alias resolution -> damage target construction
-> leakage-safe feature frame -> chronological split -> quality report.

Usage:
    python scripts/prepare_data.py --mode sample
    python scripts/prepare_data.py --mode real

`--mode real` fails gracefully (prints exactly what's missing and how to
get it) rather than crashing with a stack trace, if data/raw/ doesn't yet
have the required file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from birdstrikegeo.data.ingest_faa import ingest_faa  # noqa: E402
from birdstrikegeo.data.quality_report import QualityReportBuilder, missingness_by_column, value_counts_section  # noqa: E402
from birdstrikegeo.data.validate import valid_coordinates_mask, valid_timestamp_mask  # noqa: E402
from birdstrikegeo.features.build_damage_features import build_feature_frame, make_damage_target  # noqa: E402
from birdstrikegeo.training.splits import chronological_split  # noqa: E402

SAMPLE_FAA_PATH = REPO_ROOT / "data" / "sample" / "faa_strikes_sample.csv"
REAL_FAA_PATH = REPO_ROOT / "data" / "raw" / "faa_wildlife_strikes.csv"
REAL_FAA_PATH_XLSX = REPO_ROOT / "data" / "raw" / "faa_wildlife_strikes.xlsx"

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
INTERIM_DIR = REPO_ROOT / "data" / "interim"
RESULTS_DIR = REPO_ROOT / "results" / "damage"


def resolve_faa_source(mode: str) -> Path:
    if mode == "sample":
        return SAMPLE_FAA_PATH
    if REAL_FAA_PATH.exists():
        return REAL_FAA_PATH
    if REAL_FAA_PATH_XLSX.exists():
        return REAL_FAA_PATH_XLSX
    print("[prepare_data] --mode real requires a real FAA export. Missing file(s):")
    print(f"  - {REAL_FAA_PATH}  (or {REAL_FAA_PATH_XLSX.name})")
    print()
    print("See DATA_DOWNLOAD_GUIDE.md for where to download the FAA Wildlife")
    print("Strike Database export and what filename to save it as.")
    sys.exit(1)


def run(mode: str) -> None:
    is_sample = mode == "sample"
    source_path = resolve_faa_source(mode)

    print(f"[prepare_data] mode={mode} source={source_path}")
    result = ingest_faa(source_path)
    print(f"[prepare_data] Read {result.n_rows_read} rows. "
          f"Resolved {len(result.resolved_columns)}/{len(result.missing_columns) + len(result.resolved_columns)} "
          f"expected columns.")
    if result.missing_columns:
        print(f"[prepare_data] Columns not found in source (left null): {result.missing_columns}")

    with_target = make_damage_target(result.data)
    status_counts = with_target["damage_target_status"].value_counts().to_dict()
    print(f"[prepare_data] Damage target status counts: {status_counts}")

    eligible = with_target[with_target["damage"].notna()].copy()
    excluded = with_target[with_target["damage"].isna()].copy()
    print(f"[prepare_data] {len(eligible)} rows eligible for supervised training; "
          f"{len(excluded)} excluded (unknown/ambiguous damage outcome).")

    valid_coords = valid_coordinates_mask(eligible["latitude"], eligible["longitude"])
    valid_dates = valid_timestamp_mask(eligible["incident_date"])
    print(f"[prepare_data] Valid coordinates: {int(valid_coords.sum())}/{len(eligible)}; "
          f"valid incident dates: {int(valid_dates.sum())}/{len(eligible)}")

    features = build_feature_frame(eligible)
    features["damage"] = eligible["damage"].values
    features["incident_date"] = eligible["incident_date"].values

    split = chronological_split(features, date_column="incident_date")
    print(f"[prepare_data] Chronological split: train={len(split.train)} "
          f"(through {split.train_end_date.date() if pd.notna(split.train_end_date) else 'n/a'}), "
          f"validation={len(split.validation)} "
          f"(through {split.validation_end_date.date() if pd.notna(split.validation_end_date) else 'n/a'}), "
          f"test={len(split.test)}")

    out_dir = PROCESSED_DIR / ("sample" if is_sample else "real")
    out_dir.mkdir(parents=True, exist_ok=True)
    split.train.to_csv(out_dir / "damage_train.csv", index=False)
    split.validation.to_csv(out_dir / "damage_validation.csv", index=False)
    split.test.to_csv(out_dir / "damage_test.csv", index=False)
    excluded.to_csv(out_dir / "damage_excluded_unknown_target.csv", index=False)
    print(f"[prepare_data] Wrote train/validation/test/excluded CSVs to {out_dir}")

    report = QualityReportBuilder(
        title="FAA Damage-Prediction Data Quality Report",
        sample_data_flag=is_sample,
    )
    report.add_source("faa_strikes", source_path, result.n_rows_read)
    report.add_section("column_resolution", {
        "resolved": len(result.resolved_columns),
        "missing": result.missing_columns,
    })
    report.add_section("target_construction", status_counts)
    report.add_section("validity", {
        "valid_coordinates": int(valid_coords.sum()),
        "valid_incident_dates": int(valid_dates.sum()),
        "total_eligible_rows": len(eligible),
    })
    report.add_section("split_sizes", {
        "train": len(split.train),
        "validation": len(split.validation),
        "test": len(split.test),
        "train_end_date": str(split.train_end_date),
        "validation_end_date": str(split.validation_end_date),
    })
    report.add_section("records_by_year", value_counts_section(eligible["incident_date"].dt.year))
    report.add_section("records_by_state", value_counts_section(eligible["state"]))
    report.add_section("missingness_by_column", missingness_by_column(eligible))
    report.add_section("unresolved_airport_ids", int((eligible["airport_id"].isna() | (eligible["airport_id"] == "")).sum()))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report.write(
        RESULTS_DIR / f"data_quality_{mode}.json",
        RESULTS_DIR / f"data_quality_{mode}.md",
    )
    print(f"[prepare_data] Wrote quality report to {RESULTS_DIR}/data_quality_{mode}.{{json,md}}")
    if is_sample:
        print("[prepare_data] SAMPLE DATA — NOT A REAL RESULT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sample", "real"], default="sample")
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
