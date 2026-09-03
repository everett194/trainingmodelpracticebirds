#!/usr/bin/env python3
"""
scripts/inspect_raw_schemas.py
----------------------------------
A fast, read-only look at whatever raw or sample data files are present:
which FAA columns resolve via alias matching (and which don't), and
whether the Trektellen sites/counts files have the expected columns.
Doesn't build features or train anything - just answers "will ingestion
work, and what will it find?" before you run the full pipeline.

Usage:
    python scripts/inspect_raw_schemas.py --mode sample
    python scripts/inspect_raw_schemas.py --mode real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from birdstrikegeo.data.column_aliases import missing_required_columns, resolve_columns  # noqa: E402
from birdstrikegeo.schemas.trektellen import TREKTELLEN_COUNT_SCHEMA, TREKTELLEN_SITE_SCHEMA  # noqa: E402

PATHS = {
    "sample": {
        "faa": REPO_ROOT / "data" / "sample" / "faa_strikes_sample.csv",
        "trektellen_sites": REPO_ROOT / "data" / "sample" / "trektellen_sites_sample.csv",
        "trektellen_counts": REPO_ROOT / "data" / "sample" / "trektellen_counts_sample.csv",
        "airports": REPO_ROOT / "data" / "sample" / "airports_sample.geojson",
    },
    "real": {
        "faa": REPO_ROOT / "data" / "raw" / "faa_wildlife_strikes.csv",
        "trektellen_sites": REPO_ROOT / "data" / "raw" / "trektellen_sites.csv",
        "trektellen_counts": REPO_ROOT / "data" / "raw" / "trektellen_counts.csv",
        "airports": REPO_ROOT / "data" / "raw" / "airports.geojson",
    },
}

# faa_wildlife_strikes is also accepted as .xlsx (see ingest_faa.py) - fall
# back to it if the .csv form isn't present.
_REAL_FAA_XLSX = REPO_ROOT / "data" / "raw" / "faa_wildlife_strikes.xlsx"
if not PATHS["real"]["faa"].exists() and _REAL_FAA_XLSX.exists():
    PATHS["real"]["faa"] = _REAL_FAA_XLSX


def inspect_faa(path: Path) -> None:
    if not path.exists():
        print(f"  [faa] MISSING: {path}")
        return
    columns = list(pd.read_csv(path, nrows=1).columns) if path.suffix == ".csv" else list(pd.read_excel(path, nrows=1).columns)
    resolved = resolve_columns(columns)
    missing = missing_required_columns(resolved)
    print(f"  [faa] {path.name}: {len(columns)} raw columns, {len(resolved)} resolved, {len(missing)} unresolved")
    if missing:
        print(f"         Unresolved (will be null): {missing}")


def inspect_trektellen_table(path: Path, expected_schema: dict, label: str) -> None:
    if not path.exists():
        print(f"  [{label}] MISSING: {path}")
        return
    columns = set(pd.read_csv(path, nrows=1).columns)
    expected = set(expected_schema.keys())
    present = expected & columns
    missing = expected - columns
    print(f"  [{label}] {path.name}: {len(present)}/{len(expected)} expected columns present")
    if missing:
        print(f"         Missing: {sorted(missing)}")


def run(mode: str) -> None:
    paths = PATHS[mode]
    print(f"[inspect_raw_schemas] mode={mode}\n")
    inspect_faa(paths["faa"])
    inspect_trektellen_table(paths["trektellen_sites"], TREKTELLEN_SITE_SCHEMA, "trektellen_sites")
    inspect_trektellen_table(paths["trektellen_counts"], TREKTELLEN_COUNT_SCHEMA, "trektellen_counts")
    print(f"  [airports] {'present' if paths['airports'].exists() else 'MISSING (optional)'}: {paths['airports']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sample", "real"], default="sample")
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
