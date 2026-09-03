#!/usr/bin/env python3
"""
scripts/validate_data.py
----------------------------
Checks which files configs/data_sources.yaml expects are actually
present, and reports required-but-missing ones with instructions rather
than letting a later pipeline step fail with a confusing stack trace.
For --mode real, exits non-zero if any REQUIRED source is missing (a
graceful failure, per project requirements) - optional sources are
reported but never block anything.

Usage:
    python scripts/validate_data.py --mode sample
    python scripts/validate_data.py --mode real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "data_sources.yaml"

SAMPLE_EQUIVALENTS = {
    "data/raw/faa_wildlife_strikes.csv": "data/sample/faa_strikes_sample.csv",
    "data/raw/trektellen_sites.csv": "data/sample/trektellen_sites_sample.csv",
    "data/raw/trektellen_counts.csv": "data/sample/trektellen_counts_sample.csv",
    "data/raw/airports.geojson": "data/sample/airports_sample.geojson",
    "data/raw/hourly_weather.parquet": "data/sample/weather_sample.csv",
    # bts_t100_departures has no sample fixture yet (reserved for future
    # flight-exposure work - see README.md "Scientific limitations"), so
    # sample mode just checks the same raw path, which is expected to be
    # absent - it's optional either way.
    "data/raw/bts_t100_departures.csv": "data/raw/bts_t100_departures.csv",
}


def run(mode: str) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    print(f"[validate_data] mode={mode}\n")

    missing_required = []
    for name, source in config["sources"].items():
        raw_rel_path = source["path"]
        if mode == "sample":
            path = REPO_ROOT / SAMPLE_EQUIVALENTS[raw_rel_path]
            candidates = [path]
        else:
            base = REPO_ROOT / raw_rel_path
            # Also accept any other declared format's extension for this
            # source (e.g. faa_strikes declares [csv, xlsx] but path is
            # the .csv form) - see ingest_faa.py, which reads either.
            candidates = [base.with_suffix(f".{fmt}") for fmt in source.get("formats", [])] or [base]
        path = next((c for c in candidates if c.exists()), candidates[0])
        exists = any(c.exists() for c in candidates)
        required = source["required"] and mode == "real"
        status = "OK" if exists else ("MISSING (required)" if required else "missing (optional)")
        print(f"  [{name}] {status}: {path}")
        if required and not exists:
            missing_required.append((name, raw_rel_path))

    if missing_required:
        print("\n[validate_data] --mode real is missing required source(s):")
        for name, raw_rel_path in missing_required:
            print(f"  - {name}: expected at {raw_rel_path}")
        print("\nSee DATA_DOWNLOAD_GUIDE.md for how to obtain these.")
        sys.exit(1)

    print("\n[validate_data] All required sources present.")
    if mode == "sample":
        print("[validate_data] SAMPLE DATA — NOT A REAL RESULT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sample", "real"], default="sample")
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
