#!/usr/bin/env python3
"""
scripts/build_trektellen_annual_dataset.py
------------------------------------------------
Discovers every annual-totals Trektellen PDF in data/reference/
(pattern: trektellen*annual*.pdf), parses and merges them, and writes
the combined long-format dataset to data/processed/trektellen/.

This is the "update automatically as I add more data" entry point: drop
a new/updated annual-totals PDF into data/reference/ and re-run this
script -- no code changes needed. It's also the first step
scripts/build_eastern_shore_risk_report.py now runs automatically (see
that script), so a normal report build already picks up whatever's in
data/reference/ without a separate manual step -- this script exists
standalone too for when you just want to regenerate/inspect the merged
dataset on its own.

Usage:
    python scripts/build_trektellen_annual_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from birdstrikegeo.data.ingest_trektellen_annual_pdf import (  # noqa: E402
    aggregate_annual_local_activity,
    discover_annual_pdfs,
    parse_and_merge_annual_pdfs,
)

REFERENCE_DIR = REPO_ROOT / "data" / "reference"
OUT_DIR = REPO_ROOT / "data" / "processed" / "trektellen"


def run() -> None:
    pdf_paths = discover_annual_pdfs(REFERENCE_DIR)
    if not pdf_paths:
        print(f"[build_trektellen_annual_dataset] No annual-totals PDFs found in {REFERENCE_DIR} "
              f"(pattern: trektellen*annual*.pdf). Nothing to do.")
        return

    print(f"[build_trektellen_annual_dataset] Found {len(pdf_paths)} file(s): "
          f"{', '.join(p.name for p in pdf_paths)}")

    long_df = parse_and_merge_annual_pdfs(pdf_paths)
    conflicts = long_df.attrs.get("conflicts")
    if conflicts is not None and len(conflicts):
        print(f"[build_trektellen_annual_dataset] WARNING: {len(conflicts)} species/year pairs "
              f"had disagreeing values across source files (most-recently-modified file's value "
              f"was kept for each) -- see the printed table below.")
        print(conflicts.to_string(index=False))

    activity = aggregate_annual_local_activity(long_df)

    # pandas tries to JSON-serialize df.attrs into parquet metadata;
    # the conflicts DataFrame/source_files list attached above aren't
    # JSON-serializable and would crash to_parquet -- already
    # printed/consumed above, safe to clear before writing.
    long_df.attrs.clear()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(OUT_DIR / "annual_long.parquet", index=False)
    activity.to_csv(OUT_DIR / "annual_local_activity.csv", index=False)

    n_monitored_years = long_df.loc[long_df["monitored"], "year"].nunique()
    total_hours = long_df.loc[long_df["monitored"], ["year", "observation_hours"]].drop_duplicates()["observation_hours"].sum()
    print(f"[build_trektellen_annual_dataset] Wrote {OUT_DIR}/annual_long.parquet "
          f"({long_df['species'].nunique()} species x {long_df['year'].nunique()} years) and "
          f"annual_local_activity.csv ({len(activity)} species with >=1 monitored year, "
          f"{n_monitored_years} monitored years, {total_hours:.0f} total observation hours pooled).")


if __name__ == "__main__":
    run()
