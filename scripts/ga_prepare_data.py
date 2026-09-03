#!/usr/bin/env python3
"""
scripts/ga_prepare_data.py
------------------------------
GA conditional-damage milestone, step 1 of 2 (see scripts/ga_train_models.py
for training). Caches the raw FAA workbook as Parquet, filters to the
confirmed business/private fixed-wing GA population, builds the
damage_binary target, and writes:

  data/processed/ga/ga_filtered.parquet   - labeled GA rows (gitignored)
  reports/latest/ga_population_summary.json

Usage:
    python scripts/ga_prepare_data.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from birdstrikegeo.ga.data import (  # noqa: E402
    build_damage_binary_target,
    compute_population_counts,
    load_raw_workbook,
    normalize_raw_columns,
    POPULATION_DEFINITIONS,
    summarize_target,
)

CONFIG_PATH = REPO_ROOT / "configs" / "ga_damage.yaml"


def sha256_of_file(path: Path, chunk_size: int = 2**20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    paths = config["paths"]
    xlsx_path = REPO_ROOT / paths["raw_workbook"]
    parquet_cache = REPO_ROOT / paths["raw_cache_parquet"]

    if not xlsx_path.exists():
        print(f"[ga_prepare_data] Missing {xlsx_path}. See DATA_DOWNLOAD_GUIDE.md.")
        sys.exit(1)

    print(f"[ga_prepare_data] Loading raw workbook (cached at {parquet_cache} after first run)...")
    raw = load_raw_workbook(xlsx_path, parquet_cache)
    print(f"[ga_prepare_data] {len(raw):,} total source records, {len(raw.columns)} raw columns.")

    df = normalize_raw_columns(raw)
    counts, principal_mask = compute_population_counts(df)
    print("[ga_prepare_data] Population counts:")
    for line in counts.as_report_lines():
        print(f"  {line}")

    ga = df[principal_mask].copy()
    ga_labeled = build_damage_binary_target(ga)
    target_report = summarize_target(ga_labeled)
    print(f"[ga_prepare_data] Target report: {target_report.as_dict()}")

    eligible = ga_labeled[ga_labeled["damage_binary"].notna()].copy()
    print(f"[ga_prepare_data] {len(eligible):,} rows eligible for supervised training "
          f"({len(ga_labeled) - len(eligible):,} excluded as unknown/ambiguous).")

    processed_dir = REPO_ROOT / paths["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "ga_filtered.parquet"
    eligible.to_parquet(out_path, index=False)
    print(f"[ga_prepare_data] Wrote {out_path}")

    reports_dir = REPO_ROOT / paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_filename": xlsx_path.name,
        "source_sha256": sha256_of_file(xlsx_path),
        "total_source_records": int(len(raw)),
        "ga_filter_description": POPULATION_DEFINITIONS["confirmed_business_or_private_fixed_wing"],
        "population_counts": {name: {"count": n, "definition": POPULATION_DEFINITIONS[name]}
                               for name, n in counts.counts.items()},
        "principal_population_count": counts.counts["confirmed_business_or_private_fixed_wing"],
        "target_report": target_report.as_dict(),
        "n_eligible_for_training": int(len(eligible)),
    }
    (reports_dir / "ga_population_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[ga_prepare_data] Wrote {reports_dir / 'ga_population_summary.json'}")


if __name__ == "__main__":
    run()
