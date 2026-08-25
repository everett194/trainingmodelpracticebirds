#!/usr/bin/env python3
"""
scripts/build_geospatial_features.py
----------------------------------------
Task B (airport-period wildlife activity index) geospatial pipeline:
Trektellen ingestion -> validation -> geospatial layers -> per-airport
activity features -> activity_index -> exports (GeoJSON/GeoPackage/
GeoParquet).

Usage:
    python scripts/build_geospatial_features.py --mode sample
    python scripts/build_geospatial_features.py --mode real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from birdstrikegeo.data.ingest_airports import load_airports  # noqa: E402
from birdstrikegeo.data.ingest_trektellen import load_counts, load_sites  # noqa: E402
from birdstrikegeo.data.quality_report import QualityReportBuilder  # noqa: E402
from birdstrikegeo.features.build_activity_features import build_activity_features, prepare_counts_with_timestamps  # noqa: E402
from birdstrikegeo.geo.airport_layers import build_airport_buffers, build_airport_layer  # noqa: E402
from birdstrikegeo.geo.exports import LayerMetadata, export_layer, export_table  # noqa: E402
from birdstrikegeo.geo.trektellen_layers import build_airport_trektellen_links, build_trektellen_sites_layer  # noqa: E402
from birdstrikegeo.models.activity_index import compute_activity_index  # noqa: E402

SAMPLE_AIRPORTS_PATH = REPO_ROOT / "data" / "sample" / "airports_sample.geojson"
SAMPLE_SITES_PATH = REPO_ROOT / "data" / "sample" / "trektellen_sites_sample.csv"
SAMPLE_COUNTS_PATH = REPO_ROOT / "data" / "sample" / "trektellen_counts_sample.csv"

REAL_AIRPORTS_PATH = REPO_ROOT / "data" / "raw" / "airports.geojson"
REAL_SITES_PATH = REPO_ROOT / "data" / "raw" / "trektellen_sites.csv"
REAL_COUNTS_PATH = REPO_ROOT / "data" / "raw" / "trektellen_counts.csv"
REAL_PROVENANCE_PATH = REPO_ROOT / "data" / "raw" / "trektellen_provenance.json"

LAYERS_DIR = REPO_ROOT / "data" / "processed" / "layers"
RESULTS_DIR = REPO_ROOT / "results" / "activity"


def resolve_paths(mode: str) -> tuple[Path, Path, Path]:
    if mode == "sample":
        return SAMPLE_AIRPORTS_PATH, SAMPLE_SITES_PATH, SAMPLE_COUNTS_PATH

    missing = [p for p in (REAL_AIRPORTS_PATH, REAL_SITES_PATH, REAL_COUNTS_PATH) if not p.exists()]
    if missing:
        print("[build_geospatial_features] --mode real requires real data. Missing file(s):")
        for p in missing:
            print(f"  - {p}")
        print("\nSee DATA_DOWNLOAD_GUIDE.md for how to obtain these.")
        sys.exit(1)
    if not REAL_PROVENANCE_PATH.exists():
        print(f"[build_geospatial_features] Missing required Trektellen provenance file: {REAL_PROVENANCE_PATH}")
        print("Real Trektellen data requires a documented provenance record - see DATA_DOWNLOAD_GUIDE.md.")
        sys.exit(1)
    return REAL_AIRPORTS_PATH, REAL_SITES_PATH, REAL_COUNTS_PATH


def run(mode: str) -> None:
    is_sample = mode == "sample"
    airports_path, sites_path, counts_path = resolve_paths(mode)

    print(f"[build_geospatial_features] mode={mode}")
    airports_gdf = load_airports(airports_path)
    sites = load_sites(sites_path)
    counts, validation_report = load_counts(counts_path, sites)
    print(f"[build_geospatial_features] Loaded {len(airports_gdf)} airports, {len(sites)} Trektellen sites, "
          f"{len(counts)} count rows.")
    print(f"[build_geospatial_features] Trektellen validation: {validation_report}")

    counts_with_ts = prepare_counts_with_timestamps(counts, sites)
    n_unparseable_ts = counts_with_ts["count_timestamp_utc"].isna().sum()
    if n_unparseable_ts:
        print(f"[build_geospatial_features] WARNING: {n_unparseable_ts} count rows have an "
              f"unparseable timestamp and are excluded from all time-windowed features.")

    # --- Geospatial layers ---
    LAYERS_DIR.mkdir(parents=True, exist_ok=True)
    metadata = LayerMetadata(source=f"build_geospatial_features.py --mode {mode}", sample_data_flag=is_sample)

    export_layer(airports_gdf, LAYERS_DIR, "airports", metadata)
    sites_gdf = build_trektellen_sites_layer(sites)
    export_layer(sites_gdf, LAYERS_DIR, "trektellen_sites", metadata)

    for radius in (50, 100, 250):
        buffers = build_airport_buffers(airports_gdf, radius)
        export_layer(buffers, LAYERS_DIR, f"airport_buffers_{radius}km", metadata, formats=("geojson", "gpkg"))

    links = build_airport_trektellen_links(airports_gdf, sites, max_radius_km=250.0)
    export_table(links, LAYERS_DIR, "airport_trektellen_links", metadata)
    print(f"[build_geospatial_features] Wrote airports/trektellen_sites/buffer/link layers to {LAYERS_DIR}")

    # --- Per-airport activity features (evaluated "now", i.e. at the
    # most recent timestamp present in the count data, for demonstration) ---
    query_timestamp = pd.to_datetime(counts_with_ts["count_timestamp_utc"], utc=True).max()
    print(f"[build_geospatial_features] Computing activity features as of {query_timestamp} "
          f"(latest available observation timestamp).")

    activity_rows = []
    for _, airport in airports_gdf.iterrows():
        features = build_activity_features(
            query_lon=airport["longitude"], query_lat=airport["latitude"],
            query_timestamp_utc=query_timestamp, sites=sites, counts_with_ts=counts_with_ts,
        )
        index_result = compute_activity_index(features)
        activity_rows.append(
            {
                "airport_id": airport["airport_id"],
                "airport_name": airport["airport_name"],
                "latitude": airport["latitude"],
                "longitude": airport["longitude"],
                "query_timestamp_utc": str(query_timestamp),
                **features,
                **index_result,
            }
        )

    activity_df = pd.DataFrame(activity_rows)
    activity_gdf = build_airport_layer(activity_df)
    export_layer(activity_gdf, LAYERS_DIR, "airport_period_activity", metadata)
    print(f"[build_geospatial_features] Wrote airport_period_activity layer for {len(activity_df)} airports.")

    n_with_index = activity_df["activity_index"].notna().sum()
    print(f"[build_geospatial_features] activity_index computed for {n_with_index}/{len(activity_df)} airports "
          f"({len(activity_df) - n_with_index} had no qualifying recent observations).")

    # --- Quality report ---
    report = QualityReportBuilder(title="Trektellen / Geospatial Data Quality Report", sample_data_flag=is_sample)
    report.add_source("airports", airports_path, len(airports_gdf))
    report.add_source("trektellen_sites", sites_path, len(sites))
    report.add_source("trektellen_counts", counts_path, len(counts))
    report.add_section("trektellen_validation", vars(validation_report))
    report.add_section("unparseable_count_timestamps", int(n_unparseable_ts))
    report.add_section("activity_index_coverage", {
        "airports_with_index": int(n_with_index),
        "airports_without_index": int(len(activity_df) - n_with_index),
    })
    report.add_section("trektellen_coverage_by_airport", {
        row["airport_id"]: row["number_of_sites_within_250_km"] for row in activity_rows
    })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report.write(RESULTS_DIR / f"geospatial_quality_{mode}.json", RESULTS_DIR / f"geospatial_quality_{mode}.md")
    print(f"[build_geospatial_features] Wrote quality report to {RESULTS_DIR}/geospatial_quality_{mode}.{{json,md}}")
    if is_sample:
        print("[build_geospatial_features] SAMPLE DATA — NOT A REAL RESULT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sample", "real"], default="sample")
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
