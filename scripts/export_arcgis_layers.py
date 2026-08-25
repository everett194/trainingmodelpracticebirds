#!/usr/bin/env python3
"""
scripts/export_arcgis_layers.py
------------------------------------
Runs (or reuses) the geospatial pipeline's layer exports, then prepares
ArcGIS-compatible offline exports (Esri-JSON-shaped feature collections)
for the layers most useful to publish: airport_period_activity and
model_predictions (damage predictions, if a trained model exists).

This NEVER publishes anything automatically or requires ArcGIS
credentials - see birdstrikegeo.geo.arcgis_adapter. Output is written to
data/processed/layers/arcgis_offline/*.json for a human to review and
publish manually via ArcGIS Online/Portal later.

Usage:
    python scripts/export_arcgis_layers.py --mode sample
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import geopandas as gpd  # noqa: E402
import pandas as pd  # noqa: E402

from birdstrikegeo.geo.arcgis_adapter import (  # noqa: E402
    arcgis_credentials_configured,
    offline_export,
    prepare_airport_activity_layer,
)

LAYERS_DIR = REPO_ROOT / "data" / "processed" / "layers"
ARCGIS_OFFLINE_DIR = LAYERS_DIR / "arcgis_offline"


def run(mode: str) -> None:
    activity_path = LAYERS_DIR / "airport_period_activity.geojson"
    if not activity_path.exists():
        print(f"[export_arcgis_layers] {activity_path} not found - run "
              f"scripts/build_geospatial_features.py --mode {mode} first.")
        sys.exit(1)

    print(f"[export_arcgis_layers] ArcGIS credentials configured: {arcgis_credentials_configured()} "
          f"(not required - this script only prepares offline exports)")

    activity_gdf = gpd.read_file(activity_path)
    activity_df = pd.DataFrame(activity_gdf.drop(columns=["geometry", "source", "generated_at", "crs", "model_version"], errors="ignore"))

    prepared = prepare_airport_activity_layer(activity_df)
    print(f"[export_arcgis_layers] airport_period_activity schema validation: "
          f"{'PASSED' if prepared['validation']['passed'] else 'FAILED: ' + str(prepared['validation']['issues'])}")

    ARCGIS_OFFLINE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = offline_export(prepared, ARCGIS_OFFLINE_DIR / f"airport_period_activity_{mode}.json")
    print(f"[export_arcgis_layers] Wrote offline ArcGIS-ready export to {out_path}")
    print("[export_arcgis_layers] Nothing was published - this file is for manual review/upload only.")
    if mode == "sample":
        print("[export_arcgis_layers] SAMPLE DATA — NOT A REAL RESULT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["sample", "real"], default="sample")
    args = parser.parse_args()
    run(args.mode)


if __name__ == "__main__":
    main()
