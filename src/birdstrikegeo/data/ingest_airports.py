"""
data/ingest_airports.py
--------------------------
Loads airport point locations + metadata from a GeoJSON (preferred) or
CSV (with latitude/longitude columns). Optional data source - see
configs/data_sources.yaml. Environmental context layers (wetlands,
water, land cover, etc.) are NOT loaded here; see
birdstrikegeo.geo.airport_layers for how those are optionally attached
when the user supplies real spatial layers.
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from birdstrikegeo.schemas.airports import AIRPORT_SCHEMA


def load_airports(path: str | Path) -> gpd.GeoDataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Airports file not found: {path}")

    if path.suffix.lower() == ".geojson":
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs("EPSG:4326")
        if "latitude" not in gdf.columns or "longitude" not in gdf.columns:
            gdf["longitude"] = gdf.geometry.x
            gdf["latitude"] = gdf.geometry.y
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
        )
    else:
        raise ValueError(f"Unsupported airports file format: {path.suffix} (expected .geojson or .csv)")

    for col in AIRPORT_SCHEMA:
        if col not in gdf.columns:
            gdf[col] = pd.NA

    return gdf


def airport_lookup(gdf: gpd.GeoDataFrame) -> dict:
    """airport_id -> {latitude, longitude, airport_name, ...} for quick lookup elsewhere."""
    return gdf.set_index("airport_id")[[c for c in AIRPORT_SCHEMA if c != "airport_id"]].to_dict("index")
