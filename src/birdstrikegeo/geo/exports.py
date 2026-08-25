"""
geo/exports.py
-----------------
Exports a GeoDataFrame (or plain DataFrame with lat/lon) to every format
the project spec requires: GeoJSON, GeoPackage, GeoParquet, and CSV with
latitude/longitude columns. Every export is stamped with the required
metadata: source, generated_at, crs, sample_data_flag, model_version.

GeoPackage/GeoParquet require geopandas' fiona/pyogrio and pyarrow
backends respectively (both declared dependencies - see requirements.txt).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd


@dataclass
class LayerMetadata:
    source: str
    sample_data_flag: bool
    model_version: str = "n/a"
    crs: str = "EPSG:4326"

    def as_dict(self) -> dict:
        return {
            "source": self.source,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "crs": self.crs,
            "sample_data_flag": self.sample_data_flag,
            "model_version": self.model_version,
        }


def _stamp_metadata(gdf: gpd.GeoDataFrame, metadata: LayerMetadata) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    for key, value in metadata.as_dict().items():
        gdf[key] = value
    return gdf


def export_layer(
    gdf: gpd.GeoDataFrame,
    output_dir: str | Path,
    layer_name: str,
    metadata: LayerMetadata,
    formats: tuple[str, ...] = ("geojson", "gpkg", "geoparquet", "csv"),
) -> dict[str, Path]:
    """
    Writes `gdf` to output_dir/layer_name.<ext> for each requested
    format. Returns {format: path_written}.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamped = _stamp_metadata(gdf, metadata)

    written: dict[str, Path] = {}

    if "geojson" in formats:
        path = output_dir / f"{layer_name}.geojson"
        stamped.to_file(path, driver="GeoJSON")
        written["geojson"] = path

    if "gpkg" in formats:
        path = output_dir / f"{layer_name}.gpkg"
        stamped.to_file(path, driver="GPKG", layer=layer_name)
        written["gpkg"] = path

    if "geoparquet" in formats:
        path = output_dir / f"{layer_name}.parquet"
        stamped.to_parquet(path)
        written["geoparquet"] = path

    if "csv" in formats:
        path = output_dir / f"{layer_name}.csv"
        csv_df = pd.DataFrame(stamped.drop(columns="geometry"))
        if "longitude" not in csv_df.columns:
            csv_df["longitude"] = stamped.geometry.x
        if "latitude" not in csv_df.columns:
            csv_df["latitude"] = stamped.geometry.y
        csv_df.to_csv(path, index=False)
        written["csv"] = path

    return written


def export_table(
    df: pd.DataFrame, output_dir: str | Path, layer_name: str, metadata: LayerMetadata,
) -> Path:
    """For non-geometry tables (e.g. airport_trektellen_links) that still
    need the standard metadata stamp, written as CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamped = df.copy()
    for key, value in metadata.as_dict().items():
        stamped[key] = value
    path = output_dir / f"{layer_name}.csv"
    stamped.to_csv(path, index=False)
    return path
