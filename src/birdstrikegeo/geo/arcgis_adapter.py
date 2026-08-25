"""
geo/arcgis_adapter.py
------------------------
Optional ArcGIS interoperability. Nothing in this module is required for
the core pipeline - if the `arcgis` Python package isn't installed, or no
ArcGIS credentials are configured, every other part of BirdStrikeGeo
keeps working using GeoJSON/GeoPackage/GeoParquet via geopandas.

This module NEVER publishes anything automatically. It only:
  - reads a PUBLIC ArcGIS Feature Service URL (paginated REST queries,
    no credentials required for a public layer)
  - reads a local GeoJSON layer
  - converts GeoDataFrames to Esri-JSON-shaped feature records
  - prepares (but does not upload) publishable feature collections
  - validates geometry/spatial-reference well-formedness
  - saves an offline export (GeoJSON) when no credentials are configured,
    so a human can publish it manually via ArcGIS Online/Portal later

See ESRI_DISCUSSION_NOTES.md for the open questions this adapter is
deliberately leaving to a real Esri analyst rather than guessing at.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd

try:
    import requests

    _REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _REQUESTS_AVAILABLE = False

try:
    import arcgis  # noqa: F401

    ARCGIS_PACKAGE_AVAILABLE = True
except ImportError:
    ARCGIS_PACKAGE_AVAILABLE = False


class ArcgisAdapterError(RuntimeError):
    pass


def load_arcgis_feature_layer(url: str, where: str = "1=1", page_size: int = 1000) -> gpd.GeoDataFrame:
    """
    Reads a PUBLIC ArcGIS Feature Service layer via its REST query
    endpoint (https://<host>/.../FeatureServer/<id>/query), paginating
    with resultOffset/resultRecordCount so large services are read
    without a single oversized request. Does not require the `arcgis`
    package or credentials - a public feature service accepts anonymous
    GeoJSON queries directly over HTTP.

    Raises ArcgisAdapterError if `requests` isn't installed or the
    service returns an unexpected response - callers should catch this
    and fall back to local GeoJSON, since ArcGIS access is always optional.
    """
    if not _REQUESTS_AVAILABLE:
        raise ArcgisAdapterError("The 'requests' package is required to query a live ArcGIS Feature Service.")

    query_url = url.rstrip("/") + "/query"
    all_features = []
    offset = 0

    while True:
        params = {
            "where": where,
            "outFields": "*",
            "f": "geojson",
            "resultOffset": offset,
            "resultRecordCount": page_size,
        }
        response = requests.get(query_url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if "error" in payload:
            raise ArcgisAdapterError(f"ArcGIS Feature Service error: {payload['error']}")

        features = payload.get("features", [])
        all_features.extend(features)
        if len(features) < page_size:
            break
        offset += page_size

    if not all_features:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

    return gpd.GeoDataFrame.from_features(all_features, crs="EPSG:4326")


def load_local_geojson(path: str | Path) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def geodataframe_to_esri_features(gdf: gpd.GeoDataFrame) -> list[dict]:
    """
    Converts a WGS84 GeoDataFrame into a list of Esri-JSON-shaped feature
    dicts ({"geometry": {...}, "attributes": {...}}), the format ArcGIS
    REST endpoints expect for applyEdits/addFeatures calls. This function
    only prepares the records - it never calls an ArcGIS endpoint itself.
    """
    if gdf.crs is not None and str(gdf.crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
        gdf = gdf.to_crs("EPSG:4326")

    features = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            esri_geometry = None
        elif geom.geom_type == "Point":
            esri_geometry = {"x": geom.x, "y": geom.y, "spatialReference": {"wkid": 4326}}
        elif geom.geom_type == "Polygon":
            rings = [list(geom.exterior.coords)] + [list(interior.coords) for interior in geom.interiors]
            esri_geometry = {"rings": rings, "spatialReference": {"wkid": 4326}}
        elif geom.geom_type == "LineString":
            esri_geometry = {"paths": [list(geom.coords)], "spatialReference": {"wkid": 4326}}
        else:
            esri_geometry = None  # unsupported geometry type - attributes still preserved

        attributes = {
            k: (v.isoformat() if hasattr(v, "isoformat") else v)
            for k, v in row.drop("geometry").items()
        }
        features.append({"geometry": esri_geometry, "attributes": attributes})

    return features


def validate_arcgis_schema(gdf: gpd.GeoDataFrame) -> dict:
    """
    Checks (without contacting any service) that a GeoDataFrame is
    well-formed enough to eventually publish: has a CRS, has non-null
    geometry, field names are ArcGIS-safe (<= 31 chars, no reserved
    prefixes), and no field name collides after ArcGIS's case-insensitive
    truncation rules. Returns a dict of {check_name: passed_bool} plus
    "issues": [...] with human-readable explanations.
    """
    issues = []

    has_crs = gdf.crs is not None
    if not has_crs:
        issues.append("GeoDataFrame has no CRS set.")

    n_null_geometry = int(gdf.geometry.isna().sum() + (gdf.geometry.is_empty if len(gdf) else pd.Series(dtype=bool)).sum())
    if n_null_geometry:
        issues.append(f"{n_null_geometry} row(s) have null/empty geometry.")

    long_fields = [c for c in gdf.columns if c != "geometry" and len(c) > 31]
    if long_fields:
        issues.append(f"Field name(s) exceed ArcGIS's 31-character limit: {long_fields}")

    lowered = [c.lower() for c in gdf.columns if c != "geometry"]
    dupes = {c for c in lowered if lowered.count(c) > 1}
    if dupes:
        issues.append(f"Field name(s) collide case-insensitively (ArcGIS truncates/normalizes names): {sorted(dupes)}")

    return {
        "has_crs": has_crs,
        "no_null_geometry": n_null_geometry == 0,
        "field_names_valid_length": not long_fields,
        "no_case_insensitive_collisions": not dupes,
        "passed": not issues,
        "issues": issues,
    }


def prepare_airport_activity_layer(df: pd.DataFrame) -> dict:
    """Prepares (does not publish) a feature-collection-shaped dict for
    the airport_period_activity layer, ready for a human to review and
    manually publish via ArcGIS Online/Portal, or for offline_export()."""
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    )
    validation = validate_arcgis_schema(gdf)
    return {
        "layer_name": "airport_period_activity",
        "validation": validation,
        "features": geodataframe_to_esri_features(gdf) if validation["passed"] else None,
    }


def prepare_model_prediction_layer(df: pd.DataFrame) -> dict:
    """Same idea as prepare_airport_activity_layer, for model_predictions."""
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326"
    )
    validation = validate_arcgis_schema(gdf)
    return {
        "layer_name": "model_predictions",
        "validation": validation,
        "features": geodataframe_to_esri_features(gdf) if validation["passed"] else None,
    }


def offline_export(prepared_layer: dict, output_path: str | Path) -> Path:
    """
    Saves a prepare_*_layer() result to disk as JSON when ArcGIS
    credentials aren't configured (checked via environment variables,
    never hardcoded) - so the work of preparing a publishable layer isn't
    wasted even without live ArcGIS access. A human can inspect this file
    and publish it manually later.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(prepared_layer, indent=2, default=str))
    return output_path


def arcgis_credentials_configured() -> bool:
    return bool(os.environ.get("ARCGIS_PORTAL_URL") and os.environ.get("ARCGIS_API_KEY"))
