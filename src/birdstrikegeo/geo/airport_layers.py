"""
geo/airport_layers.py
------------------------
Builds the `airports` layer and its 50/100/250km buffer layers. Also
hosts compute_environmental_features(), which ONLY populates
distance_to_water_km / wetland_area_within_10km / etc. when the caller
supplies real spatial layers (GeoDataFrames) - per project requirements,
these are never fabricated for real airports. When a layer isn't
supplied, habitat_data_available=False and the numeric fields stay null.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from birdstrikegeo.geo.crs import buffer_geodataframe_km, region_for_point, to_geodataframe
from birdstrikegeo.schemas.airports import ENVIRONMENTAL_FEATURE_SCHEMA


def build_airport_layer(airports_df) -> gpd.GeoDataFrame:
    return to_geodataframe(airports_df)


def build_airport_buffers(airports_gdf: gpd.GeoDataFrame, radius_km: float) -> gpd.GeoDataFrame:
    """
    Buffers every airport by radius_km, grouping by region so each group
    uses its own appropriate projected CRS (see geo.crs). Airports
    outside any supported region are returned with a null geometry and
    `buffer_unavailable_reason` explaining why, rather than silently
    dropped or buffered in an inappropriate CRS.
    """
    airports_gdf = airports_gdf.copy()
    airports_gdf["_region"] = airports_gdf.apply(
        lambda r: region_for_point(r.geometry.x, r.geometry.y), axis=1
    )

    buffered_parts = []
    for region, group in airports_gdf.groupby("_region", dropna=False):
        if pd.isna(region):
            unresolved = group.copy()
            unresolved["geometry"] = None
            unresolved["buffer_unavailable_reason"] = "outside supported CRS regions (conus, europe)"
            buffered_parts.append(unresolved)
            continue
        buffered = buffer_geodataframe_km(group, radius_km, region)
        buffered["buffer_unavailable_reason"] = None
        buffered_parts.append(buffered)

    result = gpd.GeoDataFrame(pd.concat(buffered_parts, ignore_index=True), crs="EPSG:4326")
    result["buffer_radius_km"] = radius_km
    return result.drop(columns=["_region"])


def compute_environmental_features(
    airports_gdf: gpd.GeoDataFrame,
    water_layer: gpd.GeoDataFrame | None = None,
    wetland_layer: gpd.GeoDataFrame | None = None,
    agricultural_layer: gpd.GeoDataFrame | None = None,
    urban_layer: gpd.GeoDataFrame | None = None,
    landfill_layer: gpd.GeoDataFrame | None = None,
) -> pd.DataFrame:
    """
    Computes ENVIRONMENTAL_FEATURE_SCHEMA fields ONLY from layers that
    are actually supplied. Any layer left as None results in its
    corresponding feature(s) staying null - habitat_data_available is
    True only if at least one real layer was supplied and successfully
    used. This function never fabricates habitat data for a real airport.
    """
    result = pd.DataFrame(index=airports_gdf.index)
    for col in ENVIRONMENTAL_FEATURE_SCHEMA:
        result[col] = pd.NA

    any_layer_used = False

    if water_layer is not None:
        result["distance_to_water_km"] = _distance_to_nearest(airports_gdf, water_layer)
        result["water_area_within_5km"] = _area_within(airports_gdf, water_layer, 5.0)
        any_layer_used = True

    if wetland_layer is not None:
        result["wetland_area_within_10km"] = _area_within(airports_gdf, wetland_layer, 10.0)
        any_layer_used = True

    if agricultural_layer is not None:
        result["agricultural_area_within_10km"] = _area_within(airports_gdf, agricultural_layer, 10.0)
        any_layer_used = True

    if urban_layer is not None:
        result["urban_area_within_10km"] = _area_within(airports_gdf, urban_layer, 10.0)
        any_layer_used = True

    if landfill_layer is not None:
        result["landfill_distance_km"] = _distance_to_nearest(airports_gdf, landfill_layer)
        any_layer_used = True

    result["habitat_data_available"] = any_layer_used
    return result


def _distance_to_nearest(points_gdf: gpd.GeoDataFrame, features_gdf: gpd.GeoDataFrame) -> pd.Series:
    """Planar distance in the points' own CRS if projected, else left to
    the caller to have already reprojected - a thin wrapper around
    geopandas' nearest-distance join, only invoked when a real layer is
    supplied (see compute_environmental_features)."""
    joined = gpd.sjoin_nearest(points_gdf, features_gdf, distance_col="_dist", how="left")
    return joined.groupby(joined.index)["_dist"].min() / 1000.0


def _area_within(points_gdf: gpd.GeoDataFrame, features_gdf: gpd.GeoDataFrame, radius_km: float) -> pd.Series:
    areas = []
    for _, point_row in points_gdf.iterrows():
        region = region_for_point(point_row.geometry.x, point_row.geometry.y)
        if region is None:
            areas.append(pd.NA)
            continue
        buffer_gdf = buffer_geodataframe_km(
            gpd.GeoDataFrame({"id": [1]}, geometry=[point_row.geometry], crs="EPSG:4326"), radius_km, region
        )
        clipped = gpd.clip(features_gdf, buffer_gdf)
        areas.append(clipped.to_crs(buffer_gdf.crs).area.sum() if len(clipped) else 0.0)
    return pd.Series(areas, index=points_gdf.index)
