"""
geo/spatial_join.py
----------------------
Distance calculations between a query point (e.g. an airport) and a set
of Trektellen monitoring sites, plus the two spatial-aggregation options
described in the project spec:

1. nearest_site(): use only the single closest monitoring site.
2. distance_weighted_aggregate(): weight every site within a radius by a
   transparent exponential distance-decay function (see
   distance_decay_weight()) - not a fitted interpolation surface. This is
   a deliberate simplicity choice, documented in GEOSPATIAL_METHODS.md,
   to avoid overinterpreting sparse, unevenly distributed monitoring
   coverage (also one of the ESRI_DISCUSSION_NOTES.md questions).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyproj import Geod

from birdstrikegeo.geo.crs import REGIONS, distance_km, region_for_point

_GEOD = Geod(ellps="WGS84")


def distances_to_sites(query_lon: float, query_lat: float, sites: pd.DataFrame,
                        lon_col: str = "longitude", lat_col: str = "latitude") -> pd.Series:
    """
    Returns a Series of distance_km, indexed like `sites`, from
    (query_lon, query_lat) to every site. Uses the project's CRS
    strategy: projected distance for sites in the same supported region
    as the query point, geodesic distance otherwise (vectorized via
    pyproj.Geod for speed).
    """
    query_region = region_for_point(query_lon, query_lat)
    site_lons = sites[lon_col].to_numpy(dtype=float)
    site_lats = sites[lat_col].to_numpy(dtype=float)

    if query_region is not None:
        same_region_mask = np.array(
            [region_for_point(lon, lat) == query_region for lon, lat in zip(site_lons, site_lats)]
        )
    else:
        same_region_mask = np.zeros(len(sites), dtype=bool)

    distances = np.full(len(sites), np.nan)

    if same_region_mask.any():
        from shapely.geometry import Point
        import geopandas as gpd

        crs = REGIONS[query_region]["projected_crs"]
        pts = gpd.GeoDataFrame(
            geometry=[Point(lon, lat) for lon, lat in zip(site_lons[same_region_mask], site_lats[same_region_mask])],
            crs="EPSG:4326",
        ).to_crs(crs)
        query_pt = gpd.GeoSeries([Point(query_lon, query_lat)], crs="EPSG:4326").to_crs(crs).iloc[0]
        distances[same_region_mask] = pts.distance(query_pt).to_numpy() / 1000.0

    if (~same_region_mask).any():
        lons = site_lons[~same_region_mask]
        lats = site_lats[~same_region_mask]
        _, _, dist_m = _GEOD.inv(np.full(len(lons), query_lon), np.full(len(lats), query_lat), lons, lats)
        distances[~same_region_mask] = np.asarray(dist_m) / 1000.0

    return pd.Series(distances, index=sites.index, name="distance_km")


def nearest_site(query_lon: float, query_lat: float, sites: pd.DataFrame) -> dict | None:
    """Returns {site_id, distance_km} for the single closest site, or
    None if `sites` is empty."""
    if len(sites) == 0:
        return None
    dists = distances_to_sites(query_lon, query_lat, sites)
    idx = dists.idxmin()
    return {"site_id": sites.loc[idx, "site_id"], "distance_km": float(dists.loc[idx])}


def sites_within_radius(query_lon: float, query_lat: float, sites: pd.DataFrame, radius_km: float) -> pd.DataFrame:
    dists = distances_to_sites(query_lon, query_lat, sites)
    within = sites.loc[dists <= radius_km].copy()
    within["distance_km"] = dists.loc[dists <= radius_km]
    return within


def distance_decay_weight(distance_km: float | np.ndarray, half_life_km: float = 50.0) -> float | np.ndarray:
    """
    Simple, transparent exponential distance-decay:
        weight = 0.5 ** (distance_km / half_life_km)
    A site at distance=0 gets weight 1.0; a site at distance=half_life_km
    gets weight 0.5; a site at 2x half_life_km gets weight 0.25; etc.
    Deliberately NOT a fitted/kriged interpolation - see
    GEOSPATIAL_METHODS.md for why (sparse, uneven monitoring coverage
    makes a fitted surface easy to overinterpret).
    """
    return np.power(0.5, np.asarray(distance_km) / half_life_km)


def distance_weighted_aggregate(
    query_lon: float, query_lat: float, sites: pd.DataFrame, values: pd.Series,
    radius_km: float = 250.0, half_life_km: float = 50.0,
) -> dict:
    """
    values: a Series aligned with `sites` (same index) - e.g. a per-site
    total bird count for some time window.

    Returns {weighted_total, n_sites_used, total_weight} where
    weighted_total = sum(value_i * weight_i) for every site within
    radius_km, and weight_i comes from distance_decay_weight().
    """
    within = sites_within_radius(query_lon, query_lat, sites, radius_km)
    if len(within) == 0:
        return {"weighted_total": None, "n_sites_used": 0, "total_weight": 0.0}

    weights = distance_decay_weight(within["distance_km"].to_numpy(), half_life_km)
    aligned_values = values.loc[within.index].to_numpy(dtype=float)
    weighted_total = float(np.nansum(aligned_values * weights))

    return {
        "weighted_total": weighted_total,
        "n_sites_used": int(len(within)),
        "total_weight": float(np.nansum(weights)),
    }
