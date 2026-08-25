"""
geo/crs.py
------------
The project's CRS (coordinate reference system) strategy, implementing
what's documented in GEOSPATIAL_METHODS.md:

- Raw latitude/longitude is always stored in WGS84 (EPSG:4326).
- Distance and buffer calculations are NEVER done directly in degrees
  (a degree of longitude is a very different physical distance near the
  equator vs. near the poles - doing math directly on lat/lon would
  silently distort every distance and area calculation).
- A point gets reprojected into a regional, equal-area PROJECTED CRS
  (CONUS: EPSG:5070: Europe: EPSG:3035) when it falls inside that
  region's bounding box.
- Anything outside both regions, or any pair of points that straddles
  both regions (e.g. comparing a US and a European airport), falls back
  to ellipsoidal geodesic distance via pyproj.Geod - correct everywhere
  on Earth, at the cost of not supporting area/buffer geometry directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
from pyproj import Geod
from shapely.geometry import Point

STORAGE_CRS = "EPSG:4326"

REGIONS: dict[str, dict] = {
    "conus": {"projected_crs": "EPSG:5070", "bbox": (-125.0, 24.0, -66.0, 50.0)},  # (minx, miny, maxx, maxy)
    "europe": {"projected_crs": "EPSG:3035", "bbox": (-25.0, 34.0, 45.0, 72.0)},
}

_GEOD = Geod(ellps="WGS84")


def region_for_point(lon: float, lat: float) -> str | None:
    """Returns the region name ('conus', 'europe') whose bounding box
    contains (lon, lat), or None if it falls in neither (use the
    geodesic fallback for that point)."""
    for name, cfg in REGIONS.items():
        minx, miny, maxx, maxy = cfg["bbox"]
        if minx <= lon <= maxx and miny <= lat <= maxy:
            return name
    return None


def projected_crs_for_point(lon: float, lat: float) -> str | None:
    region = region_for_point(lon, lat)
    return REGIONS[region]["projected_crs"] if region else None


def to_geodataframe(df, lon_col: str = "longitude", lat_col: str = "latitude") -> gpd.GeoDataFrame:
    """Builds a WGS84 GeoDataFrame from a plain DataFrame with lon/lat columns."""
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    return gpd.GeoDataFrame(df.copy(), geometry=geometry, crs=STORAGE_CRS)


@dataclass
class DistanceResult:
    distance_km: float
    method: str  # "projected" or "geodesic"
    crs_used: str | None


def geodesic_distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Correct-everywhere ellipsoidal geodesic distance, used as the
    fallback when points aren't both inside the same supported region."""
    _, _, distance_m = _GEOD.inv(lon1, lat1, lon2, lat2)
    return distance_m / 1000.0


def distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> DistanceResult:
    """
    Picks a projected CRS if BOTH points fall in the same supported
    region (giving a fast, planar-but-accurate-for-the-region distance),
    otherwise falls back to geodesic distance. Never computes distance
    directly from raw lat/lon degrees.
    """
    region1 = region_for_point(lon1, lat1)
    region2 = region_for_point(lon2, lat2)

    if region1 is not None and region1 == region2:
        crs = REGIONS[region1]["projected_crs"]
        gdf = gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[Point(lon1, lat1), Point(lon2, lat2)],
            crs=STORAGE_CRS,
        ).to_crs(crs)
        p1, p2 = gdf.geometry.iloc[0], gdf.geometry.iloc[1]
        return DistanceResult(distance_km=p1.distance(p2) / 1000.0, method="projected", crs_used=crs)

    return DistanceResult(distance_km=geodesic_distance_km(lon1, lat1, lon2, lat2), method="geodesic", crs_used=None)


def buffer_geodataframe_km(gdf: gpd.GeoDataFrame, radius_km: float, region: str) -> gpd.GeoDataFrame:
    """
    Buffers points by radius_km using the given region's PLANAR projected
    CRS (an approximation - see GEOSPATIAL_METHODS.md's discussion of
    planar vs. geodesic buffers, one of the open questions for the Esri
    analyst), then reprojects the result back to WGS84 for storage/export.
    """
    if region not in REGIONS:
        raise ValueError(f"No projected CRS configured for region '{region}'. Supported: {list(REGIONS)}")
    crs = REGIONS[region]["projected_crs"]
    projected = gdf.to_crs(crs)
    buffered = projected.copy()
    buffered["geometry"] = projected.geometry.buffer(radius_km * 1000.0)
    return buffered.to_crs(STORAGE_CRS)
