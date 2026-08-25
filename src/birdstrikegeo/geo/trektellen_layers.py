"""
geo/trektellen_layers.py
---------------------------
Builds the `trektellen_sites` geospatial layer and `airport_trektellen_links`
(one row per airport x site pair within a configured radius) - the
building block for airport_period_activity, and a genuinely useful
standalone layer for visualizing monitoring coverage in a GIS.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from birdstrikegeo.geo.crs import to_geodataframe
from birdstrikegeo.geo.spatial_join import distances_to_sites


def build_trektellen_sites_layer(sites_df: pd.DataFrame) -> gpd.GeoDataFrame:
    return to_geodataframe(sites_df)


def build_airport_trektellen_links(
    airports_df: pd.DataFrame, sites_df: pd.DataFrame, max_radius_km: float = 250.0
) -> pd.DataFrame:
    """
    One row per (airport_id, site_id) pair within max_radius_km,
    with distance_km - a flat table suitable for a map "link lines"
    layer or a join table. Airports with zero nearby sites are NOT
    silently omitted from the *airports* table elsewhere, but simply
    won't appear here (no in-radius sites), which is itself useful
    information (see low_coverage_warning in build_activity_features).
    """
    rows = []
    for _, airport in airports_df.iterrows():
        dists = distances_to_sites(airport["longitude"], airport["latitude"], sites_df)
        within = dists[dists <= max_radius_km]
        for site_idx, distance in within.items():
            rows.append(
                {
                    "airport_id": airport["airport_id"],
                    "site_id": sites_df.loc[site_idx, "site_id"],
                    "distance_km": float(distance),
                }
            )
    return pd.DataFrame(rows)
