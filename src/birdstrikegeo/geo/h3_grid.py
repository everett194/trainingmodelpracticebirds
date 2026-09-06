"""
geo/h3_grid.py
-----------------
Phase 3 (geospatial representation): a hexagonal spatial-indexing layer
using Uber's H3 system, so bird abundance, strikes, airports, habitat,
and weather can all be aggregated onto the SAME set of cells rather than
each being aggregated ad hoc to whatever grid (or no grid) its own
pipeline happened to use.

This module only assigns points to cells and aggregates within them —
it does not replace the existing CRS/buffer strategy in `geo/crs.py`
(WGS84 storage, projected-CRS or geodesic distance for buffers/areas).
H3 cells here are a discrete AGGREGATION KEY for joining heterogeneous
sources at a shared resolution, not a substitute for exact distance
math — see `distance_km_between_h3_cells` below, which still goes
through `geo.crs`'s geodesic distance rather than approximating from
cell adjacency.

Why H3 over a raw lat/lon grid: cells are roughly equal-area worldwide
(unlike a raw degree grid, which shrinks toward the poles — see
GEOSPATIAL_METHODS.md section 1's discussion of the same distortion
problem for buffers), and H3's multi-resolution hierarchy means a
resolution-5 cell decomposes cleanly into resolution-7 children, so an
analysis can later refine resolution in dense-data regions without
re-deriving a whole new grid.

Resolution choice (see GEOSPATIAL_METHODS.md for the write-up):
  - Resolution 5 (~8.5 km edge length, ~252 km^2 average cell area):
    default for airport-buffer-scale aggregation (matches the existing
    50 km buffer radius reasonably — a 50 km buffer covers roughly a
    5-7 cell radius at this resolution).
  - Resolution 7 (~1.2 km edge length, ~5.2 km^2 average cell area):
    finer option for dense-data regions (e.g. many nearby Trektellen
    sites, or a future high-resolution eBird 3km-raster join, which
    needs a finer cell than resolution 5 to avoid conflating multiple
    raster pixels into one cell).
Both are precomputed and stored (see
`schemas.hazard_observation.GEOGRAPHIC_SCHEMA`'s `h3_cell_r5`/
`h3_cell_r7` columns) rather than computed on the fly at query time, so
a downstream join never has to guess which resolution a stored cell ID
belongs to.
"""

from __future__ import annotations

import h3
import pandas as pd

from birdstrikegeo.geo.crs import geodesic_distance_km

DEFAULT_RESOLUTION = 5
FINE_RESOLUTION = 7


def latlon_to_cell(lat: float, lon: float, resolution: int = DEFAULT_RESOLUTION) -> str | None:
    """
    Returns the H3 cell index containing (lat, lon) at the given
    resolution, or None if either coordinate is missing/invalid — never
    fabricates a cell for a point this project doesn't actually have
    coordinates for.
    """
    if lat is None or lon is None or pd.isna(lat) or pd.isna(lon):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return h3.latlng_to_cell(lat, lon, resolution)


def add_h3_columns(
    df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude",
    resolutions: tuple[int, ...] = (DEFAULT_RESOLUTION, FINE_RESOLUTION),
) -> pd.DataFrame:
    """
    Returns a copy of df with one new column per requested resolution,
    named f"h3_cell_r{resolution}" (matching
    schemas.hazard_observation.GEOGRAPHIC_SCHEMA's column names exactly).
    Rows with missing/invalid coordinates get a null cell, not a guess.
    """
    out = df.copy()
    for resolution in resolutions:
        out[f"h3_cell_r{resolution}"] = [
            latlon_to_cell(lat, lon, resolution) for lat, lon in zip(out[lat_col], out[lon_col])
        ]
    return out


def cell_center(cell: str) -> tuple[float, float]:
    """(lat, lon) of the given H3 cell's center point."""
    return h3.cell_to_latlng(cell)


def distance_km_between_h3_cells(cell_a: str, cell_b: str) -> float:
    """
    True geodesic distance in km between two H3 cells' center points —
    goes through geo.crs's geodesic distance rather than H3's own
    `grid_distance` (which counts cell-hops, not physical km, and is
    resolution-dependent) so this number means the same thing regardless
    of which resolution the two cells came from.
    """
    lat_a, lon_a = cell_center(cell_a)
    lat_b, lon_b = cell_center(cell_b)
    # geo.crs.geodesic_distance_km takes (lon, lat) pairs, not (lat, lon).
    return geodesic_distance_km(lon_a, lat_a, lon_b, lat_b)


def aggregate_points_by_cell(
    df: pd.DataFrame, value_columns: list[str], cell_col: str = f"h3_cell_r{DEFAULT_RESOLUTION}",
    agg: str = "sum",
) -> pd.DataFrame:
    """
    Groups df by its H3 cell column and aggregates value_columns (e.g.
    summing bird counts, or averaging a modeled abundance value that
    fell in the same cell from multiple nearby sources). Rows with a
    null cell (missing/invalid coordinates) are dropped from the
    aggregate — they were never assigned to any cell, so including them
    under a synthetic "unknown" cell would misrepresent coverage rather
    than surface it; callers that need to track how many rows were
    dropped should check `df[cell_col].isna().sum()` before calling.
    """
    valid = df[df[cell_col].notna()]
    return valid.groupby(cell_col, as_index=False)[value_columns].agg(agg)
