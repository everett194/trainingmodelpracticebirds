from __future__ import annotations

import pandas as pd

from birdstrikegeo.geo.h3_grid import (
    add_h3_columns,
    aggregate_points_by_cell,
    cell_center,
    distance_km_between_h3_cells,
    latlon_to_cell,
)

JFK = (40.6413, -73.7781)
LAX = (33.9416, -118.4085)


def test_latlon_to_cell_returns_a_cell_for_a_valid_point():
    cell = latlon_to_cell(*JFK, resolution=5)
    assert isinstance(cell, str)
    assert len(cell) > 0


def test_latlon_to_cell_returns_none_for_missing_coordinates():
    assert latlon_to_cell(None, -73.78) is None
    assert latlon_to_cell(40.64, None) is None
    assert latlon_to_cell(float("nan"), -73.78) is None


def test_latlon_to_cell_returns_none_for_out_of_range_coordinates():
    assert latlon_to_cell(200.0, -73.78) is None
    assert latlon_to_cell(40.64, 400.0) is None


def test_same_point_same_resolution_gives_same_cell():
    assert latlon_to_cell(*JFK, resolution=5) == latlon_to_cell(*JFK, resolution=5)


def test_finer_resolution_gives_a_different_cell():
    assert latlon_to_cell(*JFK, resolution=5) != latlon_to_cell(*JFK, resolution=7)


def test_add_h3_columns_adds_both_resolutions_and_preserves_row_count():
    df = pd.DataFrame({"latitude": [JFK[0], LAX[0], None], "longitude": [JFK[1], LAX[1], -73.0]})
    out = add_h3_columns(df)
    assert len(out) == 3
    assert "h3_cell_r5" in out.columns
    assert "h3_cell_r7" in out.columns
    assert out["h3_cell_r5"].isna().sum() == 1  # the row with a null latitude


def test_cell_center_round_trips_close_to_original_point():
    cell = latlon_to_cell(*JFK, resolution=9)  # fine resolution -> tight round trip
    lat, lon = cell_center(cell)
    assert abs(lat - JFK[0]) < 0.01
    assert abs(lon - JFK[1]) < 0.01


def test_distance_between_jfk_and_lax_cells_is_plausible():
    cell_jfk = latlon_to_cell(*JFK, resolution=5)
    cell_lax = latlon_to_cell(*LAX, resolution=5)
    km = distance_km_between_h3_cells(cell_jfk, cell_lax)
    # True JFK-LAX great-circle distance is ~3980 km; allow slack for
    # H3 cell-center vs exact-airport-point offset.
    assert 3800 < km < 4200


def test_distance_from_a_cell_to_itself_is_near_zero():
    cell = latlon_to_cell(*JFK, resolution=5)
    assert distance_km_between_h3_cells(cell, cell) < 0.001


def test_aggregate_points_by_cell_sums_within_a_cell_and_drops_null_cells():
    df = pd.DataFrame(
        {
            "latitude": [JFK[0], JFK[0] + 0.0001, LAX[0], None],
            "longitude": [JFK[1], JFK[1] + 0.0001, LAX[1], -73.0],
            "count": [3, 4, 10, 99],
        }
    )
    df = add_h3_columns(df, resolutions=(5,))
    agg = aggregate_points_by_cell(df, value_columns=["count"], cell_col="h3_cell_r5")
    assert agg["count"].sum() == 17  # 3 + 4 + 10 -- the null-coordinate row's 99 is dropped
    assert len(agg) == 2  # one cell for the two nearby JFK points, one for LAX
