import numpy as np
import pandas as pd
import pytest

from birdstrikegeo.geo.crs import distance_km, region_for_point
from birdstrikegeo.geo.spatial_join import (
    distance_decay_weight,
    distance_weighted_aggregate,
    distances_to_sites,
    nearest_site,
    sites_within_radius,
)


def test_region_for_point_conus():
    assert region_for_point(-100.0, 40.0) == "conus"


def test_region_for_point_europe():
    assert region_for_point(10.0, 50.0) == "europe"


def test_region_for_point_outside_supported_regions_returns_none():
    assert region_for_point(140.0, -30.0) is None  # Australia - unsupported


def test_distance_km_uses_projected_crs_within_same_region():
    result = distance_km(-100.0, 40.0, -100.5, 40.5)
    assert result.method == "projected"
    assert result.crs_used == "EPSG:5070"
    assert result.distance_km > 0


def test_distance_km_falls_back_to_geodesic_across_regions():
    # one CONUS-like point, one Europe-like point
    result = distance_km(-100.0, 40.0, 10.0, 50.0)
    assert result.method == "geodesic"
    assert result.crs_used is None
    assert result.distance_km > 5000  # genuinely far apart


def test_distance_km_matches_known_geodesic_distance_approximately():
    # New York City to London, well-known real-world distance ~5570 km
    result = distance_km(-74.006, 40.7128, -0.1276, 51.5074)
    assert 5400 < result.distance_km < 5750


def test_distances_to_sites_returns_one_value_per_site():
    sites = pd.DataFrame({"site_id": ["A", "B", "C"], "longitude": [-100.0, -100.2, 10.0], "latitude": [40.0, 40.1, 50.0]})
    dists = distances_to_sites(-100.0, 40.0, sites)
    assert len(dists) == 3
    assert dists.iloc[0] == pytest.approx(0.0, abs=0.01)
    assert dists.iloc[2] > dists.iloc[1]  # the European site should be much farther


def test_nearest_site_picks_the_closest_one():
    sites = pd.DataFrame({"site_id": ["FAR", "NEAR"], "longitude": [-95.0, -100.01], "latitude": [45.0, 40.01]})
    result = nearest_site(-100.0, 40.0, sites)
    assert result["site_id"] == "NEAR"


def test_nearest_site_with_no_sites_returns_none():
    empty = pd.DataFrame({"site_id": [], "longitude": [], "latitude": []})
    assert nearest_site(-100.0, 40.0, empty) is None


def test_sites_within_radius_respects_configurable_radius():
    sites = pd.DataFrame(
        {"site_id": ["A", "B", "C"], "longitude": [-100.0, -100.5, -105.0], "latitude": [40.0, 40.0, 40.0]}
    )
    within_50 = sites_within_radius(-100.0, 40.0, sites, radius_km=50)
    within_500 = sites_within_radius(-100.0, 40.0, sites, radius_km=500)
    assert len(within_50) < len(within_500)
    assert set(within_500["site_id"]) == {"A", "B", "C"}


def test_distance_decay_weight_is_1_at_zero_and_half_at_half_life():
    assert distance_decay_weight(0, half_life_km=50) == pytest.approx(1.0)
    assert distance_decay_weight(50, half_life_km=50) == pytest.approx(0.5)
    assert distance_decay_weight(100, half_life_km=50) == pytest.approx(0.25)


def test_distance_weighted_aggregate_weights_closer_sites_more():
    sites = pd.DataFrame(
        {"site_id": ["NEAR", "FAR"], "longitude": [-100.0, -101.5], "latitude": [40.0, 40.0]}
    )
    values = pd.Series([10.0, 10.0], index=sites.index)
    result = distance_weighted_aggregate(-100.0, 40.0, sites, values, radius_km=250, half_life_km=50)
    # both sites contribute value=10, but NEAR should get a much higher weight
    assert result["n_sites_used"] == 2
    assert result["weighted_total"] < 20  # decayed, not a plain sum


def test_distance_weighted_aggregate_empty_when_nothing_in_radius():
    sites = pd.DataFrame({"site_id": ["FAR"], "longitude": [-50.0], "latitude": [10.0]})
    values = pd.Series([10.0], index=sites.index)
    result = distance_weighted_aggregate(-100.0, 40.0, sites, values, radius_km=10)
    assert result["n_sites_used"] == 0
    assert result["weighted_total"] is None
