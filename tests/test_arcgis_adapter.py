import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from birdstrikegeo.geo.arcgis_adapter import (
    arcgis_credentials_configured,
    geodataframe_to_esri_features,
    offline_export,
    prepare_airport_activity_layer,
    validate_arcgis_schema,
)


def _sample_gdf():
    return gpd.GeoDataFrame(
        {"airport_id": ["A1", "A2"], "value": [1, 2]},
        geometry=[Point(-100.0, 40.0), Point(-101.0, 41.0)],
        crs="EPSG:4326",
    )


def test_geodataframe_to_esri_features_converts_points():
    features = geodataframe_to_esri_features(_sample_gdf())
    assert len(features) == 2
    assert features[0]["geometry"]["x"] == -100.0
    assert features[0]["geometry"]["y"] == 40.0
    assert features[0]["attributes"]["airport_id"] == "A1"


def test_validate_arcgis_schema_passes_for_clean_gdf():
    result = validate_arcgis_schema(_sample_gdf())
    assert result["passed"]
    assert result["has_crs"]
    assert result["no_null_geometry"]


def test_validate_arcgis_schema_flags_missing_crs():
    gdf = _sample_gdf()
    gdf.crs = None
    result = validate_arcgis_schema(gdf)
    assert not result["passed"]
    assert not result["has_crs"]


def test_validate_arcgis_schema_flags_long_field_names():
    gdf = _sample_gdf().rename(columns={"value": "a" * 40})
    result = validate_arcgis_schema(gdf)
    assert not result["passed"]
    assert not result["field_names_valid_length"]


def test_validate_arcgis_schema_flags_case_insensitive_collisions():
    gdf = _sample_gdf()
    gdf["Value"] = gdf["value"]  # collides with "value" once lowercased
    result = validate_arcgis_schema(gdf)
    assert not result["passed"]
    assert not result["no_case_insensitive_collisions"]


def test_prepare_airport_activity_layer_produces_features_when_valid():
    df = pd.DataFrame({"airport_id": ["A1"], "latitude": [40.0], "longitude": [-100.0], "activity_index": [0.5]})
    result = prepare_airport_activity_layer(df)
    assert result["layer_name"] == "airport_period_activity"
    assert result["validation"]["passed"]
    assert result["features"] is not None


def test_offline_export_writes_json_without_publishing_anything(tmp_path):
    df = pd.DataFrame({"airport_id": ["A1"], "latitude": [40.0], "longitude": [-100.0], "activity_index": [0.5]})
    prepared = prepare_airport_activity_layer(df)
    path = offline_export(prepared, tmp_path / "offline_layer.json")
    assert path.exists()
    assert "airport_period_activity" in path.read_text()


def test_arcgis_credentials_configured_false_by_default(monkeypatch):
    monkeypatch.delenv("ARCGIS_PORTAL_URL", raising=False)
    monkeypatch.delenv("ARCGIS_API_KEY", raising=False)
    assert arcgis_credentials_configured() is False


def test_geojson_export_via_exports_module(tmp_path):
    from birdstrikegeo.geo.exports import LayerMetadata, export_layer

    gdf = _sample_gdf()
    metadata = LayerMetadata(source="test", sample_data_flag=True)
    written = export_layer(gdf, tmp_path, "test_layer", metadata, formats=("geojson",))
    assert written["geojson"].exists()
    reloaded = gpd.read_file(written["geojson"])
    assert "sample_data_flag" in reloaded.columns
    assert "generated_at" in reloaded.columns
