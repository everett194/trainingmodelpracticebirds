"""
test_sample_workflow.py
--------------------------
An end-to-end smoke test of the sample-data pipeline: generate ->
prepare (FAA) -> build geo features (Trektellen) -> confirm every
expected output file exists. Runs the actual scripts as subprocesses
(the same way a user would), rather than re-implementing their logic,
so this test fails if the real CLI-facing pipeline breaks - not just an
internal function.

Deliberately slower than the rest of the suite - this is the one test
that exercises the whole system together. No network access or real
data required.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=300,
    )


@pytest.mark.slow
def test_generate_sample_data_produces_all_expected_files():
    result = _run("generate_sample_data.py")
    assert result.returncode == 0, result.stderr

    sample_dir = REPO_ROOT / "data" / "sample"
    for filename in (
        "faa_strikes_sample.csv", "trektellen_sites_sample.csv", "trektellen_counts_sample.csv",
        "airports_sample.geojson", "airports_sample.csv", "weather_sample.csv",
        "trektellen_provenance_sample.json",
    ):
        assert (sample_dir / filename).exists(), f"missing {filename}"


@pytest.mark.slow
def test_prepare_data_sample_mode_produces_splits_and_quality_report():
    result = _run("prepare_data.py", "--mode", "sample")
    assert result.returncode == 0, result.stderr

    processed_dir = REPO_ROOT / "data" / "processed" / "sample"
    for filename in ("damage_train.csv", "damage_validation.csv", "damage_test.csv", "damage_excluded_unknown_target.csv"):
        assert (processed_dir / filename).exists()

    results_dir = REPO_ROOT / "results" / "damage"
    assert (results_dir / "data_quality_sample.json").exists()
    assert (results_dir / "data_quality_sample.md").exists()


@pytest.mark.slow
def test_build_geospatial_features_sample_mode_produces_all_layer_formats():
    result = _run("build_geospatial_features.py", "--mode", "sample")
    assert result.returncode == 0, result.stderr

    layers_dir = REPO_ROOT / "data" / "processed" / "layers"
    for filename in (
        "airports.geojson", "airports.gpkg", "airports.parquet", "airports.csv",
        "trektellen_sites.geojson", "airport_period_activity.geojson",
        "airport_buffers_50km.geojson", "airport_buffers_100km.geojson", "airport_buffers_250km.geojson",
        "airport_trektellen_links.csv",
    ):
        assert (layers_dir / filename).exists(), f"missing {filename}"


@pytest.mark.slow
def test_validate_data_real_mode_exits_nonzero_with_missing_files_listed():
    result = _run("validate_data.py", "--mode", "real")
    # Real data almost certainly isn't present in a test environment -
    # this should fail GRACEFULLY (clear message, nonzero exit), not crash.
    assert result.returncode == 1
    assert "DATA_DOWNLOAD_GUIDE.md" in result.stdout


@pytest.mark.slow
def test_prepare_data_real_mode_fails_gracefully_when_data_missing():
    real_faa = REPO_ROOT / "data" / "raw" / "faa_wildlife_strikes.csv"
    if real_faa.exists():
        pytest.skip("Real FAA data is present in this environment - graceful-failure path not exercised.")
    result = _run("prepare_data.py", "--mode", "real")
    assert result.returncode == 1
    assert "DATA_DOWNLOAD_GUIDE.md" in result.stdout
