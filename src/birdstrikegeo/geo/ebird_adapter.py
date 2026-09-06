"""
geo/ebird_adapter.py
------------------------
Placeholder adapter for eBird Status & Trends data (see
BIRD_DATA_SOURCE_COMPARISON.md for the full research writeup on why this
source, and geo/arcgis_adapter.py for the "never auto-fetch/publish
without explicit credentials" pattern this module follows).

This module deliberately does NOT make any network call unless an
access key is explicitly configured, and even then only exposes a
narrow, documented download function — never a general-purpose scraper.
No eBird data has been downloaded as part of building this adapter (see
BIRD_DATA_SOURCE_COMPARISON.md: "no eBird data was downloaded ...
consistent with 'do not download large datasets unless clearly
necessary'"). This is structure only, ready for real use once:

  1. A free access key is requested at https://ebird.org/st/request
     and set as the EBIRD_ST_ACCESS_KEY environment variable (see .env).
  2. The key's Terms of Use are read and this module's usage is
     confirmed to comply with them (redistribution terms were NOT
     confirmed during Phase 4 research — see BIRD_DATA_SOURCE_COMPARISON.md).
  3. The `ebirdst`-equivalent download logic below is filled in against
     the real API (https://ebird.github.io/ebirdst/articles/api.html) —
     the two real endpoints are "list available files for a species" and
     "download a single file" (GeoTIFF).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


class EbirdNotConfiguredError(RuntimeError):
    """Raised when eBird integration is used without an access key configured."""


@dataclass(frozen=True)
class EbirdAbundanceRequest:
    """
    Describes what would be requested from eBird Status & Trends —
    constructing this does not make any network call.
    """

    species_code: str  # eBird species code, e.g. "amecro" for American Crow
    week: int  # 1-52, per eBird's weekly Status & Trends cadence
    bounding_box: tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat), WGS84


def access_key() -> str | None:
    """Returns the configured EBIRD_ST_ACCESS_KEY, or None if unset."""
    return os.environ.get("EBIRD_ST_ACCESS_KEY")


def is_configured() -> bool:
    return access_key() is not None


def fetch_abundance_raster(request: EbirdAbundanceRequest, out_dir: str | Path) -> Path:
    """
    Would download the requested week's abundance GeoTIFF for the given
    species/bounding box to out_dir and return its path.

    NOT IMPLEMENTED against the real API yet — raises EbirdNotConfiguredError
    if no access key is set (the common case today), and NotImplementedError
    if a key IS set but the real HTTP call hasn't been written in yet.
    This split exists so a future implementer's first step is exactly
    "write the HTTP call," not "figure out where this even plugs in."
    """
    if not is_configured():
        raise EbirdNotConfiguredError(
            "EBIRD_ST_ACCESS_KEY is not set. Request a free key at "
            "https://ebird.org/st/request, read its Terms of Use (redistribution "
            "terms were not confirmed during this project's Phase 4 research — see "
            "BIRD_DATA_SOURCE_COMPARISON.md), then set it in .env before calling this."
        )
    raise NotImplementedError(
        "An access key is configured, but the real eBird Status & Trends API call "
        "(https://ebird.github.io/ebirdst/articles/api.html) has not been implemented "
        "yet — this adapter is structure only. Two endpoints are needed: list "
        "available files for a species/week, then download a single GeoTIFF file."
    )


def offline_placeholder_record(request: EbirdAbundanceRequest) -> dict:
    """
    Returns a schema-conformant-shaped dict with every value field null and
    bird_data_available=False, bird_data_source="ebird_status_trends"
    (see schemas.hazard_observation.BIRD_SCHEMA) — useful for wiring up
    downstream code against the eventual real shape of this data without
    needing a real key or network access yet.
    """
    return {
        "species_or_group": request.species_code,
        "estimated_abundance": None,
        "relative_abundance_index": None,
        "seasonal_presence": None,
        "migration_intensity": None,
        "observation_effort_hours": None,
        "distance_from_bird_observation_to_airport_km": None,
        "bird_data_source": "ebird_status_trends",
        "bird_data_resolution": "weekly_3km_modeled",
        "bird_data_available": False,
    }
