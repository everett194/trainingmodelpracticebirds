#!/usr/bin/env python3
"""
scripts/generate_sample_data.py
---------------------------------
Generates small, deterministic, SYNTHETIC fixtures that exercise every
stage of the BirdStrikeGeo pipeline (ingestion, alias resolution,
leakage checks, geospatial joins, feature engineering, modeling,
evaluation) without needing any real FAA or Trektellen data.

*** THIS IS NOT REAL DATA. ***
Every row produced here is invented, using a fixed random seed, purely so
the software can be exercised end to end. Airport names, site names,
strike records, and counts are all fabricated - see data/sample/README.md.
Every generated file carries data_origin="synthetic_fixture" and
operational_use=false, and every output the pipeline later derives from
this data must be labeled "SAMPLE DATA — NOT A REAL RESULT".

Run with:
    python scripts/generate_sample_data.py
or via the CLI:
    python -m birdstrikegeo.cli generate-sample-data
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from birdstrikegeo.features.taxonomy import SPECIES_TAXONOMY  # noqa: E402
from birdstrikegeo.schemas.trektellen import KNOWN_COUNT_TYPES  # noqa: E402

SAMPLE_DIR = REPO_ROOT / "data" / "sample"
RANDOM_SEED = 42

N_AIRPORTS = 25
N_TREKTELLEN_SITES = 40
N_TREKTELLEN_COUNT_ROWS_TARGET = 5000
N_FAA_STRIKES = 500

# Two loosely CONUS-like and Europe-like bounding boxes so the generated
# data exercises the dual-region CRS strategy described in
# GEOSPATIAL_METHODS.md. These are NOT precise geographic boundaries -
# just plausible synthetic scatter regions.
CONUS_BBOX = dict(lat=(30.0, 46.0), lon=(-120.0, -75.0))
EUROPE_BBOX = dict(lat=(45.0, 60.0), lon=(-5.0, 20.0))

SPECIES_LIST = list(SPECIES_TAXONOMY.items())  # [(scientific_name, (common_name, group)), ...]

DISCLAIMER = "SAMPLE DATA — NOT A REAL RESULT"


def _rng():
    return np.random.default_rng(RANDOM_SEED)


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def generate_airports(rng) -> pd.DataFrame:
    n_conus = 15
    n_europe = N_AIRPORTS - n_conus

    rows = []
    for i in range(N_AIRPORTS):
        region = "conus" if i < n_conus else "europe"
        bbox = CONUS_BBOX if region == "conus" else EUROPE_BBOX
        lat = rng.uniform(*bbox["lat"])
        lon = rng.uniform(*bbox["lon"])
        airport_id = f"SAMPLE{i + 1:02d}"
        rows.append(
            {
                "airport_id": airport_id,
                "icao": f"K{airport_id[-3:]}" if region == "conus" else f"E{airport_id[-3:]}",
                "iata": airport_id[-3:],
                "faa_lid": airport_id if region == "conus" else "",
                "airport_name": f"Sample {'Regional' if i % 2 == 0 else 'Municipal'} Airport {i + 1:02d} (SYNTHETIC)",
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "elevation_ft": round(rng.uniform(0, 2500), 1),
                "airport_type": rng.choice(["large_airport", "medium_airport", "small_airport"], p=[0.2, 0.5, 0.3]),
                "runway_count": int(rng.integers(1, 4)),
                "longest_runway_ft": int(rng.integers(3500, 12000)),
                "state": "SS" if region == "conus" else "",
                "country": "US-SAMPLE" if region == "conus" else "EU-SAMPLE",
                "data_origin": "synthetic_fixture",
                "operational_use": False,
            }
        )
    return pd.DataFrame(rows)


def generate_trektellen_sites(rng, airports: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(N_TREKTELLEN_SITES):
        # Scatter sites around a randomly chosen airport at a random
        # distance (0-300km) so proximity features have realistic variety,
        # including some sites far enough to trigger distance_warning.
        anchor = airports.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        bearing = rng.uniform(0, 2 * np.pi)
        distance_km = rng.exponential(scale=80.0)
        distance_km = min(distance_km, 400.0)
        dlat = (distance_km / 111.0) * np.cos(bearing)
        dlon = (distance_km / (111.0 * np.cos(np.radians(anchor["latitude"])))) * np.sin(bearing)

        lat = anchor["latitude"] + dlat
        lon = anchor["longitude"] + dlon
        is_conus = anchor["country"] == "US-SAMPLE"

        count_type = rng.choice(list(KNOWN_COUNT_TYPES), p=[0.65, 0.15, 0.15, 0.05])
        active_from = datetime(2015, 1, 1, tzinfo=timezone.utc)
        rows.append(
            {
                "site_id": f"TREK{i + 1:03d}",
                "site_name": f"Sample Watch Point {i + 1:02d} (SYNTHETIC)",
                "country": "US-SAMPLE" if is_conus else "EU-SAMPLE",
                "latitude": round(lat, 5),
                "longitude": round(lon, 5),
                "count_type": count_type,
                "timezone": "America/New_York" if is_conus else "Europe/Amsterdam",
                "site_active_from": active_from.date().isoformat(),
                "site_active_to": "",
                "data_origin": "synthetic_fixture",
                "operational_use": False,
            }
        )
    return pd.DataFrame(rows)


def generate_trektellen_counts(rng, sites: pd.DataFrame) -> pd.DataFrame:
    rows = []
    count_id = 0
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2024, 12, 31)
    date_span_days = (end_date - start_date).days

    rows_per_site = max(1, N_TREKTELLEN_COUNT_ROWS_TARGET // (N_TREKTELLEN_SITES * 3))

    for _, site in sites.iterrows():
        n_sessions = max(4, rows_per_site)
        session_days = np.sort(rng.integers(0, date_span_days, size=n_sessions))

        for day_offset in session_days:
            session_date = start_date + timedelta(days=int(day_offset))
            day_of_year = session_date.timetuple().tm_yday
            # Simple seasonal bump around spring (day ~100) and fall (day ~270)
            # migration peaks, purely to give the sample data a learnable
            # signal - NOT a real phenology model.
            seasonal_factor = 1.0 + 3.0 * (
                np.exp(-((day_of_year - 100) ** 2) / (2 * 25**2))
                + np.exp(-((day_of_year - 270) ** 2) / (2 * 25**2))
            )

            start_hour = rng.integers(5, 10)
            obs_hours = round(float(rng.uniform(1.0, 6.0)), 2)
            start_time = f"{start_hour:02d}:00"
            end_time = f"{min(start_hour + int(obs_hours), 23):02d}:{int((obs_hours % 1) * 60):02d}"

            n_species_this_session = rng.integers(1, 6)
            species_sample_idx = rng.choice(len(SPECIES_LIST), size=n_species_this_session, replace=False)

            for sp_idx in species_sample_idx:
                sci_name, (common_name, _group) = SPECIES_LIST[sp_idx]
                base_count = rng.poisson(lam=3.0 * seasonal_factor)
                count_id += 1
                rows.append(
                    {
                        "count_id": f"CNT{count_id:06d}",
                        "site_id": site["site_id"],
                        "count_date": session_date.date().isoformat(),
                        "start_time_local": start_time,
                        "end_time_local": end_time,
                        "observation_hours": obs_hours,
                        "species_common_name": common_name,
                        "species_scientific_name": sci_name,
                        "species_code": sci_name.replace(" ", "_").upper()[:10],
                        "count": int(base_count),
                        "flight_direction": rng.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]),
                        "count_type": site["count_type"],
                        "weather_notes": rng.choice(["clear", "overcast", "light rain", "windy", ""]),
                        "observer_effort_available": bool(rng.random() > 0.05),
                        "data_origin": "synthetic_fixture",
                        "operational_use": False,
                    }
                )
    return pd.DataFrame(rows)


def generate_weather(rng, airports: pd.DataFrame) -> pd.DataFrame:
    rows = []
    hours = 24 * 14  # 14 days hourly, per airport - kept small deliberately
    start = datetime(2024, 6, 1, tzinfo=timezone.utc)

    for _, airport in airports.iterrows():
        base_temp = 15 if airport["country"] == "US-SAMPLE" else 12
        for h in range(hours):
            ts = start + timedelta(hours=h)
            diurnal = 6 * np.sin(2 * np.pi * (ts.hour - 6) / 24)
            rows.append(
                {
                    "station_id": f"WX-{airport['airport_id']}",
                    "timestamp_utc": ts.isoformat(),
                    "latitude": airport["latitude"],
                    "longitude": airport["longitude"],
                    "temperature_c": round(base_temp + diurnal + rng.normal(0, 2), 1),
                    "dewpoint_c": round(base_temp + diurnal - rng.uniform(2, 8), 1),
                    "relative_humidity_pct": round(float(rng.uniform(30, 95)), 1),
                    "wind_speed_knots": round(float(rng.exponential(6)), 1),
                    "wind_direction_deg": int(rng.integers(0, 360)),
                    "wind_gust_knots": round(float(rng.exponential(9)), 1),
                    "visibility_km": round(float(rng.uniform(2, 15)), 1),
                    "precipitation_mm": round(float(rng.choice([0, 0, 0, 0.5, 2, 5], p=[0.6, 0.15, 0.1, 0.07, 0.05, 0.03])), 2),
                    "pressure_hpa": round(float(rng.normal(1013, 8)), 1),
                    "cloud_ceiling_ft": int(rng.choice([500, 1500, 5000, 15000, 25000])),
                    "present_weather_code": rng.choice(["", "RA", "BR", "HZ"]),
                    "quality_flag": "sample",
                    "data_origin": "synthetic_fixture",
                    "operational_use": False,
                }
            )
    return pd.DataFrame(rows)


def generate_faa_strikes(rng, airports: pd.DataFrame) -> pd.DataFrame:
    """
    Generates rows using RAW, FAA-export-like column names (not the
    canonical schema) so the sample workflow exercises real alias
    resolution (birdstrikegeo.data.column_aliases), not just a pass-through.
    """
    rows = []
    start_date = datetime(2018, 1, 1)
    date_span_days = (datetime(2024, 12, 31) - start_date).days

    phases = ["Take-off run", "Climb", "Approach", "Landing Roll", "En Route", "Descent"]
    sky_conditions = ["No Cloud", "Some Cloud", "Overcast"]
    precip_options = ["None", "Rain", "Snow", "Fog"]
    aircraft_makes = ["Boeing", "Airbus", "Embraer", "Bombardier", "Cessna"]
    mass_classes = ["1", "2", "3", "4", "5"]  # FAA-style banded mass class codes

    for i in range(N_FAA_STRIKES):
        airport = airports.sample(1, random_state=int(rng.integers(0, 1_000_000))).iloc[0]
        incident_date = start_date + timedelta(days=int(rng.integers(0, date_span_days)))
        sci_name, (common_name, group) = SPECIES_LIST[int(rng.integers(0, len(SPECIES_LIST)))]

        phase = rng.choice(phases, p=[0.2, 0.15, 0.25, 0.15, 0.15, 0.1])
        wildlife_size = rng.choice(["Small", "Medium", "Large"], p=[0.5, 0.35, 0.15])
        number_struck_raw = rng.choice(["1", "2-10", "11-100", "Over 100"], p=[0.7, 0.2, 0.08, 0.02])
        height_agl = float(rng.exponential(800))
        speed = float(rng.uniform(80, 250))

        # Hand-designed (NOT scientifically validated) correlation so the
        # sample pipeline has a learnable signal: larger birds, higher
        # speed, and multi-bird strikes are more likely to be recorded as
        # damaging. This mirrors real intuition but is fabricated, exactly
        # like synthetic_demo/generate_data.py - see data/sample/README.md.
        score = -2.0
        score += {"Small": -1.0, "Medium": 0.3, "Large": 1.8}[wildlife_size]
        score += {"1": -0.3, "2-10": 0.4, "11-100": 1.0, "Over 100": 1.4}[number_struck_raw]
        score += 0.0015 * speed
        score += rng.normal(0, 1.0)
        damage_probability = 1 / (1 + np.exp(-score))

        # ~8% of rows get an ambiguous/blank damage indicator, to exercise
        # "exclude unknown target rows" handling downstream.
        if rng.random() < 0.08:
            damage_ind = rng.choice(["", "UNK", "N/A"])
            damage_level = ""
        else:
            is_damage = rng.random() < damage_probability
            damage_ind = "Y" if is_damage else "N"
            damage_level = rng.choice(["M", "S"], p=[0.7, 0.3]) if is_damage else ""

        effect = ""
        if damage_ind == "Y":
            effect = rng.choice(["Precautionary Landing", "Aborted Takeoff", "None", "Other"], p=[0.3, 0.15, 0.4, 0.15])

        rows.append(
            {
                "INDEX_NR": f"SAMPLE-{i + 1:05d}",
                "INCIDENT_DATE": incident_date.date().isoformat(),
                "TIME": f"{int(rng.integers(0, 24)):02d}:{int(rng.integers(0, 60)):02d}",
                "AIRPORT_ID": airport["airport_id"] if rng.random() > 0.03 else "",  # a few unresolved airports
                "AIRPORT": airport["airport_name"],
                "STATE": airport["state"],
                "LATITUDE": airport["latitude"] + rng.normal(0, 0.02),
                "LONGITUDE": airport["longitude"] + rng.normal(0, 0.02),
                "PHASE_OF_FLIGHT": phase,
                "HEIGHT": round(height_agl, 0),
                "SPEED": round(speed, 0),
                "AC_MAKE": rng.choice(aircraft_makes),
                "AC_MODEL": f"Model-{int(rng.integers(100, 900))}",
                "AC_CLASS": "Airplane",
                "AC_MASS": rng.choice(mass_classes),
                "ENG_TYPE": rng.choice(["Turbofan", "Turboprop", "Turbojet"]),
                "NUM_ENGS": int(rng.choice([2, 4], p=[0.9, 0.1])),
                "SKY": rng.choice(sky_conditions),
                "PRECIPITATION": rng.choice(precip_options, p=[0.7, 0.15, 0.05, 0.1]),
                "SPECIES": common_name,
                "SPECIES_ID": sci_name,
                "SIZE": wildlife_size,
                "NUM_SEEN": number_struck_raw,
                "NUM_STRUCK": number_struck_raw,
                "DAMAGE_IND": damage_ind,
                "DAM_LEVEL": damage_level,
                "EFFECT": effect,
                "STR_PARTS": rng.choice(["Windshield", "Engine", "Wing", "Nose", "Fuselage"]),
                "DAM_PARTS": rng.choice(["Windshield", "Engine", "Wing", "", ""]) if damage_ind == "Y" else "",
                "COST_REPAIRS": round(float(rng.exponential(15000)), 2) if damage_ind == "Y" else 0.0,
                "COST_OTHER": round(float(rng.exponential(2000)), 2) if damage_ind == "Y" else 0.0,
                "NR_INJURIES": int(rng.choice([0, 0, 0, 0, 1], p=[0.9, 0.05, 0.03, 0.01, 0.01])),
                "NR_FATALITIES": 0,
                "REMARKS": "Synthetic sample record generated for pipeline testing.",
                "data_origin": "synthetic_fixture",
                "operational_use": False,
            }
        )
    return pd.DataFrame(rows)


def airports_to_geojson(airports: pd.DataFrame) -> dict:
    features = []
    for _, row in airports.iterrows():
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [row["longitude"], row["latitude"]]},
                "properties": {k: (None if pd.isna(v) else v) for k, v in row.items() if k not in ("latitude", "longitude")},
            }
        )
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}},
        "metadata": {
            "source": "generate_sample_data.py",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sample_data_flag": True,
            "disclaimer": DISCLAIMER,
        },
        "features": features,
    }


def write_provenance_sidecar():
    provenance = {
        "source_name": "synthetic_fixture (generate_sample_data.py)",
        "source_url": "",
        "download_date": "",
        "access_method": "generated_locally",
        "permission_or_license": "not_applicable_synthetic_data",
        "contact_or_citation": "",
        "geographic_scope": "synthetic CONUS-like and Europe-like bounding boxes",
        "temporal_scope": "2023-01-01 to 2024-12-31",
        "notes": (
            "This is a SYNTHETIC FIXTURE, not a real Trektellen export. It exists "
            "only to exercise the ingestion/feature/model pipeline end to end. "
            "Real Trektellen data requires its own provenance record with a real "
            "source_url, download_date, and permission_or_license - see "
            "DATA_DOWNLOAD_GUIDE.md."
        ),
        "data_origin": "synthetic_fixture",
        "operational_use": False,
    }
    (SAMPLE_DIR / "trektellen_provenance_sample.json").write_text(json.dumps(provenance, indent=2))


def main():
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    rng = _rng()

    print(f"[generate_sample_data] {DISCLAIMER}")
    print(f"[generate_sample_data] Using fixed random seed {RANDOM_SEED} for determinism.")

    airports = generate_airports(rng)
    airports.drop(columns=["data_origin", "operational_use"]).to_csv(
        SAMPLE_DIR / "airports_sample.csv", index=False
    )
    (SAMPLE_DIR / "airports_sample.geojson").write_text(json.dumps(airports_to_geojson(airports), indent=2))
    print(f"[generate_sample_data] Wrote {len(airports)} sample airports.")

    sites = generate_trektellen_sites(rng, airports)
    sites.to_csv(SAMPLE_DIR / "trektellen_sites_sample.csv", index=False)
    print(f"[generate_sample_data] Wrote {len(sites)} sample Trektellen sites.")

    counts = generate_trektellen_counts(rng, sites)
    counts.to_csv(SAMPLE_DIR / "trektellen_counts_sample.csv", index=False)
    print(f"[generate_sample_data] Wrote {len(counts)} sample Trektellen count rows.")

    write_provenance_sidecar()

    weather = generate_weather(rng, airports)
    weather.to_csv(SAMPLE_DIR / "weather_sample.csv", index=False)
    print(f"[generate_sample_data] Wrote {len(weather)} sample hourly weather rows.")

    faa = generate_faa_strikes(rng, airports)
    faa.to_csv(SAMPLE_DIR / "faa_strikes_sample.csv", index=False)
    print(f"[generate_sample_data] Wrote {len(faa)} sample FAA strike records "
          f"(raw FAA-export-style column names, for alias-resolution testing).")

    print(f"\n[generate_sample_data] Done. {DISCLAIMER}")
    print(f"[generate_sample_data] All files written to {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
