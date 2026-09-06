# Data Download Guide

Everything you need to download to move this project from sample data to
real data. Only the FAA and Trektellen sections are required to run
`--mode real` on Task A and Task B respectively — airports, weather, and
flight-exposure data are optional/future enhancements.

After downloading, run `python -m birdstrikegeo.cli validate-data --mode real`
to confirm everything is in place before running the pipeline.

---

## FAA Wildlife Strike Database

```text
Expected filename: data/raw/faa_wildlife_strikes.csv
Download URL: [USER TO COMPLETE]
Download date: [USER TO COMPLETE]
Date range: [USER TO COMPLETE]
Filters applied: [USER TO COMPLETE]
```

The FAA publishes this database publicly (search "FAA Wildlife Strike
Database" for the current portal — the exact URL has moved over time, so
it's intentionally left for you to fill in with the live link).
`.xlsx` exports are also accepted at
`data/raw/faa_wildlife_strikes.xlsx`.

Column names have varied across export vintages — this project resolves
them via `src/birdstrikegeo/data/column_aliases.py`. Run
`python -m birdstrikegeo.cli inspect-data --mode real` after downloading
to see exactly which columns resolved and which didn't; extend
`FAA_COLUMN_ALIASES` if your export uses names not yet covered.

## Trektellen

```text
Expected count filename: data/raw/trektellen_counts.csv
Expected site filename: data/raw/trektellen_sites.csv
Access/export method: [USER TO COMPLETE]
Permission/license: [USER TO COMPLETE]
Contact/citation: [USER TO COMPLETE]
Geographic scope: [USER TO COMPLETE]
Date range: [USER TO COMPLETE]
Count types: [USER TO COMPLETE]
```

**This project never scrapes Trektellen.** Real data requires:

1. A user-authorized export (CSV/XLSX) from trektellen.org, obtained
   according to their terms of use, OR a documented API response if you
   have an authorized endpoint.
2. A completed provenance record at
   `data/raw/trektellen_provenance.json` with every field listed in
   `src/birdstrikegeo/schemas/trektellen.py:TREKTELLEN_PROVENANCE_FIELDS`
   (`source_name`, `source_url`, `download_date`, `access_method`,
   `permission_or_license`, `contact_or_citation`, `geographic_scope`,
   `temporal_scope`, `notes`). `scripts/build_geospatial_features.py
   --mode real` refuses to proceed without this file — see
   `data/sample/trektellen_provenance_sample.json` for the shape (it's
   marked as a synthetic fixture; yours should describe your real export).

Your export may be in long format (one row per site × session × species)
or wide format (one column per species) — both are accepted and
converted automatically (`birdstrikegeo/data/ingest_trektellen.py`).

## Airports

```text
Expected filename: data/raw/airports.geojson (or airports.csv, see below)
Provider: OurAirports (https://ourairports.com/data/), mirrored at
          https://github.com/davidmegginson/ourairports-data
Layer URL: https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/airports.csv
License: Public domain (Unlicense / Open Data Commons PDDL) -- no
         attribution required, commercial use and redistribution permitted.
Downloaded: 2026-09-06, 86,051 airports worldwide (all types, including
            heliports and closed airports).
```

**Status as of 2026-09-06: downloaded to `data/raw/airports.csv`, NOT
yet wired into the pipeline.** `ingest_airports.py` expects
`AIRPORT_SCHEMA` column names (`airport_id`, `icao`, `iata`, `faa_lid`,
`airport_name`, `latitude`, `longitude`, `elevation_ft`, `airport_type`,
`runway_count`, `longest_runway_ft`, `state`, `country`); the raw
OurAirports export uses different names (`ident`, `icao_code`,
`iata_code`, `local_code`, `name`, `latitude_deg`, `longitude_deg`,
`type`, `iso_region` (e.g. `"US-PA"`, needs the `US-` prefix stripped),
`iso_country`) and has no `runway_count`/`longest_runway_ft` columns at
all (those live in the separate `runways.csv`, not yet downloaded). A
small column-mapping/filter step is needed before this can be passed to
`load_airports()` -- left as future integration work, not done here.

See `ESRI_DISCUSSION_NOTES.md` question 15 for what an Esri
analyst might recommend here (e.g. a public FAA/Eurocontrol airport
layer, or an ArcGIS Living Atlas layer). Without this file, FAA-reported
lat/lon on individual strikes is still usable directly.

## Weather

```text
Expected filename: data/raw/hourly_weather.parquet
Provider: NOAA or other documented provider
Download method: [TO COMPLETE]
```

Optional. `NOAA_TOKEN` in `.env` is used if you build a real download
step against NOAA's API (not implemented in this version — the ingestion
side, `birdstrikegeo/data/ingest_weather.py`, only reads an already-
downloaded Parquet/CSV file).

## Flight exposure

```text
Expected filename: data/raw/bts_t100_departures.csv
Provider: BTS T-100
Date range: [TO COMPLETE]
Known coverage limitations: [TO COMPLETE]
```

Not used by either task in this version — reserved as the future
flight-exposure denominator needed before a true strike-*probability*
(as opposed to Task A's damage-*given-a-reported-strike*) model could be
built responsibly. See `README.md` "Scientific limitations".
