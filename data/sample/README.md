# data/sample/

> **SAMPLE DATA — NOT A REAL RESULT.** Every file in this folder is
> synthetic. It exists only so the BirdStrikeGeo pipeline can be
> installed and run end to end without downloading real, license-
> restricted FAA or Trektellen data first.

Generated deterministically (fixed random seed `42`) by
`scripts/generate_sample_data.py`. Regenerate at any time with:

```bash
python scripts/generate_sample_data.py
# or: python -m birdstrikegeo.cli generate-sample-data
```

Every row in every file below carries `data_origin: "synthetic_fixture"`
and `operational_use: false`. Every metric, chart, map, or web page the
pipeline produces from this data is labeled
`SAMPLE DATA — NOT A REAL RESULT`.

## Files

| File | Rows (approx) | Purpose |
|---|---|---|
| `faa_strikes_sample.csv` | 500 | Exercises `ingest_faa` end to end, **including column-alias resolution** - column names deliberately mimic a raw FAA export (`INDEX_NR`, `DAMAGE_IND`, etc.), not the canonical schema. |
| `trektellen_sites_sample.csv` | 40 | Trektellen monitoring-site metadata (not listed in the original file-structure sketch, but required for `site_id` resolution during count ingestion - see `birdstrikegeo/schemas/trektellen.py`). |
| `trektellen_counts_sample.csv` | ~5,000 | One row per site x count-session x species, already in canonical long format. |
| `trektellen_provenance_sample.json` | - | Provenance sidecar clearly marked as a synthetic fixture, not a real authorized export - real Trektellen data requires filling in the real fields documented in `DATA_DOWNLOAD_GUIDE.md`. |
| `airports_sample.geojson` / `airports_sample.csv` | 25 | 15 airports scattered in a CONUS-like bounding box, 10 in a Europe-like bounding box, to exercise the dual-region CRS strategy (see `GEOSPATIAL_METHODS.md`). |
| `weather_sample.csv` | ~8,400 | 14 days of hourly weather at each sample airport. |

## What's fabricated vs. what's real

- **Airport, site, and strike-record identities are entirely invented.**
  Names are suffixed `(SYNTHETIC)` and IDs use a `SAMPLE`/`TREK` prefix
  that will never collide with real FAA airport IDs or Trektellen site
  IDs.
- **Species names are real bird species** (drawn from
  `birdstrikegeo/features/taxonomy.py`) so taxonomic-group features have
  something meaningful to compute, but their assignment to sample
  strike/count records is random.
- **The relationship between strike circumstances (bird size, number
  struck, speed) and the synthetic `damage` outcome is hand-designed**,
  the same way `synthetic_demo/generate_data.py`'s labels are - it gives
  the sample pipeline a learnable signal but is **not derived from or
  validated against any real strike data.**
- **~8% of FAA sample rows have a deliberately blank/ambiguous damage
  indicator** so the "exclude unknown target rows" path is exercised.
- **A few sample strike rows have a blank `AIRPORT_ID`** so the
  "unresolved airports" quality-report path is exercised.

Do not use metrics, trained weights, or maps produced from this data as
evidence of real-world model performance.
