# BirdStrikeGeo

A portfolio/educational geospatial machine-learning project exploring
reported wildlife strikes to aircraft and bird-migration monitoring
data. **Not operational aviation-safety software.**

---

## The problem

Wildlife strikes to aircraft are a real, ongoing aviation-safety
concern. Two genuinely different questions come up when studying them:

1. **Given that a strike was reported, was it damaging?** Some
   information is usually known at strike-report time (species, flight
   phase, speed, conditions) that correlates with whether the aircraft
   was damaged.
2. **Is wildlife activity around a given airport, right now, elevated or
   not?** Birds migrate seasonally and daily; geographic context
   (proximity to water, migration corridors, monitored bird-activity
   data) plausibly says something about relative risk exposure.

These are **not the same question**, and conflating them is a common
mistake. BirdStrikeGeo keeps them explicitly separate.

## What this project actually predicts

**As of the Phase 1/2 audit and consolidation** (see
`DATA_AUDIT_REPORT.md`, `LEGACY_SYSTEMS.md`), each task now has a
PRIMARY (current, authoritative) implementation and a preserved LEGACY
one — the same treatment `synthetic_demo/` has always gotten.

| Task | Question answered | Primary implementation | Legacy implementation |
|---|---|---|---|
| **A — Damage model** | *Given a reported wildlife strike*, estimated probability it caused aircraft damage. | `/ga-damage` — calibrated CatBoost, confirmed GA population only | `/damage` — PyTorch NN, all aircraft types (see `LEGACY_SYSTEMS.md`) |
| **B — Bird-hazard / activity index** | A relative measure of bird presence/hazard for an airport during a time window, from nearby bird-migration monitoring. | `/hazard` — FAA-derived species-severity risk joined to local Trektellen activity | `/activity` — a transparent `activity_index` formula (see `LEGACY_SYSTEMS.md`) |

**Neither task predicts the probability that a wildlife strike will
occur.** That would require knowing how many flights *didn't* have a
strike — a flight-exposure denominator (e.g. BTS T-100 departures) this
project does not yet have. See "Scientific limitations" below.

## Why bird migration and geography matter

Bird activity near an airport isn't uniform across the year or the day:
migration season, dawn/dusk timing, proximity to water and wetlands, and
recent local activity all plausibly shift risk exposure. Task B exists
to explore whether *publicly available bird-migration monitoring data*
can say anything useful about that — while being honest that monitoring
coverage is sparse and uneven, so the resulting index is only as good as
the nearest available observations (surfaced explicitly via
`distance_warning` / `low_coverage_warning` on every result).

## Current development status

Functional skeleton, running end-to-end **on synthetic sample data**.
Real-data mode (`--mode real`) is fully wired up in every script and
fails gracefully — with an exact list of missing files and a pointer to
`DATA_DOWNLOAD_GUIDE.md` — until you supply real FAA/Trektellen data.
90 automated tests pass without any real data or network access.

---

## Architecture diagram (Task A — DamageNet)

```
raw strike record (34 FAA fields, alias-resolved)
        │
        ▼
leakage-safe feature selection (damage_flag, cost, injuries, ... BLOCKED)
        │
        ▼
time/season features (cyclical month/hour, dawn/dusk via sunrise-sunset)
        │
        ▼
preprocessing (fit on TRAIN split only)
  numeric  → impute (median) → standardize
  categorical → rare-category bucketing → one-hot encode
        │
        ▼
   encoded_input (84 features, sample data)
        │
        ▼
   Linear(64) → ReLU → Dropout(0.20)
        │
   Linear(32) → ReLU → Dropout(0.10)
        │
   Linear(1)                                    ← raw logit
        │
   sigmoid (inference only)
        │
        ▼
"Estimated probability of aircraft damage,
 conditional on a reported wildlife strike"
```

## Data-flow diagram (both tasks)

```
data/raw/*  (real, gitignored)  ──┐
data/sample/*  (synthetic, committed)  ──┤
                                          ▼
                              scripts/*.py  (birdstrikegeo.cli wraps these)
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        ▼                                 ▼                                 ▼
 prepare_data.py                build_geospatial_features.py       train_models.py
 (FAA ingest → target →         (Trektellen ingest → CRS-aware      (baselines + DamageNet
  leakage-safe features →        joins → activity features →        → threshold tuning →
  chronological split)           GeoJSON/GPKG/GeoParquet exports)    evaluation → checkpoint)
        │                                 │                                 │
        ▼                                 ▼                                 ▼
data/processed/*                data/processed/layers/*          models/damage/*.pt
results/damage/*_quality*.json  results/activity/*_quality*.json  results/damage/*
                                          │
                                          ▼
                                       app.py  (Flask, 3 modes)
                                  /synthetic  /damage  /activity
```

---

## Quick start (sample data — no downloads required)

> **If this repo lives under iCloud Drive (e.g. `~/Documents`)**, exclude
> `.venv` from sync BEFORE installing anything, or every Python
> invocation can silently degrade to 100-250x slower (confirmed: a bare
> `import pandas` took 255 seconds with `.venv` iCloud-synced vs. under
> 1 second with it excluded) and large files under `data/` can get
> evicted to 0-byte placeholders that hang on read until re-downloaded
> (`brctl download <path>`):
> ```bash
> mv .venv .venv.nosync && ln -s .venv.nosync .venv
> ```
> Do this every time `.venv` is recreated from scratch in this location.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .

python -m birdstrikegeo.cli generate-sample-data
python -m birdstrikegeo.cli prepare-data --mode sample
python -m birdstrikegeo.cli build-geo-features --mode sample
python -m birdstrikegeo.cli train --task damage --mode sample
python -m birdstrikegeo.cli evaluate --task damage --mode sample
python -m birdstrikegeo.cli export-layers --mode sample
python -m birdstrikegeo.cli run-app
# open http://localhost:5001
```

Or run the test suite (no data generation needed first — it generates
its own fixtures):

```bash
pytest tests/ -v
```

## Moving to real data

```bash
# after downloading real files into data/raw/ - see DATA_DOWNLOAD_GUIDE.md
python -m birdstrikegeo.cli validate-data --mode real
python -m birdstrikegeo.cli prepare-data --mode real
python -m birdstrikegeo.cli build-geo-features --mode real
python -m birdstrikegeo.cli train --task damage --mode real
```

Every `--mode real` command checks for its required files first and
exits with an exact list of what's missing (never a raw stack trace) if
something isn't there yet.

---

## The five kinds of data in this repository

| Kind | Where | Real or synthetic? | Used by |
|---|---|---|---|
| **Synthetic educational data** | `synthetic_demo/` | 100% synthetic, hand-designed | The original 8-input neural-network demo — unrelated to BirdStrikeGeo's real tasks. |
| **Sample integration fixtures** | `data/sample/` | 100% synthetic, deterministic (seed 42) | Exercises the full BirdStrikeGeo pipeline (`--mode sample`) without downloads. |
| **Real FAA strike data** | `data/raw/faa_wildlife_strikes.csv` | Real, user-downloaded | Task A, `--mode real`. |
| **Real Trektellen data** | `data/raw/trektellen_*.csv` | Real, user-downloaded, requires provenance record | Task B, `--mode real`. |
| **Future flight-exposure data** | `data/raw/bts_t100_departures.csv` (placeholder) | Real, not yet integrated | A future, more rigorous strike-*probability* model — see limitations below. |

## Project structure

```
synthetic_demo/          Original educational demo (preserved, self-contained)
src/birdstrikegeo/       The BirdStrikeGeo package
  schemas/                 Canonical field definitions (FAA, Trektellen, airports, weather)
  data/                     Ingestion, column-alias resolution, validation, quality reports
  geo/                      CRS strategy, spatial/temporal joins, layer exports, ArcGIS adapter
  features/                 Target construction, leakage guard, damage/activity feature building
  models/                   DamageNet (PyTorch), baselines, activity_index formula, checkpoints
  training/                 Chronological splitting, DamageNet training loop, threshold tuning
  evaluation/                Metrics, calibration, bootstrap CIs, explainability, plots
  inference/                 predict_damage(), calculate_activity_for_airport()
  cli.py                    python -m birdstrikegeo.cli ...
scripts/                  Standalone, independently-runnable pipeline scripts
configs/                  YAML configuration (no important choices hardcoded)
data/{raw,sample,interim,processed}/
templates/, static/, app.py    Flask app: 3 modes (synthetic / damage / activity)
tests/                    90 tests, no real data or network access required
.github/workflows/tests.yml    CI: sample pipeline + tests on every push
```

## Documentation map

- **`DATABASE.md`** — the consolidated SQLite database (`data/birdstrikegeo.db`): schema, example queries, how to rebuild it. Start here for the fastest way to explore this project's data.
- **`DATA_AUDIT_REPORT.md`** — Phase 1 repository/data audit: what exists, what's duplicated, what's unverified.
- **`LEGACY_SYSTEMS.md`** — which pipeline is primary vs. legacy for each task, and why.
- **`BIRD_DATA_SOURCE_COMPARISON.md`** — eBird/BirdCast/USGS/Movebank/Motus/Audubon/Esri source comparison for future bird-data integration.
- **`DATA_CARD.md`** — every dataset, real and synthetic, and its known limitations.
- **`MODEL_CARD.md`** — both models' architecture, training, sample-data results, and limitations.
- **`GEOSPATIAL_METHODS.md`** — CRS strategy, distance-decay weighting, layer exports, ArcGIS interoperability, H3 grid indexing.
- **`ESRI_DISCUSSION_NOTES.md`** — a project summary and 20 open GIS design questions for a real analyst.
- **`DATA_DOWNLOAD_GUIDE.md`** — exactly what to download and where to put it.
- **`reports/latest/data_inventory.json`** — machine-readable data inventory (row counts, columns, missingness per file).

## Scientific limitations

- FAA wildlife-strike reporting is not a complete census of all strikes.
- Trektellen sites are not uniformly distributed.
- Observer effort varies session to session.
- Visible migration counts, captures, and nocturnal flight calls measure
  different processes and are never silently combined.
- A distant monitoring site may not represent activity at a given
  airport.
- Missing observations do not mean zero birds.
- Reported damage may have missingness or reporting bias.
- Correlation does not demonstrate causation.
- An activity index is not a certified operational risk forecast.
- Results from Europe (denser Trektellen coverage) may not generalize to
  North America.
- Trektellen's usefulness for a given airport depends entirely on
  spatial and temporal proximity of the nearest monitoring.
- A true strike-*probability* model requires defensible non-strike or
  flight-exposure observations this project does not yet have.

This project is a demonstration of a responsible approach to a hard,
data-limited problem — not a finished operational tool.
