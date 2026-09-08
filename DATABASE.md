# Consolidated Database

One SQLite file (`data/birdstrikegeo.db`, gitignored, regeneratable)
consolidating every real, structured data source in this project — FAA
strike data, airports, all three Trektellen sources, and key report
outputs — so a human or an AI agent can query everything with plain SQL
instead of hunting across `data/raw/`, `data/reference/`,
`data/processed/`, and `reports/latest/` in four different file formats.

## Build / rebuild it

```bash
python scripts/build_consolidated_database.py
```

Always a full rebuild (not an incremental merge) — it drops and
recreates the file from whatever source files currently exist, and
skips (with a clear message, never a fake row) any source that isn't
present locally. Safe to re-run any time; takes a few seconds.

## Query it

```bash
sqlite3 data/birdstrikegeo.db          # interactive shell
sqlite3 data/birdstrikegeo.db ".tables"
sqlite3 data/birdstrikegeo.db ".schema faa_strikes_ga"
```

From Python:

```python
import sqlite3, pandas as pd
conn = sqlite3.connect("data/birdstrikegeo.db")
df = pd.read_sql("SELECT * FROM trektellen_counts WHERE species = 'Northern Cardinal'", conn)
```

**Start here if you don't know the schema yet:**

```sql
SELECT table_name, row_count, description FROM _table_metadata;
```

Every table's purpose, row count, source file, and full column list is
in that one metadata table — read it before writing any query against
an unfamiliar table.

## Tables

| Table | Rows | Grain | Status |
|---|---|---|---|
| `faa_strikes_ga` | 29,522 | one row per reported strike, GA population only | **primary** — see `LEGACY_SYSTEMS.md` |
| `airports` | 86,050 | one row per airport, worldwide | raw OurAirports columns, not yet remapped to `AIRPORT_SCHEMA` |
| `trektellen_counts` | 93,450 | species x year x period preset x metric | **authoritative** Trektellen source |
| `trektellen_period_index` | 37 | one row per period preset | validates `trektellen_counts` against |
| `trektellen_career_summary` | 150 | one row per species (career totals) | fields not in `trektellen_counts` (year_max, career newly_ringed/retraps) |
| `trektellen_single_season_2025_legacy` | 129 | species x month, 2025 only | **legacy** — superseded by `trektellen_counts` |
| `national_species_risk` | 484 | one row per species | FAA-wide severity risk score |
| `legacy_damage_model_comparison` | 4 | one row per model | sample-data-only, see `LEGACY_SYSTEMS.md` |
| `ga_population_definitions` | 4 | one row per population filter | which rows count as GA, and how many |
| `ga_population_summary` | 1 | single row | source hash, target class balance |
| `severity_model_metrics` | 1 | single row | severity regressor validation metrics |

## Why `trektellen_counts` is the one to use

Three real Trektellen sources exist in this repo (see `DATA_CARD.md` §2
for the full provenance of each), at increasing resolution:

1. `trektellen_single_season_2025_legacy` — one season, monthly, combined metric only.
2. `trektellen_career_summary` — 10 years, but only career TOTALS per species (no per-year breakdown except what's re-derivable from source #1's single year).
3. `trektellen_counts` — 10 years (2017-2026) x 37 period presets (every month, two named seasons, every rolling 2- and 3-month window, "all months") x 3 metrics (combined / newly ringed / retraps). **This strictly supersedes both other sources in resolution.**

`trektellen_counts` is deliberately kept SEPARATE from `faa_strikes_ga`
— they're different populations at different grains (one strike-level,
one station-activity-level) and must never be joined as if they were
the same kind of row. See `README.md` "The problem" for why conflating
strike-conditioned and activity-conditioned data is exactly the mistake
this project is built to avoid.

## A worked example: local species risk, done in SQL

The same join `hazard/species_risk.py` does in pandas, expressed as SQL
against the consolidated database:

```sql
SELECT
  t.species,
  SUM(t.count) / 3389.0 AS local_activity_per_hour,  -- 3,389 = pooled 2022-2025 observation hours
  r.risk_score,
  (SUM(t.count) / 3389.0) * r.risk_score AS local_risk_contribution
FROM trektellen_counts t
LEFT JOIN national_species_risk r ON UPPER(r.species) = UPPER(t.species)
WHERE t.period_code = 0 AND t.metric = 'combined' AND t.year IN (2022, 2023, 2024, 2025)
GROUP BY t.species, r.risk_score
ORDER BY local_risk_contribution DESC
LIMIT 15;
```

## Adding more data later

- **More FBBO crosstab years/presets**: re-export
  `foremans_branch_bbo_yeartotals_all_presets.xlsx` from Trektellen and
  replace the copy in the repo root (or wherever it's picked up from —
  see `scripts/build_consolidated_database.py`'s glob), then re-run the
  build script.
- **More annual-totals PDFs**: drop into `data/reference/` (pattern
  `trektellen*annual*.pdf`) — picked up automatically, see
  `data/reference/README.md`.
- **A new data source entirely**: add one `_add_<source>()` function to
  `scripts/build_consolidated_database.py` following the existing
  pattern (load, `_write_table()` with a real description, done) — the
  metadata table and indexes are handled generically.
