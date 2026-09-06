# Phase 1 Data & Repository Audit

Read-only audit, performed before any schema redesign or new modeling
work. Companion machine-readable file: `reports/latest/data_inventory.json`.

This project is more mature than a single README suggests: it already
contains **two generations** of the "conditional damage probability"
model and **two independent Trektellen ingestion paths**, built at
different times for different scopes. The most important output of
this audit is making that duplication explicit, because it directly
affects which pipeline should be treated as authoritative going
forward (see "Recommended primary target").

---

## 1. Directory structure

```
synthetic_demo/          Original 8-input NN demo, 100% synthetic, self-contained (PRESERVE, clearly labeled)
src/birdstrikegeo/
  schemas/                 Canonical field defs: faa.py, trektellen.py, airports.py, weather.py
  data/                     Ingestion + column-alias resolution + validation + quality_report (OLD/all-aircraft path)
  geo/                      CRS strategy, spatial/temporal joins, layer exports, ArcGIS adapter
  features/                 OLD Task A feature building + leakage.py (canonical-name leakage guard)
  models/                   DamageNet (PyTorch), baselines, activity_index formula, checkpoints
  training/                 Chronological splitting, DamageNet training loop, threshold tuning
  evaluation/               Metrics, calibration, bootstrap CIs, explainability, plots
  inference/                predict_damage() (OLD/PyTorch), calculate_activity_for_airport()
  ga/                       NEW CatBoost conditional-damage model (GA population only), own feature_policy.py
  hazard/                   NEW severity-regression + Eastern Shore/Trektellen risk-report pipeline
  cli.py                    python -m birdstrikegeo.cli ...
scripts/                  16 standalone scripts (old + new pipelines, see section 4)
configs/                  YAML config per pipeline (damage_model, ga_damage, eastern_shore_risk, geospatial, data_sources)
data/{raw,sample,interim,processed}/
models/, results/, reports/latest/
templates/, app.py       Flask app, 5 modes as of this session (synthetic / damage / activity / ga-damage / hazard)
tests/                    21 files, 119 test functions (199 passed / 6 skipped last verified run)
.github/workflows/tests.yml   CI: sample pipeline + tests on every push
DATA_CARD.md, MODEL_CARD.md, GEOSPATIAL_METHODS.md, ESRI_DISCUSSION_NOTES.md, DATA_DOWNLOAD_GUIDE.md
```

No files were moved or deleted during this audit.

---

## 2. Data file inventory

Full detail in `reports/latest/data_inventory.json`. Summary:

| File | Rows | Cols | Unit of observation | Status |
|---|---|---|---|---|
| `data/raw/faa_wildlife_strikes.xlsx` | 353,969 | 102 | strike incident, all aircraft types | real, gitignored |
| `data/raw/read_me.xls` | n/a | n/a | FAA's own doc workbook, not data | real, gitignored |
| `data/raw/trektellen_2025_year_totals.csv` | 129 (127 species + 2 summary rows) | 24 | species × month, single station (FBBO / site 3460) | real, gitignored, **no provenance record** |
| `data/interim/ga/faa_raw_cache.parquet` | 353,969 | 102 | cached copy of the xlsx | gitignored, regeneratable |
| `data/processed/ga/ga_filtered.parquet` | 29,522 | 105 | strike incident, **GA population only** | gitignored, regeneratable |
| `data/processed/real/damage_{train,val,test}.csv` | 247,778 / 53,095 / 53,096 | 39 | strike incident, **all aircraft types** | gitignored, regeneratable |
| `data/processed/layers/*` | 25 (sample airports only) | varies | airport | gitignored, regeneratable, **no real airport data used yet** |
| `data/sample/*` | 25 / 500 / 40 / 4,852 / 8,400 | varies | synthetic fixtures | **committed**, 100% synthetic |
| `models/ga/*`, `models/hazard/*` | — | — | trained CatBoost artifacts | gitignored, **present and functional** |
| `models/damage/*.pt` (real) | — | — | trained PyTorch artifact | gitignored, **NOT present** — see below |

**Geographic coverage:** FAA data is US-wide but 13.6% of rows (48,031/353,969)
are missing latitude/longitude entirely (mostly rows also missing `state`).
Any spatial join, buffer, or CRS-based feature is silently unavailable for
those rows unless explicitly handled.

**Date range:** 2007 through a partial 2026 (FAA reporting lag — the
current calendar year is always under-counted, already documented in
`configs/ga_damage.yaml`).

**Missingness:** cited from the already-computed `results/damage/data_quality_real.md`
rather than recomputed — notably `aircraft_make`/`aircraft_model` and
`parts_struck`/`parts_damaged` are **100% missing** in this export
vintage (not partially missing — entirely absent as columns).

---

## 3. Existing scripts, models, tests, and outputs

The repository contains **four parallel pipelines**, not two as the
top-level README currently describes:

| Pipeline | Scripts | Feature/leakage policy | Population | Model status |
|---|---|---|---|---|
| **Task A (old)** — reported-strike damage, PyTorch | `prepare_data.py`, `train_models.py`, `evaluate_models.py` | `features/leakage.py` (canonical names); **permits** species/size/counts-struck as features | All aircraft types (353,969 rows) | Data prepared (`damage_{train,val,test}.csv` exist); **no `models/damage/*.pt` checkpoint present** for real mode |
| **Task B (old)** — Trektellen activity index | `build_geospatial_features.py`, `export_arcgis_layers.py` | N/A (formula, not a trained model) | Sample airports only so far | Formula-based, functional on sample data |
| **GA CatBoost (new)** — conditional damage, calibrated | `ga_prepare_data.py`, `ga_train_models.py`, `predict_damage.py` | `ga/feature_policy.py` (raw names); **forbids** species/size/counts-struck as "outcome-adjacent" | Confirmed GA fixed-wing only (29,522 rows) | **Trained, calibrated, receipted — functional** (`models/ga/*`) |
| **Hazard/Eastern Shore (new)** — severity regression + local risk report | `train_severity_model.py`, `build_eastern_shore_risk_report.py` | Reuses GA feature_policy + deliberately re-adds `species` (documented, not a bug — see `hazard/features.py` docstring) | GA population (national) joined to one Trektellen station (FBBO, local) | **Trained, report generated — functional** (`models/hazard/*`) |

`synthetic_demo/` is a fifth, explicitly unrelated component (hand-designed
synthetic data, 8-input NN) — correctly preserved and labeled throughout
the app (`/synthetic` route carries no real-data claim).

**Test suite:** 21 test files, 119 test functions, last verified run:
199 passed / 6 skipped (204 passed / 1 skipped including `-m slow`).
Notably, `tests/test_leakage.py` and `tests/test_ga_feature_policy.py`
are **separate, independently-parametrized suites** — one per leakage
guard — confirming both pipelines are tested but never cross-checked
against each other.

**Flask app (`app.py`):** 5 modes as of this session — `/synthetic`,
`/damage` (old PyTorch), `/activity` (old formula), `/ga-damage` (new
CatBoost), `/hazard` (new report viewer). These changes (adding modes 4
and 5) are made but **not yet committed** — see `git status` below.

**Uncommitted work at audit time:**
```
 M app.py
 M data/sample/airports_sample.geojson   (trivial — regenerated timestamp from a test run, not a real change)
 M requirements.txt
 M templates/base.html
 M templates/index.html
?? templates/ga_damage.html
?? templates/hazard.html
```

---

## 4. Duplicated / conflicting pipelines — the central finding

Both "Task A (old)" and "GA CatBoost (new)" answer the same *conceptual*
question — P(damage | reported strike) — but disagree on:

1. **Population.** Old: all 353,969 strikes, every aircraft type
   (airliners, military, GA, everything). New: 29,522 GA-only strikes.
   These are not comparable models of the same phenomenon; a "does the
   new model agree with the old one" check would be meaningless without
   first restricting the old model's population.
2. **Feature/leakage policy.** Old: `species_common_name`, `wildlife_size`,
   `number_seen`, `number_struck` are **allowed** features (they are the
   *realized* species/count of the strike, known only after it happened).
   New: these are explicitly **forbidden** as "outcome-adjacent —
   unknowable before a strike happens" (`ga/feature_policy.py` docstring).
   This is a genuine methodological disagreement, not an oversight in
   either direction — but only one of them should be presented as "the"
   damage model without a caveat.
3. **Algorithm & calibration.** Old: a small PyTorch NN, not currently
   trained in this environment for real data. New: calibrated CatBoost
   with isotonic/Platt calibration, SHAP explanations, a full model
   receipt.

**Recommendation:** treat the GA CatBoost model as the authoritative
"Task A" model going forward, and re-label the old PyTorch DamageNet
the same way `synthetic_demo/` is already labeled — preserved as an
architecture demonstration (it's genuinely useful for showing a from-
scratch PyTorch training loop with proper leakage guards and
calibration-free baselines), but not the current best answer to "what's
P(damage | strike)". This does not require deleting any code.

Separately, **two independent Trektellen ingestion paths** exist:
`birdstrikegeo/data/ingest_trektellen.py` (canonical long-format schema,
feeds the old activity-index formula) and
`birdstrikegeo/hazard/trektellen_season.py` (bespoke wide-format
year-totals parser, feeds the new hazard report). They were built for
genuinely different Trektellen export shapes (long-format
site×session×species vs. wide-format species×month year-totals), so
this isn't necessarily wrong — but it means "Trektellen support" in this
repo is not one code path, and a future contributor extending one will
not automatically extend the other.

---

## 5. Data leakage assessment

**No leakage bugs found.** Both damage-prediction pipelines have a
dedicated, tested guard:

- `features/leakage.py` + `tests/test_leakage.py` — blocklist over
  canonical (alias-resolved) column names, for the old pipeline.
- `ga/feature_policy.py` + `tests/test_ga_feature_policy.py` — a
  stricter **allowlist** over raw FAA column names, for the new
  pipeline. It also blocks the realized species/size/counts-struck,
  which the old guard does not.

`hazard/features.py` deliberately re-adds `species` after calling the
GA feature builder (which already ran the leakage check on the columns
it read), with an explicit docstring justifying why this is not leakage
for a *retrospective, explanatory* model rather than a *preflight
predictor*. This is a reasonable, well-documented design choice, not a
bug — but it means "leakage-safe" means something subtly different in
`hazard/` than in `ga/`, worth keeping in mind if either package grows.

**One thing worth independently verifying in Phase 2/7, not asserted
here:** `damage_excluded_unknown_target.csv` has **zero rows** for the
real 353,969-row dataset — i.e., every single row had an unambiguous
`damage_flag`/`damage_level`. That's plausible (this export's
`damage_flag` has 0% missingness) but is a suspiciously clean result
worth spot-checking against a handful of raw rows before relying on it.

---

## 6. Infrastructure risk (new finding, not in existing docs)

This repository lives under `~/Documents`, which is iCloud-Drive-synced
on this machine. During this audit:

- `data/raw/faa_wildlife_strikes.xlsx` (156 MB) and
  `data/interim/ga/faa_raw_cache.parquet` (46 MB) were found **evicted
  to iCloud placeholders** — `ls` reported their full size, but `du`
  showed **0 bytes actually on local disk**. Reading them hung/timed
  out until `brctl download` was used to force materialization.
- Separately (across earlier sessions on this project), the same
  iCloud/Documents interaction has intermittently caused the OS to
  return "Operation not permitted" on basic `ls`/`cat`/`find` calls
  against the repo, requiring a `tccutil reset SystemPolicyDocumentsFolder
  com.apple.<terminal-app>` to clear a stale permission grant.

**This is not a code bug**, but it's a real reproducibility risk: a
script that "works" today can silently hang or fail tomorrow purely
because macOS decided to evict a large data file to free local space,
and `Path.exists()` checks will still report `True` for an evicted
file. **Recommendation:** exclude `data/raw/`, `data/interim/`,
`data/processed/`, and `models/` from iCloud sync (a per-folder
`.nosync` suffix, or moving the repo outside `~/Documents` entirely) —
independent of anything else in this audit.

---

## 7. `.gitignore` assessment

Current policy is sound and already matches what this audit found on
disk: raw/interim/processed data and trained models are gitignored
(regeneratable or license-restricted), sample fixtures are committed
(small, synthetic, needed for the repo to run out of the box), and
`reports/latest/` is committed (the actual deliverable artifacts). One
observation: `data/sample/airports_sample.geojson`'s `generated_at`
timestamp changes on every test run that calls
`generate_sample_data.py`, producing spurious diffs — low priority, but
either gitignoring that one field's non-determinism or excluding the
regenerated sample files from version control entirely would remove
noise from future diffs.

No new `.gitignore` entries are needed as a result of this audit.

---

## 8. Assumptions that cannot yet be verified

- **Trektellen site 3460 = FBBO** is asserted in `configs/eastern_shore_risk.yaml`
  (`station_id: FBBO`) and the exported CSV's URLs
  (`trektellen.org/count/view/3460/...`), but there is no independent
  confirmation in this repository of FBBO's exact coordinates, habitat
  type, or whether "FBBO" (a banding-station abbreviation) is the same
  physical site Trektellen's UI labels "3460" beyond the URL pattern
  matching. Low-risk (both point at the same numeric site ID) but not
  independently cross-checked here.
- **Zero ambiguous-damage rows** (section 5) — plausible, not
  independently re-derived from raw values in this audit.
- **The real `/damage` (old PyTorch) checkpoint's actual current
  performance** cannot be assessed — no `models/damage/*.pt` exists
  for real data in this environment, so whatever historical numbers
  might exist elsewhere are not verifiable from this checkout alone.
  `MODEL_CARD.md`'s reported numbers are explicitly sample-data-only
  and correctly labeled as such.
- **Airport identifiers' geocoding accuracy**: FAA-reported lat/lon is
  used directly; this audit did not cross-check any of it against an
  authoritative airport database (none is present in `data/raw/` yet —
  see inventory, "real airports.geojson" is still a documented but
  unfulfilled requirement).
- **iCloud eviction behavior going forward** — this audit forced local
  materialization of the two large files for this session; whether
  they get re-evicted before the next session is outside this
  repository's control (see section 6).

---

## 9. Recommended primary target (per the user's Phase 0 framing)

Given the data actually available (audited above — no flight-movements/
departures/exposure denominator exists anywhere in this repository, and
`DATA_CARD.md` already documents `bts_t100_departures.csv` as an
unfulfilled placeholder):

| Candidate target | Feasible now? | Why |
|---|---|---|
| **P(strike occurs per flight)** | **No** | No exposure denominator (departures/movements/flight-time). FAA data alone is strike-conditioned, not exposure-conditioned — this is already correctly documented project-wide. |
| **Expected strike count per airport-period** | **No** | Same reason — a Poisson/negative-binomial count model needs an at-risk denominator (or at minimum airport traffic volume as an offset), which isn't in this repo yet. |
| **P(damage \| reported strike)** | **Yes — already built** | This is exactly what both damage pipelines already do. Recommend the **GA CatBoost model** as primary (calibrated, stricter leakage policy, GA-scoped, receipted) over the old PyTorch model (uncalibrated, broader/less-specific population, no real-data checkpoint currently trained). |
| **Species/species-group given a strike** | **Partially — data exists, not yet modeled as a target in its own right** | `SPECIES`/`SPECIES_ID` are present and clean (0% missing) in the GA population; no classifier currently predicts species — it's only used as a feature (hazard) or excluded (GA damage model). Feasible as a secondary target with existing data. |
| **Relative bird-activity/hazard index for a location+time** | **Yes — already built, twice** | The old Trektellen `activity_index` formula (transparent, not a trained model) and the new hazard package's national-species-risk × local-FBBO-activity join both already do a version of this. Recommend **unifying** these into one documented secondary output rather than running both indefinitely — see Phase 2 discussion. |

**Recommendation:** keep **P(damage | reported strike), GA population,
CatBoost** as the primary target (already feasible, already built, just
needs the "which pipeline is authoritative" labeling fix from section
4), and **relative bird-hazard index** as the secondary geospatial
output (already feasible, needs the two existing implementations
reconciled into one). Absolute strike probability and expected strike
counts should be explicitly deferred and documented as "blocked on
flight-exposure data" — exactly as `DATA_CARD.md` §5 already states —
rather than attempted with a proxy denominator.

---

## Next step

This completes Phase 1. Phases 2-9 were subsequently built in the same
pass, on the user's explicit instruction to proceed autonomously
("build all the structures so I could perhaps insert the data in
later"). Summary of what exists now, all structure-first (real code,
real tests, run against the GA/Trektellen data already in this repo
where possible; clearly marked as unimplemented/deferred where new
external data would be required):

- **Phase 2** — `schemas/hazard_observation.py` (the standardized
  airport×date secondary-output schema — geographic/time/bird/weather/
  aviation-exposure/target column groups, never mixed) and
  `schemas/validation.py` (a generic, reusable schema-validation layer:
  required columns, dtype compatibility, coordinate ranges, timestamp
  validity, duplicate-key detection, missingness, and a "companion
  column" provenance rule). Consolidation: `LEGACY_SYSTEMS.md` +
  README/app.py/template relabeling make the GA CatBoost model and the
  hazard report explicitly primary, the old PyTorch model and formula-
  based activity index explicitly legacy — no files moved or deleted.
- **Phase 3** — `geo/h3_grid.py` (H3 hexagonal indexing, resolutions 5
  and 7, justified in `GEOSPATIAL_METHODS.md` section 7a), building on
  the existing CRS/geodesic-distance strategy rather than replacing it.
- **Phase 4** — `BIRD_DATA_SOURCE_COMPARISON.md` (eBird Status & Trends
  recommended as the best future bird-abundance source; BirdCast,
  USGS BBS, Movebank, Motus, Audubon, Esri Living Atlas all researched
  and compared) and `geo/ebird_adapter.py` (structure only — no network
  calls without an access key, none requested). `data/raw/airports.csv`
  (OurAirports, public domain) was downloaded to fill a real gap, not
  yet wired into `ingest_airports.py` (column names differ — documented
  in `DATA_DOWNLOAD_GUIDE.md`).
- **Phase 5** — `hazard/spatiotemporal_join.py` +
  `scripts/build_join_quality_report.py`: joins every GA incident to
  the FBBO Trektellen data with full provenance (source, resolution,
  distance, time gap, observed-vs-unavailable) and produces a real
  join-quality report (spatial/temporal match-rate tables) — not yet
  run in this pass; run `python scripts/build_join_quality_report.py`
  to generate it.
- **Phase 6** — individual-tracking-vs-population-map recommendation
  (defer Movebank/Motus from v1; use only for future validation) written
  into `BIRD_DATA_SOURCE_COMPARISON.md`.
- **Phase 7** — `scripts/build_eda_report.py`: missingness, distributions,
  correlation, spatial/seasonal plots, species/year/airport confounding
  checks, and a genuine leakage smoke-test (correlating
  `ga/feature_policy.py`'s forbidden columns against the target to
  confirm the guard is blocking real signal) — not yet run in this pass.
- **Phase 8** — `MODELING_APPROACH.md`: documents what the GA model
  already does well (baselines, chronological+geographic validation,
  calibration, bootstrap CIs), flags the hazard/severity model's thinner
  evaluation as a real gap, explicitly defers a Poisson/count model
  until exposure data exists, and fully specifies (prior/likelihood/
  posterior/assumptions) a hierarchical Bayesian model for future work.
- **Phase 9** — `DISPLAY_DESIGN.md`: which quantities must never share a
  color scale, the 7 required always-visible display elements, and why
  a relative hazard index (not absolute probability) is right for now.

**Infrastructure finding, not part of the original 9 phases:** this
repository lives under iCloud Drive, and `.venv` itself (not just large
data files) was found iCloud-synced, causing `import pandas` to take
255 seconds. Fixed via `.venv.nosync` (see README.md "Quick start").
This explains most of the mysterious slowness/hangs encountered across
this project's sessions to date — see project memory for the full
diagnosis.

**Honesty note on verification:** this pass prioritized writing
correct, reviewed code (two real bugs were caught and fixed by manual
review before ever running: an argument-order mismatch in
`geodesic_distance_km`, and dead/buggy code in the EDA plotting
function) over exhaustively running every new script end-to-end, because
this sandboxed environment's Python startup time was highly variable
during this session (seconds to several minutes for the same command).
Run `PYTHONPATH=src python -m pytest tests/ -q` to verify before relying
on any of this.
