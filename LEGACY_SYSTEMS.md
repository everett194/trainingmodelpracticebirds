# Legacy Systems

This project went through a Phase 1 repository audit
(`DATA_AUDIT_REPORT.md`) that found two "generations" of both the
damage model and the bird-activity output. Neither older system is
wrong or broken — both are preserved, working, and tested — but this
document makes explicit which one is now authoritative, so a reader
never has to guess which number to trust.

This is the same treatment `synthetic_demo/` has always gotten: kept,
clearly labeled, never presented as evidence for the current system.

---

## Damage model

| | **Legacy** (`/damage`) | **Primary** (`/ga-damage`) |
|---|---|---|
| Population | All FAA-reported strikes, every aircraft type (353,969 rows) | Confirmed business/private fixed-wing GA only (29,522 rows) |
| Algorithm | PyTorch NN (`DamageNet`), uncalibrated | CatBoost, Platt/isotonic-calibrated |
| Feature policy | `features/leakage.py` — permits realized `species`, `wildlife_size`, `number_seen`, `number_struck` as features | `ga/feature_policy.py` — forbids those as "outcome-adjacent, unknowable before a strike happens" |
| Current status | Real-mode data prepared, but **no trained checkpoint exists** for real data in this environment | Trained, calibrated, receipted (`reports/latest/model_receipt.md`) |

**Why keep it:** it's a complete, tested, from-scratch PyTorch training
loop (leakage guard, chronological split, class-weighted loss,
early stopping, calibration-free baselines) — genuinely useful as an
architecture reference, and it answers a *different* question (P(damage
| strike) across ALL aviation, not just GA) that the primary model
deliberately does not attempt.

**Why it's not primary:** broader/less-specific population for a
project whose current milestone is GA-specific; a more permissive
feature policy that a stricter later analysis (`ga/feature_policy.py`)
judged too lenient; and, in this environment, no trained real-data
checkpoint to actually serve.

## Bird-activity / hazard index

| | **Legacy** (`/activity`) | **Primary** (`/hazard`) |
|---|---|---|
| Method | Transparent formula: `tanh(log1p(rate)/4) * (0.5 + 0.5*coverage)` | CatBoost severity regression → national per-species risk scores → joined against local Trektellen activity |
| Grounding | Hand-picked compression constant, not fit to any outcome data | Tied to actual FAA-reported damage severity by species |
| Trektellen ingestion | `data/ingest_trektellen.py`, long-format site×session×species | `hazard/trektellen_season.py`, wide-format species×month year-totals |
| Scope | Any airport within reach of a Trektellen site (formula runs regardless) | Currently exactly one station (FBBO, Eastern Shore MD, 2025 season) |

**Why keep it:** simpler, more general — it will produce *some* answer
for any airport near any Trektellen site, whereas the primary hazard
report today is scoped to one station's one season. It's also a good
example of the "don't fit an interpolation surface you can't justify"
philosophy documented in `GEOSPATIAL_METHODS.md`.

**Why it's not primary:** the formula's constants aren't tied to any
real outcome data — it's a reasonable, documented guess, not a fitted
relationship. The hazard report at least ties bird presence to an
actual FAA-derived severity signal, even though it's presently
single-station.

**Next step recommended, not yet done:** extend the hazard report's
approach to more stations as real Trektellen exports become available,
and consider whether `eBird Status & Trends` (see
`BIRD_DATA_SOURCE_COMPARISON.md`) should eventually replace
single-station Trektellen data as the primary bird-activity input,
once its access terms are confirmed.

---

## What was NOT done as part of this consolidation

No files were moved or deleted. No old tests were removed — the legacy
systems' full test coverage (`tests/test_leakage.py`,
`tests/test_target.py`, `tests/test_splits.py`, `tests/test_inference.py`,
`tests/test_preprocessing.py`, `tests/test_temporal_join.py`,
`tests/test_spatial_join.py`) still runs and still needs to pass. This
was a labeling/documentation change (README, MODEL_CARD, Flask nav and
page headings) plus this explanatory document — a deliberately low-risk
way to resolve the Phase 1 audit's "which pipeline is authoritative"
finding without touching working code.
