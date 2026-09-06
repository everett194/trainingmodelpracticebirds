# Modeling Approach (Phase 8)

What's already built, what's deliberately deferred, and — per project
requirement — a fully specified (not just labeled) Bayesian approach for
future work, so "Bayesian" never becomes a decorative word on a display.

## What the primary model (GA CatBoost) already does

Verified against `reports/latest/model_receipt.md` — this is not
aspirational, it's already implemented and receipted:

- **Baselines compared**: dummy, logistic regression, a PyTorch neural
  net, and the principal CatBoost model, all on identical leakage-safe
  features (`model_comparison.csv`-equivalent for GA: `model_receipt.md`'s
  comparison table).
- **Validation strategy**: chronological split (train ≤2019, validation
  2020-2022, test 2023+) — NOT a random split, so the test set can never
  contain information from the model's future. PLUS a geographic
  robustness holdout (FAA regions AAL/Alaska and FGN/Foreign, 37 rows,
  never used in training) — the same trained model re-evaluated on
  regions it never trained on, honestly labeled as "a lighter check than
  full region-holdout retraining," not oversold as full spatial
  cross-validation.
- **Metrics**: ROC-AUC, PR-AUC, Brier score, precision/recall/F1,
  balanced accuracy, confusion matrix — appropriate for a rare-event
  (16.8% positive rate in test) binary classification target, with
  bootstrap 95% confidence intervals on the headline metrics.
- **Calibration**: isotonic regression, selected by validation (never
  test) Brier score — this is discrimination AND calibration, not just
  ROC-AUC.
- **Class imbalance**: handled via the calibration/threshold choice
  (0.213, tuned) rather than resampling, and reported alongside raw
  prevalence rather than hidden.

**Gap found in this audit**: the secondary severity/hazard model
(`models/hazard/severity_catboost.cbm`) has none of this rigor yet —
`severity_model_metrics.json` has only `val_rmse`, `val_r2`,
`best_iteration`, `test_rmse`. No calibration check (meaningful for a
regression target via e.g. a reliability diagram of predicted vs.
observed severity by bucket), no bootstrap CI, no baseline comparison
(e.g. "predict the species' mean severity" as a trivial baseline). This
is real, deferred work, not a design decision — the receipt-writing
infrastructure already exists (`ga/receipt.py`, `evaluation/bootstrap.py`,
`evaluation/calibration.py`) and could be pointed at the severity model
directly.

## Deliberately deferred: Poisson / negative-binomial count model

**Not attempted, and should not be, until exposure data exists.** An
expected-strike-count model needs an at-risk denominator (movements,
departures, or flight-hours) as either the outcome's implicit
denominator or an offset term — this project has documented, since
`DATA_CARD.md` §5, that it does not have one (`bts_t100_departures.csv`
is a placeholder). Fitting a count model on strike counts alone, without
an exposure offset, would silently model "which airports report the
most strikes" (confounded by airport size/traffic volume) rather than
"which airports have elevated bird-strike risk relative to their
traffic" — exactly the false-precision trap this project's whole
philosophy is built to avoid. **Do not build this until
`bts_t100_departures.csv` (or an equivalent) is integrated.**

## The Bayesian approach, fully specified (not yet implemented)

Per project requirement: this is a specification for a **future**
model, written out completely so it can be built later without
re-deriving the design, and so "Bayesian" is never just a display label.

- **Unit of observation**: airport × date (the secondary hazard
  schema — `schemas.hazard_observation`), NOT per-strike.
- **Prior**: each airport's historical strike-per-monitored-period rate
  (or, for an airport with little/no history, its FAA region's pooled
  rate — a hierarchical/partial-pooling prior, so a low-traffic airport
  with 2 years of data doesn't get an overconfident airport-specific
  estimate). Concretely: a Gamma prior on each airport's latent strike
  rate λ_airport, hierarchically centered on a region-level rate drawn
  from its own hyperprior — a standard Bayesian hierarchical
  (partial-pooling) structure, not a flat/uninformative prior per
  airport.
- **Likelihood**: given λ_airport and a period's covariates (season,
  local bird-activity estimate, weather severity flag, and — once
  available — traffic volume as an offset), a Poisson (or, if
  overdispersion is present once exposure data exists,
  negative-binomial) likelihood for the observed strike count in that
  airport-period.
- **Posterior quantity**: the posterior distribution over λ_airport
  given that period's observed covariates — i.e., a posterior-predictive
  distribution over strike counts for a NEW period at that airport,
  from which a relative-hazard estimate AND its credible interval both
  fall out naturally (unlike CatBoost's point estimate, which needs a
  separate bootstrap to get any uncertainty band).
- **What this buys over the current CatBoost approach**: honest
  uncertainty for airports with little history (the credible interval
  widens automatically, rather than the model confidently extrapolating
  from a handful of rows) — directly useful for Phase 9's requirement to
  show "uncertainty or insufficient-data regions" rather than implying
  false precision everywhere.
- **What it does NOT solve**: the same fundamental limitation as
  everything else in this project — without an exposure offset, λ still
  conflates "risky" with "busy." The Bayesian structure improves
  uncertainty quantification and partial pooling across sparse airports;
  it does not manufacture an exposure denominator this project doesn't have.

**Recommended library, when this is built**: `PyMC` or `numpyro` for the
actual sampling (NUTS) — not implemented here; this section is the
design a future implementer should start from, per the "just build the
structures" instruction for this pass.

## A calibrated classification model, restated

The GA CatBoost model already IS this (isotonic-calibrated, per above).
No separate "calibrated classification model" comparison entry is
needed beyond what's already in `model_receipt.md`.
