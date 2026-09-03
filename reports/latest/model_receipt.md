# GA Conditional-Damage Model — Receipt

**Predicts:** given that a wildlife strike occurs, under the supplied aircraft and encounter conditions, the probability it causes reported aircraft damage.

**Does NOT predict** whether a strike will occur. The FAA workbook has no non-strike flight/airport-day denominator - see `birdstrikegeo.ga.future_schemas` for the planned separate occurrence-risk model.

## Dataset
- Source: FAA Wildlife Strike Database export (`faa_wildlife_strikes.xlsx`)
- SHA-256 fingerprint: `a2b21612b0abde685b9b6772cbd6d659d1fd93299c0d6d28cb96fa485a1ce99f`
- Total records in source workbook: 353,969
- GA filter: `OPERATOR in {BUSINESS, PRIVATELY OWNED} AND AC_CLASS == 'A' (airplane) - PRINCIPAL population`
- GA population (principal): 29,522 records

| Population | Count | Definition |
|---|---|---|
| confirmed_business_or_private | 33,474 | OPERATOR in {BUSINESS, PRIVATELY OWNED} (all aircraft classes) |
| confirmed_business_or_private_fixed_wing | 29,522 | OPERATOR in {BUSINESS, PRIVATELY OWNED} AND AC_CLASS == 'A' (airplane) - PRINCIPAL population |
| light_fixed_wing_ga | 19,516 | principal population AND AC_MASS in {1, 2} (<= 5,700 kg) |
| light_piston_ga | 14,017 | light fixed-wing GA AND TYPE_ENG == 'A' (reciprocating/piston) |

## Target
`damage_binary` = 1 only when `INDICATED_DAMAGE` affirmatively indicates damage; = 0 only when it affirmatively indicates no damage. Unknown/ambiguous rows are excluded, never coerced to 0. `DAMAGE_LEVEL` is cross-checked for internal consistency but does not override `INDICATED_DAMAGE`.

- Positive (damage): 6,883
- Negative (no damage): 22,639
- Excluded (unknown/ambiguous): 0
- Inconsistent with `DAMAGE_LEVEL`: 2
- Positive class rate: 23.3%

## Split
- Train: through 2019 (20,754 rows)
- Validation: 2020–2022 (3,719 rows)
- Test: 2023+ (5,049 rows, untouched)
- Geographic robustness holdout regions: ['AAL', 'FGN'] (37 rows)

## Features
- Feature mode evaluated as principal: **operational_core**
- Operational-core feature count: 19
- Reduced-preflight feature count: 14
- Excluded for leakage/administrative reasons: 75 raw columns (full list in `feature_list.json`)

## Model
- Principal model: CatBoostClassifier (gradient-boosted trees, native categorical + missing-value handling)
- Selected hyperparameters: {'depth': 8, 'learning_rate': 0.1, 'l2_leaf_reg': 6, 'best_iteration': 311}
- Random seed: 42
- Training timestamp: 2026-09-02T04:00:51.541417+00:00
- Package versions: {'python': '3.12.10', 'pandas': '3.0.5', 'numpy': '2.5.2', 'scikit-learn': '1.9.0', 'torch': '2.13.0', 'catboost': '1.2.10'}
- Git commit: aa8d7f93bee0c4c19586207fe77bc7228e310bc3

## Model comparison (test split, principal feature mode)
| Model | ROC-AUC | PR-AUC | Log loss | Brier | F1 | Balanced acc. |
|---|---|---|---|---|---|---|
| dummy | 0.5128 | 0.1722 | 11.9075 | 0.3304 | 0.2198 | 0.5128 |
| logistic_regression | 0.6772 | 0.281 | 0.6273 | 0.2183 | 0.3529 | 0.6153 |
| neural_net | 0.7433 | 0.387 | 0.5237 | 0.1745 | 0.4256 | 0.6686 |
| catboost | 0.7409 | 0.3797 | 0.5349 | 0.1799 | 0.4158 | 0.6542 |

## Principal model (CatBoost) — test metrics
- N = 5,049, positive rate = 16.8%
- Threshold used: 0.213 (0.5 metrics also in `metrics.json`)
- ROC-AUC: 0.7394  (95% CI 0.7228–0.7553)
- PR-AUC: 0.3509  (95% CI 0.3230–0.3816)
- Brier score: 0.1249  (95% CI 0.1181–0.1311)
- Precision: 0.378, Recall (sensitivity): 0.464, Specificity: 0.846, F1: 0.417, Balanced accuracy: 0.655
- Confusion matrix: {'tn': 3552, 'fp': 647, 'fn': 456, 'tp': 394}
- **Efron pseudo-R² for probability predictions: 0.1577** (NOT ordinary OLS R² - an educational heuristic only; do not use as the primary quality measure)

## Calibration
- Selected method: **isotonic** (chosen by validation Brier score, never test)
- Validation Brier by method: {'raw': 0.17471575887959076, 'platt': 0.12394503914249504, 'isotonic': 0.12157638120503186}

## Presets

**`c172_day_clear_takeoff`** — Cessna 172, daytime, clear, takeoff roll
- Prediction: 12.9% probability of reported damage, conditional on a strike
- Risk band: Moderate relative conditional severity
- Top contributing inputs: ['light_condition (-0.220)', 'aircraft_type (-0.179)', 'season (-0.147)']

**`c172_night_clear_approach`** — Cessna 172, nighttime, clear, approach
- Prediction: 15.0% probability of reported damage, conditional on a strike
- Risk band: Moderate relative conditional severity
- Top contributing inputs: ['aircraft_type (-0.225)', 'light_condition (+0.216)', 'month_cos (+0.135)']

**`pa28_day_clear_approach`** — Piper PA-28, daytime, clear, approach
- Prediction: 15.0% probability of reported damage, conditional on a strike
- Risk band: Moderate relative conditional severity
- Top contributing inputs: ['season (-0.176)', 'faa_region (-0.158)', 'height_agl_ft (+0.148)']

**`cirrus_day_clear_cruise`** — Cirrus SR20/22, daytime, clear, en route (cruise)
- Prediction: 32.4% probability of reported damage, conditional on a strike
- Risk band: Elevated relative conditional severity
- Top contributing inputs: ['speed_ias_knots (+0.367)', 'aircraft_type (-0.346)', 'height_agl_ft (+0.333)']

**`king_air_day_clear_approach`** — Beechcraft King Air 200, daytime, clear, approach
- Prediction: 32.4% probability of reported damage, conditional on a strike
- Risk band: Elevated relative conditional severity
- Top contributing inputs: ['height_agl_ft (+0.407)', 'speed_ias_knots (+0.312)', 'light_condition (+0.173)']

**`business_jet_day_clear_approach`** — Bombardier Challenger 300 (business jet), daytime, clear, approach
- Prediction: 10.3% probability of reported damage, conditional on a strike
- Risk band: Moderate relative conditional severity
- Top contributing inputs: ['aircraft_type (-0.441)', 'speed_ias_knots (+0.369)', 'height_agl_ft (+0.299)']

> Every preset prediction is conditional damage probability assuming a strike occurs. It is NOT the probability that a strike will occur.

## Known limitations
- FAA wildlife-strike reporting is voluntary and not a complete census.
- This model predicts damage conditional on a reported strike - NOT the probability a strike occurs.
- The GA population filter depends on OPERATOR/AC_CLASS text fields that have varied across FAA export vintages - verify against a fresh export before reuse.
- WARNED (bird-warning) may reflect reporting thoroughness as much as true warning-system performance.
- Season is derived from calendar month only (Northern-hemisphere convention); FGN (foreign)/AAL (Alaska) region rows may not fit that convention.
- The current calendar year is under-counted due to normal FAA reporting lag (353,969 total source rows span 1990-2026).
- Efron pseudo-R^2 is an educational heuristic, not a substitute for ROC-AUC/PR-AUC/Brier/calibration.
- Geographic robustness check evaluates the SAME trained model on held-out regions (not retrained without them) - a lighter check than full region-holdout retraining.
- Feature importances and SHAP contributions describe ASSOCIATIONS the model found in reported FAA strike data, not causal effects. Reporting bias, confounding (e.g. airport, aircraft type, and season all correlate with each other), and voluntary-reporting patterns can all produce a strong association without a causal relationship.
