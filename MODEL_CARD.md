# Model Card — BirdStrikeGeo

Two models, two different guarantees. Read the "what this predicts"
section for each carefully before using either one for anything.

---

## Task A: Reported-strike damage model (`DamageNet`)

### What this predicts

> **Estimated probability of aircraft damage, conditional on a reported
> wildlife strike.**

This is **not** the probability that a wildlife strike will occur. That
is a fundamentally different question this project does not attempt to
answer yet — see "Scientific limitations" below and
`configs/data_sources.yaml`'s `bts_t100_departures` placeholder for what
would be needed.

### Architecture

```
encoded_input
  → Linear(64) → ReLU → Dropout(0.20)
  → Linear(32) → ReLU → Dropout(0.10)
  → Linear(1)
```

No sigmoid inside the model — trained with `BCEWithLogitsLoss`, sigmoid
applied only at inference (`birdstrikegeo/inference/predict_damage.py`),
same pattern as `synthetic_demo/model.py`. Optimizer: `AdamW`. Early
stopping on validation loss (patience configurable in
`configs/damage_model.yaml`). Class weighting (`pos_weight` in the loss)
is derived from the training split's own class balance, not hand-tuned.

For the **sample-data run** referenced below, the encoded input is 84
features wide (after one-hot encoding + rare-category bucketing), giving
**7,553 trainable parameters** exactly (84×64+64 + 64×32+32 + 32×1+1).

### Baselines

Trained on the identical leakage-safe, train-only-fit preprocessing:
dummy (stratified), logistic regression (`class_weight="balanced"`),
histogram gradient boosting (`class_weight="balanced"`).

### Preventing target leakage

`damage_flag`, `damage_level`, `effect_on_flight`, `parts_damaged`,
`repair_cost`, `other_cost`, `injuries`, `fatalities`, and `narrative`
all describe the strike's *outcome* and are hard-blocked from the
feature matrix by `birdstrikegeo/features/leakage.py`, enforced by
`tests/test_leakage.py` (parametrized over every forbidden column).

### Sample-data results

> **SAMPLE DATA — NOT A REAL RESULT.** These numbers come from
> `data/sample/faa_strikes_sample.csv` (synthetic, hand-designed
> relationships) — see `data/sample/README.md`. They exist to prove the
> pipeline works end to end, not to claim any real-world performance.
> Regenerate with `python scripts/train_models.py --task damage --mode sample`.

| Model | Accuracy | Balanced Acc. | Precision | Recall | F1 | ROC-AUC | Brier |
|---|---|---|---|---|---|---|---|
| dummy | 0.700 | 0.496 | 0.125 | 0.222 | 0.160 | 0.496 | 0.300 |
| logistic_regression | 0.400 | 0.514 | 0.133 | 0.667 | 0.222 | 0.550 | 0.175 |
| histogram_gradient_boosting | 0.300 | 0.551 | 0.143 | 0.889 | 0.246 | **0.690** | 0.133 |
| neural_net | 0.457 | 0.594 | 0.163 | 0.778 | 0.269 | 0.554 | 0.228 |

(test split: n=70, 12.9% positive prevalence — a bootstrap confidence
interval for ROC-AUC was **not computed**: the split has only 9 positive
cases, below this project's minimum of 10 for a meaningful interval, per
`birdstrikegeo/evaluation/bootstrap.py`. Reporting "unavailable" here
instead of a misleadingly precise interval is deliberate.)

**Honest comparison note:** on this sample split, **histogram gradient
boosting had the best ROC-AUC — the neural network was not the
best-performing model.** This project reports that outcome as-is; a
9-trainable-parameter improvement or architecture change wasn't made
just to "win" the comparison table. With more data (especially real FAA
data, orders of magnitude larger than this 500-row fixture), the
relative ranking of these models may well change.

### Explainability

`birdstrikegeo/evaluation/explainability.py` provides logistic-regression
coefficients, permutation importance, and grouped feature importance
(one-hot dummy columns summed back to their conceptual feature). SHAP is
attempted only if the `shap` package is installed and runs without
error — otherwise it's skipped, never faked.

> **Feature importance shows predictive association, not causation.**

**Integration limitation:** the project spec calls for including
Trektellen coverage variables in this analysis so the model's behavior
under sparse monitoring is visible. Task A is documented (and built) to
be trainable from FAA data **alone** — Trektellen features are not part
of its default feature set. `grouped_feature_importance()` will group
and report any Trektellen coverage columns that get added to the feature
matrix, but no such enrichment ships by default in this version.

---

## Task B: Airport-period wildlife activity index

### What this predicts

> A relative **`activity_index`** (0–1) for an airport during a
> specified time window, based on nearby Trektellen monitoring
> observations. **Not** a calibrated probability of anything, and
> **not** a strike-probability forecast.

### Why this is a formula, not a trained classifier

Per project requirements, this project does **not** train a
strike-vs-no-strike classifier without a defensible non-strike exposure
denominator (see `DATA_CARD.md` §5, BTS T-100). Until that exists,
`compute_activity_index()` (`birdstrikegeo/models/activity_index.py`) is
a transparent, documented formula:

```
raw_index = tanh(log1p(birds_per_observation_hour_previous_7d) / 4)
activity_index = raw_index * (0.5 + 0.5 * data_coverage_score)
```

- `tanh(log1p(rate) / 4)` log-compresses the effort-normalized 7-day
  bird rate into `[0, 1)` so a handful of unusually large counts don't
  dominate the index. The constant `4` is a hand-picked compression
  factor (documented, not fitted).
- Multiplying by `(0.5 + 0.5 * data_coverage_score)` down-weights the
  index when monitoring is sparse, old, or distant — coverage can damp
  the index by at most 50%, never zero it out entirely.
- If no Trektellen observation session occurred in the relevant window,
  `activity_index` is `None` (not `0.0`) — absence of data is never
  presented as absence of activity.

### Coverage transparency

Every activity-index result carries: `trektellen_available`,
`distance_to_nearest_trektellen_site_km`, `nearest_site_observation_age_hours`,
`number_of_sites_within_{50,100,250}_km`, `observation_effort_available`,
`distance_warning`, `low_coverage_warning`, and `data_coverage_score` —
see the "Coverage details" table on the `/activity` page.

---

## Scientific limitations (both tasks)

- FAA wildlife-strike reporting is not a complete census of all strikes.
- Trektellen sites are not uniformly distributed; most U.S. airports
  have none nearby.
- Observer effort varies session to session.
- Visible migration counts, captures, and nocturnal flight calls
  measure different processes and are never silently combined.
- A distant monitoring site may not represent activity at a given
  airport — `distance_warning`/`low_coverage_warning` exist precisely
  because of this.
- Missing observations do not mean zero birds.
- Reported damage may have missingness or reporting bias.
- **Correlation does not demonstrate causation** — feature importance is
  association, not a causal claim.
- An activity index is not a certified operational risk forecast.
- Results from Europe (where Trektellen has denser coverage) may not
  generalize to North America.
- A true strike-*probability* model (as opposed to Task A's
  damage-*given-a-reported-strike* model) requires defensible non-strike
  or flight-exposure observations this project does not yet have.
