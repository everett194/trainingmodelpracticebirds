"""
birdstrikegeo.ga.receipt
----------------------------
Writes the small, Git-trackable "digital receipt" under reports/latest/:
a human-readable model_receipt.md plus the JSON/CSV/PNG files it points
to. Deliberately excludes raw records, PII, and serialized model
binaries - see project requirements Section 13.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm: dict, output_path: str | Path, model_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    matrix = [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i][j]:,}", ha="center", va="center")
    ax.set_xticks([0, 1], labels=["Predicted: no damage", "Predicted: damage"])
    ax.set_yticks([0, 1], labels=["Actual: no damage", "Actual: damage"])
    ax.set_title(f"Confusion Matrix — {model_name} (test)")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return Path(output_path)


def plot_feature_importance(importance_rows: list[dict], output_path: str | Path, top_n: int = 15) -> Path:
    top = importance_rows[:top_n][::-1]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.35 * len(top))))
    ax.barh([r["feature"] for r in top], [r["importance"] for r in top])
    ax.set_xlabel("CatBoost feature importance (PredictionValuesChange)")
    ax.set_title("Global Feature Importance — GA Conditional-Damage GBT")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return Path(output_path)


def write_json(obj, path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, default=str))


def render_markdown_receipt(ctx: dict) -> str:
    """ctx: see scripts/ga_train_models.py for the exact keys populated."""
    lines = []
    a = lines.append

    a("# GA Conditional-Damage Model — Receipt\n")
    a(
        "**Predicts:** given that a wildlife strike occurs, under the supplied aircraft "
        "and encounter conditions, the probability it causes reported aircraft damage.\n"
    )
    a(
        "**Does NOT predict** whether a strike will occur. The FAA workbook has no "
        "non-strike flight/airport-day denominator - see `birdstrikegeo.ga.future_schemas` "
        "for the planned separate occurrence-risk model.\n"
    )

    a("## Dataset")
    a(f"- Source: FAA Wildlife Strike Database export (`{ctx['source_filename']}`)")
    a(f"- SHA-256 fingerprint: `{ctx['source_sha256']}`")
    a(f"- Total records in source workbook: {ctx['total_source_records']:,}")
    a(f"- GA filter: `{ctx['ga_filter_description']}`")
    a(f"- GA population (principal): {ctx['ga_population_count']:,} records")
    a("")
    a("| Population | Count | Definition |")
    a("|---|---|---|")
    for row in ctx["population_counts_table"]:
        a(f"| {row['name']} | {row['count']:,} | {row['definition']} |")

    a("\n## Target")
    a("`damage_binary` = 1 only when `INDICATED_DAMAGE` affirmatively indicates damage; "
      "= 0 only when it affirmatively indicates no damage. Unknown/ambiguous rows are excluded, "
      "never coerced to 0. `DAMAGE_LEVEL` is cross-checked for internal consistency but does not "
      "override `INDICATED_DAMAGE`.\n")
    t = ctx["target_report"]
    a(f"- Positive (damage): {t['n_positive']:,}")
    a(f"- Negative (no damage): {t['n_negative']:,}")
    a(f"- Excluded (unknown/ambiguous): {t['n_excluded_unknown']:,}")
    a(f"- Inconsistent with `DAMAGE_LEVEL`: {t['n_inconsistent_indicated_vs_level']:,}")
    a(f"- Positive class rate: {t['positive_class_rate']:.1%}")

    a("\n## Split")
    a(f"- Train: through {ctx['split']['train_end_year']} ({ctx['split']['n_train']:,} rows)")
    a(f"- Validation: {ctx['split']['train_end_year']+1}–{ctx['split']['validation_end_year']} "
      f"({ctx['split']['n_validation']:,} rows)")
    a(f"- Test: {ctx['split']['validation_end_year']+1}+ ({ctx['split']['n_test']:,} rows, untouched)")
    a(f"- Geographic robustness holdout regions: {ctx['geo_holdout']['regions']} "
      f"({ctx['geo_holdout']['n_holdout']:,} rows)")

    a("\n## Features")
    a(f"- Feature mode evaluated as principal: **{ctx['principal_feature_mode']}**")
    a(f"- Operational-core feature count: {len(ctx['feature_lists']['operational_core'])}")
    a(f"- Reduced-preflight feature count: {len(ctx['feature_lists']['reduced_preflight'])}")
    a(f"- Excluded for leakage/administrative reasons: {len(ctx['forbidden_columns'])} raw columns "
      f"(full list in `feature_list.json`)")

    a("\n## Model")
    a(f"- Principal model: CatBoostClassifier (gradient-boosted trees, native categorical + missing-value handling)")
    a(f"- Selected hyperparameters: {ctx['catboost_best_params']}")
    a(f"- Random seed: {ctx['random_seed']}")
    a(f"- Training timestamp: {ctx['training_timestamp']}")
    a(f"- Package versions: {ctx['package_versions']}")
    a(f"- Git commit: {ctx.get('git_commit', 'unavailable')}")

    a("\n## Model comparison (test split, principal feature mode)")
    a("| Model | ROC-AUC | PR-AUC | Log loss | Brier | F1 | Balanced acc. |")
    a("|---|---|---|---|---|---|---|")
    for row in ctx["comparison_table"]:
        a(f"| {row['model']} | {row['roc_auc']} | {row['average_precision']} | {row.get('log_loss','n/a')} "
          f"| {row['brier_score']} | {row['f1']} | {row['balanced_accuracy']} |")

    a("\n## Principal model (CatBoost) — test metrics")
    m = ctx["catboost_test_metrics"]
    a(f"- N = {m['sample_count']:,}, positive rate = {m['positive_class_prevalence']:.1%}")
    a(f"- Threshold used: {m['threshold_used']:.3f} (0.5 metrics also in `metrics.json`)")
    a(f"- ROC-AUC: {m['roc_auc']:.4f}  (95% CI {ctx['bootstrap_ci']['roc_auc']['ci_low']:.4f}"
      f"–{ctx['bootstrap_ci']['roc_auc']['ci_high']:.4f})")
    a(f"- PR-AUC: {m['average_precision']:.4f}  (95% CI {ctx['bootstrap_ci']['pr_auc']['ci_low']:.4f}"
      f"–{ctx['bootstrap_ci']['pr_auc']['ci_high']:.4f})")
    a(f"- Brier score: {m['brier_score']:.4f}  (95% CI {ctx['bootstrap_ci']['brier']['ci_low']:.4f}"
      f"–{ctx['bootstrap_ci']['brier']['ci_high']:.4f})")
    a(f"- Precision: {m['precision']:.3f}, Recall (sensitivity): {m['recall']:.3f}, "
      f"Specificity: {m['specificity']:.3f}, F1: {m['f1']:.3f}, Balanced accuracy: {m['balanced_accuracy']:.3f}")
    a(f"- Confusion matrix: {m['confusion_matrix']}")
    a(f"- **Efron pseudo-R² for probability predictions: {ctx['efron_pseudo_r2']:.4f}** "
      f"(NOT ordinary OLS R² - an educational heuristic only; do not use as the primary quality measure)")

    a("\n## Calibration")
    a(f"- Selected method: **{ctx['calibration']['selected_method']}** "
      f"(chosen by validation Brier score, never test)")
    a(f"- Validation Brier by method: {ctx['calibration']['validation_brier_by_method']}")

    a("\n## Presets")
    for name, result in ctx["preset_predictions"].items():
        a(f"\n**`{name}`** — {result['description']}")
        a(f"- Prediction: {result['probability']:.1%} probability of reported damage, conditional on a strike")
        a(f"- Risk band: {result['risk_band']}")
        a(f"- Top contributing inputs: {result['top_features']}")
    a(
        "\n> Every preset prediction is conditional damage probability assuming a strike "
        "occurs. It is NOT the probability that a strike will occur."
    )

    a("\n## Known limitations")
    for item in ctx["limitations"]:
        a(f"- {item}")

    return "\n".join(lines) + "\n"
