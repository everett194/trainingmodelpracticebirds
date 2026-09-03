#!/usr/bin/env python3
"""
scripts/ga_train_models.py
------------------------------
GA conditional-damage milestone, step 2 of 2 (see scripts/ga_prepare_data.py
first). Trains dummy/logistic-regression/neural-net baselines plus the
principal CatBoost gradient-boosted-tree model, in both feature modes
(operational_core / reduced_preflight), on a chronological GA split; picks
a validation-selected probability calibration for the principal model;
evaluates everything on the untouched chronological test split plus a
geographic robustness subset; runs every preset; and writes the
reports/latest/ receipt.

Usage:
    python scripts/ga_train_models.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import catboost  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402

from birdstrikegeo.evaluation.calibration import compute_calibration_curve  # noqa: E402
from birdstrikegeo.evaluation.metrics import compute_metrics, model_comparison_table  # noqa: E402
from birdstrikegeo.evaluation.plots import plot_calibration_curve, plot_precision_recall_curve, plot_roc_curve  # noqa: E402
from birdstrikegeo.ga.evaluation import (  # noqa: E402
    apply_calibration,
    bootstrap_metric_ci,
    efron_pseudo_r2,
    risk_decile_table,
    select_calibration,
)
from birdstrikegeo.ga.explain import (  # noqa: E402
    WARNING_NOT_CAUSAL,
    global_feature_importance,
    permutation_importance,
)
from birdstrikegeo.ga.feature_policy import FORBIDDEN_COLUMNS  # noqa: E402
from birdstrikegeo.ga.features import FEATURE_BUILDERS, chronological_split_by_year  # noqa: E402
from birdstrikegeo.ga.inference import GaModelBundle, predict_scenario, save_bundle  # noqa: E402
from birdstrikegeo.ga.modeling import (  # noqa: E402
    build_dummy_and_logistic,
    prepare_dense,
    prepare_tree,
    predict_proba_catboost,
    predict_proba_sklearn,
    train_catboost_with_search,
)
from birdstrikegeo.ga.presets import PRESETS  # noqa: E402
from birdstrikegeo.ga.receipt import plot_confusion_matrix, plot_feature_importance, render_markdown_receipt, write_json  # noqa: E402
from birdstrikegeo.training.splits import holdout_by_region  # noqa: E402
from birdstrikegeo.training.train_damage import (  # noqa: E402
    PreparedData,
    predict_proba_neural_net,
    set_seed,
    train_neural_net,
)
from birdstrikegeo.training.tune_threshold import tune_threshold  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "ga_damage.yaml"
PRINCIPAL_MODE = "operational_core"


def git_commit_hash() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unavailable"


def metrics_with_log_loss(y_true, proba, threshold) -> dict:
    m = compute_metrics(y_true, proba, threshold)
    m["log_loss"] = float(log_loss(y_true, proba, labels=[0, 1]))
    return m


def train_and_evaluate_mode(mode: str, splits: dict, config: dict, random_seed: int) -> dict:
    """splits: {'train': raw_df, 'validation': raw_df, 'test': raw_df} (already GA-filtered, labeled)."""
    build_features = FEATURE_BUILDERS[mode]
    feature_sets = {name: build_features(df) for name, df in splits.items()}
    numeric_cols = feature_sets["train"].numeric_columns
    categorical_cols = feature_sets["train"].categorical_columns
    y = {name: splits[name]["damage_binary"].astype(int).to_numpy() for name in splits}

    print(f"[ga_train_models:{mode}] train={len(y['train'])} val={len(y['validation'])} test={len(y['test'])} "
          f"features={len(numeric_cols) + len(categorical_cols)}")

    # --- dense track: dummy, logistic regression, neural net ---
    dense = prepare_dense(feature_sets["train"].features, feature_sets["validation"].features,
                           feature_sets["test"].features, numeric_cols, categorical_cols,
                           config["rare_category_threshold"])

    baselines = build_dummy_and_logistic(random_seed)
    fitted_baselines = {}
    for name, model in baselines.items():
        model.fit(dense.X_train, y["train"])
        fitted_baselines[name] = model

    prepared = PreparedData(
        X_train=dense.X_train, X_val=dense.X_val, X_test=dense.X_test,
        y_train=y["train"], y_val=y["validation"], y_test=y["test"],
        preprocessor=dense.preprocessor, feature_names_out=dense.feature_names_out,
        frequent_categories=dense.frequent_categories,
    )
    set_seed(random_seed)
    nn_cfg = config["neural_net"]
    nn_result = train_neural_net(
        prepared, hidden_sizes=tuple(nn_cfg["hidden_layer_sizes"]), dropout=tuple(nn_cfg["dropout"]),
        learning_rate=nn_cfg["learning_rate"], batch_size=nn_cfg["batch_size"],
        maximum_epochs=nn_cfg["maximum_epochs"], early_stopping_patience=nn_cfg["early_stopping_patience"],
        random_seed=random_seed,
    )
    print(f"[ga_train_models:{mode}] DamageNet: {len(nn_result.train_losses)} epochs, best={nn_result.best_epoch}, "
          f"train_loss={nn_result.train_losses[-1]:.4f} val_loss={nn_result.val_losses[-1]:.4f}")

    val_proba = {
        "dummy": predict_proba_sklearn(fitted_baselines["dummy"], dense.X_val),
        "logistic_regression": predict_proba_sklearn(fitted_baselines["logistic_regression"], dense.X_val),
        "neural_net": predict_proba_neural_net(nn_result.model, dense.X_val),
    }
    test_proba = {
        "dummy": predict_proba_sklearn(fitted_baselines["dummy"], dense.X_test),
        "logistic_regression": predict_proba_sklearn(fitted_baselines["logistic_regression"], dense.X_test),
        "neural_net": predict_proba_neural_net(nn_result.model, dense.X_test),
    }

    # --- tree track: CatBoost ---
    tree_train, tree_val, tree_test, cat_features = prepare_tree(
        feature_sets["train"].features, feature_sets["validation"].features, feature_sets["test"].features,
        numeric_cols, categorical_cols,
    )
    search = train_catboost_with_search(
        tree_train, y["train"], tree_val, y["validation"], cat_features,
        base_config=config["catboost"], search_grid=config["catboost"]["search_grid"], random_seed=random_seed,
    )
    print(f"[ga_train_models:{mode}] CatBoost search: {len(search.search_table)} combos, "
          f"best={search.best_params}, train/val = {search.train_val_metrics}")

    val_proba["catboost"] = predict_proba_catboost(search.best_model, tree_val, cat_features)
    test_proba["catboost"] = predict_proba_catboost(search.best_model, tree_test, cat_features)

    thresholds = {name: tune_threshold(y["validation"], proba, objective=config["threshold_objective"])
                  for name, proba in val_proba.items()}
    test_metrics = {name: metrics_with_log_loss(y["test"], proba, thresholds[name]["threshold"])
                     for name, proba in test_proba.items()}

    return {
        "mode": mode, "feature_sets": feature_sets, "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols, "y": y,
        "dense": dense, "fitted_baselines": fitted_baselines, "nn_result": nn_result,
        "tree_frames": (tree_train, tree_val, tree_test), "cat_features": cat_features,
        "catboost_search": search, "val_proba": val_proba, "test_proba": test_proba,
        "thresholds": thresholds, "test_metrics": test_metrics,
    }


def run() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    random_seed = config["random_seed"]
    paths = config["paths"]
    set_seed(random_seed)

    processed_path = REPO_ROOT / paths["processed_dir"] / "ga_filtered.parquet"
    if not processed_path.exists():
        print(f"[ga_train_models] Missing {processed_path}. Run scripts/ga_prepare_data.py first.")
        sys.exit(1)
    ga = pd.read_parquet(processed_path)

    split_cfg = config["split"]
    split = chronological_split_by_year(ga, "INCIDENT_YEAR", split_cfg["train_end_year"], split_cfg["validation_end_year"])
    splits = {"train": split.train, "validation": split.validation, "test": split.test}
    print(f"[ga_train_models] Chronological split: train(<= {split.train_end_year})={len(split.train)} "
          f"validation({split.train_end_year+1}-{split.validation_end_year})={len(split.validation)} "
          f"test(> {split.validation_end_year})={len(split.test)}")

    geo_cfg = config["geographic_holdout"]
    non_holdout_test, geo_holdout_test = holdout_by_region(split.test, "FAAREGION", geo_cfg["held_out_regions"])
    print(f"[ga_train_models] Geographic holdout (test split, regions={geo_cfg['held_out_regions']}): "
          f"{len(geo_holdout_test)} rows")

    results_by_mode = {}
    for mode in config["feature_modes"]:
        results_by_mode[mode] = train_and_evaluate_mode(mode, splits, config, random_seed)

    principal = results_by_mode[PRINCIPAL_MODE]

    # --- calibration (principal CatBoost model only, validation-selected) ---
    calibration = select_calibration(principal["y"]["validation"], principal["val_proba"]["catboost"],
                                      config["calibration"]["isotonic_min_validation_rows"])
    print(f"[ga_train_models] Calibration selected: {calibration['selected_method']} "
          f"(validation Brier by method: {calibration['validation_brier_by_method']})")

    calibrated_val_proba = apply_calibration(calibration["selected_method"], calibration["calibrator"],
                                              principal["val_proba"]["catboost"])
    calibrated_test_proba = apply_calibration(calibration["selected_method"], calibration["calibrator"],
                                               principal["test_proba"]["catboost"])
    calibrated_threshold = tune_threshold(principal["y"]["validation"], calibrated_val_proba,
                                           objective=config["threshold_objective"])
    catboost_calibrated_test_metrics = metrics_with_log_loss(
        principal["y"]["test"], calibrated_test_proba, calibrated_threshold["threshold"]
    )
    catboost_calibrated_test_metrics_at_05 = metrics_with_log_loss(principal["y"]["test"], calibrated_test_proba, 0.5)

    # --- Efron pseudo-R^2 + bootstrap CIs (calibrated principal model, test split) ---
    y_train_mean = float(principal["y"]["train"].mean())
    efron_r2 = efron_pseudo_r2(principal["y"]["test"], calibrated_test_proba, y_train_mean)
    n_boot = config["bootstrap"]["n_resamples"]
    conf = config["bootstrap"]["confidence_level"]
    bootstrap_ci = {
        "roc_auc": bootstrap_metric_ci(principal["y"]["test"], calibrated_test_proba, "roc_auc", n_boot, conf, random_seed),
        "pr_auc": bootstrap_metric_ci(principal["y"]["test"], calibrated_test_proba, "pr_auc", n_boot, conf, random_seed),
        "brier": bootstrap_metric_ci(principal["y"]["test"], calibrated_test_proba, "brier", n_boot, conf, random_seed),
    }
    deciles = risk_decile_table(principal["y"]["test"], calibrated_test_proba)

    # --- geographic robustness subset (calibrated principal model) ---
    geo_features = FEATURE_BUILDERS[PRINCIPAL_MODE](geo_holdout_test)
    geo_tree = geo_features.features[principal["numeric_columns"] + principal["categorical_columns"]].copy()
    for col in principal["categorical_columns"]:
        geo_tree[col] = geo_tree[col].astype("string").fillna("missing_value")
    geo_raw_proba = predict_proba_catboost(principal["catboost_search"].best_model, geo_tree, principal["cat_features"])
    geo_calibrated_proba = apply_calibration(calibration["selected_method"], calibration["calibrator"], geo_raw_proba)
    geo_y = geo_holdout_test["damage_binary"].astype(int).to_numpy()
    geo_metrics = (
        metrics_with_log_loss(geo_y, geo_calibrated_proba, calibrated_threshold["threshold"])
        if len(set(geo_y.tolist())) >= 2 else {"_warning": "insufficient class diversity in geographic holdout"}
    )

    # --- explainability ---
    tree_train, tree_val, tree_test = principal["tree_frames"]
    feature_names = principal["numeric_columns"] + principal["categorical_columns"]
    importance = global_feature_importance(principal["catboost_search"].best_model, feature_names)
    perm_importance = permutation_importance(
        principal["catboost_search"].best_model, tree_val, principal["y"]["validation"], principal["cat_features"],
    )

    # --- presets ---
    bundle = GaModelBundle(
        catboost_model=principal["catboost_search"].best_model,
        calibration_method=calibration["selected_method"], calibrator=calibration["calibrator"],
        numeric_columns=principal["numeric_columns"], categorical_columns=principal["categorical_columns"],
        seen_categories={col: set(tree_train[col].dropna().unique()) | set(tree_val[col].dropna().unique())
                          for col in principal["categorical_columns"]},
        numeric_ranges={col: (float(tree_train[col].min(skipna=True)), float(tree_train[col].max(skipna=True)))
                        for col in principal["numeric_columns"]},
        threshold=calibrated_threshold["threshold"], model_version="ga-damage-v0.1.0-dev", feature_mode=PRINCIPAL_MODE,
    )
    save_bundle(bundle, REPO_ROOT / paths["models_dir"])

    preset_predictions = {}
    for name, scenario in PRESETS.items():
        result = predict_scenario(scenario, bundle, FEATURE_BUILDERS[PRINCIPAL_MODE])
        preset_predictions[name] = {"description": scenario["description"], **result}
        print(f"[ga_train_models] preset {name}: {result['probability']:.1%} ({result['risk_band']})")

    # --- write results/ga/ (full detail, not the lean receipt) ---
    results_dir = REPO_ROOT / paths["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    full_results = {
        mode: {
            "catboost_search_table": r["catboost_search"].search_table,
            "catboost_best_params": r["catboost_search"].best_params,
            "catboost_train_val_metrics": r["catboost_search"].train_val_metrics,
            "test_metrics": r["test_metrics"],
            "thresholds": r["thresholds"],
        }
        for mode, r in results_by_mode.items()
    }
    write_json(full_results, results_dir / "all_modes_results.json")

    # --- receipt ---
    reports_dir = REPO_ROOT / paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    population_summary = json.loads((reports_dir / "ga_population_summary.json").read_text())

    comparison_rows = model_comparison_table(principal["test_metrics"])
    for row in comparison_rows:
        row["log_loss"] = round(principal["test_metrics"][row["model"]]["log_loss"], 4)
    pd.DataFrame(comparison_rows).to_csv(reports_dir / "model_comparison.csv", index=False)

    calibration_curve = compute_calibration_curve(principal["y"]["test"], calibrated_test_proba)
    plot_calibration_curve(calibration_curve, reports_dir / "calibration_curve.png", False, "catboost_calibrated")
    plot_confusion_matrix(catboost_calibrated_test_metrics["confusion_matrix"], reports_dir / "confusion_matrix.png",
                           "catboost_calibrated")
    plot_feature_importance(importance, reports_dir / "feature_importance.png")
    plot_roc_curve(principal["y"]["test"], calibrated_test_proba, results_dir / "roc_catboost_calibrated.png", False, "catboost_calibrated")
    plot_precision_recall_curve(principal["y"]["test"], calibrated_test_proba, results_dir / "pr_catboost_calibrated.png", False, "catboost_calibrated")

    feature_list = {
        "operational_core": principal["numeric_columns"] + principal["categorical_columns"],
        "reduced_preflight": results_by_mode["reduced_preflight"]["numeric_columns"]
                              + results_by_mode["reduced_preflight"]["categorical_columns"]
                              if "reduced_preflight" in results_by_mode else [],
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
    }
    write_json(feature_list, reports_dir / "feature_list.json")

    metrics_json = {
        "principal_feature_mode": PRINCIPAL_MODE,
        "catboost_best_params": principal["catboost_search"].best_params,
        "catboost_calibrated_test_metrics": catboost_calibrated_test_metrics,
        "catboost_calibrated_test_metrics_at_0.5": catboost_calibrated_test_metrics_at_05,
        "calibration": {k: v for k, v in calibration.items() if k != "calibrator"},
        "efron_pseudo_r2": efron_r2,
        "bootstrap_confidence_intervals": bootstrap_ci,
        "risk_deciles": deciles,
        "geographic_holdout": {"regions": geo_cfg["held_out_regions"], "n": len(geo_holdout_test), "metrics": geo_metrics},
        "global_feature_importance": importance,
        "permutation_importance_validation": perm_importance,
        "not_causal_warning": WARNING_NOT_CAUSAL,
        "reduced_preflight_test_metrics": results_by_mode.get("reduced_preflight", {}).get("test_metrics", {}),
        "all_baseline_test_metrics": principal["test_metrics"],
    }
    write_json(metrics_json, reports_dir / "metrics.json")

    receipt_ctx = {
        "source_filename": population_summary["source_filename"],
        "source_sha256": population_summary["source_sha256"],
        "total_source_records": population_summary["total_source_records"],
        "ga_filter_description": population_summary["ga_filter_description"],
        "ga_population_count": population_summary["principal_population_count"],
        "population_counts_table": [{"name": n, **d} for n, d in population_summary["population_counts"].items()],
        "target_report": population_summary["target_report"],
        "split": {"train_end_year": split.train_end_year, "validation_end_year": split.validation_end_year,
                   "n_train": len(split.train), "n_validation": len(split.validation), "n_test": len(split.test)},
        "geo_holdout": {"regions": geo_cfg["held_out_regions"], "n_holdout": len(geo_holdout_test)},
        "principal_feature_mode": PRINCIPAL_MODE,
        "feature_lists": {"operational_core": feature_list["operational_core"],
                           "reduced_preflight": feature_list["reduced_preflight"]},
        "forbidden_columns": feature_list["forbidden_columns"],
        "catboost_best_params": principal["catboost_search"].best_params,
        "random_seed": random_seed,
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "package_versions": {
            "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
            "scikit-learn": sklearn.__version__, "torch": torch.__version__, "catboost": catboost.__version__,
        },
        "git_commit": git_commit_hash(),
        "comparison_table": comparison_rows,
        "catboost_test_metrics": catboost_calibrated_test_metrics,
        "bootstrap_ci": bootstrap_ci,
        "efron_pseudo_r2": efron_r2,
        "calibration": {k: v for k, v in calibration.items() if k != "calibrator"},
        "preset_predictions": preset_predictions,
        "limitations": [
            "FAA wildlife-strike reporting is voluntary and not a complete census.",
            "This model predicts damage conditional on a reported strike - NOT the probability a strike occurs.",
            "The GA population filter depends on OPERATOR/AC_CLASS text fields that have varied across FAA export vintages - verify against a fresh export before reuse.",
            "WARNED (bird-warning) may reflect reporting thoroughness as much as true warning-system performance.",
            "Season is derived from calendar month only (Northern-hemisphere convention); FGN (foreign)/AAL (Alaska) region rows may not fit that convention.",
            f"The current calendar year is under-counted due to normal FAA reporting lag ({population_summary['total_source_records']:,} total source rows span 1990-2026).",
            "Efron pseudo-R^2 is an educational heuristic, not a substitute for ROC-AUC/PR-AUC/Brier/calibration.",
            "Geographic robustness check evaluates the SAME trained model on held-out regions (not retrained without them) - a lighter check than full region-holdout retraining.",
            WARNING_NOT_CAUSAL,
        ],
    }
    (reports_dir / "model_receipt.md").write_text(render_markdown_receipt(receipt_ctx))
    print(f"[ga_train_models] Wrote receipt to {reports_dir}")

    print("\n=== Model comparison (test, principal feature mode, uncalibrated) ===")
    for row in comparison_rows:
        print(f"  {row['model']:<22} roc_auc={row['roc_auc']} pr_auc={row['average_precision']} "
              f"log_loss={row['log_loss']} brier={row['brier_score']} f1={row['f1']}")
    print(f"\nCatBoost CALIBRATED ({calibration['selected_method']}) test ROC-AUC="
          f"{catboost_calibrated_test_metrics['roc_auc']:.4f}, Efron pseudo-R^2={efron_r2:.4f}")


if __name__ == "__main__":
    run()
