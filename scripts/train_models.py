#!/usr/bin/env python3
"""
scripts/train_models.py
---------------------------
Trains all Task A baselines plus DamageNet on the chronologically split
data produced by scripts/prepare_data.py, tunes the decision threshold
on validation data, evaluates everything on the held-out test split, and
saves the neural net checkpoint + a model-comparison table.

Usage:
    python scripts/train_models.py --task damage --mode sample
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from birdstrikegeo.evaluation.calibration import compute_calibration_curve  # noqa: E402
from birdstrikegeo.evaluation.metrics import compute_metrics, model_comparison_table  # noqa: E402
from birdstrikegeo.evaluation.plots import (  # noqa: E402
    plot_calibration_curve,
    plot_precision_recall_curve,
    plot_roc_curve,
    plot_training_loss,
)
from birdstrikegeo.models.checkpoints import save_checkpoint  # noqa: E402
from birdstrikegeo.training.train_damage import (  # noqa: E402
    predict_proba_neural_net,
    predict_proba_sklearn,
    prepare_datasets,
    set_seed,
    train_baselines,
    train_neural_net,
    tune_all_thresholds,
)

CONFIG_PATH = REPO_ROOT / "configs" / "damage_model.yaml"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models" / "damage"
RESULTS_DIR = REPO_ROOT / "results" / "damage"
MODEL_VERSION = "damage-v0.1.0-dev"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def run(task: str, mode: str) -> None:
    if task != "damage":
        raise NotImplementedError("Only --task damage is implemented; Task B has no trained classifier by design.")

    is_sample = mode == "sample"
    config = load_config()
    set_seed(config["random_seed"])

    split_dir = PROCESSED_DIR / mode
    for name in ("damage_train.csv", "damage_validation.csv", "damage_test.csv"):
        if not (split_dir / name).exists():
            print(f"[train_models] Missing {split_dir / name}. Run scripts/prepare_data.py --mode {mode} first.")
            sys.exit(1)

    train_df = pd.read_csv(split_dir / "damage_train.csv")
    val_df = pd.read_csv(split_dir / "damage_validation.csv")
    test_df = pd.read_csv(split_dir / "damage_test.csv")
    print(f"[train_models] mode={mode} train={len(train_df)} validation={len(val_df)} test={len(test_df)}")

    prepared = prepare_datasets(train_df, val_df, test_df, config["rare_category_threshold"])
    print(f"[train_models] Encoded feature matrix width: {prepared.X_train.shape[1]}")

    # --- Baselines ---
    baseline_models = train_baselines(prepared, config["random_seed"])
    baseline_val_proba = {name: predict_proba_sklearn(m, prepared.X_val) for name, m in baseline_models.items()}
    baseline_test_proba = {name: predict_proba_sklearn(m, prepared.X_test) for name, m in baseline_models.items()}

    # --- Neural net ---
    model_cfg = config["model"]
    nn_result = train_neural_net(
        prepared,
        hidden_sizes=tuple(model_cfg["hidden_layer_sizes"]),
        dropout=tuple(model_cfg["dropout"]),
        learning_rate=model_cfg["learning_rate"],
        batch_size=model_cfg["batch_size"],
        maximum_epochs=model_cfg["maximum_epochs"],
        early_stopping_patience=model_cfg["early_stopping_patience"],
        random_seed=config["random_seed"],
    )
    print(f"[train_models] DamageNet trained for {len(nn_result.train_losses)} epochs "
          f"(best epoch {nn_result.best_epoch}), {nn_result.trainable_parameters} trainable parameters.")

    nn_val_proba = predict_proba_neural_net(nn_result.model, prepared.X_val)
    nn_test_proba = predict_proba_neural_net(nn_result.model, prepared.X_test)

    all_val_proba = {**baseline_val_proba, "neural_net": nn_val_proba}
    all_test_proba = {**baseline_test_proba, "neural_net": nn_test_proba}

    # --- Threshold tuning (validation only) ---
    thresholds = tune_all_thresholds(all_val_proba, prepared.y_val, config["threshold_objective"])
    print(f"[train_models] Tuned thresholds ({config['threshold_objective']}): "
          f"{ {k: round(v['threshold'], 3) for k, v in thresholds.items()} }")

    # --- Final evaluation (test split) ---
    test_metrics = {
        name: compute_metrics(prepared.y_test, proba, thresholds[name]["threshold"])
        for name, proba in all_test_proba.items()
    }
    comparison_table = model_comparison_table(test_metrics)

    best_model_name = max(test_metrics, key=lambda n: (test_metrics[n]["roc_auc"] or -1))
    baseline_names = [n for n in test_metrics if n != "neural_net"]
    best_baseline_name = max(baseline_names, key=lambda n: (test_metrics[n]["roc_auc"] or -1))
    honest_comparison = (
        f"Best model on test ROC-AUC: '{best_model_name}'. "
        + (
            "The neural network was NOT the best-performing model on this split."
            if best_model_name != "neural_net"
            else f"The neural network outperformed the best baseline ('{best_baseline_name}')."
        )
    )
    print(f"[train_models] {honest_comparison}")

    # --- Plots + calibration (neural net) ---
    plots_dir = RESULTS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_training_loss(nn_result.train_losses, nn_result.val_losses, plots_dir / f"loss_curve_{mode}.png", is_sample)

    calibration = None
    if len(set(prepared.y_test.tolist())) >= 2:
        plot_roc_curve(prepared.y_test, nn_test_proba, plots_dir / f"roc_neural_net_{mode}.png", is_sample, "neural_net")
        plot_precision_recall_curve(prepared.y_test, nn_test_proba, plots_dir / f"pr_neural_net_{mode}.png", is_sample, "neural_net")
        calibration = compute_calibration_curve(prepared.y_test, nn_test_proba)
        plot_calibration_curve(calibration, plots_dir / f"calibration_neural_net_{mode}.png", is_sample, "neural_net")

    # --- Save checkpoint ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"damage_net_{mode}.pt"
    save_checkpoint(
        model=nn_result.model, preprocessor=prepared.preprocessor,
        feature_columns=list(train_df.columns.drop(["damage", "incident_date"], errors="ignore")),
        threshold=thresholds["neural_net"]["threshold"], model_version=MODEL_VERSION,
        hidden_sizes=tuple(model_cfg["hidden_layer_sizes"]), dropout=tuple(model_cfg["dropout"]),
        path=checkpoint_path, frequent_categories=prepared.frequent_categories,
    )
    print(f"[train_models] Saved DamageNet checkpoint to {checkpoint_path}")

    # --- Save results ---
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "sample_data_flag": is_sample,
        "disclaimer": "SAMPLE DATA — NOT A REAL RESULT" if is_sample else None,
        "model_version": MODEL_VERSION,
        "trainable_parameters_neural_net": nn_result.trainable_parameters,
        "best_epoch": nn_result.best_epoch,
        "epochs_trained": len(nn_result.train_losses),
        "thresholds": thresholds,
        "model_comparison_table": comparison_table,
        "honest_comparison_note": honest_comparison,
        "test_metrics_full": test_metrics,
        "calibration_neural_net": calibration,
        "prediction_semantics": (
            "Estimated probability of aircraft damage, conditional on a reported "
            "wildlife strike. This is NOT the probability that a strike will occur."
        ),
    }
    (RESULTS_DIR / f"training_report_{mode}.json").write_text(json.dumps(report, indent=2, default=str))
    pd.DataFrame(comparison_table).to_csv(RESULTS_DIR / f"model_comparison_{mode}.csv", index=False)
    print(f"[train_models] Wrote training report and model comparison table to {RESULTS_DIR}")

    print("\nTraining complete.")
    for row in comparison_table:
        print(f"  {row['model']:<28} accuracy={row['accuracy']:.3f}  roc_auc={row['roc_auc']}  f1={row['f1']:.3f}")
    if is_sample:
        print("\nSAMPLE DATA — NOT A REAL RESULT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["damage"], default="damage")
    parser.add_argument("--mode", choices=["sample", "real"], default="sample")
    args = parser.parse_args()
    run(args.task, args.mode)


if __name__ == "__main__":
    main()
