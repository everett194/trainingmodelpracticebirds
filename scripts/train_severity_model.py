#!/usr/bin/env python3
"""
scripts/train_severity_model.py
-----------------------------------
Exploratory bird-strike risk analysis, step 1 of 2 (see
scripts/build_eastern_shore_risk_report.py). Trains a CatBoost regressor
on the GA-filtered FAA data to predict damage_severity_score (0-3), then
aggregates per-species national risk scores from its predictions.

NOT the calibrated GA conditional-damage milestone
(scripts/ga_train_models.py) - a lighter, explanatory risk-ranking
analysis over the same population. See configs/eastern_shore_risk.yaml.

Usage:
    python scripts/train_severity_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from birdstrikegeo.ga.data import build_damage_binary_target  # noqa: E402
from birdstrikegeo.ga.features import chronological_split_by_year  # noqa: E402
from birdstrikegeo.ga.modeling import prepare_tree  # noqa: E402
from birdstrikegeo.hazard.features import build_severity_features  # noqa: E402
from birdstrikegeo.hazard.modeling import predict_severity, train_severity_regressor  # noqa: E402
from birdstrikegeo.hazard.severity_target import build_damage_severity_target  # noqa: E402
from birdstrikegeo.hazard.species_risk import aggregate_species_risk  # noqa: E402

CONFIG_PATH = REPO_ROOT / "configs" / "eastern_shore_risk.yaml"


def run() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    paths = config["paths"]

    processed_path = REPO_ROOT / paths["ga_filtered_parquet"]
    if not processed_path.exists():
        print(f"[train_severity_model] Missing {processed_path}. Run scripts/ga_prepare_data.py first.")
        sys.exit(1)

    ga = pd.read_parquet(processed_path)
    # damage_binary/damage_target_status/damage_level_inconsistent aren't
    # persisted redundantly in every cache vintage - rebuild them here so
    # this script doesn't depend on ga_filtered.parquet's exact columns.
    ga = build_damage_binary_target(ga)
    ga = build_damage_severity_target(ga)

    labeled = ga[ga["damage_severity_score"].notna()].copy()
    print(f"[train_severity_model] {len(labeled):,} / {len(ga):,} GA rows have a usable severity target")

    split_cfg = config["split"]
    split = chronological_split_by_year(labeled, "INCIDENT_YEAR", split_cfg["train_end_year"], split_cfg["validation_end_year"])
    splits = {"train": split.train, "validation": split.validation, "test": split.test}
    print(f"[train_severity_model] train={len(split.train)} validation={len(split.validation)} test={len(split.test)}")

    feature_sets = {name: build_severity_features(df) for name, df in splits.items()}
    numeric_cols = feature_sets["train"].numeric_columns
    categorical_cols = feature_sets["train"].categorical_columns
    y = {name: splits[name]["damage_severity_score"].astype(float).to_numpy() for name in splits}

    tree_train, tree_val, tree_test, cat_features = prepare_tree(
        feature_sets["train"].features, feature_sets["validation"].features, feature_sets["test"].features,
        numeric_cols, categorical_cols,
    )

    model, metrics = train_severity_regressor(
        tree_train, y["train"], tree_val, y["validation"], cat_features, config["catboost"], config["random_seed"],
    )
    test_pred = predict_severity(model, tree_test, cat_features)
    metrics["test_rmse"] = float(((test_pred - y["test"]) ** 2).mean() ** 0.5)
    print(f"[train_severity_model] val_rmse={metrics['val_rmse']:.3f} val_r2={metrics['val_r2']:.3f} "
          f"test_rmse={metrics['test_rmse']:.3f} best_iteration={metrics['best_iteration']}")

    # Per-species national risk score, from predictions over every row
    # this model saw (train+val+test) - more stable species-level
    # aggregates than the test split alone, since this is a descriptive
    # risk ranking, not a held-out generalization claim.
    all_features = pd.concat([tree_train, tree_val, tree_test], ignore_index=True)
    all_species = pd.concat([splits["train"]["SPECIES"], splits["validation"]["SPECIES"], splits["test"]["SPECIES"]], ignore_index=True)
    all_predictions = predict_severity(model, all_features, cat_features)
    predictions_df = pd.DataFrame({"SPECIES": all_species.to_numpy(), "predicted_severity": all_predictions})

    species_risk = aggregate_species_risk(predictions_df)
    species_risk = species_risk.sort_values("risk_score", ascending=False).reset_index(drop=True)
    print(f"[train_severity_model] Top 5 national species risk:\n{species_risk.head(5).to_string(index=False)}")

    models_dir = REPO_ROOT / paths["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(models_dir / "severity_catboost.cbm"))

    reports_dir = REPO_ROOT / paths["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    species_risk.to_csv(reports_dir / "national_species_risk.csv", index=False)
    pd.Series(metrics).to_json(reports_dir / "severity_model_metrics.json", indent=2)
    print(f"[train_severity_model] Wrote model to {models_dir}, species risk + metrics to {reports_dir}")


if __name__ == "__main__":
    run()
