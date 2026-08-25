#!/usr/bin/env python3
"""
scripts/evaluate_models.py
------------------------------
Re-evaluates an already-trained DamageNet checkpoint against the test
split, without retraining. Useful for re-checking a saved model (e.g.
after downloading real data and running train_models.py separately) and
for exercising the "evaluate" CLI command distinctly from "train".

Usage:
    python scripts/evaluate_models.py --task damage --mode sample
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

from birdstrikegeo.evaluation.bootstrap import bootstrap_confidence_interval  # noqa: E402
from birdstrikegeo.evaluation.calibration import compute_calibration_curve  # noqa: E402
from birdstrikegeo.evaluation.metrics import compute_metrics  # noqa: E402
from birdstrikegeo.features.build_damage_features import apply_rare_categories  # noqa: E402
from birdstrikegeo.models.checkpoints import load_checkpoint  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
import torch  # noqa: E402

PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models" / "damage"
RESULTS_DIR = REPO_ROOT / "results" / "damage"


def run(task: str, mode: str) -> None:
    if task != "damage":
        raise NotImplementedError("Only --task damage is implemented.")

    is_sample = mode == "sample"
    checkpoint_path = MODELS_DIR / f"damage_net_{mode}.pt"
    test_path = PROCESSED_DIR / mode / "damage_test.csv"

    if not checkpoint_path.exists():
        print(f"[evaluate_models] No trained checkpoint at {checkpoint_path}. Run scripts/train_models.py first.")
        sys.exit(1)
    if not test_path.exists():
        print(f"[evaluate_models] No test split at {test_path}. Run scripts/prepare_data.py first.")
        sys.exit(1)

    checkpoint = load_checkpoint(checkpoint_path)
    test_df = pd.read_csv(test_path)
    print(f"[evaluate_models] Loaded checkpoint {checkpoint['model_version']} and {len(test_df)} test rows.")

    ready = apply_rare_categories(test_df, checkpoint["frequent_categories"])
    for col in checkpoint["frequent_categories"]:
        if col in ready.columns:
            ready[col] = ready[col].astype("string").fillna("missing")

    X_test = checkpoint["preprocessor"].transform(ready[checkpoint["feature_columns"]])
    y_test = test_df["damage"].astype(int).to_numpy()

    with torch.no_grad():
        logits = checkpoint["model"](torch.tensor(X_test, dtype=torch.float32))
        proba = torch.sigmoid(logits).squeeze(1).numpy()

    metrics = compute_metrics(y_test, proba, checkpoint["threshold"])
    print(f"[evaluate_models] Test metrics: accuracy={metrics['accuracy']:.3f} "
          f"roc_auc={metrics['roc_auc']} f1={metrics['f1']:.3f}")

    ci = bootstrap_confidence_interval(y_test, proba, roc_auc_score)
    if ci["available"]:
        print(f"[evaluate_models] ROC-AUC 95% bootstrap CI: [{ci['lower']:.3f}, {ci['upper']:.3f}]")
    else:
        print(f"[evaluate_models] Bootstrap CI unavailable: {ci['reason']}")

    calibration = compute_calibration_curve(y_test, proba) if len(set(y_test.tolist())) >= 2 else None

    report = {
        "sample_data_flag": is_sample,
        "disclaimer": "SAMPLE DATA — NOT A REAL RESULT" if is_sample else None,
        "model_version": checkpoint["model_version"],
        "metrics": metrics,
        "roc_auc_bootstrap_ci": ci,
        "calibration": calibration,
        "prediction_semantics": (
            "Estimated probability of aircraft damage, conditional on a reported "
            "wildlife strike. This is NOT the probability that a strike will occur."
        ),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"evaluation_{mode}.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[evaluate_models] Wrote {out_path}")
    if is_sample:
        print("[evaluate_models] SAMPLE DATA — NOT A REAL RESULT")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["damage"], default="damage")
    parser.add_argument("--mode", choices=["sample", "real"], default="sample")
    args = parser.parse_args()
    run(args.task, args.mode)


if __name__ == "__main__":
    main()
