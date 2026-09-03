#!/usr/bin/env python3
"""
scripts/predict_damage.py
-----------------------------
GA conditional-damage preset predictions. Loads the calibrated CatBoost
model trained by scripts/ga_train_models.py and runs one of the named
scenarios in birdstrikegeo.ga.presets.

Usage:
    python scripts/predict_damage.py --list-presets
    python scripts/predict_damage.py --show-preset c172_day_clear_approach
    python scripts/predict_damage.py --preset c172_day_clear_takeoff
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import yaml  # noqa: E402

from birdstrikegeo.ga.features import build_operational_core_features  # noqa: E402
from birdstrikegeo.ga.inference import load_bundle, predict_scenario  # noqa: E402
from birdstrikegeo.ga.presets import (  # noqa: E402
    ScenarioValidationError,
    get_preset,
    list_presets,
    validate_scenario,
)

CONFIG_PATH = REPO_ROOT / "configs" / "ga_damage.yaml"


def _load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def cmd_list_presets() -> int:
    for name in list_presets():
        print(f"  {name}  —  {get_preset(name)['description']}")
    return 0


def cmd_show_preset(name: str) -> int:
    try:
        scenario = get_preset(name)
    except ScenarioValidationError as e:
        print(f"Error: {e}")
        return 1
    print(f"Preset: {name}")
    print(f"  {scenario.pop('description')}")
    for k, v in scenario.items():
        print(f"  {k}: {v}")
    return 0


def cmd_predict(name: str) -> int:
    config = _load_config()
    models_dir = REPO_ROOT / config["paths"]["models_dir"]
    try:
        bundle = load_bundle(models_dir, feature_mode="operational_core")
    except FileNotFoundError:
        print(f"Error: no trained GA model found at {models_dir}. Run scripts/ga_train_models.py first.")
        return 1

    try:
        scenario = get_preset(name)
    except ScenarioValidationError as e:
        print(f"Error: {e}")
        return 1

    scenario_fields = {k: v for k, v in scenario.items() if k != "description"}
    try:
        validate_scenario(scenario_fields, bundle.seen_categories, bundle.numeric_ranges)
    except ScenarioValidationError as e:
        print(f"Error: invalid scenario — {e}")
        return 1

    result = predict_scenario(scenario_fields, bundle, build_operational_core_features)

    print(f"Model: GA conditional-damage gradient-boosted tree (CatBoost, calibration={bundle.calibration_method})")
    print(f"Scenario: {scenario['description']}")
    print(f"Prediction: {result['probability']:.1%} probability of reported damage conditional on a strike")
    print(f"Risk band: {result['risk_band']}")
    print(f"Important inputs: {', '.join(c['feature'] for c in result['shap_explanation']['contributions'][:3])}")
    print(f"\n{result['disclaimer']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list-presets", action="store_true")
    group.add_argument("--show-preset", metavar="NAME")
    group.add_argument("--preset", metavar="NAME")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of text (with --preset).")
    args = parser.parse_args()

    if args.list_presets:
        return cmd_list_presets()
    if args.show_preset:
        return cmd_show_preset(args.show_preset)
    if args.preset:
        if not args.json:
            return cmd_predict(args.preset)
        config = _load_config()
        bundle = load_bundle(REPO_ROOT / config["paths"]["models_dir"], feature_mode="operational_core")
        scenario = {k: v for k, v in get_preset(args.preset).items() if k != "description"}
        validate_scenario(scenario, bundle.seen_categories, bundle.numeric_ranges)
        result = predict_scenario(scenario, bundle, build_operational_core_features)
        print(json.dumps({k: v for k, v in result.items() if k != "shap_explanation"}, indent=2, default=str))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
