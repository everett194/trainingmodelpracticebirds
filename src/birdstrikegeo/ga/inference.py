"""
birdstrikegeo.ga.inference
------------------------------
Turns a scenario dict (a preset, or future custom JSON input) - keyed by
the ENGINEERED feature names in birdstrikegeo.ga.features - into a single
tree-native input row and a calibrated damage-probability prediction,
using the exact model bundle saved by scripts/ga_train_models.py.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier

from birdstrikegeo.ga.evaluation import apply_calibration
from birdstrikegeo.ga.explain import local_shap_explanation

PREDICTION_DISCLAIMER = (
    "This is conditional damage probability assuming a strike occurs. "
    "It is not the probability that a strike will occur."
)

RISK_BANDS = [
    (0.10, "Low relative conditional severity"),
    (0.25, "Moderate relative conditional severity"),
    (0.50, "Elevated relative conditional severity"),
    (1.01, "High relative conditional severity"),
]


def risk_band_for(probability: float) -> str:
    for threshold, label in RISK_BANDS:
        if probability < threshold:
            return label
    return RISK_BANDS[-1][1]


# scenario field (engineered name) -> raw FAA column name expected by
# birdstrikegeo.ga.features.build_operational_core_features.
SCENARIO_TO_RAW_COLUMN = {
    "aircraft_type": "AIRCRAFT", "aircraft_mass_class": "AC_MASS", "engine_type": "TYPE_ENG",
    "engine_count": "NUM_ENGS", "engine_position_primary": "ENG_1_POS", "engine_position_secondary": "ENG_2_POS",
    "state": "STATE", "faa_region": "FAAREGION", "sky_condition": "SKY", "precipitation": "PRECIPITATION",
    "month": "INCIDENT_MONTH", "phase_of_flight": "PHASE_OF_FLIGHT", "height_agl_ft": "HEIGHT",
    "speed_ias_knots": "SPEED", "light_condition": "TIME_OF_DAY", "bird_warned": "WARNED",
}


def scenario_to_raw_row(scenario: dict) -> pd.DataFrame:
    row = {}
    for field, raw_col in SCENARIO_TO_RAW_COLUMN.items():
        value = scenario.get(field)
        if raw_col == "ENG_2_POS" and value == "NONE":
            value = None
        row[raw_col] = [value]
    return pd.DataFrame(row)


@dataclass
class GaModelBundle:
    catboost_model: CatBoostClassifier
    calibration_method: str
    calibrator: object
    numeric_columns: list[str]
    categorical_columns: list[str]
    seen_categories: dict
    numeric_ranges: dict
    threshold: float
    model_version: str
    feature_mode: str


MODEL_FILENAME = "ga_catboost_{mode}.cbm"
BUNDLE_FILENAME = "ga_model_bundle_{mode}.pkl"


def save_bundle(bundle: GaModelBundle, models_dir: str | Path) -> None:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    bundle.catboost_model.save_model(str(models_dir / MODEL_FILENAME.format(mode=bundle.feature_mode)))
    metadata = {k: v for k, v in bundle.__dict__.items() if k != "catboost_model"}
    with open(models_dir / BUNDLE_FILENAME.format(mode=bundle.feature_mode), "wb") as f:
        pickle.dump(metadata, f)


def load_bundle(models_dir: str | Path, feature_mode: str = "operational_core") -> GaModelBundle:
    models_dir = Path(models_dir)
    model = CatBoostClassifier()
    model.load_model(str(models_dir / MODEL_FILENAME.format(mode=feature_mode)))
    with open(models_dir / BUNDLE_FILENAME.format(mode=feature_mode), "rb") as f:
        metadata = pickle.load(f)
    return GaModelBundle(catboost_model=model, **metadata)


def predict_scenario(scenario: dict, bundle: GaModelBundle, feature_builder) -> dict:
    """
    feature_builder: birdstrikegeo.ga.features.build_operational_core_features
    (passed in rather than imported, since reduced_preflight scenarios
    would use the other builder in a future extension).
    """
    raw_row = scenario_to_raw_row(scenario)
    engineered = feature_builder(raw_row).features

    tree_row = engineered[bundle.numeric_columns + bundle.categorical_columns].copy()
    for col in bundle.categorical_columns:
        tree_row[col] = tree_row[col].astype("string").fillna("missing_value")

    raw_proba = bundle.catboost_model.predict_proba(tree_row)[:, 1][0]
    calibrated_proba = float(apply_calibration(bundle.calibration_method, bundle.calibrator, [raw_proba])[0])

    explanation = local_shap_explanation(bundle.catboost_model, tree_row, bundle.categorical_columns)
    top_features = [f"{c['feature']} ({c['shap_contribution']:+.3f})" for c in explanation["contributions"][:3]]

    return {
        "probability_raw": float(raw_proba),
        "probability": calibrated_proba,
        "risk_band": risk_band_for(calibrated_proba),
        "top_features": top_features,
        "shap_explanation": explanation,
        "disclaimer": PREDICTION_DISCLAIMER,
    }
