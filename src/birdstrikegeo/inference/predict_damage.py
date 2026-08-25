"""
inference/predict_damage.py
-------------------------------
Task A inference: loads a saved DamageNet checkpoint and turns a single
new (raw, un-encoded) strike record into a probability.

The prediction is ALWAYS described as: "Estimated probability of
aircraft damage, conditional on a reported wildlife strike." It must
never be presented as the probability that a strike will occur -
that is a fundamentally different, unanswered question (see README.md
"Scientific limitations" and configs/data_sources.yaml's bts_t100
placeholder for what would be needed to answer it).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from birdstrikegeo.features.build_damage_features import apply_rare_categories
from birdstrikegeo.geo.feature_engineering import build_time_features
from birdstrikegeo.models.checkpoints import load_checkpoint

PREDICTION_SEMANTICS = (
    "Estimated probability of aircraft damage, conditional on a reported wildlife strike. "
    "This is NOT the probability that a strike will occur."
)


def predict_damage(raw_record: dict, checkpoint_path: str) -> dict:
    """
    raw_record: a dict of raw (un-encoded) FAA-schema field values for
    ONE strike, e.g. {"latitude": 40.0, "phase_of_flight": "Approach", ...}.
    Missing fields are fine - the pipeline's imputers handle them the
    same way they were fit to handle missing training data.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    model = checkpoint["model"]
    preprocessor = checkpoint["preprocessor"]
    threshold = checkpoint["threshold"]

    df = pd.DataFrame([raw_record])
    if "incident_date" in df.columns:
        combined = pd.to_datetime(df["incident_date"], errors="coerce")
        if "incident_time_local" in df.columns:
            time_str = df["incident_time_local"].astype("string").fillna("00:00")
            parsed_time = pd.to_timedelta(time_str.str.strip().replace("", "00:00") + ":00", errors="coerce")
            combined = combined + parsed_time.fillna(pd.Timedelta(0))
        df["_incident_timestamp"] = combined
        df = build_time_features(df, "_incident_timestamp")
        df = df.drop(columns=["_incident_timestamp"])

    for col in checkpoint["feature_columns"]:
        if col not in df.columns:
            # np.nan (not pd.NA) - SimpleImputer/numpy can't convert a
            # pd.NA-filled object column to a numeric array directly.
            df[col] = np.nan

    # Reapply the SAME rare-category bucketing fit on the training data
    # (saved in the checkpoint) before encoding, so a previously-seen-but-
    # infrequent category is treated identically to how it was treated
    # during training, rather than falling through OneHotEncoder's
    # handle_unknown="ignore" as if it were completely novel.
    df = apply_rare_categories(df, checkpoint["frequent_categories"])
    for col in checkpoint["frequent_categories"]:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("missing")

    encoded = preprocessor.transform(df[checkpoint["feature_columns"]])
    with torch.no_grad():
        logit = model(torch.tensor(encoded, dtype=torch.float32)).item()
    probability = torch.sigmoid(torch.tensor(logit)).item()

    return {
        "logit": logit,
        "probability_of_damage_given_reported_strike": probability,
        "classification": "ELEVATED" if probability >= threshold else "LOWER",
        "threshold_used": threshold,
        "model_version": checkpoint["model_version"],
        "prediction_semantics": PREDICTION_SEMANTICS,
    }
