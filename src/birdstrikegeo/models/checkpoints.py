"""
models/checkpoints.py
-------------------------
Saves/loads a DamageNet checkpoint together with everything needed to
reproduce its exact input pipeline at inference time: the fitted
preprocessing (encoder + rare-category buckets), feature column order,
chosen decision threshold, and a model_version stamp - the same pattern
synthetic_demo/train.py uses (bundling feature_mean/feature_std with the
weights), extended with the extra state this bigger pipeline needs.
"""

from __future__ import annotations

from pathlib import Path

import torch

from birdstrikegeo.models.neural_net import DamageNet


def save_checkpoint(
    model: DamageNet,
    preprocessor,
    feature_columns: list[str],
    threshold: float,
    model_version: str,
    hidden_sizes: tuple[int, int],
    dropout: tuple[float, float],
    path: str | Path,
    frequent_categories: dict | None = None,
) -> None:
    """
    frequent_categories: the dict returned by
    birdstrikegeo.features.build_damage_features.fit_rare_categories()
    on the TRAINING split. Must be saved alongside the preprocessor and
    reapplied at inference time (see inference/predict_damage.py) - the
    fitted OneHotEncoder's categories were learned AFTER rare-category
    bucketing, so skipping this step at inference time would treat a
    below-threshold-but-previously-seen category differently than it was
    treated during training.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": model.layer1.in_features,
            "hidden_sizes": hidden_sizes,
            "dropout": dropout,
            "preprocessor": preprocessor,  # a fitted sklearn ColumnTransformer (pickled via torch.save)
            "feature_columns": feature_columns,
            "frequent_categories": frequent_categories or {},
            "threshold": threshold,
            "model_version": model_version,
        },
        path,
    )


def load_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")
    checkpoint = torch.load(path, weights_only=False)

    model = DamageNet(
        input_size=checkpoint["input_size"],
        hidden_sizes=tuple(checkpoint["hidden_sizes"]),
        dropout=tuple(checkpoint["dropout"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return {
        "model": model,
        "preprocessor": checkpoint["preprocessor"],
        "feature_columns": checkpoint["feature_columns"],
        "frequent_categories": checkpoint.get("frequent_categories", {}),
        "threshold": checkpoint["threshold"],
        "model_version": checkpoint["model_version"],
    }
