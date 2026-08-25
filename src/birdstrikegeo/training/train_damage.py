"""
training/train_damage.py
----------------------------
Task A end-to-end training: preprocessing (fit on train only) ->
baselines -> DamageNet (PyTorch, AdamW, BCEWithLogitsLoss, early
stopping, training-set-derived class weighting) -> threshold tuning on
validation -> evaluation on test -> explainability -> checkpoint + plots.

Called by scripts/train_models.py, kept as an importable module (rather
than a standalone script) so both scripts/train_models.py and tests can
drive it without shelling out.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from birdstrikegeo.features.build_damage_features import apply_rare_categories, fit_rare_categories
from birdstrikegeo.models.baselines import build_baselines
from birdstrikegeo.models.neural_net import DamageNet, count_trainable_parameters
from birdstrikegeo.training.tune_threshold import tune_threshold

NUMERIC_COLUMNS = [
    "height_agl_ft", "speed_ias_knots", "engine_count", "year", "month", "day_of_year", "hour_local",
    "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos", "hour_sin", "hour_cos",
]
CATEGORICAL_COLUMNS = [
    "state", "airport_id", "phase_of_flight", "aircraft_make", "aircraft_model", "aircraft_type",
    "aircraft_mass_class", "engine_type", "sky_condition", "precipitation", "species_common_name",
    "species_scientific_name", "wildlife_size", "number_seen", "number_struck", "parts_struck",
    "season", "is_migration_season", "incident_time_local",
]
BOOLEAN_LIKE_COLUMNS = ["is_dawn", "is_dusk", "is_night"]  # nullable booleans -> treated as tri-state categoricals


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    feature_names_out: list[str]
    frequent_categories: dict


def build_preprocessor(train_df: pd.DataFrame, rare_category_threshold: int) -> tuple[ColumnTransformer, dict]:
    present_categorical = [c for c in CATEGORICAL_COLUMNS + BOOLEAN_LIKE_COLUMNS if c in train_df.columns]
    present_numeric = [c for c in NUMERIC_COLUMNS if c in train_df.columns]

    frequent_values = fit_rare_categories(train_df, present_categorical, rare_category_threshold)

    numeric_pipeline = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_pipeline = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [("numeric", numeric_pipeline, present_numeric), ("categorical", categorical_pipeline, present_categorical)]
    )
    return preprocessor, frequent_values


def _apply_rare_and_stringify(df: pd.DataFrame, frequent_values: dict) -> pd.DataFrame:
    df = apply_rare_categories(df, frequent_values)
    for col in frequent_values:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("missing")
    return df


def prepare_datasets(train_df, val_df, test_df, rare_category_threshold: int) -> PreparedData:
    preprocessor, frequent_values = build_preprocessor(train_df, rare_category_threshold)

    train_ready = _apply_rare_and_stringify(train_df, frequent_values)
    val_ready = _apply_rare_and_stringify(val_df, frequent_values)
    test_ready = _apply_rare_and_stringify(test_df, frequent_values)

    X_train = preprocessor.fit_transform(train_ready)
    X_val = preprocessor.transform(val_ready)
    X_test = preprocessor.transform(test_ready)

    feature_names_out = list(preprocessor.get_feature_names_out())

    return PreparedData(
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=train_df["damage"].astype(int).to_numpy(),
        y_val=val_df["damage"].astype(int).to_numpy(),
        y_test=test_df["damage"].astype(int).to_numpy(),
        preprocessor=preprocessor, feature_names_out=feature_names_out,
        frequent_categories=frequent_values,
    )


def train_baselines(prepared: PreparedData, random_seed: int) -> dict:
    baselines = build_baselines(random_seed)
    fitted = {}
    for name, model in baselines.items():
        model.fit(prepared.X_train, prepared.y_train)
        fitted[name] = model
    return fitted


@dataclass
class NeuralNetTrainingResult:
    model: DamageNet
    train_losses: list[float]
    val_losses: list[float]
    best_epoch: int
    trainable_parameters: int


def train_neural_net(
    prepared: PreparedData,
    hidden_sizes: tuple[int, int] = (64, 32),
    dropout: tuple[float, float] = (0.20, 0.10),
    learning_rate: float = 0.001,
    batch_size: int = 64,
    maximum_epochs: int = 200,
    early_stopping_patience: int = 15,
    random_seed: int = 42,
) -> NeuralNetTrainingResult:
    set_seed(random_seed)

    input_size = prepared.X_train.shape[1]
    model = DamageNet(input_size=input_size, hidden_sizes=hidden_sizes, dropout=dropout)

    # Training-set-derived class weighting: pos_weight = n_negative / n_positive,
    # so BCEWithLogitsLoss compensates for class imbalance using only
    # statistics observed in the training split (never validation/test).
    n_pos = max(int(prepared.y_train.sum()), 1)
    n_neg = max(len(prepared.y_train) - n_pos, 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    X_train_t = torch.tensor(prepared.X_train, dtype=torch.float32)
    y_train_t = torch.tensor(prepared.y_train, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(prepared.X_val, dtype=torch.float32)
    y_val_t = torch.tensor(prepared.y_val, dtype=torch.float32).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(dataset)
        train_losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    return NeuralNetTrainingResult(
        model=model, train_losses=train_losses, val_losses=val_losses,
        best_epoch=best_epoch, trainable_parameters=count_trainable_parameters(model),
    )


def predict_proba_neural_net(model: DamageNet, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32))
        return torch.sigmoid(logits).squeeze(1).numpy()


def predict_proba_sklearn(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


def tune_all_thresholds(models_val_proba: dict[str, np.ndarray], y_val, objective: str) -> dict[str, dict]:
    return {name: tune_threshold(y_val, proba, objective=objective) for name, proba in models_val_proba.items()}
