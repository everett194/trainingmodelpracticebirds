"""
inference.py
-------------
Small shared helpers for running the trained model on new inputs. Used by
predict.py (terminal), app.py (web page), and inspect_weights.py, so the
"load the model" and "turn 8 numbers into a probability" logic lives in
exactly one place.
"""

import torch

from config import MODEL_PATH
from dataset import normalize_single_example
from model import BirdStrikeNet


def load_model():
    """
    Recreates an empty BirdStrikeNet and loads the learned weights/biases
    and normalization stats saved by train.py.
    """
    checkpoint = torch.load(MODEL_PATH, weights_only=False)

    model = BirdStrikeNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # switch to inference mode

    return model, checkpoint["feature_mean"], checkpoint["feature_std"]


def predict(raw_values, model, feature_mean, feature_std):
    """
    raw_values: list of 8 raw (un-normalized) feature numbers, in the same
                order as config.FEATURE_NAMES.
    Returns (logit, probability) - the model's raw output and its
    sigmoid-converted 0-1 probability.
    """
    normalized = normalize_single_example(raw_values, feature_mean, feature_std)
    input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)  # shape (1, 8): a batch of one

    with torch.no_grad():
        logit = model(input_tensor).item()

    probability = torch.sigmoid(torch.tensor(logit)).item()
    return logit, probability
