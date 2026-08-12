"""
predict.py
----------
The main way to actually use the trained network: run this script, answer
8 questions about current conditions, and it prints an elevated-bird-
strike-risk probability.

DIFFERENCE BETWEEN TRAINING AND INFERENCE
Training (train.py) is the process of showing the network many labeled
examples and adjusting its weights/biases so its predictions improve.
Inference (this script) is the opposite: the weights are already fixed
(loaded from models/bird_strike_model.pt), and we just run new, unlabeled
inputs forward through the network to get a prediction. No learning, no
backpropagation, no weight updates happen during inference.

DIFFERENCE BETWEEN A LOGIT AND A PROBABILITY
BirdStrikeNet's final layer (model.py) has no activation function, so
calling the model directly returns a raw number called a "logit" - it can
be any real number (negative, zero, positive) and isn't a probability by
itself. To turn it into an easy-to-read 0-100% probability, we apply the
sigmoid function, which squashes any real number into the range (0, 1):

    probability = 1 / (1 + e^(-logit))

A logit of 0 corresponds to exactly 50% probability. Positive logits mean
"more likely elevated risk", negative logits mean "more likely lower
risk" - the further from 0, the more confident the model is.

WHAT'S INSIDE THE .pt FILE?
models/bird_strike_model.pt is a PyTorch checkpoint saved with
torch.save(). It's a dictionary containing:
  - "model_state_dict": every learned weight and bias in the network
    (a mapping from layer name to tensor of numbers).
  - "feature_mean" / "feature_std": the exact normalization statistics
    computed from the training data, so we can normalize new inputs
    exactly the same way the training data was normalized.
It does NOT contain the model's code/architecture - that's why we still
need to import BirdStrikeNet from model.py and re-create an empty network
before loading the saved weights into it.
"""

import torch

from config import (
    CLASSIFICATION_THRESHOLD,
    FEATURE_NAMES,
    FEATURE_RANGES,
    MODEL_PATH,
)
from dataset import normalize_single_example
from model import BirdStrikeNet

# Human-friendly prompts and short hints shown for each feature, in the
# same order as FEATURE_NAMES so answers line up with the right feature.
PROMPTS = {
    "temperature_c": "Temperature (Celsius)",
    "wind_speed_knots": "Wind speed (knots)",
    "visibility_km": "Visibility (km)",
    "precipitation": "Precipitation (0 = no, 1 = yes)",
    "hour_of_day": "Hour of day (0-23)",
    "month": "Month (1-12)",
    "distance_to_water_km": "Distance to water (km)",
    "recent_bird_activity": "Recent bird activity (0-10)",
}


def ask_for_value(feature_name):
    """Repeatedly prompts until the user enters a valid number for a feature."""
    low, high = FEATURE_RANGES[feature_name]
    prompt_text = f"{PROMPTS[feature_name]} [{low}-{high}]: "

    while True:
        raw = input(prompt_text).strip()
        try:
            value = float(raw)
        except ValueError:
            print("  Please enter a number.")
            continue
        return value


def load_model():
    checkpoint = torch.load(MODEL_PATH, weights_only=False)

    model = BirdStrikeNet()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()  # switch to inference mode

    return model, checkpoint["feature_mean"], checkpoint["feature_std"]


def main():
    print("Bird-Strike Risk Predictor")
    print("(Educational demo only - NOT an operational aviation-safety tool)\n")
    print("Enter the current conditions:\n")

    raw_values = [ask_for_value(name) for name in FEATURE_NAMES]

    model, feature_mean, feature_std = load_model()

    normalized = normalize_single_example(raw_values, feature_mean, feature_std)
    input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)  # shape (1, 8): a batch of one

    with torch.no_grad():
        logit = model(input_tensor).item()

    probability = torch.sigmoid(torch.tensor(logit)).item()
    classification = "ELEVATED" if probability >= CLASSIFICATION_THRESHOLD else "LOWER"

    print("\n## Neural network output")
    print(f"Raw model output (logit): {logit:.4f}")
    print(f"Elevated bird-strike-risk probability: {probability * 100:.1f}%")
    print(f"Classification: {classification}")


if __name__ == "__main__":
    main()
