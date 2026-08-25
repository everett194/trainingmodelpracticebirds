"""
config.py
---------
All the "knobs" for this project live in one place so you don't have to go
hunting through every file to find a constant. Nothing in here is fancy -
it's just plain Python variables that the other scripts import.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# BASE_DIR is the folder this config.py file lives in. Using it means the
# project works no matter which directory you happen to run it from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "bird_strike_model.pt")
METRICS_PATH = os.path.join(RESULTS_DIR, "training_metrics.json")
LOSS_CURVE_PATH = os.path.join(RESULTS_DIR, "loss_curve.png")

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
# The order of this list matters: it defines the order in which features are
# fed into the neural network as the 8 input numbers. Every other file
# (dataset.py, train.py, predict.py) uses this same list so the order always
# matches.
FEATURE_NAMES = [
    "temperature_c",
    "wind_speed_knots",
    "visibility_km",
    "precipitation",
    "hour_of_day",
    "month",
    "distance_to_water_km",
    "recent_bird_activity",
]

LABEL_NAME = "elevated_risk"

# Reasonable real-world-ish ranges used both to generate synthetic data and
# to validate/prompt for user input in predict.py. These are NOT scientific
# constants - they are plausible bounds picked for an educational demo.
FEATURE_RANGES = {
    "temperature_c": (-20.0, 40.0),        # degrees Celsius
    "wind_speed_knots": (0.0, 50.0),       # knots
    "visibility_km": (0.0, 15.0),          # kilometers
    "precipitation": (0, 1),               # 0 = no precipitation, 1 = precipitation
    "hour_of_day": (0, 23),                # 24-hour clock
    "month": (1, 12),                      # calendar month
    "distance_to_water_km": (0.0, 30.0),   # kilometers to nearest water body
    "recent_bird_activity": (0, 10),       # subjective 0-10 activity scale
}

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# Using the same "seed" every time makes randomness repeatable: shuffles,
# random weight initialization, and data generation all come out the same
# way each run, which makes debugging and comparisons much easier.
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------
NUM_SAMPLES = 5000
TEST_SPLIT_FRACTION = 0.2  # 20% of samples held out for testing

# ---------------------------------------------------------------------------
# Model / training hyperparameters
# ---------------------------------------------------------------------------
INPUT_SIZE = 8      # number of input features
HIDDEN1_SIZE = 6    # neurons in first hidden layer
HIDDEN2_SIZE = 4    # neurons in second hidden layer
OUTPUT_SIZE = 1     # a single number: probability of "elevated risk"

NUM_EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 32

# Threshold used to turn a probability into a 0/1 classification.
CLASSIFICATION_THRESHOLD = 0.5
