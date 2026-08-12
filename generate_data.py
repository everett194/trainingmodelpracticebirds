"""
generate_data.py
-----------------
Generates a SYNTHETIC, EDUCATIONAL dataset of bird-strike-risk examples.

*** IMPORTANT DISCLAIMER ***
There is no real aviation dataset behind this project. The relationships
coded below between the 8 input features and the "elevated risk" label are
made up by hand, for teaching purposes only, so that the neural network has
*something plausible* to learn. They are NOT scientifically validated
bird-strike-risk relationships and must never be used for real aviation
safety decisions.

How the fake labels are built:
1. Start each simulated observation with 8 random feature values, each
   drawn from the ranges defined in config.py.
2. Compute a "risk score" by combining the features with hand-picked
   weights that encode simple, intuitive assumptions, e.g.:
     - more recent bird activity -> higher risk
     - closer to water -> higher risk (birds congregate near water)
     - dawn/dusk hours -> higher risk (peak bird activity times)
     - spring/fall migration months -> higher risk
     - poor visibility -> higher risk
     - very high wind speed -> lower risk (birds fly less in high wind)
3. Squash that risk score into a 0-1 probability with a sigmoid function.
4. Flip a weighted coin using that probability to decide the final 0/1
   label. This keeps the data "noisy" and realistic instead of a dataset
   that's perfectly, artificially separable.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    FEATURE_RANGES,
    LABEL_NAME,
    NUM_SAMPLES,
    RANDOM_SEED,
    TEST_CSV_PATH,
    TEST_SPLIT_FRACTION,
    TRAIN_CSV_PATH,
)


def sigmoid(x):
    """Squashes any real number into the range (0, 1)."""
    return 1.0 / (1.0 + np.exp(-x))


def generate_features(rng, num_samples):
    """Draw random values for each of the 8 features from their ranges."""
    data = {}

    t_lo, t_hi = FEATURE_RANGES["temperature_c"]
    data["temperature_c"] = rng.uniform(t_lo, t_hi, num_samples)

    w_lo, w_hi = FEATURE_RANGES["wind_speed_knots"]
    data["wind_speed_knots"] = rng.uniform(w_lo, w_hi, num_samples)

    v_lo, v_hi = FEATURE_RANGES["visibility_km"]
    data["visibility_km"] = rng.uniform(v_lo, v_hi, num_samples)

    p_lo, p_hi = FEATURE_RANGES["precipitation"]
    # Most hours have no precipitation, so bias towards 0.
    data["precipitation"] = rng.choice([p_lo, p_hi], size=num_samples, p=[0.75, 0.25])

    h_lo, h_hi = FEATURE_RANGES["hour_of_day"]
    data["hour_of_day"] = rng.integers(h_lo, h_hi + 1, num_samples)

    m_lo, m_hi = FEATURE_RANGES["month"]
    data["month"] = rng.integers(m_lo, m_hi + 1, num_samples)

    d_lo, d_hi = FEATURE_RANGES["distance_to_water_km"]
    data["distance_to_water_km"] = rng.uniform(d_lo, d_hi, num_samples)

    a_lo, a_hi = FEATURE_RANGES["recent_bird_activity"]
    data["recent_bird_activity"] = rng.integers(a_lo, a_hi + 1, num_samples)

    return pd.DataFrame(data)


def compute_risk_labels(df, rng):
    """
    Hand-designed (NOT scientific) scoring rules that turn the 8 features
    into a binary "elevated risk" label. See the module docstring above.
    """
    score = np.zeros(len(df))

    # Recent bird activity is the single strongest signal: 0-10 scale,
    # centered and scaled so it can swing the score a lot.
    score += (df["recent_bird_activity"] - 5.0) * 0.6

    # Closer to water -> more risk. distance ranges 0-30km; invert and
    # scale so "0 km away" contributes the most positive score.
    d_lo, d_hi = FEATURE_RANGES["distance_to_water_km"]
    score += (1.0 - (df["distance_to_water_km"] - d_lo) / (d_hi - d_lo)) * 2.0

    # Dawn (5-8) and dusk (18-21) hours are peak bird activity times.
    is_dawn_or_dusk = df["hour_of_day"].between(5, 8) | df["hour_of_day"].between(18, 21)
    score += np.where(is_dawn_or_dusk, 1.5, -0.3)

    # Spring (3-5) and fall (8-10) migration months raise risk.
    is_migration_season = df["month"].between(3, 5) | df["month"].between(8, 10)
    score += np.where(is_migration_season, 1.2, -0.2)

    # Poor visibility raises risk (harder to spot and avoid birds).
    v_lo, v_hi = FEATURE_RANGES["visibility_km"]
    score += (1.0 - (df["visibility_km"] - v_lo) / (v_hi - v_lo)) * 1.5

    # Light-to-moderate wind doesn't suppress bird activity much, but very
    # high wind tends to ground birds, lowering risk.
    w_lo, w_hi = FEATURE_RANGES["wind_speed_knots"]
    score += np.where(df["wind_speed_knots"] > 30, -1.2, 0.2)

    # Mild temperatures (roughly 5-25C) correspond to more bird activity
    # than extreme cold or heat.
    is_mild_temp = df["temperature_c"].between(5, 25)
    score += np.where(is_mild_temp, 0.6, -0.2)

    # Precipitation can push insects (and the birds that eat them) lower
    # to the ground, near flight paths.
    score += np.where(df["precipitation"] == 1, 0.5, 0.0)

    # A constant offset to roughly balance the two classes (without this,
    # the sum of the positive contributions above skews heavily towards
    # "elevated risk"). This is just a knob for making the demo dataset
    # more instructive - it has no real-world meaning.
    score -= 2.3

    # Add random noise so the relationship isn't perfectly deterministic -
    # real-world outcomes are never fully predictable from a few inputs.
    score += rng.normal(loc=0.0, scale=1.0, size=len(df))

    probabilities = sigmoid(score)

    # Sample the final binary label as a weighted coin flip using the
    # computed probability, rather than a hard threshold. This keeps the
    # dataset realistically noisy.
    labels = rng.binomial(n=1, p=probabilities)

    return labels


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"Generating {NUM_SAMPLES} synthetic bird-strike-risk observations...")
    df = generate_features(rng, NUM_SAMPLES)
    df[LABEL_NAME] = compute_risk_labels(df, rng)

    print(f"Label balance: {df[LABEL_NAME].mean():.1%} elevated risk, "
          f"{1 - df[LABEL_NAME].mean():.1%} lower risk")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SPLIT_FRACTION,
        random_state=RANDOM_SEED,
        stratify=df[LABEL_NAME],
    )

    train_df.to_csv(TRAIN_CSV_PATH, index=False)
    test_df.to_csv(TEST_CSV_PATH, index=False)

    print(f"Saved {len(train_df)} training examples to {TRAIN_CSV_PATH}")
    print(f"Saved {len(test_df)} testing examples to {TEST_CSV_PATH}")
    print("\nReminder: these relationships are artificially designed for this "
          "educational demo and are NOT scientifically validated.")


if __name__ == "__main__":
    main()
