"""
models/baselines.py
-----------------------
Comparable baselines for Task A, trained on the exact same
leakage-safe, train-only-fit preprocessing as the neural network (see
birdstrikegeo.training.train_damage). A neural network that can't beat a
dummy classifier or plain logistic regression is telling you something
important - this project reports that honestly rather than hiding it.
"""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def build_baselines(random_seed: int) -> dict:
    return {
        "dummy": DummyClassifier(strategy="stratified", random_state=random_seed),
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_seed),
        "histogram_gradient_boosting": HistGradientBoostingClassifier(
            random_state=random_seed, class_weight="balanced"
        ),
    }
