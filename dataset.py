"""
dataset.py
----------
Wraps a CSV file of bird-strike-risk observations in a PyTorch Dataset, so
train.py can hand it to a DataLoader and pull out (features, label) pairs
in batches.

WHY NORMALIZE FEATURES?
Our 8 features live on very different scales - e.g. "hour_of_day" ranges
0-23 while "temperature_c" ranges roughly -20 to 40 and "precipitation" is
just 0 or 1. If we fed those raw numbers straight into the network, the
larger-scale features would dominate the early weighted sums purely
because of their size, not because they're actually more important. This
makes training slower and less stable.

The standard fix is "standardization": for each feature, subtract its mean
and divide by its standard deviation, so every feature ends up roughly
centered around 0 with a spread of about 1.

    normalized_value = (raw_value - mean) / std

IMPORTANT: the mean/std must be computed from the TRAINING data only, and
then that exact same mean/std must be reused to normalize the test data
and any new predictions. If you recompute mean/std on the test set, you
leak information about the test set into your evaluation and get an
overly optimistic (and dishonest) sense of how well the model works.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from config import FEATURE_NAMES, LABEL_NAME


class BirdStrikeDataset(Dataset):
    def __init__(self, csv_path, mean=None, std=None):
        """
        csv_path: path to a CSV with the 8 feature columns plus the label.
        mean/std: optional numpy arrays of shape (8,). Pass these in for a
                  test set (or a single prediction) so it gets normalized
                  using the training set's statistics rather than its own.
                  Leave as None for the training set - the mean/std will be
                  computed from this data and exposed as self.mean/self.std
                  so callers can reuse them later.
        """
        df = pd.read_csv(csv_path)

        # Pull out just the 8 feature columns, in the fixed order defined
        # in config.py, as a plain numpy array of numbers.
        raw_features = df[FEATURE_NAMES].values.astype(np.float32)
        self.labels = df[LABEL_NAME].values.astype(np.float32)

        if mean is None or std is None:
            self.mean = raw_features.mean(axis=0)
            self.std = raw_features.std(axis=0)
            # Guard against a feature that happens to have zero variance,
            # which would otherwise cause a divide-by-zero.
            self.std[self.std == 0] = 1.0
        else:
            self.mean = mean
            self.std = std

        self.features = (raw_features - self.mean) / self.std

    def __len__(self):
        # Required by PyTorch's Dataset interface: how many examples there are.
        return len(self.labels)

    def __getitem__(self, idx):
        # Required by PyTorch's Dataset interface: return one (features,
        # label) pair as tensors, given an index.
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


def normalize_single_example(raw_values, mean, std):
    """
    Helper used by predict.py to normalize one hand-entered observation
    (a list/array of 8 raw numbers) the same way the training data was
    normalized, using the training set's saved mean/std.
    """
    raw_values = np.asarray(raw_values, dtype=np.float32)
    return (raw_values - mean) / std
