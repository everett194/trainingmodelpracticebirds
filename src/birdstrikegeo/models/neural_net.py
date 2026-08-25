"""
models/neural_net.py
------------------------
Task A (damage prediction) PyTorch neural network. Exactly the
architecture specified in the project requirements - no extra layers
added for the sake of looking sophisticated:

    encoded_input -> Linear(64) -> ReLU -> Dropout(0.20)
                   -> Linear(32) -> ReLU -> Dropout(0.10)
                   -> Linear(1)

Like synthetic_demo/model.py, this has NO sigmoid in forward() - it
outputs a raw logit, trained with BCEWithLogitsLoss and converted to a
probability only at inference time (see
birdstrikegeo.inference.predict_damage).
"""

from __future__ import annotations

import torch.nn as nn


class DamageNet(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: tuple[int, int] = (64, 32),
                 dropout: tuple[float, float] = (0.20, 0.10)):
        super().__init__()
        h1, h2 = hidden_sizes
        d1, d2 = dropout

        self.layer1 = nn.Linear(input_size, h1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(d1)

        self.layer2 = nn.Linear(h1, h2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(d2)

        self.layer3 = nn.Linear(h2, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.layer3(x)
        return x


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
