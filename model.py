"""
model.py
--------
This file defines the neural network itself: BirdStrikeNet.

------------------------------------------------------------------------
ML CONCEPTS, EXPLAINED
------------------------------------------------------------------------

INPUT FEATURE
    A single measurable number we feed into the network, e.g. temperature
    in Celsius. This network takes 8 input features (see config.py).

LAYER
    A layer is a transformation step the data passes through. Our network
    has three "Linear" layers, each followed (except the last) by a "ReLU"
    activation. Data flows through the layers in order: layer 1, then
    layer 2, then layer 3.

WEIGHT
    Every connection between an input number and a neuron has an
    associated weight - a single number that scales that input's
    contribution. A Linear(8, 6) layer has 8 x 6 = 48 weights, one for
    every input-to-neuron connection. Weights start out random and are
    adjusted during training so the network's predictions get better.

BIAS
    Each neuron also has one extra number, called a bias, that gets added
    on top of the weighted sum of inputs. The bias lets a neuron shift its
    output up or down independent of the inputs, which gives the network
    more flexibility (similar to the "+b" in the line equation y = mx + b).

OUTPUT NEURON
    A single neuron whose value we care about as the network's answer.
    Our final layer, Linear(4, 1), has exactly one output neuron: a raw
    number (a "logit") that represents how strongly the network believes
    the conditions are elevated-risk. See predict.py for how this raw
    number is converted into a 0-100% probability.

ACTIVATION FUNCTION (ReLU)
    After a Linear layer computes (weights * inputs + bias), we pass the
    result through an activation function. Without activation functions,
    stacking multiple Linear layers would mathematically collapse into
    one single Linear layer, no matter how many we stack - the network
    could only ever learn straight-line relationships. ReLU (Rectified
    Linear Unit) is a simple activation function: it keeps positive
    numbers unchanged and turns negative numbers into 0.
        ReLU(x) = max(0, x)
    This small nonlinearity is what lets the network learn curved,
    complex relationships between the 8 inputs and the risk label.

------------------------------------------------------------------------
"""

import torch.nn as nn

from config import HIDDEN1_SIZE, HIDDEN2_SIZE, INPUT_SIZE, OUTPUT_SIZE


class BirdStrikeNet(nn.Module):
    """
    A tiny feed-forward ("fully connected") neural network:

        8 inputs -> Linear -> ReLU -> 6 -> Linear -> ReLU -> 4 -> Linear -> 1 output

    Note: this class does NOT apply a Sigmoid at the end. During training
    we use nn.BCEWithLogitsLoss, which expects the model's raw, un-squashed
    output (called a "logit") and applies the sigmoid math internally in a
    more numerically stable way than doing it ourselves. When we want an
    actual 0-1 probability (e.g. in predict.py), we apply torch.sigmoid()
    to this model's output ourselves, after the fact.
    """

    def __init__(self):
        # Always call the parent class's __init__ first. This wires up all
        # of PyTorch's internal bookkeeping (tracking parameters, enabling
        # .to(device), etc.) before we add our own layers.
        super().__init__()

        # Layer 1: takes our 8 raw input features and produces 6 numbers.
        # nn.Linear(in_features, out_features) internally stores a
        # (6 x 8) weight matrix and a (6,) bias vector.
        self.layer1 = nn.Linear(INPUT_SIZE, HIDDEN1_SIZE)

        # Layer 2: takes the 6 numbers from layer 1 and produces 4 numbers.
        self.layer2 = nn.Linear(HIDDEN1_SIZE, HIDDEN2_SIZE)

        # Layer 3 (output layer): takes the 4 numbers from layer 2 and
        # produces a single raw number (the logit).
        self.layer3 = nn.Linear(HIDDEN2_SIZE, OUTPUT_SIZE)

        # ReLU has no weights or biases of its own, so one shared instance
        # is enough to reuse after layer1 and after layer2.
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Defines how data flows through the network. PyTorch calls this
        method automatically whenever you do `model(x)`.

        x: a tensor of shape (batch_size, 8) - a batch of input feature
           vectors, 8 numbers each.
        """
        x = self.layer1(x)   # 8 -> 6
        x = self.relu(x)     # nonlinearity
        x = self.layer2(x)   # 6 -> 4
        x = self.relu(x)     # nonlinearity
        x = self.layer3(x)   # 4 -> 1 (raw logit, no activation here)
        return x
