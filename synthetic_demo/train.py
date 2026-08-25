"""
train.py
--------
Trains BirdStrikeNet on data/train.csv and evaluates it on data/test.csv.

WHAT IS AN EPOCH?
One epoch = one full pass through the entire training dataset. We train
for many epochs (see config.NUM_EPOCHS) because a single pass usually
isn't enough for the network's weights to converge on good values - the
network sees the same examples repeatedly, adjusting its weights a little
more each time.

WHAT IS A LOSS FUNCTION?
A loss function is a single number that measures how wrong the model's
predictions are compared to the true labels - lower is better. We use
BCEWithLogitsLoss ("Binary Cross-Entropy with Logits"), the standard loss
for binary (0/1) classification. It:
  1. Internally applies a sigmoid to the model's raw output (the logit),
     turning it into a 0-1 probability.
  2. Compares that probability to the true 0/1 label and penalizes the
     model more heavily the more confidently wrong it was.
We use the "with logits" version (rather than applying sigmoid ourselves
and using plain BCELoss) because it's more numerically stable - the sigmoid
and the loss math are combined into one operation.

WHAT IS AN OPTIMIZER, AND WHAT DOES ADAM DO?
The optimizer is the algorithm that actually updates the model's weights
and biases after each batch, using the gradients computed by
backpropagation (see below). Adam is a popular optimizer that adapts the
"step size" for each individual weight based on the recent history of its
gradients, which in practice tends to converge faster and more reliably
than plain gradient descent, especially with little hyperparameter tuning.

WHAT IS BACKPROPAGATION?
After computing the loss for a batch, PyTorch's autograd automatically
works backwards through the network (loss.backward()) to compute the
gradient of the loss with respect to every single weight and bias - i.e.
"if I nudge this particular weight up slightly, does the loss go up or
down, and by how much?" The optimizer then uses those gradients to nudge
each weight in the direction that reduces the loss (optimizer.step()).
This backward pass is what "backpropagation" refers to.

WHAT DOES TRAINING ACTUALLY CHANGE?
Only the numbers inside the Linear layers - the weights and biases. The
architecture itself (how many layers, how many neurons) never changes;
training just searches for weight/bias values that make the network's
predictions match the training labels as closely as possible.
"""

import json
import random

import matplotlib
matplotlib.use("Agg")  # write plots straight to a file, no GUI window needed
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score
from torch.utils.data import DataLoader

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    LOSS_CURVE_PATH,
    METRICS_PATH,
    MODEL_PATH,
    NUM_EPOCHS,
    RANDOM_SEED,
    TEST_CSV_PATH,
    TRAIN_CSV_PATH,
)
from dataset import BirdStrikeDataset
from model import BirdStrikeNet


def set_seed(seed):
    """
    Makes randomness reproducible. Random weight initialization, data
    shuffling, etc. all use Python/NumPy/PyTorch's random number
    generators - seeding all three means re-running this script produces
    the same results every time.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, data_loader, criterion):
    """
    Runs the model over an entire dataset in evaluation mode (no weight
    updates) and returns the average loss plus accuracy/precision/recall.
    """
    model.eval()  # tells layers like Dropout/BatchNorm to behave differently at inference time (we have none here, but it's good practice)

    total_loss = 0.0
    all_preds = []
    all_labels = []

    # torch.no_grad() turns off gradient tracking, since we're not going
    # to call .backward() here - this saves memory and computation.
    with torch.no_grad():
        for features, labels in data_loader:
            labels = labels.unsqueeze(1)  # shape (batch,) -> (batch, 1) to match model output
            logits = model(features)
            loss = criterion(logits, labels)
            total_loss += loss.item() * features.size(0)

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()

            all_preds.extend(predictions.squeeze(1).tolist())
            all_labels.extend(labels.squeeze(1).tolist())

    average_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)

    return average_loss, accuracy, precision, recall


def main():
    set_seed(RANDOM_SEED)

    # Load the training set first so its mean/std get computed from
    # training data only, then reuse those exact stats to normalize the
    # test set (see dataset.py for why this matters).
    train_dataset = BirdStrikeDataset(TRAIN_CSV_PATH)
    test_dataset = BirdStrikeDataset(TEST_CSV_PATH, mean=train_dataset.mean, std=train_dataset.std)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = BirdStrikeNet()
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Training BirdStrikeNet for {NUM_EPOCHS} epochs...\n")

    # Record the average loss after every epoch so we can plot how it
    # changes over time - a falling curve is the visual signature of a
    # network that's actually learning.
    epoch_losses = []

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()  # tells the model it's in training mode
        running_loss = 0.0

        for features, labels in train_loader:
            labels = labels.unsqueeze(1)  # shape (batch,) -> (batch, 1) to match model output

            optimizer.zero_grad()          # clear gradients from the previous step
            logits = model(features)       # forward pass: compute predictions
            loss = criterion(logits, labels)  # compare predictions to true labels
            loss.backward()                # backpropagation: compute gradients
            optimizer.step()               # update weights and biases using those gradients

            running_loss += loss.item() * features.size(0)

        epoch_loss = running_loss / len(train_dataset)
        epoch_losses.append(epoch_loss)

        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}/{NUM_EPOCHS} - Loss: {epoch_loss:.4f}")

    # Plot the recorded loss curve and save it as an image. This is the
    # same "loss" printed above each epoch, just visualized over time.
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, NUM_EPOCHS + 1), epoch_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss (BCEWithLogitsLoss)")
    plt.title("BirdStrikeNet training loss over time")
    plt.grid(True, alpha=0.3)
    plt.savefig(LOSS_CURVE_PATH)
    plt.close()

    # Final evaluation on the held-out test set - data the model never
    # trained on, so this is an honest measure of how well it generalizes.
    test_loss, accuracy, precision, recall = evaluate(model, test_loader, criterion)

    metrics = {
        "test_loss": test_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save the trained weights AND the normalization stats together, since
    # predict.py needs both to make a correct prediction later.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_mean": train_dataset.mean,
            "feature_std": train_dataset.std,
        },
        MODEL_PATH,
    )

    print("\nTraining complete.")
    print(f"Test accuracy: {accuracy * 100:.1f}%")
    print(f"Test precision: {precision * 100:.1f}%")
    print(f"Test recall: {recall * 100:.1f}%")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    print(f"Loss curve saved to {LOSS_CURVE_PATH}")


if __name__ == "__main__":
    main()
