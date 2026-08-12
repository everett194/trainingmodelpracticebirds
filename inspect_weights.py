"""
inspect_weights.py
-------------------
Two things in one script:

1. Prints every weight and bias number that training actually learned,
   so you can see what's really inside the "black box."

2. Demonstrates manually editing a single weight by hand - overwriting a
   learned number instead of getting there through training - and shows
   how that one change moves a prediction. This is NOT how the model
   normally learns (that's backpropagation + the optimizer, see
   train.py); it's a hands-on way to build intuition for what a weight
   actually does.

Run it with:
    python inspect_weights.py
"""

import torch

from config import FEATURE_NAMES, MODEL_PATH
from inference import load_model


def print_layer(name, layer):
    print(f"\n{name}  (nn.Linear({layer.in_features}, {layer.out_features}))")

    print("  weights (shape: out_features x in_features):")
    for row in layer.weight.data.tolist():
        formatted = ", ".join(f"{v:7.4f}" for v in row)
        print(f"    [{formatted}]")

    print("  biases (shape: out_features):")
    formatted = ", ".join(f"{v:7.4f}" for v in layer.bias.data.tolist())
    print(f"    [{formatted}]")


def predict_probability(model, normalized_input):
    with torch.no_grad():
        logit = model(normalized_input).item()
    return torch.sigmoid(torch.tensor(logit)).item()


def layer2_activations(model, normalized_input):
    """Runs the input through layer1+ReLU and layer2+ReLU by hand, mirroring
    model.py's forward() up to (but not including) the output layer, so we
    can see what layer2 actually outputs for this input."""
    with torch.no_grad():
        h1 = torch.relu(model.layer1(normalized_input))
        h2 = torch.relu(model.layer2(h1))
    return h2


def main():
    model, feature_mean, feature_std = load_model()

    # -----------------------------------------------------------------
    # Part 1: print every learned weight and bias
    # -----------------------------------------------------------------
    print("=" * 70)
    print("LEARNED WEIGHTS AND BIASES")
    print("=" * 70)
    print(f"Input features, in order: {FEATURE_NAMES}")

    print_layer("layer1 (8 -> 6)", model.layer1)
    print_layer("layer2 (6 -> 4)", model.layer2)
    print_layer("layer3 (4 -> 1)", model.layer3)

    # -----------------------------------------------------------------
    # Part 2: manually edit one weight and see the effect
    # -----------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MANUALLY EDITING A WEIGHT")
    print("=" * 70)

    # A fixed, made-up set of conditions so we have something consistent
    # to compare "before" and "after" on.
    example_raw = torch.tensor(
        [10.0, 5.0, 2.0, 1.0, 6.0, 4.0, 1.0, 9.0], dtype=torch.float32
    )
    example_normalized = ((example_raw - torch.tensor(feature_mean)) / torch.tensor(feature_std)).unsqueeze(0)

    before = predict_probability(model, example_normalized)
    print(f"\nExample input: {dict(zip(FEATURE_NAMES, example_raw.tolist()))}")
    print(f"Predicted probability BEFORE editing: {before * 100:.2f}%")

    # ReLU means some hidden neurons output exactly 0 for a given input
    # ("off" for this particular example) - editing a weight connected to
    # an "off" neuron would have zero visible effect. So instead of
    # picking a fixed weight, we compute each of layer3's 4 weights'
    # actual contribution to the logit for THIS input (weight * hidden
    # activation) and flip the sign of whichever one currently matters
    # most - guaranteeing a visible change.
    h2 = layer2_activations(model, example_normalized)
    print(f"\nlayer2's hidden activations for this input: {h2.squeeze(0).tolist()}")

    contributions = (model.layer3.weight.data[0] * h2.squeeze(0)).tolist()
    col = max(range(len(contributions)), key=lambda i: abs(contributions[i]))
    row = 0

    old_value = model.layer3.weight.data[row, col].item()
    model.layer3.weight.data[row, col] = old_value * -1.0
    new_value = model.layer3.weight.data[row, col].item()

    after = predict_probability(model, example_normalized)

    print(f"\nManually flipped the sign of layer3.weight[{row}][{col}] "
          f"(connects layer2's hidden neuron {col} -> the output logit; "
          f"this was the biggest contributor to the logit for this input)")
    print(f"  old value: {old_value:.4f}")
    print(f"  new value: {new_value:.4f}")
    print(f"Predicted probability AFTER editing:  {after * 100:.2f}%")

    print("\nThis is the same arithmetic training does automatically, millions of "
          "times, guided by gradients - here we just did it once, by hand, to "
          "see the effect directly.")

    print("\nNote: this edited model was NOT saved back to "
          f"{MODEL_PATH} - it only existed in this script's memory. "
          "To edit-and-keep a change, save it yourself with, e.g.:\n"
          "    torch.save({'model_state_dict': model.state_dict(), "
          "'feature_mean': feature_mean, 'feature_std': feature_std}, "
          "'models/bird_strike_model_edited.pt')")


if __name__ == "__main__":
    main()
