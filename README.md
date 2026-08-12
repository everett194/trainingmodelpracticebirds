# Bird-Strike Risk Predictor (Educational Neural Network Demo)

A tiny PyTorch neural network that predicts "elevated" vs "lower"
bird-strike risk from 8 numerical inputs.

**This is an educational demo for learning neural network fundamentals.
It is NOT an operational aviation-safety system.** The training data is
synthetic, and the relationships between features and risk were made up
by hand for teaching purposes — they are not scientifically validated.
Do not use this project, or anything derived from it, to make real
aviation safety decisions.

If you're new to machine learning, this README explains every concept
used in the code, in plain language, alongside the actual numbers this
project produces when you run it.

---

## 1. What this project does

1. `generate_data.py` invents 5,000 fake "observations" (weather +
   bird-activity conditions) and a made-up 0/1 risk label for each one.
2. `train.py` trains a small neural network (`BirdStrikeNet`, defined in
   `model.py`) to predict that label from the 8 numbers.
3. `predict.py` lets you type in 8 numbers by hand and see what the
   trained network predicts.

Nothing here talks to a real weather feed, a real airport, or real bird
observation data. It's a self-contained sandbox for learning how a
neural network is built, trained, and used.

---

## 2. What is a neural network?

A neural network is a function built out of many small, simple pieces —
here, three **layers** — stacked so the output of one feeds into the
next. Each layer does simple arithmetic (multiply, add, and optionally
squash through a nonlinear function). No single layer is smart on its
own; what makes the *whole* network able to learn complicated patterns is
the combination of many weighted connections, trained together.

Concretely, this network is a plain Python class (`BirdStrikeNet` in
`model.py`) built from three `nn.Linear` layers.

---

## 3. Architecture

```
8 INPUT FEATURES
   |
   v
[8 values]
   |
   v
Linear 8 -> 6      (layer1: 8*6 = 48 weights + 6 biases)
   |
   v
ReLU
   |
   v
Linear 6 -> 4      (layer2: 6*4 = 24 weights + 4 biases)
   |
   v
ReLU
   |
   v
Linear 4 -> 1      (layer3: 4*1 = 4 weights + 1 bias)
   |
   v
[raw logit]
   |
   v
Sigmoid  (applied only in predict.py, not inside the model)
   |
   v
Probability (0-100%)
   |
   v
Lower Risk  <——— 50% threshold ———>  Elevated Risk
```

---

## 4. Why 8 input neurons?

Because we chose 8 measurements to describe the conditions at a given
moment. Each measurement becomes one number fed into the network, so the
input layer's size is fixed at 8 — one "slot" per feature. If we added a
9th feature, we'd need to change `INPUT_SIZE` in `config.py` and retrain.

## 5. What each input neuron represents

Defined in `config.py` (`FEATURE_NAMES`), always in this order:

| # | Feature | Meaning | Typical range used here |
|---|---------|---------|--------------------------|
| 1 | `temperature_c` | Air temperature | -20°C to 40°C |
| 2 | `wind_speed_knots` | Wind speed | 0 to 50 knots |
| 3 | `visibility_km` | Visibility distance | 0 to 15 km |
| 4 | `precipitation` | Is it precipitating? | 0 = no, 1 = yes |
| 5 | `hour_of_day` | Hour on a 24-hour clock | 0 to 23 |
| 6 | `month` | Calendar month | 1 to 12 |
| 7 | `distance_to_water_km` | Distance to nearest water body | 0 to 30 km |
| 8 | `recent_bird_activity` | Subjective recent bird-activity level | 0 to 10 |

These ranges are plausible, hand-picked numbers for a demo — not
official aviation or ornithological standards.

## 6. What the 6-neuron hidden layer does

It's the first place the network can combine the 8 raw inputs into new,
learned combinations. Each of its 6 neurons computes its own weighted sum
of all 8 inputs (plus a bias), so this layer can learn things like "high
bird activity AND close to water matters more than either alone" — a
combination no single raw input can express by itself.

## 7. What the 4-neuron hidden layer does

It takes the 6 numbers from the previous layer and combines them further
into 4 new numbers. Each additional layer lets the network represent
progressively more abstract combinations of the original inputs. Going
from 6 down to 4 also gradually funnels the information toward the single
final answer.

## 8. Why only 1 output neuron?

Because we're solving a **binary** classification problem — there are
only two possible answers ("lower risk" or "elevated risk"), so we only
need one number that says how strongly the network leans toward
"elevated." (A problem with, say, 5 possible categories would typically
use 5 output neurons instead.)

## 9. What are weights?

Every connection between one layer's outputs and the next layer's neurons
has an associated **weight**: a single number the network multiplies that
value by. `nn.Linear(8, 6)` stores a 6×8 grid of weights — one weight for
every (input, neuron) pair. Weights start out random and are adjusted
during training so the network's predictions get better. Big weights mean
"this input strongly influences this neuron"; near-zero weights mean
"this input barely matters here."

## 10. What are biases?

Each neuron also has one extra learned number, its **bias**, added after
the weighted sum of inputs. It works like the `+ b` in the line equation
`y = mx + b`: it lets a neuron shift its output up or down regardless of
the inputs, giving the network more flexibility than weights alone would.

## 11. What does ReLU do?

ReLU ("Rectified Linear Unit") is an **activation function**:

```
ReLU(x) = max(0, x)
```

It keeps positive numbers unchanged and turns negative numbers into 0.
Without a nonlinear function like this between layers, stacking multiple
`Linear` layers would mathematically collapse into a single `Linear`
layer no matter how many you stack — the network could only ever learn
straight-line relationships. ReLU's small "bend" at zero is what lets the
network learn curved, complex patterns.

## 12. What is a loss function?

A **loss function** produces a single number that measures how wrong the
model's predictions are — lower is better. It's what training is
*minimizing*. Without a loss function, there'd be no way to tell the
optimizer which direction "better" even is.

## 13. What does BCEWithLogitsLoss do?

`BCEWithLogitsLoss` ("Binary Cross-Entropy with Logits") is the loss
function used to train `BirdStrikeNet`. For each example it:

1. Takes the model's raw output (the **logit** — see section 19).
2. Internally applies a sigmoid to turn it into a 0-1 probability.
3. Compares that probability to the true 0/1 label, penalizing the model
   more heavily the more confidently wrong it was.

We use the "with logits" version instead of applying `sigmoid()`
ourselves plus plain `BCELoss` because combining both steps into one
PyTorch operation is more numerically stable. This is exactly why
`model.py`'s `forward()` does *not* end with a `Sigmoid` layer — the loss
function expects the raw logit.

## 14. What is an optimizer?

The **optimizer** is the algorithm that updates the model's weights and
biases after each batch of examples, using the gradients computed by
backpropagation. The loss function says "how wrong are we"; the optimizer
decides "how do we change the weights to be less wrong."

## 15. What does Adam do?

`Adam` is the optimizer used in `train.py`. It's a popular, general-purpose
choice because it automatically adapts the "step size" for each
individual weight based on the recent history of that weight's gradients,
which in practice converges faster and more reliably than plain gradient
descent, with little manual tuning required.

## 16. What is an epoch?

One **epoch** = one full pass through the entire training dataset. This
project trains for 100 epochs (`NUM_EPOCHS` in `config.py`), meaning the
network sees all 4,000 training examples 100 times, adjusting its weights
a little more after each batch, each pass.

## 17. What does backpropagation mean?

After computing the loss for a batch, PyTorch automatically works
*backwards* through the network (`loss.backward()`) to compute the
gradient of the loss with respect to every single weight and bias — i.e.
"if I nudge this one weight up slightly, does the loss go up or down, and
by how much?" The optimizer then uses those gradients to nudge each
weight in the direction that reduces the loss (`optimizer.step()`). This
backward pass is what "backpropagation" refers to.

## 18. What does training actually change inside the model?

Only the numbers inside the three `Linear` layers — the weights and
biases. The architecture itself (3 layers, sized 8→6→4→1) never changes
during training; training just searches for weight/bias values that make
the network's predictions match the training labels as closely as
possible.

## 19. Training vs. inference

- **Training** (`train.py`): shown labeled examples, computes loss,
  backpropagates, and updates weights. This is the "learning" phase.
- **Inference** (`predict.py`): the weights are already fixed (loaded
  from the saved `.pt` file); we just run new, unlabeled inputs forward
  through the network to get a prediction. No learning, no
  backpropagation, no weight updates happen during inference.

## 20. Logit vs. probability

`BirdStrikeNet`'s final layer has no activation function, so calling the
model directly returns a raw number called a **logit** — it can be any
real number (negative, zero, or positive) and isn't a probability by
itself. `predict.py` applies `torch.sigmoid()` to convert it into an
easy-to-read 0-100% **probability**:

```
probability = 1 / (1 + e^(-logit))
```

A logit of 0 corresponds to exactly 50% probability. Positive logits lean
toward "elevated risk," negative logits lean toward "lower risk" — the
further from 0, the more confident the model is.

## What's inside the `.pt` model file?

`models/bird_strike_model.pt` is a PyTorch checkpoint saved with
`torch.save()`. It's a dictionary containing:

- `model_state_dict` — every learned weight and bias in the network (a
  mapping from layer name to a tensor of numbers).
- `feature_mean` / `feature_std` — the exact normalization statistics
  computed from the training data, so new inputs at prediction time get
  normalized exactly the same way the training data was.

It does **not** contain the model's code/architecture — that's why
`predict.py` still imports `BirdStrikeNet` from `model.py` and re-creates
an empty network before loading the saved weights into it.

---

## File structure

```
config.py            All constants: feature names, ranges, seed, hyperparameters, paths
model.py              BirdStrikeNet neural network definition
dataset.py            PyTorch Dataset: reads CSV, normalizes features, returns tensors
generate_data.py       Creates the synthetic dataset (data/train.csv, data/test.csv)
train.py              Trains the model, evaluates it, saves weights + metrics
predict.py             Interactive CLI: enter 8 values, get a prediction
data/                 Generated CSV datasets
models/               Saved trained model weights (.pt)
results/              Saved training/evaluation metrics (.json)
```

---

## How to run it

```bash
pip install -r requirements.txt
python generate_data.py
python train.py
python predict.py
```

### What actually happened when we ran this

`python generate_data.py`:

```
Generating 5000 synthetic bird-strike-risk observations...
Label balance: 50.4% elevated risk, 49.6% lower risk
Saved 4000 training examples to data/train.csv
Saved 1000 testing examples to data/test.csv
```

`python train.py`:

```
Training BirdStrikeNet for 100 epochs...

Epoch 1/100 - Loss: 0.6948
Epoch 10/100 - Loss: 0.5133
Epoch 20/100 - Loss: 0.5100
...
Epoch 100/100 - Loss: 0.5053

Training complete.
Test accuracy: 75.4%
Test precision: 74.3%
Test recall: 78.2%
Test loss: 0.4972
Model saved to models/bird_strike_model.pt
Metrics saved to results/training_metrics.json
```

These numbers are honest — not artificially inflated. ~75% accuracy on
a deliberately noisy synthetic dataset (see `generate_data.py` — the
labels are sampled probabilistically, not a hard deterministic rule)
is a reasonable, believable result for a 3-layer network this small.

`python predict.py` then asks for the 8 values interactively and prints
something like:

```
## Neural network output
Raw model output (logit): 3.7181
Elevated bird-strike-risk probability: 97.6%
Classification: ELEVATED
```

---

## A note on the synthetic data

Because there's no real aviation bird-strike dataset here, `generate_data.py`
invents one. It hand-codes plausible-sounding rules (e.g. "closer to
water raises risk," "dawn/dusk hours raise risk," "spring/fall migration
months raise risk," "poor visibility raises risk") into a risk score,
squashes that score into a probability with a sigmoid, and then samples
the final 0/1 label as a weighted coin flip — so the data is realistically
noisy rather than a perfectly separable toy problem.

**These relationships are made up for this demo and are not scientifically
validated bird-strike-risk relationships.** See the comments in
`generate_data.py` for the exact rules used.

---

## Ideas for extending this (optional, for further learning)

- Add more epochs or a learning-rate schedule and see how the loss curve changes.
- Try a wider or deeper network and compare test accuracy.
- Plot the training loss over epochs.
- Add a validation split to watch for overfitting separately from the test set.

None of these are required — the current project already demonstrates
the full train → evaluate → predict loop end to end.
