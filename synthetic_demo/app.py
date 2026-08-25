"""
app.py
------
A minimal local website for talking to the trained BirdStrikeNet model in
a browser, instead of typing values into a terminal one at a time
(predict.py). Built with Flask, a small Python web framework.

Flask's job here is narrow and mechanical: turn an incoming form
submission into 8 numbers, hand them to inference.predict() (the exact
same function predict.py uses), and render the result as HTML. All of the
actual machine-learning code - the network architecture, normalization,
sigmoid conversion - is unchanged and lives in model.py / dataset.py /
inference.py, same as before.

Run with:
    python app.py
Then open:
    http://localhost:5001
in a browser. (Train the model first with `python train.py` if you
haven't - this page will tell you if the model file is missing.)

Note: this uses port 5001, not the more common default of 5000, because
5000 is claimed by macOS's AirPlay Receiver by default and would refuse
the connection.
"""

import os

from flask import Flask, render_template, request

from config import CLASSIFICATION_THRESHOLD, FEATURE_NAMES, FEATURE_RANGES, MODEL_PATH
from inference import load_model, predict

app = Flask(__name__)

# Human-friendly labels shown next to each input box, in the same order
# as config.FEATURE_NAMES so they always line up with the right feature.
FIELD_LABELS = {
    "temperature_c": "Temperature (°C)",
    "wind_speed_knots": "Wind speed (knots)",
    "visibility_km": "Visibility (km)",
    "precipitation": "Precipitation (0 = no, 1 = yes)",
    "hour_of_day": "Hour of day (0-23)",
    "month": "Month (1-12)",
    "distance_to_water_km": "Distance to water (km)",
    "recent_bird_activity": "Recent bird activity (0-10)",
}


def build_fields(submitted_values=None):
    """
    Builds the list of field dicts the HTML template loops over to draw
    the form, optionally re-filled with whatever the user just submitted
    (so a validation error doesn't wipe out their entries).
    """
    submitted_values = submitted_values or {}
    fields = []
    for name in FEATURE_NAMES:
        low, high = FEATURE_RANGES[name]
        fields.append(
            {
                "name": name,
                "label": FIELD_LABELS[name],
                "low": low,
                "high": high,
                "value": submitted_values.get(name, ""),
            }
        )
    return fields


@app.route("/", methods=["GET"])
def index():
    model_missing = not os.path.exists(MODEL_PATH)
    return render_template("index.html", fields=build_fields(), result=None, error=None, model_missing=model_missing)


@app.route("/predict", methods=["POST"])
def predict_route():
    if not os.path.exists(MODEL_PATH):
        return render_template("index.html", fields=build_fields(), result=None, error=None, model_missing=True)

    submitted_values = {}
    errors = []
    raw_values = []

    for name in FEATURE_NAMES:
        raw = request.form.get(name, "").strip()
        submitted_values[name] = raw
        try:
            raw_values.append(float(raw))
        except ValueError:
            errors.append(f"'{FIELD_LABELS[name]}' must be a number.")

    if errors:
        error_message = " ".join(errors)
        return render_template(
            "index.html", fields=build_fields(submitted_values), result=None, error=error_message, model_missing=False
        )

    model, feature_mean, feature_std = load_model()
    logit, probability = predict(raw_values, model, feature_mean, feature_std)
    classification = "ELEVATED" if probability >= CLASSIFICATION_THRESHOLD else "LOWER"

    result = {
        "logit": logit,
        "probability_pct": probability * 100,
        "classification": classification,
    }

    return render_template(
        "index.html", fields=build_fields(submitted_values), result=result, error=None, model_missing=False
    )


if __name__ == "__main__":
    # Port 5000 is macOS's default AirPlay Receiver port and will refuse
    # connections from anything else, so we use 5001 instead.
    app.run(debug=True, port=5001)
