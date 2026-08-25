"""
app.py
------
The main local website for BirdStrikeGeo, offering THREE distinct modes
so it's always clear which system produced a given number:

1. /synthetic - the original, unrelated educational neural-network demo
   preserved in synthetic_demo/ (entirely synthetic, hand-designed data).
2. /damage - Task A: estimated probability of aircraft damage,
   CONDITIONAL ON a reported wildlife strike (never "probability a
   strike occurs").
3. /activity - Task B: airport-period wildlife activity index, built
   from nearby Trektellen monitoring observations.

None of this is operational aviation-safety software. Every page that
reflects sample data displays a prominent "SAMPLE DATA — NOT A REAL
RESULT" banner - see the *_uses_sample_data() checks below.

Run with:
    python app.py
Then open http://localhost:5001 (port 5001, not 5000, because macOS's
AirPlay Receiver claims port 5000 by default).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from flask import Flask, render_template, request

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from birdstrikegeo.data.ingest_airports import load_airports  # noqa: E402
from birdstrikegeo.features.taxonomy import SPECIES_TAXONOMY  # noqa: E402
from birdstrikegeo.inference.calculate_activity import calculate_activity_for_airport  # noqa: E402
from birdstrikegeo.inference.predict_damage import predict_damage  # noqa: E402

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Synthetic demo: loaded from synthetic_demo/ WITHOUT modifying any of its
# files. Those files use bare `from config import ...` / `from model import
# ...` imports, so we temporarily add synthetic_demo/ to sys.path and import
# them under their own plain module names, exactly as running
# `python synthetic_demo/app.py` directly would.
# ---------------------------------------------------------------------------
SYNTHETIC_DEMO_DIR = REPO_ROOT / "synthetic_demo"


def _load_synthetic_demo_modules():
    sys.path.insert(0, str(SYNTHETIC_DEMO_DIR))
    try:
        import config as syn_config  # noqa
        import inference as syn_inference  # noqa
    finally:
        sys.path.remove(str(SYNTHETIC_DEMO_DIR))
    return syn_config, syn_inference


SYN_CONFIG, SYN_INFERENCE = _load_synthetic_demo_modules()

# ---------------------------------------------------------------------------
# Damage model (Task A) paths/config
# ---------------------------------------------------------------------------
DAMAGE_CHECKPOINT_SAMPLE = REPO_ROOT / "models" / "damage" / "damage_net_sample.pt"
DAMAGE_CHECKPOINT_REAL = REPO_ROOT / "models" / "damage" / "damage_net_real.pt"

DAMAGE_FORM_FIELDS = [
    ("incident_date", "Incident date", "date"),
    ("incident_time_local", "Time (local, HH:MM)", "time"),
    ("phase_of_flight", "Phase of flight", "select", ["Take-off run", "Climb", "Approach", "Landing Roll", "En Route", "Descent"]),
    ("height_agl_ft", "Height above ground (ft)", "number"),
    ("speed_ias_knots", "Indicated airspeed (knots)", "number"),
    ("aircraft_mass_class", "Aircraft mass class", "select", ["1", "2", "3", "4", "5"]),
    ("engine_type", "Engine type", "select", ["Turbofan", "Turboprop", "Turbojet"]),
    ("engine_count", "Number of engines", "number"),
    ("sky_condition", "Sky condition", "select", ["No Cloud", "Some Cloud", "Overcast"]),
    ("precipitation", "Precipitation", "select", ["None", "Rain", "Snow", "Fog"]),
    ("species_scientific_name", "Species", "select", sorted(SPECIES_TAXONOMY.keys())),
    ("wildlife_size", "Wildlife size", "select", ["Small", "Medium", "Large"]),
    ("number_struck", "Number struck", "select", ["1", "2-10", "11-100", "Over 100"]),
    ("state", "State/region code", "text"),
    ("airport_id", "Airport ID", "text"),
]

# ---------------------------------------------------------------------------
# Activity index (Task B) paths/config
# ---------------------------------------------------------------------------
SAMPLE_AIRPORTS_PATH = REPO_ROOT / "data" / "sample" / "airports_sample.geojson"
SAMPLE_SITES_PATH = REPO_ROOT / "data" / "sample" / "trektellen_sites_sample.csv"
SAMPLE_COUNTS_PATH = REPO_ROOT / "data" / "sample" / "trektellen_counts_sample.csv"
REAL_AIRPORTS_PATH = REPO_ROOT / "data" / "raw" / "airports.geojson"
REAL_SITES_PATH = REPO_ROOT / "data" / "raw" / "trektellen_sites.csv"
REAL_COUNTS_PATH = REPO_ROOT / "data" / "raw" / "trektellen_counts.csv"


def _activity_data_available() -> tuple[bool, bool]:
    """Returns (sample_available, real_available)."""
    return (
        SAMPLE_AIRPORTS_PATH.exists() and SAMPLE_SITES_PATH.exists() and SAMPLE_COUNTS_PATH.exists(),
        REAL_AIRPORTS_PATH.exists() and REAL_SITES_PATH.exists() and REAL_COUNTS_PATH.exists(),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    sample_activity, real_activity = _activity_data_available()
    return render_template(
        "index.html",
        synthetic_model_available=Path(SYN_CONFIG.MODEL_PATH).exists(),
        damage_model_available=DAMAGE_CHECKPOINT_SAMPLE.exists() or DAMAGE_CHECKPOINT_REAL.exists(),
        activity_data_available=sample_activity or real_activity,
    )


# --- Mode 1: synthetic demo ---
@app.route("/synthetic", methods=["GET"])
def synthetic_index():
    model_missing = not Path(SYN_CONFIG.MODEL_PATH).exists()
    fields = [
        {"name": name, "low": SYN_CONFIG.FEATURE_RANGES[name][0], "high": SYN_CONFIG.FEATURE_RANGES[name][1], "value": ""}
        for name in SYN_CONFIG.FEATURE_NAMES
    ]
    return render_template("synthetic.html", fields=fields, result=None, error=None, model_missing=model_missing)


@app.route("/synthetic/predict", methods=["POST"])
def synthetic_predict():
    model_missing = not Path(SYN_CONFIG.MODEL_PATH).exists()
    submitted = {name: request.form.get(name, "") for name in SYN_CONFIG.FEATURE_NAMES}
    fields = [
        {"name": name, "low": SYN_CONFIG.FEATURE_RANGES[name][0], "high": SYN_CONFIG.FEATURE_RANGES[name][1], "value": submitted[name]}
        for name in SYN_CONFIG.FEATURE_NAMES
    ]

    if model_missing:
        return render_template("synthetic.html", fields=fields, result=None, error=None, model_missing=True)

    try:
        raw_values = [float(submitted[name]) for name in SYN_CONFIG.FEATURE_NAMES]
    except ValueError:
        return render_template("synthetic.html", fields=fields, result=None,
                                error="All fields must be numbers.", model_missing=False)

    model, feature_mean, feature_std = SYN_INFERENCE.load_model()
    logit, probability = SYN_INFERENCE.predict(raw_values, model, feature_mean, feature_std)
    result = {
        "logit": logit, "probability_pct": probability * 100,
        "classification": "ELEVATED" if probability >= SYN_CONFIG.CLASSIFICATION_THRESHOLD else "LOWER",
    }
    return render_template("synthetic.html", fields=fields, result=result, error=None, model_missing=False)


# --- Mode 2: damage prediction ---
@app.route("/damage", methods=["GET"])
def damage_index():
    checkpoint_path = DAMAGE_CHECKPOINT_SAMPLE if DAMAGE_CHECKPOINT_SAMPLE.exists() else DAMAGE_CHECKPOINT_REAL
    return render_template(
        "damage.html", fields=DAMAGE_FORM_FIELDS, submitted={}, result=None, error=None,
        model_missing=not checkpoint_path.exists(), using_sample_model=checkpoint_path == DAMAGE_CHECKPOINT_SAMPLE,
    )


@app.route("/damage/predict", methods=["POST"])
def damage_predict():
    checkpoint_path = DAMAGE_CHECKPOINT_SAMPLE if DAMAGE_CHECKPOINT_SAMPLE.exists() else DAMAGE_CHECKPOINT_REAL
    submitted = {name: request.form.get(name, "") for name, *_ in DAMAGE_FORM_FIELDS}

    if not checkpoint_path.exists():
        return render_template("damage.html", fields=DAMAGE_FORM_FIELDS, submitted=submitted, result=None,
                                error=None, model_missing=True, using_sample_model=False)

    raw_record = dict(submitted)
    if raw_record.get("species_scientific_name"):
        raw_record["species_common_name"] = SPECIES_TAXONOMY.get(raw_record["species_scientific_name"], (None,))[0]

    try:
        result = predict_damage(raw_record, str(checkpoint_path))
        error = None
    except Exception as exc:  # surfaced to the user rather than a 500 page
        result = None
        error = f"Could not compute a prediction: {exc}"

    return render_template(
        "damage.html", fields=DAMAGE_FORM_FIELDS, submitted=submitted, result=result, error=error,
        model_missing=False, using_sample_model=checkpoint_path == DAMAGE_CHECKPOINT_SAMPLE,
    )


# --- Mode 3: activity explorer ---
@app.route("/activity", methods=["GET"])
def activity_index_page():
    sample_available, real_available = _activity_data_available()
    airports_path = REAL_AIRPORTS_PATH if real_available else SAMPLE_AIRPORTS_PATH
    airports = []
    if sample_available or real_available:
        gdf = load_airports(airports_path)
        airports = gdf[["airport_id", "airport_name"]].to_dict("records")
    return render_template(
        "activity.html", airports=airports, result=None, error=None,
        data_missing=not (sample_available or real_available), using_sample_data=not real_available,
        submitted={},
    )


@app.route("/activity/predict", methods=["POST"])
def activity_predict():
    sample_available, real_available = _activity_data_available()
    using_sample = not real_available
    airports_path = REAL_AIRPORTS_PATH if real_available else SAMPLE_AIRPORTS_PATH
    sites_path = REAL_SITES_PATH if real_available else SAMPLE_SITES_PATH
    counts_path = REAL_COUNTS_PATH if real_available else SAMPLE_COUNTS_PATH

    airports = []
    if sample_available or real_available:
        gdf = load_airports(airports_path)
        airports = gdf[["airport_id", "airport_name"]].to_dict("records")

    submitted = {
        "airport_id": request.form.get("airport_id", ""),
        "date": request.form.get("date", ""),
        "time": request.form.get("time", "12:00"),
        "radius_km": request.form.get("radius_km", "100"),
        "lookback": request.form.get("lookback", "7d"),
    }

    if not (sample_available or real_available):
        return render_template("activity.html", airports=airports, result=None, error=None,
                                data_missing=True, using_sample_data=using_sample, submitted=submitted)

    try:
        query_timestamp = f"{submitted['date']}T{submitted['time']}:00Z"
        result = calculate_activity_for_airport(
            airport_id=submitted["airport_id"], query_timestamp=query_timestamp,
            airports_path=airports_path, sites_path=sites_path, counts_path=counts_path,
            max_coverage_radius_km=float(submitted["radius_km"]),
        )
        result["lookback_label"] = submitted["lookback"]
        result["lookback_total"] = result["features"].get(f"total_birds_previous_{submitted['lookback']}")

        # For the map: every Trektellen site within the selected radius.
        from birdstrikegeo.data.ingest_trektellen import load_sites
        from birdstrikegeo.geo.spatial_join import sites_within_radius

        airport_row = next(a for a in airports if a["airport_id"] == submitted["airport_id"])
        gdf = load_airports(airports_path)
        airport_full = gdf[gdf["airport_id"] == submitted["airport_id"]].iloc[0]
        sites_df = load_sites(sites_path)
        nearby = sites_within_radius(airport_full["longitude"], airport_full["latitude"], sites_df, float(submitted["radius_km"]))
        result["airport_lat"] = float(airport_full["latitude"])
        result["airport_lon"] = float(airport_full["longitude"])
        result["nearby_sites"] = [
            {"site_id": r["site_id"], "site_name": r.get("site_name", r["site_id"]),
             "latitude": float(r["latitude"]), "longitude": float(r["longitude"]),
             "distance_km": round(float(r["distance_km"]), 1)}
            for _, r in nearby.iterrows()
        ]
        error = None
    except Exception as exc:
        result = None
        error = f"Could not compute an activity index: {exc}"

    return render_template("activity.html", airports=airports, result=result, error=error,
                            data_missing=False, using_sample_data=using_sample, submitted=submitted)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
