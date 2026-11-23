"""
app.py
------
The main local website for BirdStrikeGeo, offering FIVE modes so it's
always clear which system produced a given number. As of the Phase 1/2
audit and consolidation (see DATA_AUDIT_REPORT.md, LEGACY_SYSTEMS.md),
modes 4 and 5 are the PRIMARY, current models; modes 2 and 3 are
preserved, working, but LEGACY — kept for their own architecture/formula
demonstration value, the same way /synthetic has always been preserved,
not because they're wrong, but because modes 4/5 supersede them for
actually answering "what's the current best estimate."

1. /synthetic - the original, unrelated educational neural-network demo
   preserved in synthetic_demo/ (entirely synthetic, hand-designed data).
2. /damage - LEGACY: estimated probability of aircraft damage given a
   reported strike, PyTorch NN, trained on ALL aircraft types. See
   LEGACY_SYSTEMS.md for why mode 4 supersedes this.
3. /activity - LEGACY: airport-period wildlife activity index, a
   transparent formula over Trektellen observations. See
   LEGACY_SYSTEMS.md for why mode 5 supersedes this.
4. /ga-damage - PRIMARY: calibrated CatBoost conditional-damage model,
   confirmed GA (business/private fixed-wing) population only, stricter
   leakage policy than mode 2 (see ga/feature_policy.py).
5. /hazard - PRIMARY: relative bird-hazard index report, joining
   FAA-wide species severity risk against local Trektellen activity.

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
from collections import Counter
from pathlib import Path

import markdown
from flask import Flask, render_template, request, send_from_directory

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from birdstrikegeo.dashboard.actions import generate_actions  # noqa: E402
from birdstrikegeo.dashboard.activity import aggregate_activity_for_taxon, monthly_activity_for_taxon  # noqa: E402
from birdstrikegeo.dashboard.citations import CITATIONS  # noqa: E402
from birdstrikegeo.dashboard.data_access import (  # noqa: E402
    load_eastern_shore_airport_roster,
    load_ga_strike_rows,
    load_ops_rows,
    load_trektellen_effort_rows,
)
from birdstrikegeo.dashboard.evidence import assemble_evidence  # noqa: E402
from birdstrikegeo.dashboard.exposure import compute_strike_rate  # noqa: E402
from birdstrikegeo.dashboard.query import (  # noqa: E402
    COHORT_LABELS,
    filter_by_cohort,
    filter_by_scope,
    query_taxon_rows,
    recommend_default_cohort,
    species_id_share,
)
from birdstrikegeo.dashboard.taxonomy import classify  # noqa: E402
from birdstrikegeo.data.ingest_airports import load_airports  # noqa: E402
from birdstrikegeo.features.taxonomy import SPECIES_TAXONOMY  # noqa: E402
from birdstrikegeo.ga.features import build_operational_core_features  # noqa: E402
from birdstrikegeo.ga.inference import load_bundle, predict_scenario  # noqa: E402
from birdstrikegeo.ga.presets import ScenarioValidationError, get_preset, list_presets, validate_scenario  # noqa: E402
from birdstrikegeo.inference.calculate_activity import calculate_activity_for_airport  # noqa: E402
from birdstrikegeo.inference.predict_damage import predict_damage  # noqa: E402

app = Flask(__name__)

# ---------------------------------------------------------------------------
# BirdStrikeNet dashboard (Eastern Shore MD) paths/config
# ---------------------------------------------------------------------------
DASHBOARD_DB_PATH = REPO_ROOT / "data" / "birdstrikegeo.db"
DASHBOARD_AIRPORTS_CONFIG_PATH = REPO_ROOT / "configs" / "eastern_shore_airports.yaml"
DASHBOARD_TREKTELLEN_CSV_PATH = REPO_ROOT / "data" / "raw" / "trektellen_2025_year_totals.csv"
DASHBOARD_OPERATIONS_DIR = REPO_ROOT / "data" / "raw" / "operations"
DASHBOARD_SAMPLE_STRIKES_PATH = REPO_ROOT / "data" / "sample" / "faa_strikes_sample.csv"

TREKTELLEN_MONTH_COLUMNS = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov"]
MONTH_CODE_BY_NUMBER = {3: "mar", 4: "apr", 5: "may", 6: "jun", 7: "jul", 8: "aug", 9: "sep", 10: "oct", 11: "nov"}
MONTH_ABBR = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
              7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
SEASON_MONTHS = {"spring": [3, 4, 5], "summer": [6, 7, 8], "fall": [9, 10, 11], "winter": [12, 1, 2]}
PERIOD_OPTIONS = (
    [("all", "All months"), ("spring", "Spring (Mar-May)"), ("summer", "Summer (Jun-Aug)"),
     ("fall", "Fall (Sep-Nov)"), ("winter", "Winter (Dec-Feb)")]
    + [(str(m), MONTH_ABBR[m]) for m in range(1, 13)]
)

_dashboard_cache: dict = {}


def _dashboard_data_available() -> bool:
    return DASHBOARD_DB_PATH.exists() and DASHBOARD_AIRPORTS_CONFIG_PATH.exists()


def _load_dashboard_data() -> dict:
    """
    Loads once per process (this is a local single-user Flask dev
    server; the underlying files don't change while it's running) and
    caches in-memory. Returns the same dict every call.
    """
    if _dashboard_cache:
        return _dashboard_cache

    roster = load_eastern_shore_airport_roster(DASHBOARD_AIRPORTS_CONFIG_PATH)
    airport_ids = [a["faa_lid"] for a in roster]
    region_rows = load_ga_strike_rows(DASHBOARD_DB_PATH, airport_ids=airport_ids)
    national_rows = load_ga_strike_rows(DASHBOARD_DB_PATH, airport_ids=None)
    effort_rows = (
        load_trektellen_effort_rows(DASHBOARD_TREKTELLEN_CSV_PATH, TREKTELLEN_MONTH_COLUMNS)
        if DASHBOARD_TREKTELLEN_CSV_PATH.exists() else []
    )
    atads_rows, annual_rows = load_ops_rows(DASHBOARD_OPERATIONS_DIR)
    years_present = [r["incident_year"] for r in region_rows if r.get("incident_year")]

    _dashboard_cache.update({
        "roster": roster, "airport_ids": airport_ids, "region_rows": region_rows,
        "national_rows": national_rows, "effort_rows": effort_rows,
        "atads_rows": atads_rows, "annual_rows": annual_rows,
        "default_cohort": recommend_default_cohort(region_rows),
        "year_range_bounds": (min(years_present), max(years_present)) if years_present else (1990, 2026),
    })
    return _dashboard_cache


def _parse_period(period: str) -> list[int] | None:
    if period == "all":
        return None
    if period in SEASON_MONTHS:
        return SEASON_MONTHS[period]
    try:
        month = int(period)
    except ValueError:
        return None
    return [month] if 1 <= month <= 12 else None


def _period_label(period: str) -> str:
    return dict(PERIOD_OPTIONS).get(period, "All months")


def _parse_filters(data: dict) -> dict:
    airport_param = request.args.get("airport", "region")
    period_param = request.args.get("period", "all")
    cohort_param = request.args.get("cohort", "")
    grouped = request.args.get("grouped") == "1"
    include_mammals = request.args.get("include_mammals") == "1"

    lo_bound, hi_bound = data["year_range_bounds"]
    try:
        year_from = int(request.args.get("year_from", lo_bound))
    except ValueError:
        year_from = lo_bound
    try:
        year_to = int(request.args.get("year_to", hi_bound))
    except ValueError:
        year_to = hi_bound

    return {
        "airport": None if airport_param in ("region", "") else airport_param,
        "airport_param": airport_param,
        "period": period_param,
        "months": _parse_period(period_param),
        "year_range": (year_from, year_to),
        "cohort": cohort_param if cohort_param in COHORT_LABELS else data["default_cohort"],
        "grouped": grouped,
        "include_mammals": include_mammals,
    }


def _filters_as_query_args(filters: dict) -> dict:
    return {
        "airport": filters["airport_param"], "period": filters["period"],
        "year_from": filters["year_range"][0], "year_to": filters["year_range"][1],
        "cohort": filters["cohort"], "grouped": "1" if filters["grouped"] else "",
        "include_mammals": "1" if filters["include_mammals"] else "",
    }


def _airport_name_lookup(roster: list[dict]) -> dict:
    return {a["faa_lid"]: a["name"] for a in roster}


def _trektellen_months_for(months: list[int] | None) -> list[str]:
    scope = months if months is not None else list(range(1, 13))
    return [MONTH_CODE_BY_NUMBER[m] for m in scope if m in MONTH_CODE_BY_NUMBER]

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
# GA conditional-damage model (Task 4, CatBoost) paths/config
# ---------------------------------------------------------------------------
GA_MODELS_DIR = REPO_ROOT / "models" / "ga"
GA_FEATURE_MODE = "operational_core"


def _ga_model_available() -> bool:
    return (GA_MODELS_DIR / f"ga_catboost_{GA_FEATURE_MODE}.cbm").exists() and \
        (GA_MODELS_DIR / f"ga_model_bundle_{GA_FEATURE_MODE}.pkl").exists()

# ---------------------------------------------------------------------------
# Eastern Shore hazard report (Task 5) paths/config
# ---------------------------------------------------------------------------
HAZARD_REPORTS_DIR = REPO_ROOT / "reports" / "latest"
HAZARD_REPORT_PATH = HAZARD_REPORTS_DIR / "eastern_shore_birdstrike_risk_analysis.md"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def dashboard_index():
    if not _dashboard_data_available():
        return render_template("dashboard_missing.html", active_mode="dashboard")

    data = _load_dashboard_data()
    filters = _parse_filters(data)
    trektellen_months = _trektellen_months_for(filters["months"])

    evidence = assemble_evidence(
        data["region_rows"], data["national_rows"], airport=filters["airport"],
        months=filters["months"], year_range=filters["year_range"], cohort=filters["cohort"],
        grouped=filters["grouped"], include_mammals=filters["include_mammals"],
    )

    for taxon_row in evidence["primary"]["taxon_rows"]:
        taxon_row["fbbo"] = aggregate_activity_for_taxon(
            data["effort_rows"], taxon_row["taxon_key"], filters["grouped"], trektellen_months,
        )

    primary_airports = [filters["airport"]] if filters["airport"] else None
    scoped_rows = filter_by_scope(
        filter_by_cohort(data["region_rows"], filters["cohort"]),
        airports=primary_airports, months=filters["months"], year_range=filters["year_range"],
    )
    total_reports = len(scoped_rows)
    known = [r for r in scoped_rows if r.get("indicated_damage") in (0, 1)]
    n_damaging = sum(1 for r in known if r["indicated_damage"] == 1)

    monthly_counts = Counter(r["incident_month"] for r in scoped_rows if r.get("incident_month"))
    max_month_n = max(monthly_counts.values(), default=0)
    month_chart = [
        {"month": m, "label": MONTH_ABBR[m], "n": monthly_counts.get(m, 0),
         "low_n": 0 < monthly_counts.get(m, 0) < 3,
         "height_pct": round(100 * monthly_counts.get(m, 0) / max_month_n) if max_month_n else 0}
        for m in range(1, 13)
    ]

    exposure = None
    if filters["airport"]:
        exposure = compute_strike_rate(
            total_reports, filters["airport"], filters["months"], filters["year_range"],
            data["atads_rows"], data["annual_rows"],
        )

    actions = generate_actions(evidence["primary"]["taxon_rows"], scoped_rows)

    airports_with_data = sorted({r["airport_id"] for r in data["region_rows"]})
    airport_names = _airport_name_lookup(data["roster"])

    return render_template(
        "dashboard.html", active_mode="dashboard", filters=filters, evidence=evidence,
        total_reports=total_reports, species_id_share_value=species_id_share(scoped_rows),
        pct_known_damage=(len(known) / total_reports) if total_reports else None,
        n_damaging=n_damaging, month_chart=month_chart, exposure=exposure, actions=actions,
        airports_with_data=airports_with_data, airport_names=airport_names,
        cohort_labels=COHORT_LABELS, period_options=PERIOD_OPTIONS,
        year_range_bounds=data["year_range_bounds"], query_args=_filters_as_query_args(filters),
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


# --- Mode 4: GA conditional-damage model (CatBoost) ---
@app.route("/ga-damage", methods=["GET"])
def ga_damage_index():
    presets = [(name, get_preset(name)["description"]) for name in list_presets()]
    return render_template(
        "ga_damage.html", presets=presets, submitted_preset="", preset_description=None,
        result=None, error=None, model_missing=not _ga_model_available(),
    )


@app.route("/ga-damage/predict", methods=["POST"])
def ga_damage_predict():
    presets = [(name, get_preset(name)["description"]) for name in list_presets()]
    submitted_preset = request.form.get("preset", "")

    if not _ga_model_available():
        return render_template("ga_damage.html", presets=presets, submitted_preset=submitted_preset,
                                preset_description=None, result=None, error=None, model_missing=True)

    try:
        scenario = get_preset(submitted_preset)
        preset_description = scenario.pop("description")
        bundle = load_bundle(GA_MODELS_DIR, feature_mode=GA_FEATURE_MODE)
        validate_scenario(scenario, bundle.seen_categories, bundle.numeric_ranges)
        result = predict_scenario(scenario, bundle, build_operational_core_features)
        error = None
    except ScenarioValidationError as exc:
        result, preset_description, error = None, None, str(exc)
    except Exception as exc:  # surfaced to the user rather than a 500 page
        result, preset_description, error = None, None, f"Could not compute a prediction: {exc}"

    return render_template("ga_damage.html", presets=presets, submitted_preset=submitted_preset,
                            preset_description=preset_description, result=result, error=error, model_missing=False)


# --- Research models overview (moved off the landing page) ---
@app.route("/research-models", methods=["GET"])
def research_models_index():
    sample_activity, real_activity = _activity_data_available()
    return render_template(
        "research_models.html", active_mode="research_models",
        synthetic_model_available=Path(SYN_CONFIG.MODEL_PATH).exists(),
        damage_model_available=DAMAGE_CHECKPOINT_SAMPLE.exists() or DAMAGE_CHECKPOINT_REAL.exists(),
        activity_data_available=sample_activity or real_activity,
        ga_damage_model_available=_ga_model_available(),
    )


# --- Hazard index (relative screening measure) ---
@app.route("/methods/hazard-index", methods=["GET"])
def hazard_index():
    if not HAZARD_REPORT_PATH.exists():
        return render_template("methods_hazard_index.html", active_mode="hazard", report_missing=True, report_html=None)

    report_text = HAZARD_REPORT_PATH.read_text()
    # Rewrite the report's relative image reference to go through
    # /report-assets/, since reports/latest/ isn't Flask's static folder.
    report_text = report_text.replace(
        "(eastern_shore_risk_scatter.png)", "(/report-assets/eastern_shore_risk_scatter.png)"
    )
    report_html = markdown.markdown(report_text, extensions=["tables"])
    return render_template("methods_hazard_index.html", active_mode="hazard", report_missing=False, report_html=report_html)


@app.route("/report-assets/<path:filename>")
def report_assets(filename):
    return send_from_directory(HAZARD_REPORTS_DIR, filename)


# --- BirdStrikeNet dashboard: species drill-down ---
@app.route("/species/<taxon>", methods=["GET"])
def species_detail(taxon):
    if not _dashboard_data_available():
        return render_template("dashboard_missing.html", active_mode="dashboard")

    data = _load_dashboard_data()
    filters = _parse_filters(data)
    primary_airports = [filters["airport"]] if filters["airport"] else None

    # The drill-down's 12-month chart is always full-year, regardless of
    # the dashboard's period filter - only airport/cohort/year carry over.
    year_scoped_rows = filter_by_scope(
        filter_by_cohort(data["region_rows"], filters["cohort"]),
        airports=primary_airports, year_range=filters["year_range"],
    )

    def _matches_taxon(row: dict) -> bool:
        info = classify(row.get("species"))
        return (info.group == taxon) if filters["grouped"] else (row.get("species") == taxon)

    matching_rows = [r for r in year_scoped_rows if _matches_taxon(r)]
    taxon_label = info_label = taxon
    if matching_rows:
        info = classify(matching_rows[0]["species"])
        taxon_label = info.group_label if filters["grouped"] else matching_rows[0]["species"]

    monthly_report_counts = Counter(r["incident_month"] for r in matching_rows if r.get("incident_month"))
    all_months_codes = _trektellen_months_for(None)
    fbbo_by_month = {
        m["month"]: m for m in monthly_activity_for_taxon(data["effort_rows"], taxon, filters["grouped"], all_months_codes)
    }
    month_number_by_code = {v: k for k, v in MONTH_CODE_BY_NUMBER.items()}
    chart = []
    for month_num in range(1, 13):
        code = MONTH_CODE_BY_NUMBER.get(month_num)
        fbbo_row = fbbo_by_month.get(code) if code else {"status": "not_monitored", "count": None}
        chart.append({
            "month": month_num, "label": MONTH_ABBR[month_num],
            "n_reports": monthly_report_counts.get(month_num, 0),
            "fbbo_status": fbbo_row["status"] if fbbo_row else "not_monitored",
            "fbbo_count": fbbo_row.get("count") if fbbo_row else None,
        })

    known = [r for r in matching_rows if r.get("indicated_damage") in (0, 1)]
    damage_level_counts = Counter(r.get("damage_level") for r in known if r.get("damage_level"))

    n_detail_min = 10
    phase_counts = (Counter(r.get("phase_of_flight") for r in matching_rows if r.get("phase_of_flight"))
                    if len(matching_rows) >= n_detail_min else None)

    def _height_band(height) -> str | None:
        if height is None:
            return None
        if height <= 0:
            return "Ground level"
        if height <= 500:
            return "1-500 ft"
        if height <= 1500:
            return "501-1,500 ft"
        return "Over 1,500 ft"

    height_counts = None
    if len(matching_rows) >= n_detail_min:
        height_counts = Counter(_height_band(r.get("height")) for r in matching_rows if r.get("height") is not None)

    report_list = sorted(
        ({"date": r.get("incident_date"), "airport": r.get("airport_id"),
          "phase": r.get("phase_of_flight"), "damage": "Yes" if r.get("indicated_damage") == 1
          else ("No" if r.get("indicated_damage") == 0 else "Unknown")} for r in matching_rows),
        key=lambda r: r["date"] or "", reverse=True,
    )

    national_taxon_rows = query_taxon_rows(
        data["national_rows"], year_range=filters["year_range"], cohort=filters["cohort"],
        grouped=filters["grouped"], include_mammals=True,
    )
    national_match = next((r for r in national_taxon_rows if r["taxon_key"] == taxon), None)

    return render_template(
        "species_detail.html", active_mode="dashboard", filters=filters, taxon=taxon,
        taxon_label=taxon_label, chart=chart, n_reports=len(matching_rows), n_known=len(known),
        n_damaging=sum(1 for r in known if r["indicated_damage"] == 1), damage_level_counts=damage_level_counts,
        phase_counts=phase_counts, height_counts=height_counts, report_list=report_list,
        national_match=national_match, query_args=_filters_as_query_args(filters),
    )


# --- Methods ---
GA_POPULATION_SUMMARY_PATH = REPO_ROOT / "reports" / "latest" / "ga_population_summary.json"


@app.route("/methods", methods=["GET"])
def methods_index():
    import json

    data = _load_dashboard_data() if _dashboard_data_available() else None
    cohort_counts = {}
    if data is not None:
        for key in COHORT_LABELS:
            cohort_counts[key] = len(filter_by_cohort(data["region_rows"], key))

    population_summary = None
    if GA_POPULATION_SUMMARY_PATH.exists():
        population_summary = json.loads(GA_POPULATION_SUMMARY_PATH.read_text())

    ops_available = bool(data and (data["atads_rows"] or data["annual_rows"]))

    return render_template(
        "methods.html", active_mode="methods", data_available=data is not None,
        data=data, cohort_labels=COHORT_LABELS, cohort_counts=cohort_counts,
        citations=CITATIONS, population_summary=population_summary, ops_available=ops_available,
    )


if __name__ == "__main__":
    app.run(debug=True, port=5001)
