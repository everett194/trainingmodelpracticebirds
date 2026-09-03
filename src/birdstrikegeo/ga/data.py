"""
birdstrikegeo.ga.data
----------------------
Raw-workbook caching, GA population filtering, and damage-target
construction for the GA conditional-damage milestone.

This deliberately reads the FAA export's RAW column names directly
(OPERATOR, AC_CLASS, DAMAGE_LEVEL, ...) rather than going through
birdstrikegeo.data.ingest_faa's narrower canonical schema, because the
GA population filter and several requested features (engine positions,
light condition, bird-warning flag) need raw fields that schema doesn't
carry. Column-name handling here is intentionally direct, not
alias-resolved - this module is scoped to one specific, inspected export
(data/raw/faa_wildlife_strikes.xlsx), not to "any FAA export vintage".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

RAW_XLSX_CANDIDATES = ("faa_wildlife_strikes.xlsx", "Public.xlsx")


def load_raw_workbook(xlsx_path: str | Path, parquet_cache_path: str | Path, force_reload: bool = False) -> pd.DataFrame:
    """
    Reads the full raw FAA workbook (~350k rows x 102 columns, all as
    strings - dates/numbers are coerced downstream, not here, so the
    cache is a faithful copy of what's in the sheet) and caches it as
    Parquet so repeated runs don't re-parse a 150MB Excel file.
    """
    parquet_cache_path = Path(parquet_cache_path)
    xlsx_path = Path(xlsx_path)

    if parquet_cache_path.exists() and not force_reload:
        return pd.read_parquet(parquet_cache_path)

    if not xlsx_path.exists():
        raise FileNotFoundError(
            f"Raw FAA workbook not found at {xlsx_path}. See DATA_DOWNLOAD_GUIDE.md."
        )

    raw = pd.read_excel(xlsx_path, dtype=str, engine="openpyxl")
    parquet_cache_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(parquet_cache_path, index=False)
    return raw


def _norm(series: pd.Series) -> pd.Series:
    """Trim whitespace and upper-case a raw string column. Leaves NaN as NaN."""
    return series.astype("string").str.strip().str.upper()


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# GA population filter definitions - each a (label, predicate) pair over
# a dataframe already normalized by normalize_raw_columns(). Every filter
# is evaluated (not just the principal one) so the milestone's requested
# population counts can all be reported from one pass over the data.
POPULATION_DEFINITIONS: dict[str, str] = {
    "confirmed_business_or_private": "OPERATOR in {BUSINESS, PRIVATELY OWNED} (all aircraft classes)",
    "confirmed_business_or_private_fixed_wing": "OPERATOR in {BUSINESS, PRIVATELY OWNED} AND AC_CLASS == 'A' (airplane) - PRINCIPAL population",
    "light_fixed_wing_ga": "principal population AND AC_MASS in {1, 2} (<= 5,700 kg)",
    "light_piston_ga": "light fixed-wing GA AND TYPE_ENG == 'A' (reciprocating/piston)",
}


def normalize_raw_columns(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy with the handful of raw columns the GA population
    filter and feature set depend on normalized (trimmed/upper-cased
    strings, coerced numerics). Does not touch or rename other columns -
    downstream code still reads original FAA column names.
    """
    df = raw.copy()

    string_cols = [
        "OPERATOR", "AC_CLASS", "TYPE_ENG", "AC_MASS", "PHASE_OF_FLIGHT", "SKY", "PRECIPITATION",
        "TIME_OF_DAY", "WARNED", "STATE", "FAAREGION", "AIRPORT_ID", "ENG_1_POS", "ENG_2_POS",
        "ENG_3_POS", "ENG_4_POS", "INDICATED_DAMAGE", "DAMAGE_LEVEL", "AIRCRAFT",
    ]
    for col in string_cols:
        if col in df.columns:
            df[col] = _norm(df[col])

    numeric_cols = ["HEIGHT", "SPEED", "NUM_ENGS", "LATITUDE", "LONGITUDE", "INCIDENT_YEAR", "INCIDENT_MONTH"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    if "INCIDENT_DATE" in df.columns:
        df["INCIDENT_DATE"] = pd.to_datetime(df["INCIDENT_DATE"], errors="coerce")

    return df


@dataclass
class PopulationCounts:
    counts: dict[str, int] = field(default_factory=dict)

    def as_report_lines(self) -> list[str]:
        return [f"{name}: {n:,}  ({POPULATION_DEFINITIONS.get(name, '')})" for name, n in self.counts.items()]


def compute_population_counts(df: pd.DataFrame) -> tuple[PopulationCounts, pd.Series]:
    """
    Returns (counts-for-every-definition, boolean mask for the PRINCIPAL
    population). Every count is computed from the actual data, never
    hardcoded, so a data refresh naturally changes these numbers.
    """
    operator = df.get("OPERATOR", pd.Series(dtype="string"))
    ac_class = df.get("AC_CLASS", pd.Series(dtype="string"))
    ac_mass = df.get("AC_MASS", pd.Series(dtype="string"))
    type_eng = df.get("TYPE_ENG", pd.Series(dtype="string"))

    # This export vintage spells OPERATOR out in full ("BUSINESS",
    # "PRIVATELY OWNED") rather than using FAA's documented 3-letter ICAO
    # codes (BUS/PVT) - verified against the actual data, not assumed.
    is_biz_or_pvt = operator.isin(["BUSINESS", "PRIVATELY OWNED"])
    is_fixed_wing = ac_class == "A"
    is_light_mass = ac_mass.isin(["1", "2"])
    is_piston = type_eng == "A"

    masks = {
        "confirmed_business_or_private": is_biz_or_pvt,
        "confirmed_business_or_private_fixed_wing": is_biz_or_pvt & is_fixed_wing,
        "light_fixed_wing_ga": is_biz_or_pvt & is_fixed_wing & is_light_mass,
        "light_piston_ga": is_biz_or_pvt & is_fixed_wing & is_light_mass & is_piston,
    }
    counts = PopulationCounts(counts={name: int(mask.sum()) for name, mask in masks.items()})
    principal_mask = masks["confirmed_business_or_private_fixed_wing"]
    return counts, principal_mask


@dataclass
class TargetReport:
    n_positive: int
    n_negative: int
    n_excluded_unknown: int
    n_inconsistent: int
    positive_rate: float

    def as_dict(self) -> dict:
        return {
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_excluded_unknown": self.n_excluded_unknown,
            "n_inconsistent_indicated_vs_level": self.n_inconsistent,
            "positive_class_rate": self.positive_rate,
        }


# INDICATED_DAMAGE is FAA's own Y/N summary flag. DAMAGE_LEVEL is the
# ICAO damage-severity code (blank/N/M/M?/S/D - see read_me.xls "DAMAGE"
# sheet). A record is affirmatively damaged only if INDICATED_DAMAGE=Y;
# affirmatively undamaged only if INDICATED_DAMAGE=N. DAMAGE_LEVEL is used
# only to flag INTERNAL INCONSISTENCY (e.g. indicated=N but level=S), not
# to override INDICATED_DAMAGE - two disagreeing FAA fields are ambiguous
# evidence, not a tiebreak, so those rows are excluded, not guessed.
_DAMAGE_LEVEL_IMPLIES_DAMAGE = {"M", "M?", "S", "D"}
_DAMAGE_LEVEL_IMPLIES_NO_DAMAGE = {"N", ""}


def build_damage_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `damage_binary` (nullable Int64: 1 / 0 / <NA>) and
    `damage_target_status` to a copy of df. Only affirmative
    INDICATED_DAMAGE values decide the label; unknown/blank is left
    missing, never coerced to 0. DAMAGE_LEVEL is cross-checked and
    disagreements are reported (see TargetReport) but do not change the
    label - see module docstring above the constant definitions.
    """
    df = df.copy()
    indicated = df["INDICATED_DAMAGE"].astype("string")
    level = df.get("DAMAGE_LEVEL", pd.Series("", index=df.index, dtype="string")).astype("string").fillna("")

    # This export vintage codes INDICATED_DAMAGE as "1"/"0", not "Y"/"N" -
    # verified against the actual data. Accept both forms defensively.
    is_yes = indicated.isin(["1", "Y"])
    is_no = indicated.isin(["0", "N"])

    status = pd.Series("missing_or_ambiguous", index=df.index, dtype="object")
    status[is_yes] = "explicit_positive"
    status[is_no] = "explicit_negative"

    damage_binary = pd.Series(pd.NA, index=df.index, dtype="Int64")
    damage_binary[is_yes] = 1
    damage_binary[is_no] = 0

    level_implies_damage = level.isin(_DAMAGE_LEVEL_IMPLIES_DAMAGE)
    level_implies_no_damage = level.isin(_DAMAGE_LEVEL_IMPLIES_NO_DAMAGE)
    inconsistent = (is_yes & level_implies_no_damage) | (is_no & level_implies_damage)

    df["damage_binary"] = damage_binary
    df["damage_target_status"] = status
    df["damage_level_inconsistent"] = inconsistent

    return df


def summarize_target(df_with_target: pd.DataFrame) -> TargetReport:
    status = df_with_target["damage_target_status"]
    n_pos = int((status == "explicit_positive").sum())
    n_neg = int((status == "explicit_negative").sum())
    n_excluded = int((status == "missing_or_ambiguous").sum())
    n_inconsistent = int(df_with_target["damage_level_inconsistent"].sum())
    total_labeled = n_pos + n_neg
    positive_rate = n_pos / total_labeled if total_labeled else float("nan")
    return TargetReport(
        n_positive=n_pos, n_negative=n_neg, n_excluded_unknown=n_excluded,
        n_inconsistent=n_inconsistent, positive_rate=positive_rate,
    )
