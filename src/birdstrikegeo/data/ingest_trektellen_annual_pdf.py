"""
data/ingest_trektellen_annual_pdf.py
----------------------------------------
Parses a Trektellen-derived "annual banding totals" PDF (e.g.
`data/reference/trektellen_fbbo_annual_banding_totals_2016-2025.pdf`) into the
same row_type-based long/wide convention `hazard.trektellen_season`
already uses for the single-season monthly export — just with YEAR
period columns instead of MONTH period columns, so the two sources can
be handled by generalized, shared downstream code.

Why a PDF and not a CSV: this is exactly what Trektellen/the observer
exported for a multi-year annual summary; this project never scrapes
Trektellen (see DATA_DOWNLOAD_GUIDE.md) so the PDF is used as supplied,
parsed here rather than requesting the observer re-export as CSV.

Critical honesty rule carried over from trektellen_season.py: a year
with NO reported observation hours must never be treated as a real
zero-activity year. This PDF's own caption states "Years 2016-2021 are
retained as reported; the active series begins in 2022" and its
Observation-hours row is blank for 2016-2021 -- this module marks those
years `monitored=False` and callers must not compute a rate (e.g.
count/hours) for them, exactly as `effort_normalize_monthly` already
refuses to for a zero-observation-hours month.

Requires the `poppler` `pdftotext` binary on PATH (brew install poppler
/ apt-get install poppler-utils) -- this module shells out to it rather
than adding a new pure-Python PDF dependency, since `pdftotext -layout`
already produces exactly the column-aligned text this parser expects.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

_SPECIES_ROW_RE = re.compile(
    r"^\s*\d+\s+(?P<species>.+?)\s+"
    r"(?P<years>(?:\d+\s+){9}\d+)\s+"
    r"(?P<total>\d+)\s+"
    r"(?P<average>[\d.]+)\s+"
    r"(?P<newly_ringed>\d+)\s+"
    r"(?P<retraps>\d+)\s+"
    r"(?P<year_max_count>\d+)\s*\((?P<year_max_year>\d{4})\)\s*$"
)

_TOTALS_ROW_RE = re.compile(
    r"^\s*Totals\s+"
    r"(?P<years>(?:\d+\s+){9}\d+)\s+"
    r"(?P<total>\d+)\s+"
    r"(?P<average>[\d.]+)\s+"
    r"(?P<newly_ringed>\d+)\s+"
    r"(?P<retraps>\d+)\s*$"
)

_OBS_HOURS_ROW_RE = re.compile(r"^\s*Observation hours\s+(?P<rest>.+)$")

_HHMM_RE = re.compile(r"[\d,]+:\d{2}")


def _parse_hhmm(token: str) -> float:
    """"1,018:12" -> 1018.2 (hours, decimal). Comma thousands-separators stripped."""
    token = token.replace(",", "")
    hours_str, minutes_str = token.split(":")
    return int(hours_str) + int(minutes_str) / 60


def extract_text(pdf_path: str | Path) -> str:
    if shutil.which("pdftotext") is None:
        raise RuntimeError(
            "pdftotext not found on PATH. Install poppler (`brew install poppler` on macOS, "
            "`apt-get install poppler-utils` on Linux) to parse Trektellen annual PDF exports."
        )
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"], capture_output=True, text=True, check=True
    )
    return result.stdout


def parse_annual_pdf_text(text: str, source_file: str) -> dict:
    """
    Returns {"species_rows": DataFrame, "totals_row": dict, "observation_hours": dict, "years": [int, ...]}.

    species_rows columns: species, year_2016..year_2025 (raw wide, for
    parity with the row_type convention), total, average, newly_ringed,
    retraps, year_max_count, year_max_year, source_file.
    """
    years = list(range(2016, 2026))  # this PDF's fixed column set; see module docstring
    species_records = []
    totals_record = None
    obs_hours_by_year: dict[int, float] = {}

    for line in text.splitlines():
        species_match = _SPECIES_ROW_RE.match(line)
        if species_match:
            year_values = [int(v) for v in species_match.group("years").split()]
            species_records.append(
                {
                    "species": species_match.group("species").strip(),
                    **{f"year_{y}": v for y, v in zip(years, year_values)},
                    "total": int(species_match.group("total")),
                    "average": float(species_match.group("average")),
                    "newly_ringed": int(species_match.group("newly_ringed")),
                    "retraps": int(species_match.group("retraps")),
                    "year_max_count": int(species_match.group("year_max_count")),
                    "year_max_year": int(species_match.group("year_max_year")),
                    "source_file": source_file,
                }
            )
            continue

        totals_match = _TOTALS_ROW_RE.match(line)
        if totals_match:
            year_values = [int(v) for v in totals_match.group("years").split()]
            totals_record = {
                **{f"year_{y}": v for y, v in zip(years, year_values)},
                "total": int(totals_match.group("total")),
                "average": float(totals_match.group("average")),
                "newly_ringed": int(totals_match.group("newly_ringed")),
                "retraps": int(totals_match.group("retraps")),
                "source_file": source_file,
            }
            continue

        obs_match = _OBS_HOURS_ROW_RE.match(line)
        if obs_match:
            tokens = _HHMM_RE.findall(obs_match.group("rest"))
            # This PDF reports observation hours only for the years with
            # nonzero effort (2022-2025 in the current export) -- the
            # LAST two tokens are always Total/Average, the rest are
            # per-year values for however many years actually have data.
            # We don't assume a fixed count; we align from the right.
            if len(tokens) >= 2:
                per_year_tokens = tokens[:-2]
                # Align to the most recent len(per_year_tokens) years,
                # since blank/unmonitored years are omitted entirely
                # from this row (not zero-padded) in the source PDF.
                monitored_years = years[-len(per_year_tokens):] if per_year_tokens else []
                for y, tok in zip(monitored_years, per_year_tokens):
                    obs_hours_by_year[y] = _parse_hhmm(tok)

    return {
        "species_rows": pd.DataFrame(species_records),
        "totals_row": totals_record,
        "observation_hours_by_year": obs_hours_by_year,
        "years": years,
        "source_file": source_file,
    }


def to_long_format(parsed: dict) -> pd.DataFrame:
    """
    Wide (species_rows, one row per species with year_2016..year_2025
    columns) -> long (one row per species x year), with `monitored`
    (whether that year has reported observation hours) and
    `count_per_hour` (null when unmonitored -- never a guessed rate).
    """
    df = parsed["species_rows"]
    obs_hours = parsed["observation_hours_by_year"]
    years = parsed["years"]

    rows = []
    for _, row in df.iterrows():
        for year in years:
            count = row[f"year_{year}"]
            hours = obs_hours.get(year)
            monitored = hours is not None and hours > 0
            rows.append(
                {
                    "species": row["species"],
                    "year": year,
                    "count": count,
                    "observation_hours": hours,
                    "monitored": monitored,
                    "count_per_hour": (count / hours) if monitored else pd.NA,
                    "source_file": row["source_file"],
                }
            )
    return pd.DataFrame(rows)


def parse_annual_pdf(pdf_path: str | Path) -> dict:
    pdf_path = Path(pdf_path)
    text = extract_text(pdf_path)
    return parse_annual_pdf_text(text, source_file=pdf_path.name)


def discover_annual_pdfs(raw_dir: str | Path, pattern: str = "trektellen*annual*.pdf") -> list[Path]:
    """
    Globs raw_dir for annual-totals PDFs matching `pattern`. This is the
    "update automatically as I add more data" mechanism: drop another
    file matching this pattern into data/reference/ (e.g. an updated
    export covering more years) and re-running
    scripts/build_trektellen_annual_dataset.py picks it up with no code
    change. Sorted by modification time, oldest first, so downstream
    merge logic can treat "later in this list" as "more recent upload."
    """
    raw_dir = Path(raw_dir)
    return sorted(raw_dir.glob(pattern), key=lambda p: p.stat().st_mtime)


def parse_and_merge_annual_pdfs(pdf_paths: list[Path]) -> pd.DataFrame:
    """
    Parses every PDF in pdf_paths (oldest-to-newest, per
    discover_annual_pdfs's sort) into long format and merges them.

    Merge policy: for any (species, year) reported by more than one
    file, the value from the MOST RECENTLY MODIFIED file wins (a
    re-export superseding an older one). Every case where two files
    actually DISAGREE on that (species, year)'s value is recorded, not
    silently dropped -- attached to the returned DataFrame as
    `.attrs["conflicts"]` (a small DataFrame of species/year pairs with
    >1 distinct reported value), so a caller can inspect or log them
    without every caller needing to unpack a tuple return.
    """
    if not pdf_paths:
        return pd.DataFrame(
            columns=["species", "year", "count", "observation_hours", "monitored", "count_per_hour", "source_file"]
        )

    all_long = []
    for path in pdf_paths:
        parsed = parse_annual_pdf(path)
        all_long.append(to_long_format(parsed))
    combined = pd.concat(all_long, ignore_index=True)

    # Detect genuine conflicts (same species+year, different count)
    # BEFORE deduplicating, so they're never silently lost.
    conflicts = (
        combined.groupby(["species", "year"])["count"]
        .nunique()
        .reset_index(name="n_distinct_values")
        .query("n_distinct_values > 1")
    )

    # "Last occurrence wins" == the most-recently-modified file's value,
    # since pdf_paths (and therefore all_long/combined) is sorted
    # oldest-to-newest.
    merged = combined.drop_duplicates(subset=["species", "year"], keep="last").reset_index(drop=True)
    merged.attrs["conflicts"] = conflicts
    merged.attrs["source_files"] = [p.name for p in pdf_paths]
    return merged


def aggregate_annual_local_activity(long_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per species: total_count and total_hours pooled across
    MONITORED years only, and local_activity = total_count / total_hours
    - a multi-year-pooled per-hour encounter rate. Mirrors
    hazard.trektellen_season.aggregate_season_local_activity exactly,
    but pooling across years (this module's grain) instead of months
    within one season (that module's grain). Same shape as that
    function's output (species, total_count, total_hours,
    local_activity), so both are drop-in compatible with
    hazard.species_risk.merge_local_risk.

    Multi-year pooling is preferable to a single season where both are
    available: more monitored hours (e.g. ~3,388 hours across 2022-2025
    here vs. ~899 hours for 2025 alone) means a less noisy per-hour rate
    estimate for less-common species.
    """
    monitored = long_df[long_df["monitored"]]
    grouped = monitored.groupby("species").agg(
        total_count=("count", "sum"), total_hours=("observation_hours", "sum")
    ).reset_index()
    grouped["local_activity"] = grouped["total_count"] / grouped["total_hours"]
    return grouped
