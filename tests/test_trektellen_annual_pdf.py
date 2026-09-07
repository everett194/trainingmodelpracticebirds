from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest

from birdstrikegeo.data.ingest_trektellen_annual_pdf import (
    aggregate_annual_local_activity,
    discover_annual_pdfs,
    parse_and_merge_annual_pdfs,
    parse_annual_pdf_text,
    to_long_format,
)

SAMPLE_TEXT = """\
  FBBO Banding Totals, 2016-2025
  Species-level annual totals for Trektellen site 3460. Years 2016-2021 are retained as reported; the active series begins in 2022. Average values reflect the source table.

    No.                        Species              2016     2017     2018    2019     2020     2021      2022       2023       2024       2025         Total        Average   Newly ringed   Retraps   Year maximum
     1    Hooded Merganser                            0       0        0        0       0        0         0          0           1          0            1            0.3          1           0         1 (2024)
   148    Northern Cardinal                            0       0        0        0       0        0         370        654         597        651         2272          568          888         1384      654 (2023)
          Totals                                0      0      0      0      0      0     13421     19188     22428    19594     74631     18658        51358       23273
          Observation hours                                                              532:59   1,018:12   937:56   899:25   3,388:32   847:08
"""


def test_species_rows_parsed_correctly():
    parsed = parse_annual_pdf_text(SAMPLE_TEXT, source_file="test.pdf")
    df = parsed["species_rows"]
    assert len(df) == 2
    cardinal = df[df["species"] == "Northern Cardinal"].iloc[0]
    assert cardinal["year_2016"] == 0
    assert cardinal["year_2022"] == 370
    assert cardinal["year_2025"] == 651
    assert cardinal["total"] == 2272
    assert cardinal["average"] == 568
    assert cardinal["newly_ringed"] == 888
    assert cardinal["retraps"] == 1384
    assert cardinal["year_max_count"] == 654
    assert cardinal["year_max_year"] == 2023
    assert cardinal["source_file"] == "test.pdf"


def test_totals_row_parsed_correctly():
    parsed = parse_annual_pdf_text(SAMPLE_TEXT, source_file="test.pdf")
    totals = parsed["totals_row"]
    assert totals["year_2016"] == 0
    assert totals["year_2022"] == 13421
    assert totals["year_2025"] == 19594
    assert totals["total"] == 74631
    assert totals["newly_ringed"] == 51358
    assert totals["retraps"] == 23273


def test_observation_hours_parsed_and_aligned_to_recent_years():
    parsed = parse_annual_pdf_text(SAMPLE_TEXT, source_file="test.pdf")
    hours = parsed["observation_hours_by_year"]
    assert hours[2022] == 532 + 59 / 60
    assert hours[2023] == 1018 + 12 / 60  # comma thousands-separator stripped
    assert hours[2024] == 937 + 56 / 60
    assert hours[2025] == 899 + 25 / 60
    assert 2016 not in hours
    assert 2021 not in hours


def test_to_long_format_marks_unmonitored_years_with_null_rate_not_zero():
    parsed = parse_annual_pdf_text(SAMPLE_TEXT, source_file="test.pdf")
    long_df = to_long_format(parsed)
    cardinal = long_df[long_df["species"] == "Northern Cardinal"].set_index("year")

    # 2016: raw count is 0 (as reported), but the year was unmonitored --
    # this must be `monitored=False` with a NULL rate, never a computed 0.
    assert cardinal.loc[2016, "count"] == 0
    assert cardinal.loc[2016, "monitored"] == False  # noqa: E712
    assert cardinal.loc[2016, "count_per_hour"] is None or str(cardinal.loc[2016, "count_per_hour"]) == "<NA>"

    # 2022: monitored, real rate computed.
    assert cardinal.loc[2022, "monitored"] == True  # noqa: E712
    assert abs(cardinal.loc[2022, "count_per_hour"] - (370 / (532 + 59 / 60))) < 1e-9


def test_to_long_format_row_count_is_species_times_years():
    parsed = parse_annual_pdf_text(SAMPLE_TEXT, source_file="test.pdf")
    long_df = to_long_format(parsed)
    assert len(long_df) == 2 * 10  # 2 species x 10 years (2016-2025)


def test_aggregate_annual_local_activity_pools_monitored_years_only():
    parsed = parse_annual_pdf_text(SAMPLE_TEXT, source_file="test.pdf")
    long_df = to_long_format(parsed)
    activity = aggregate_annual_local_activity(long_df)
    cardinal = activity[activity["species"] == "Northern Cardinal"].iloc[0]
    # Only 2022-2025 are monitored for Northern Cardinal in the sample text.
    assert cardinal["total_count"] == 370 + 654 + 597 + 651
    expected_hours = (532 + 59 / 60) + (1018 + 12 / 60) + (937 + 56 / 60) + (899 + 25 / 60)
    assert abs(cardinal["total_hours"] - expected_hours) < 1e-6
    assert abs(cardinal["local_activity"] - cardinal["total_count"] / expected_hours) < 1e-9


def test_discover_annual_pdfs_finds_matching_files_sorted_by_mtime(tmp_path):
    old_file = tmp_path / "trektellen_fbbo_annual_banding_totals_2016-2024.pdf"
    old_file.write_text("old")
    time.sleep(0.01)
    new_file = tmp_path / "trektellen_fbbo_annual_banding_totals_2016-2025.pdf"
    new_file.write_text("new")
    (tmp_path / "unrelated_file.pdf").write_text("nope")
    (tmp_path / "trektellen_2025_year_totals.csv").write_text("nope, wrong extension")

    found = discover_annual_pdfs(tmp_path)
    assert [p.name for p in found] == [old_file.name, new_file.name]


def test_discover_annual_pdfs_returns_empty_list_when_none_present(tmp_path):
    assert discover_annual_pdfs(tmp_path) == []


def test_parse_and_merge_annual_pdfs_returns_empty_frame_for_no_files():
    merged = parse_and_merge_annual_pdfs([])
    assert len(merged) == 0
    assert list(merged.columns) == [
        "species", "year", "count", "observation_hours", "monitored", "count_per_hour", "source_file",
    ]


@pytest.mark.skipif(shutil.which("pdftotext") is None, reason="requires poppler's pdftotext binary")
def test_parse_and_merge_annual_pdfs_end_to_end_with_real_pdf(tmp_path):
    real_pdf = (
        Path(__file__).resolve().parent.parent
        / "data" / "reference" / "trektellen_fbbo_annual_banding_totals_2016-2025.pdf"
    )
    if not real_pdf.exists():
        pytest.skip("real reference PDF not present in this checkout")
    merged = parse_and_merge_annual_pdfs([real_pdf])
    assert len(merged) == 150 * 10  # 150 species x 10 years, per the known PDF content
    assert merged.attrs["source_files"] == [real_pdf.name]
    assert len(merged.attrs["conflicts"]) == 0  # only one source file -- no conflicts possible
