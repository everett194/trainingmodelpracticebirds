from __future__ import annotations

import openpyxl
import pytest

from birdstrikegeo.data.ingest_trektellen_crosstabs_xlsx import parse_crosstabs_xlsx, read_index


def _build_sample_workbook(path) -> None:
    wb = openpyxl.Workbook()
    index_ws = wb.active
    index_ws.title = "Index"
    index_ws.append(["Foreman's Branch Bird Observatory (MD) -- Banding Year Totals 2017-2026"])
    index_ws.append(["Source: trektellen.org site 3460"])
    index_ws.append([])
    index_ws.append(["Period code", "Period label", "Months included", "Species recorded",
                      "Total (combined)", "Total (newly ringed)", "Total (retraps)"])
    index_ws.append([10, "October", "10", 1, 5, 3, 2])

    oct_ws = wb.create_sheet("October")
    oct_ws.append(["Foreman's Branch Bird Observatory -- October (period code 10)"])
    oct_ws.append([])
    oct_ws.append(["Metric", "Species", "2024", "2025", "Total 2017-2026"])
    oct_ws.append(["Combined (newly ringed + retraps)", "Northern Cardinal", 2.0, 3.0, 5.0])
    oct_ws.append(["TOTAL — Combined (newly ringed + retraps)", None, None, None, 5.0])
    oct_ws.append([])
    oct_ws.append(["Newly ringed", "Northern Cardinal", 1.0, 2.0, 3.0])
    oct_ws.append(["TOTAL — Newly ringed", None, None, None, 3.0])
    oct_ws.append([])
    oct_ws.append(["Retraps", "Northern Cardinal", 1.0, 1.0, 2.0])
    oct_ws.append(["TOTAL — Retraps", None, None, None, 2.0])

    wb.save(path)


@pytest.fixture()
def sample_xlsx(tmp_path):
    path = tmp_path / "sample_all_presets.xlsx"
    _build_sample_workbook(path)
    return path


def test_read_index(sample_xlsx):
    index = read_index(sample_xlsx)
    assert len(index) == 1
    row = index.iloc[0]
    assert row["period_code"] == 10
    assert row["period_label"] == "October"
    assert row["total_combined"] == 5
    assert row["total_newly_ringed"] == 3
    assert row["total_retraps"] == 2


def test_parse_crosstabs_extracts_all_three_metrics(sample_xlsx):
    df = parse_crosstabs_xlsx(sample_xlsx)
    assert len(df) == 6  # 1 species x 2 years x 3 metrics

    combined_2025 = df.query("metric == 'combined' and year == 2025").iloc[0]
    assert combined_2025["count"] == 3.0
    assert combined_2025["species"] == "Northern Cardinal"
    assert combined_2025["period_code"] == 10
    assert combined_2025["period_label"] == "October"

    newly_ringed_2024 = df.query("metric == 'newly_ringed' and year == 2024").iloc[0]
    assert newly_ringed_2024["count"] == 1.0

    retraps_2025 = df.query("metric == 'retraps' and year == 2025").iloc[0]
    assert retraps_2025["count"] == 1.0


def test_total_rows_excluded_from_output(sample_xlsx):
    df = parse_crosstabs_xlsx(sample_xlsx)
    assert df["species"].isna().sum() == 0  # TOTAL rows (species=None) never make it into the output


def test_parsed_metric_sums_match_index_totals(sample_xlsx):
    index = read_index(sample_xlsx)
    df = parse_crosstabs_xlsx(sample_xlsx)
    for metric, index_col in [("combined", "total_combined"), ("newly_ringed", "total_newly_ringed"),
                               ("retraps", "total_retraps")]:
        parsed_sum = df[df["metric"] == metric]["count"].sum()
        assert parsed_sum == index.iloc[0][index_col]
