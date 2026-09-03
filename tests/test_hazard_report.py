import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from birdstrikegeo.hazard.report import build_markdown_report, plot_local_risk_scatter  # noqa: E402


def _merged_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "species": ["Mourning Dove", "Canada Goose", "Some Obscure Warbler"],
            "risk_score": [4.0, 9.0, None],
            "n_strikes": [500, 300, None],
            "mean_predicted_severity": [1.0, 2.0, None],
            "local_activity": [2.0, 0.1, 1.0],
            "local_risk_contribution": [8.0, 0.9, None],
            "matched_faa_species": [True, True, False],
        }
    )


def test_plot_local_risk_scatter_writes_a_nonempty_png_file(tmp_path):
    output_path = tmp_path / "scatter.png"
    plot_local_risk_scatter(_merged_df(), output_path)
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_build_markdown_report_includes_species_names_and_metrics(tmp_path):
    output_path = tmp_path / "report.md"
    context = {
        "model_metrics": {"val_rmse": 0.65, "val_r2": 0.31},
        "national_top_species": _merged_df().sort_values("risk_score", ascending=False).to_dict("records"),
        "local_top_species": _merged_df().sort_values("local_risk_contribution", ascending=False).to_dict("records"),
        "scatter_image_relpath": "scatter.png",
        "n_unmatched_local_species": 1,
    }
    build_markdown_report(context, output_path)

    text = output_path.read_text()
    assert "Canada Goose" in text
    assert "0.65" in text
    assert "scatter.png" in text
