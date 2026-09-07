"""
birdstrikegeo.hazard.report
--------------------------------
Renders the Eastern Shore bird-strike risk analysis: a scatterplot of
national species risk vs. local (Trektellen) activity, and the
accompanying markdown writeup.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless - never try to open a window
import matplotlib.pyplot as plt  # noqa: E402

_TOP_N_TABLE = 15
_TOP_N_LABELS = 12


def plot_local_risk_scatter(
    merged_df, output_path: str | Path, top_n_labels: int = _TOP_N_LABELS,
    activity_label: str = "FBBO 2025",
) -> None:
    """
    x = national risk_score, y = local activity (effort-normalized
    Trektellen abundance). Species in the top-right are both nationally
    high-severity/frequency AND locally common - the biggest local
    concern. Unmatched-to-FAA species (no risk_score) are excluded from
    the plot itself, since there is nothing to place on the x-axis.

    activity_label: short description of the local-activity data's
    basis (e.g. "FBBO 2025" for the single-season monthly source, or
    "FBBO 2016-2025" for the multi-year annual source) - shown on the
    y-axis and title so a reader never has to guess which basis
    produced the plot.
    """
    matched = merged_df[merged_df["matched_faa_species"]].copy()

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(
        matched["risk_score"], matched["local_activity"],
        s=60, alpha=0.7, c=matched["local_risk_contribution"], cmap="viridis",
    )

    labeled = matched.sort_values("local_risk_contribution", ascending=False).head(top_n_labels)
    for _, row in labeled.iterrows():
        ax.annotate(row["species"], (row["risk_score"], row["local_activity"]), fontsize=8, alpha=0.85)

    ax.set_xlabel("National risk score (mean predicted severity x log1p(strike count))")
    ax.set_ylabel(f"Local activity (effort-normalized count, {activity_label})")
    ax.set_title(f"Eastern Shore (FBBO) species: national strike risk vs. local abundance ({activity_label})")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _species_table(records: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, separator]
    for record in records[:_TOP_N_TABLE]:
        cells = []
        for col in columns:
            value = record.get(col)
            if isinstance(value, float):
                cells.append(f"{value:.2f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def build_markdown_report(context: dict, output_path: str | Path) -> None:
    metrics = context["model_metrics"]
    national_columns = ["species", "n_strikes", "mean_predicted_severity", "risk_score"]
    local_columns = ["species", "local_activity", "risk_score", "local_risk_contribution"]

    text = f"""# Eastern Shore Bird-Strike Risk Analysis

## Methodology and limitations

A CatBoost gradient-boosted-tree regressor was trained on the FAA
Wildlife Strike Database (confirmed business/private fixed-wing GA
population) to predict an ordinal damage-severity score (0 = no damage,
1 = minor, 2 = substantial, 3 = destroyed) from strike conditions
(species, engine type, phase of flight, sky/visibility, precipitation,
light condition, warning status, height, speed, season). Validation
RMSE: {metrics['val_rmse']:.2f}, R-squared: {metrics['val_r2']:.2f}.

Per-species national risk scores (mean predicted severity x
log1p(strike count)) were then joined against effort-normalized local
activity from a single Eastern Shore of Maryland banding station
(Trektellen/FBBO): {context.get('local_activity_basis_description',
'2025 season only')}. This is exploratory analysis, not a calibrated
probability: local data comes from a single station, species-name
matching is approximate for composite/subspecies-tagged Trektellen
entries, and a species' local abundance is not the same as its exposure
risk to aircraft. {context['n_unmatched_local_species']} locally
observed species had no FAA strike-record match and are excluded from
the risk scatterplot below (but are noted separately). {context.get('additional_station_years_note',
'More station-years are expected to refine this analysis.')}

## Nationwide findings: highest species risk (FAA-wide)

{_species_table(context['national_top_species'], national_columns)}

## Eastern Shore (FBBO) local risk read

![Local risk scatterplot]({context['scatter_image_relpath']})

{_species_table(context['local_top_species'], local_columns)}

## Concluding risk statement

The species combining the highest national strike-severity risk with
the highest local activity at FBBO ({context.get('local_activity_basis_description',
'2025 season only')}) represent the greatest plausible bird-strike
concern for GA aircraft operating near this Eastern Shore station - see
the top-right of the scatterplot and the local risk table above for the
specific species driving that read.
"""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
