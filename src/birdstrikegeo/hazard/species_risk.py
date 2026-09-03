"""
birdstrikegeo.hazard.species_risk
-------------------------------------
Aggregates the trained severity model's per-strike predictions into a
per-species national risk score, then joins that against local
(Trektellen) activity to estimate a relative local risk contribution.

risk_score = mean_predicted_severity * log1p(n_strikes): a species that
is both frequently struck AND predicted to cause worse damage when
struck ranks highest. log1p on frequency keeps a handful of very
high-volume species (e.g. gulls) from mechanically dominating purely on
count while still rewarding higher strike volume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_species_risk(faa_df: pd.DataFrame) -> pd.DataFrame:
    """
    faa_df: one row per strike, with SPECIES and predicted_severity
    columns (predicted_severity from the trained severity regressor).
    Returns one row per species: species, n_strikes, mean_predicted_severity,
    risk_score.
    """
    grouped = faa_df.groupby("SPECIES")["predicted_severity"].agg(["count", "mean"]).reset_index()
    grouped.columns = ["species", "n_strikes", "mean_predicted_severity"]
    grouped["risk_score"] = grouped["mean_predicted_severity"] * np.log1p(grouped["n_strikes"])
    return grouped


def merge_local_risk(species_risk: pd.DataFrame, local_activity: pd.DataFrame) -> pd.DataFrame:
    """
    local_activity: one row per species with a `local_activity` column
    (e.g. effort-normalized Trektellen count). Left-joins species_risk
    onto it so every locally-observed species is retained even when it
    has no FAA match - matched_faa_species flags which is which, rather
    than silently dropping unmatched species.

    Joined case-insensitively: FAA exports capitalize only the first
    word of a common name ("Mourning dove") while Trektellen title-cases
    both ("Mourning Dove") - an exact-string join would drop nearly every
    species. The local (Trektellen) casing is kept as the display name.
    """
    local = local_activity.copy()
    local["_join_key"] = local["species"].str.upper()
    risk = species_risk.copy()
    risk["_join_key"] = risk["species"].str.upper()
    risk = risk.drop(columns=["species"])

    merged = local.merge(risk, on="_join_key", how="left").drop(columns=["_join_key"])
    merged["matched_faa_species"] = merged["risk_score"].notna()
    merged["local_risk_contribution"] = merged["risk_score"] * merged["local_activity"]
    return merged
