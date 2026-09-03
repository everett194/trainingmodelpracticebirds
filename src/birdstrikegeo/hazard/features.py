"""
birdstrikegeo.hazard.features
---------------------------------
Feature set for the severity-regression model in this package.

This is a deliberately DIFFERENT feature policy from
birdstrikegeo.ga.feature_policy's operational GA damage-probability
model, which forbids SPECIES because it's unknowable before a strike
happens (see that module's docstring). This model asks a different,
retrospective question - "given the strikes FAA already recorded, which
species/conditions were associated with worse damage" - so species is
exactly the point, not a leak: it's an explanatory/ranking model over
historical strikes, never a preflight predictor. True outcome/cost
columns (STR_*, DAM_*, COST_*, EFFECT, injuries) are still excluded, via
reuse of birdstrikegeo.ga.features' toggle-variable set.
"""

from __future__ import annotations

from birdstrikegeo.ga.features import FeatureSet, build_operational_core_features


def build_severity_features(df) -> FeatureSet:
    core = build_operational_core_features(df)
    features = core.features.copy()
    features["species"] = df.get("SPECIES")

    return FeatureSet(
        mode="severity_with_species",
        features=features,
        numeric_columns=list(core.numeric_columns),
        categorical_columns=list(core.categorical_columns) + ["species"],
    )
