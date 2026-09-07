# Eastern Shore Bird-Strike Risk Analysis

## Methodology and limitations

A CatBoost gradient-boosted-tree regressor was trained on the FAA
Wildlife Strike Database (confirmed business/private fixed-wing GA
population) to predict an ordinal damage-severity score (0 = no damage,
1 = minor, 2 = substantial, 3 = destroyed) from strike conditions
(species, engine type, phase of flight, sky/visibility, precipitation,
light condition, warning status, height, speed, season). Validation
RMSE: 0.35, R-squared: 0.32.

Per-species national risk scores (mean predicted severity x
log1p(strike count)) were then joined against effort-normalized local
activity from a single Eastern Shore of Maryland banding station
(Trektellen/FBBO): 2022-2025 pooled (4 monitored years, 3389 total observation hours, source: trektellen_fbbo_annual_banding_totals_2016-2025.pdf). This is exploratory analysis, not a calibrated
probability: local data comes from a single station, species-name
matching is approximate for composite/subspecies-tagged Trektellen
entries, and a species' local abundance is not the same as its exposure
risk to aircraft. 55 locally
observed species had no FAA strike-record match and are excluded from
the risk scatterplot below (but are noted separately). This already pools multiple monitored years (see basis above); further station-years will continue to refine it, but the single-season noise concern from earlier versions of this report no longer applies at the same scale.

## Nationwide findings: highest species risk (FAA-wide)

| species | n_strikes | mean_predicted_severity | risk_score |
| --- | --- | --- | --- |
| White-tailed deer | 1042 | 1.24 | 8.60 |
| Unknown bird - large | 704 | 0.89 | 5.83 |
| Turkey vulture | 469 | 0.88 | 5.40 |
| Geese | 175 | 1.00 | 5.16 |
| Mule deer | 70 | 1.21 | 5.15 |
| Canada goose | 784 | 0.76 | 5.06 |
| New World vultures | 195 | 0.93 | 4.92 |
| Black vulture | 126 | 0.99 | 4.78 |
| Snow goose | 31 | 1.16 | 4.04 |
| Unknown bird - medium | 2631 | 0.44 | 3.46 |
| Ducks, geese, swans | 32 | 0.99 | 3.46 |
| Ducks | 222 | 0.64 | 3.45 |
| Double-crested cormorant | 30 | 0.94 | 3.23 |
| Anhinga | 24 | 0.95 | 3.07 |
| Bald eagle | 144 | 0.61 | 3.04 |

## Eastern Shore (FBBO) local risk read

![Local risk scatterplot](eastern_shore_risk_scatter.png)

| species | local_activity | risk_score | local_risk_contribution |
| --- | --- | --- | --- |
| White-throated Sparrow | 4.26 | 1.11 | 4.71 |
| Common Yellowthroat | 1.42 | 0.70 | 1.00 |
| Song Sparrow | 2.05 | 0.45 | 0.92 |
| Hermit Thrush | 0.67 | 1.14 | 0.76 |
| Ruby-crowned Kinglet | 0.92 | 0.42 | 0.38 |
| Swamp Sparrow | 0.87 | 0.39 | 0.34 |
| Northern Cardinal | 0.67 | 0.32 | 0.21 |
| Red-winged Blackbird | 0.71 | 0.28 | 0.20 |
| Ovenbird | 0.35 | 0.44 | 0.15 |
| Indigo Bunting | 0.57 | 0.24 | 0.14 |
| Wood Thrush | 0.23 | 0.56 | 0.13 |
| American Robin | 0.11 | 1.05 | 0.12 |
| Magnolia Warbler | 0.19 | 0.49 | 0.09 |
| Red-eyed Vireo | 0.10 | 0.95 | 0.09 |
| Yellow-breasted Chat | 0.19 | 0.47 | 0.09 |

## Concluding risk statement

The species combining the highest national strike-severity risk with
the highest local activity at FBBO (2022-2025 pooled (4 monitored years, 3389 total observation hours, source: trektellen_fbbo_annual_banding_totals_2016-2025.pdf)) represent the greatest plausible bird-strike
concern for GA aircraft operating near this Eastern Shore station - see
the top-right of the scatterplot and the local risk table above for the
specific species driving that read.
