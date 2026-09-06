# Display Design (Phase 9)

How a map/report display should distinguish between fundamentally
different quantities this project produces, so it never implies false
precision. This is a design document — the existing `/hazard` Flask page
already implements a text/table version of some of this; a future map
UI should follow this structure.

## The layers that must never be visually conflated

| Layer | What it actually is | Current implementation |
|---|---|---|
| **Observed bird abundance** | A raw count from an actual monitoring session (Trektellen) | `trektellen_season.py`'s per-species monthly counts — real observations, effort-normalized |
| **Modeled bird activity** | A statistical estimate, not a direct count (eBird Status & Trends, once integrated) | Not yet integrated — `geo/ebird_adapter.py` is structure only |
| **Historical strike density** | Raw count of REPORTED strikes in an area/period — biased by reporting behavior, not a hazard estimate on its own | `national_species_risk.csv`'s `n_strikes` column |
| **Exposure-adjusted strike rate** | Strikes per unit of flight activity — requires the exposure denominator this project does not have | **Not computable today** — must be shown as "unavailable," never approximated |
| **Predicted relative hazard** | The `relative_hazard_index` (0-100) — model output, not observation | `hazard/report.py`'s local risk read |
| **Absolute probability** | Only justified for the conditional-damage model (P(damage \| strike)), never for strike occurrence | `/ga-damage`'s calibrated probability output |
| **Uncertainty / insufficient-data regions** | Anywhere confidence is low or data is sparse — must be visually distinct, not just numerically footnoted | Partially done (`distance_warning`/`low_coverage_warning` text banners); no map-based visual treatment yet |

**The single most important rule**: "historical strike density" and
"exposure-adjusted strike rate" must NEVER share a color scale or be
placed on the same map without a clear label — an airport with high
strike density and an airport with high exposure-adjusted rate can be
completely different airports (one is just busy, the other is
genuinely more hazardous per unit of exposure) and conflating them is
exactly the mistake this whole project exists to avoid.

## Required display elements (every map/report view)

Per project requirement, every view must show, not bury in a linked
document:

1. **Legend with units** — "relative hazard index (0-100, not a
   probability)" as literal legend text, not just a color ramp.
2. **Data source** — e.g. "FAA Wildlife Strike Database + Trektellen
   FBBO 2025" (already present in `eastern_shore_birdstrike_risk_analysis.md`'s
   methodology section; not yet a persistent on-screen element).
3. **Time period** — the season/date range the underlying data covers
   (e.g. "2025 season, Mar-Nov").
4. **Spatial resolution** — e.g. "single station point" vs. "H3
   resolution-5 cell (~8.5km)" once grid aggregation is used.
5. **Model version** — already tracked in every exported layer's
   metadata (`geo/exports.py`'s `model_version` stamp); needs to be
   surfaced in the UI, not just the file.
6. **Uncertainty** — at minimum, a visually distinct treatment (e.g.
   hatching, reduced opacity, or a dedicated "insufficient data" color)
   for any cell/airport below a defined minimum-observation threshold —
   not yet implemented for any map view (the current `/hazard` page is
   tables and one scatterplot, not a hazard map).
7. **Exposure-data-missing warning** — a persistent, impossible-to-miss
   banner (not a footnote) on ANY view that could be mistaken for an
   exposure-adjusted rate, given this project has no exposure data at
   all today. The existing `.sample-banner` CSS pattern in
   `templates/base.html` is the right visual weight to reuse for this —
   same treatment, different message ("NO FLIGHT-EXPOSURE DATA — THIS
   IS NOT AN ADJUSTED RATE").

## Recommendation for the first version

**A relative hazard index (0-100), not an absolute probability, for the
secondary output** — already the current design
(`hazard/inference.py`'s `RISK_BANDS`-equivalent for the primary model;
the secondary hazard index is unitless 0-100 by construction). This is
more defensible than an absolute strike probability given the complete
absence of exposure data, and should stay that way until
`bts_t100_departures.csv` or an equivalent is integrated.

**Concrete next step, not yet built**: a real map view (Leaflet/Mapbox/
ArcGIS JS, or Folium given it's already a dependency) showing H3 cells
(once populated — see `geo/h3_grid.py`) colored by `relative_hazard_index`,
with a visually distinct "insufficient data" treatment for any cell
below a minimum-observation-count threshold, and the 7 required
elements above always visible, not hidden behind a click. The current
Flask app's `/hazard` page (tables + one static scatterplot image) is a
reasonable v1 given the single-station data available today; a map view
is deferred structure, not yet implemented in this pass.
