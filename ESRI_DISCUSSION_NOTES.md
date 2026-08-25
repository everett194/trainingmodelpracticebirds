# Discussion Notes for an Esri / GIS Analyst

This document has two parts: a one-page project description you can
hand to an analyst, and the specific open design questions this project
has deliberately left unresolved rather than guessing at.

---

## One-page project description

**BirdStrikeGeo** is a portfolio/educational project exploring two
related but distinct questions about wildlife and aviation:

1. **Given a reported wildlife strike**, can available information about
   the flight, the wildlife, and conditions at the time predict whether
   the strike caused aircraft damage? (A supervised classification
   problem, trainable from the public FAA Wildlife Strike Database
   alone.)
2. **Given an airport and a time period**, can nearby bird-migration
   monitoring data (Trektellen) produce a useful *relative* activity
   index? (A geospatial aggregation problem, not yet a trained model —
   see `MODEL_CARD.md` for why.)

The project is explicitly **not** operational aviation-safety software,
and does not (yet, and possibly not ever, without a flight-exposure
denominator) predict the probability that a strike will occur.

The current version runs entirely on synthetic sample data out of the
box and is structured so real FAA/Trektellen/airport/weather data can be
dropped into `data/raw/` and used identically (`--mode real` everywhere).
The geospatial layer (CRS handling, distance-decay-weighted monitoring
aggregation, buffer generation, multi-format export) is deliberately
kept simple and transparent rather than fitting an interpolation
surface across sparse, unevenly-distributed monitoring coverage — see
`GEOSPATIAL_METHODS.md`.

The questions below are exactly the design decisions where a real GIS/
Esri analyst's judgment would materially improve this project beyond
what a software engineer working alone chose as a reasonable default.

---

## Open questions

1. Should the project use hosted feature layers, GeoParquet, GeoPackage
   or another storage format?
2. What projection is appropriate for distance-based airport analysis
   across the United States and Europe?
3. Should airport buffers use geodesic or planar geometry?
4. What is the best way to represent time-enabled bird-count
   observations?
5. How should spatial uncertainty and sparse monitoring coverage be
   visualized?
6. Would a space-time cube or emerging-hot-spot analysis be appropriate?
7. How should the model avoid overinterpreting interpolated activity
   between distant monitoring stations?
8. What ArcGIS tools are best for joining irregular monitoring sites to
   airports?
9. Should activity be represented as points, hexagonal bins, rasters or
   airport-centered summaries?
10. What cell size would be defensible for a migration-activity surface?
11. How can the feature layer retain source provenance and
    model-version metadata?
12. How should historical observations be separated from model
    predictions?
13. What symbology best communicates uncertainty rather than false
    precision?
14. How can this later be made compatible with an ArcGIS Dashboard or
    Experience Builder application?
15. What public airport, land-cover, wetland or water-body layers would
    Esri recommend?
16. Is ArcGIS Notebooks appropriate for scheduled feature generation?
17. How should paginated feature-service queries and service limits be
    handled?
18. What is the cleanest method for publishing read-only portfolio
    layers without exposing credentials?
19. How should cross-Atlantic CRS and distance calculations be managed?
20. What analysis would make this project genuinely useful to an
    airport wildlife-hazard manager?

## A concrete finding worth discussing (question 1 / 11)

Running `scripts/export_arcgis_layers.py` against the sample
`airport_period_activity` layer surfaced a real schema problem: several
feature names this project generates —
`distance_to_nearest_trektellen_site_km`,
`nearest_site_observation_age_hours`,
`birds_per_observation_hour_previous_7d`,
`nocturnal_flight_call_count_previous_3d` — **exceed ArcGIS's
31-character field-name limit**
(`birdstrikegeo/geo/arcgis_adapter.py:validate_arcgis_schema()` flags
this rather than silently truncating). A real publish target would need
either a documented field-name abbreviation convention or a field-alias
mapping preserved alongside the short names — exactly the kind of
provenance-retention question in item 11 above.
