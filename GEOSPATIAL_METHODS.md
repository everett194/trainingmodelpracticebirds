# Geospatial Methods

How BirdStrikeGeo handles coordinate reference systems, distance/buffer
math, and spatial aggregation — and the tradeoffs behind each choice.
Several of these are exactly the kind of decisions worth validating with
a real Esri/GIS analyst — see `ESRI_DISCUSSION_NOTES.md`.

---

## 1. CRS strategy

Implemented in `src/birdstrikegeo/geo/crs.py`.

- **Storage CRS:** all raw latitude/longitude is stored and exported in
  **WGS84 (EPSG:4326)** — the universal, degrees-based standard everyone
  can read.
- **Distance/buffer math is never done directly in degrees.** A degree
  of longitude is ~111 km at the equator and ~0 km at the poles — doing
  arithmetic on raw lat/lon would silently distort every distance and
  area calculation depending on where on Earth the points happen to be.
- **Two regional, equal-area projected CRSs**, selected by a point's
  bounding box:

  | Region | Projected CRS | Why |
  |---|---|---|
  | CONUS | `EPSG:5070` (NAD83 / Conus Albers) | Standard USGS/Esri choice for CONUS-wide equal-area analysis. |
  | Europe | `EPSG:3035` (ETRS89-extended / LAEA Europe) | Standard INSPIRE/Esri choice for pan-European equal-area analysis. |

- **Fallback:** anything outside both bounding boxes, or any pair of
  points that spans both regions (e.g. comparing a US and a European
  airport), falls back to **ellipsoidal geodesic distance** via
  `pyproj.Geod` — correct anywhere on Earth, at the cost of not
  supporting true buffer/area geometry the way a projected CRS does.

### Tradeoffs

- **Two hardcoded regional bounding boxes is a simplification.** A
  production system covering more of the world would need a proper
  UTM-zone-selection strategy or per-country projected CRS lookup — see
  `ESRI_DISCUSSION_NOTES.md` question 2.
- **Planar buffers in a projected CRS are an approximation**, not
  exact geodesic buffers. For the buffer radii this project uses
  (50/100/250 km), the distortion is small but non-zero, especially
  near the edges of each region's bounding box. See
  `ESRI_DISCUSSION_NOTES.md` question 3 (geodesic vs. planar buffers).

## 2. Distance calculation (`geo/spatial_join.py`, `geo/crs.py`)

`distances_to_sites()` picks, per point pair: the region's projected CRS
if both points fall in the same supported region, otherwise vectorized
geodesic distance (`pyproj.Geod.inv`, which supports array inputs for
speed). This means a single "distance to every Trektellen site" query
can mix projected and geodesic distances across different sites — each
distance is still individually correct for its pair of points.

## 3. Spatial aggregation: two options, deliberately simple

Real Trektellen coverage is sparse and unevenly distributed. Rather than
fit a kriging/interpolation surface across large gaps (which would
imply more spatial precision than the underlying data supports), this
project offers exactly two transparent options
(`birdstrikegeo/geo/spatial_join.py`):

1. **`nearest_site()`** — use only the single closest monitoring site.
   Simple, but ignores everything else nearby.
2. **`distance_weighted_aggregate()`** — weight every site within a
   configurable radius by an **exponential distance-decay function**:

   ```
   weight = 0.5 ** (distance_km / half_life_km)
   ```

   A site at `distance = 0` gets weight `1.0`; at `distance =
   half_life_km` (default 50 km), weight `0.5`; at `2×half_life_km`,
   weight `0.25`; and so on. This is a standard, easily-explained decay
   curve — not a fitted interpolation model. See
   `ESRI_DISCUSSION_NOTES.md` question 7 for the open question of
   whether ArcGIS's built-in interpolation tools would be more
   appropriate for a production version.

Both options are documented, not hidden behind a single "best guess"
default — `configs/geospatial.yaml`'s `spatial_aggregation_default` lets
a caller choose.

## 4. Temporal joins: no future data, ever (by default)

`geo/temporal_join.py`:

- `counts_before_timestamp()` — strictly `<` the query timestamp, never
  `<=`. A strike or activity query on a given date can never see bird
  observations recorded after it.
- `nearest_prior_weather()` — same rule for weather: the nearest station
  within a configurable distance, then the most recent observation *at
  or before* the query time within a configurable age tolerance.
  `allow_future=True` exists as an explicit, opt-in symmetric research
  window — never the default for either task's predictions.

## 5. Layers produced

`birdstrikegeo/geo/{airport_layers,trektellen_layers,exports}.py`, built
by `scripts/build_geospatial_features.py`:

```
airports
trektellen_sites
airport_buffers_50km / 100km / 250km
airport_trektellen_links      (flat airport×site distance table)
airport_period_activity       (per-airport activity_index + coverage)
model_predictions             (prepared, not auto-published — see arcgis_adapter)
```

Every spatial layer is exported to **GeoJSON, GeoPackage (.gpkg),
GeoParquet (.parquet), and CSV** (with explicit `latitude`/`longitude`
columns), and every export is stamped with:

```
source
generated_at
crs
sample_data_flag
model_version
```

## 6. Environmental context layers (wetlands, water, land cover, ...)

`birdstrikegeo/geo/airport_layers.py`'s `compute_environmental_features()`
only computes `distance_to_water_km`, `wetland_area_within_10km`, etc.
when a real spatial layer is actually supplied by the caller — **never**
fabricated for a real airport. `habitat_data_available` distinguishes
"no habitat nearby" from "we don't have the data."

## 7. ArcGIS interoperability, without requiring ArcGIS

`birdstrikegeo/geo/arcgis_adapter.py` is entirely optional:

- reads a **public** ArcGIS Feature Service via paginated REST queries
  (no `arcgis` package or credentials required for a public layer)
- converts a GeoDataFrame to Esri-JSON-shaped feature records
- validates geometry/CRS/field-name well-formedness **before** anyone
  tries to publish anything
- never publishes automatically — `offline_export()` writes a reviewable
  JSON file when no ArcGIS credentials are configured

Real finding from development: running `scripts/export_arcgis_layers.py`
on the sample `airport_period_activity` layer caught that several of
this project's own feature names (e.g.
`distance_to_nearest_trektellen_site_km`, `nearest_site_observation_age_hours`)
**exceed ArcGIS's 31-character field-name limit**. That's flagged, not
silently truncated — see `ESRI_DISCUSSION_NOTES.md` for the naming
convention question this raises for a real publish target.
