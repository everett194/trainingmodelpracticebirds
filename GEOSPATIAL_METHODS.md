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

## 7a. H3 grid indexing (`geo/h3_grid.py`, added in the Phase 2/3 audit consolidation)

A hexagonal spatial-indexing layer (Uber's H3) so bird abundance,
strikes, airports, habitat, and weather can all be aggregated onto the
SAME set of cells, rather than each pipeline picking its own ad hoc
aggregation. This does not replace the CRS/buffer strategy in section 1
above — H3 cells are a discrete aggregation key for JOINING
heterogeneous sources, not a substitute for exact distance math (see
`distance_km_between_h3_cells`, which still goes through this module's
`geodesic_distance_km`, not H3's own hop-count `grid_distance`).

**Resolution choice:**

| Resolution | Edge length | Average cell area | When used |
|---|---|---|---|
| 5 | ~8.5 km | ~252 km² | Default — matches the existing 50 km airport-buffer scale reasonably (a 50 km buffer ≈ a 5-7 cell radius) |
| 7 | ~1.2 km | ~5.2 km² | Finer option for dense-data regions, or joining a fine-grained raster source (e.g. eBird's 3 km grid) without conflating multiple source pixels into one cell |

Both are precomputed and stored per row (`h3_cell_r5`/`h3_cell_r7` in
`schemas.hazard_observation.GEOGRAPHIC_SCHEMA`), never computed on the
fly at query time, so a downstream join never has to guess which
resolution a stored cell ID belongs to. A row with missing/invalid
coordinates gets a null cell, never a guessed one — the same "don't
fabricate what you don't have" rule as everywhere else in this project.

**Raster vs. vector treatment:** point sources (strikes, airports,
Trektellen sites) are assigned to their containing cell directly. A
raster source (e.g. eBird's GeoTIFF, once integrated —
`geo/ebird_adapter.py`) would be resampled to per-cell mean/max values
at the SAME resolution before joining, rather than sampled at each
point's exact pixel — this is future work, not yet implemented, since no
raster source is integrated yet.

**Boundary handling:** a point near a cell edge is assigned to whichever
cell contains its exact coordinate — no special edge-smoothing is
applied. This is a known simplification (a bird just across a cell
boundary from an airport is treated as "not in this cell" even though
it may be closer than a bird well inside the cell) — acceptable at
resolution 5's ~8.5 km scale for an exploratory index, but worth
revisiting if resolution 7 or finer is used for a more precision-
sensitive future analysis.

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
