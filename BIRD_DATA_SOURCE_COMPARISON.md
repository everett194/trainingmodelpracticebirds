# Bird Distribution/Migration Data Source Comparison

Phase 4 research deliverable: what actually underlies the public bird
migration/abundance maps this project might use, before integrating any
of them. **Nothing large was downloaded as part of this research** —
only `data/raw/airports.csv` (small, public-domain, unrelated to bird
distribution) was pulled in; see `DATA_DOWNLOAD_GUIDE.md`. No
interactive map was scraped — every source below was checked for a
documented API/bulk-download path first, per project policy (see
[[feedback-no-trektellen-scraping]]-style reasoning applied here too).

## Comparison table

| Source | Variable represented | Spatial resolution | Temporal resolution | Coverage | Access method | Licensing | Strengths | Limitations | Suitability |
|---|---|---|---|---|---|---|---|---|---|
| **eBird Status & Trends** | Modeled relative abundance (a statistical estimate from checklist data + remote-sensing covariates, NOT raw counts) | 3 km raster | Weekly (52 weeks/year) | The Americas (most species); global for some | Free API + `ebirdst` R package; requires a personal access key (`ebird.org/st/request`) | Research/non-commercial use documented; **exact redistribution terms need to be read directly from the access-key agreement before any value from this is embedded in a redistributable model artifact** — not confirmed in this research pass (eBird's FAQ page blocked automated fetching) | By far the best-documented, highest-resolution, most defensible population-level abundance surface available for the whole US, not just near sparse Trektellen sites | Requires per-user access key (a real access gate, like Trektellen's authorized-export requirement); modeled, not observed, values — presenting them as ground truth would overstate precision; weekly not hourly/daily | **High** — best candidate for the secondary hazard/activity index's bird-abundance input, once the access-key terms are read and accepted |
| **BirdCast (Cornell Lab)** | Nocturnal migration *intensity/traffic rate* (birds actively in flight, not ground abundance) | Radar-network based; underlying vertical profiles from NEXRAD at ~200m altitude bins, 5-min cadence | Forecast maps: every 6 hours; dashboard: nightly | CONUS only (143 NEXRAD stations) | **No simple bulk-download API found** for the public dashboard/forecast maps in this research pass. The underlying quantity (migration traffic rate) is reproducible from public-domain NEXRAD Level II/III data via the open-source `bioRad`/`vol2bird` pipeline (used in peer-reviewed aeroecology literature), but that is a substantial signal-processing engineering effort, not a simple download. | Answers a genuinely different, arguably more strike-relevant question than eBird (birds actively flying vs. birds present on the ground); public-domain raw radar data | The dashboard itself is a public-facing interactive map — per project policy this was NOT scraped; no confirmed simple API means this is a **build, not a download** | **Deferred** — worth a dedicated future spike into `bioRad`, not a Phase 4 integration |
| **USGS North American Breeding Bird Survey (BBS)** | Point-count abundance by species, along fixed roadside routes | Route-level (routes ~39 km long, 50 stops per route) | **Annual**, June only (breeding season) | US + Canada | Public FTP / `bbsAssistant` R package | US Government work — **public domain** | Long time series (1966–2025), unambiguous license, route-level geographic precision | Once-a-year snapshot only — no within-season or migration-timing signal at all; roadside-route sampling may not be near any given airport; breeding-season bias (won't capture migration-period activity, which is what matters most for strike risk) | **Low** for the hazard index (wrong temporal grain); **possible** as a coarse regional species-presence prior |
| **Movebank** | Individual tagged-animal GPS/sensor tracks | Point-level (per individual, per fix) | Per-fix (varies by tag, often sub-hourly) | Wherever a contributing study tagged animals — patchy, not systematic | Public REST API (JSON/CSV), per-study permission (much is open, some restricted by the data owner) | Varies per study — must be checked per dataset, no single blanket license | Real observed movement, not modeled; can show migration corridors/timing directly | Small number of tagged individuals per species/region — not a population-density surface; strong selection bias (only species/regions someone chose to tag); not usable to estimate "how many birds are near this airport" | **Low** as a direct model input; see Phase 6 — better suited to validation/interpretation of a population-level index than as a training feature |
| **Motus Wildlife Tracking System** | Automated radio-telemetry detections of tagged individuals at fixed receiver stations | Point-level (receiver station locations) | Per-detection (continuous, station-dependent) | Uneven — depends entirely on receiver station placement (denser in eastern N. America) | `motus` R package, most data openly accessible | Per-project, generally open for research use | Good for migration timing/route studies at tagged sites; free | Same individual-vs-population caveat as Movebank; receiver coverage is not uniform and mostly not airport-colocated | **Low** as a direct model input; same Phase 6 role as Movebank |
| **Audubon Bird Migration Explorer** | A compiled visualization drawing on ~720 external studies (incl. eBird, Movebank, and others) plus Audubon's own peer-reviewed climate-vulnerability analysis (Bateman et al., ~140M observations, 604 species) | Varies by underlying layer (inherits eBird's 3km for range/abundance layers) | Varies by underlying layer | Western Hemisphere (varies by species/study) | **No single bulk-download API for the explorer itself** — it is a visualization aggregating other sources; the page states "requests for data should be directed to the contributing data holder." The climate-vulnerability analysis specifically is a separate published dataset with its own citation. | Mixed — inherits each underlying source's license; Audubon's own vulnerability analysis has its own terms (check the publication) | Good single place to see *what exists*; genuinely useful as a discovery layer | **Confirms the Phase 4 question's premise: the map itself is not a primary source.** For abundance/range, the real primary source is eBird Status & Trends (see above) — go there directly rather than through this map. For climate vulnerability specifically, go to the underlying published analysis, not the map. | **Low** as a direct integration target; **high** as a pointer to which primary sources to use instead |
| **Esri Living Atlas (bird layers)** | Hosted ArcGIS feature/image services, themselves built from eBird Status & Trends data | Inherits eBird's 3km | Inherits eBird's weekly cadence | Inherits eBird's coverage | ArcGIS REST API (this project's existing `geo/arcgis_adapter.py` already knows how to read a public ArcGIS Feature Service) | Esri Living Atlas terms of use (separate from eBird's own terms — layer-dependent) | This project **already has working code** (`arcgis_adapter.py`) to consume a public ArcGIS Feature Service with no ArcGIS license required | Still ultimately eBird data underneath — no informational advantage over going to eBird directly, only an integration-convenience one (reuses existing adapter code) | **Medium** — worth checking whether a specific Living Atlas bird layer is easier to integrate than eBird's own API, given the adapter already exists, but it doesn't change the underlying data-quality picture |

## Answering Phase 4's specific questions

- **What underlies the Audubon map?** Not one thing — eBird Status &
  Trends (range/abundance layers) plus ~720 independently contributed
  studies (including Movebank tracking data) plus Audubon's own
  published climate-vulnerability analysis. There is no single
  "Audubon dataset" to download; the map is a visualization layer.
- **Is there a better primary source behind the visual map?** Yes —
  for abundance/range, **eBird Status & Trends directly**. For climate
  vulnerability, the underlying published Audubon analysis directly
  (not yet located as a downloadable table in this pass — would need a
  follow-up search for the paper's supplementary data).
- **Model-generated values as inputs, legally and statistically?**
  eBird S&T values are themselves model outputs (not raw counts) — using
  them as an input to *this* project's model would be modeling on top
  of a model. That's not disqualifying (it's normal in ecology — this
  project's own hazard index would do the same thing one layer further
  up), but it must be disclosed exactly that way, the same way this
  project already discloses "this is an index, not a probability."
  Legal redistribution terms are **not yet confirmed** — must be read
  from eBird's actual access-key Terms of Use before use.
- **Uncertainty estimates?** eBird Status & Trends does publish
  confidence-region information for some products (not independently
  verified in this pass — confirm against the `ebirdst` package
  documentation before relying on it). BirdCast forecasts are
  probabilistic by construction but no per-pixel uncertainty band was
  confirmed. Movebank/Motus are raw detections (no "uncertainty" in the
  modeled sense — the uncertainty there is about sample representativeness,
  not measurement error).

## Recommendation for Phase 2/5

**eBird Status & Trends is the clear best candidate** to eventually
replace or supplement the single-station Trektellen FBBO data as the
bird-abundance input to the secondary hazard index — nationwide
coverage (vs. one station), weekly resolution (vs. monthly totals), and
a well-documented public methodology. It requires: (1) requesting a free
access key, (2) reading its actual Terms of Use for redistribution
before any derived value is committed to this repo or served publicly,
and (3) treating its abundance values explicitly as *model output*, not
ground truth, in every place this project's own documentation makes
that distinction for its own outputs.

This is a **recommendation to bring to the still-open Phase 2 temporal-
resolution decision**, not something implemented in this pass — no
eBird data was downloaded (an access key was not requested), consistent
with "do not download large datasets unless clearly necessary."

## Phase 6: individual tracking vs. population maps

Two genuinely different data types are on the table, and this project
treats them differently on purpose:

- **Individual banded/tracked-bird records** (Movebank, Motus): a
  relatively small number of specific tagged animals, observed wherever
  a contributing study happened to tag and follow them. Excellent for
  studying *movement, timing, and route* for the species/individuals
  actually tagged — useless for estimating "how many birds of any
  species are near this airport right now," because the sample is not
  a population survey, it's whichever individuals someone chose to tag.
- **Population-density / relative-abundance surfaces** (Trektellen
  counts, eBird Status & Trends): estimate how many birds are present
  in an area, from either direct counts (Trektellen) or a statistical
  model over community-science checklists (eBird). This is the right
  data type for a bird-*activity* index — it answers "how much bird
  presence" questions, not "which individual went where."

**Recommendation: defer individual tracking data (Movebank/Motus) from
this project's first version.** Reasons:

- **Sample size & coverage**: a handful to a few hundred tagged
  individuals per species/region cannot support a population-density
  estimate near an arbitrary airport — most airports would have zero
  tagged individuals ever detected nearby, which is a coverage gap, not
  a "zero activity" signal.
- **Selection bias**: which species and individuals get tagged reflects
  researcher interest and logistics (accessibility, funding, species of
  conservation concern), not a representative sample of aviation-
  relevant bird activity (e.g. common airport-hazard species like
  gulls, geese, and vultures are not the primary focus of most tracking
  studies).
- **Temporal coverage**: tracking devices have battery/attachment
  lifespans measured in months to a few years per individual — this
  does not build up to a stable multi-year climatology the way repeated
  Trektellen seasons or eBird's compiled checklist history does.
  Computational feasibility is not the limiting factor here (Movebank's
  API is straightforward); the limiting factor is what the data
  actually represents.

**Where individual tracking data WOULD be useful, once other pieces are
in place:** validation/interpretation of a population-level hazard
index (e.g., "does a documented migration corridor from Motus detections
pass near this airport, consistent with what the hazard index already
shows from Trektellen/eBird?"), not as a direct model input. This is
explicitly deferred, not rejected — revisit once a population-level
surface (Trektellen expansion or eBird integration) is the model's
actual bird-abundance input and a specific validation question arises.

## Sources consulted: [eBird Status and Trends FAQ](https://science.ebird.org/en/status-and-trends/faq), [eBird Status and Trends API docs](https://ebird.github.io/ebirdst/articles/api.html), [BirdCast](https://birdcast.org/), [BirdCast Products, Data, and Interpretation](https://birdcast.info/about/review/), [bioRad](https://adriaandokter.com/bioRad/), [USGS Breeding Bird Survey](https://www.usgs.gov/centers/eesc/science/north-american-breeding-bird-survey), [Movebank API docs](https://github.com/movebank/movebank-api-doc), [Motus data access](https://motus.org/explore/), [Audubon Bird Migration Explorer — Data Providers](https://explorer.audubon.org/about/dataproviders), [Esri × Audubon Bird Migration Explorer](https://www.esri.com/about/newsroom/blog/audubon-bird-migration-explorer).
