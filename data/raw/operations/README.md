# Operations (exposure) data

BirdStrikeNet only computes a "strikes per 10,000 operations" rate when
one of these two files has a row that actually matches the airport,
year/month, and population being displayed. No file, or no matching
row → the dashboard says exactly what's missing, never an
interpolated/estimated rate. See `birdstrikegeo/dashboard/exposure.py`.

Neither file is here yet (both are header-only templates) - this
project has no exposure data as of this writing. See `DATA_CARD.md`
section 6 and `README.md`'s "Scientific limitations" for why a strike-
*rate* model has always been deliberately withheld until this exists.

## 1. `atads_monthly.csv` — FAA ATADS monthly operations (towered fields)

**Columns:** `airport_id,year,month,ga_ops` — `ga_ops` is total GA
(itinerant + local) operations for that airport-month. Supports a
seasonal (multi-month) rate, since monthly counts can be summed.

**Where to get it:** FAA's Air Traffic Activity Data System, served
through the ASPM portal — https://www.aspm.faa.gov/ . As of this
writing, FAA's own pages give **conflicting signals** about whether
ATADS is still the current public-release path or has been superseded
by OPSNET (which requires a login): the ATADS "About" page
(https://www.aspm.faa.gov/getInfo.asp?id=atads) still describes it as
the official public-release NAS operations count, while the ASPM
homepage points to OPSNET instead. **Verify on aspm.faa.gov which
module currently serves public monthly operations counts before
pulling data** — don't trust this README's description of the exact
click-path, since the portal has clearly changed since these pages were
written. ATADS has historically only covered towered fields (it's an
air-traffic-control reporting system) — of the Eastern Shore roster in
`configs/eastern_shore_airports.yaml`, that's **KSBY** and **KESN**
only.

## 2. `faa_5010_annual.csv` — FAA Form 5010 annual GA operations estimate

**Columns:** `airport_id,year,ga_ops_estimate` — a **self-reported,
annual-only** estimate. Never use this for a seasonal/sub-year rate;
`dashboard.exposure.compute_strike_rate` enforces that by construction.

**Where to get it:** the FAA's Airport Data and Information Portal
(ADIP) — https://adip.faa.gov — Facility Search, then the airport's
Airport Master Record (5010). As of this writing, FAA's paper Form
5010-1 was cancelled 2024-07-25 and its data now lives only in ADIP
directly, so **the exact portal navigation may have changed since this
was written** — search ADIP's facility search for each airport's LID
and look for GA itinerant/local operations fields in its current Airport
Master Record. This covers every airport in the roster, towered or not,
since every public-use airport has a 5010 record (private strips like
Massey Aerodrome generally do not, and will have no match here).

## Adding data

1. Download the real values for the airport(s)/period(s) you need from
   the sources above.
2. Append rows to the matching CSV, keeping the existing header.
3. Re-run the app — `dashboard.exposure` picks up new rows automatically,
   no code change needed.
