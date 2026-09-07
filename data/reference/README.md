# data/reference/

Unlike `data/raw/` (real datasets, always gitignored, never committed —
enforced by CI), this directory holds source documents the project
**deliberately commits to version control**: things the user
specifically chose to check in (e.g. by uploading directly to GitHub),
usually because they're small, not bulk/license-restricted in the way
`data/raw/`'s datasets are, and valuable to keep alongside the code that
parses them.

Currently:

- `trektellen_fbbo_annual_banding_totals_2016-2025.pdf` — a multi-year
  (2016-2025) annual banding-totals export for Trektellen site 3460
  (FBBO), parsed by
  `src/birdstrikegeo/data/ingest_trektellen_annual_pdf.py`. See
  `DATA_CARD.md` for details and `scripts/build_trektellen_annual_dataset.py`
  to regenerate the derived dataset from it.

If you add another annual-totals PDF here (e.g. an updated export
covering more years), name it so it matches the glob pattern
`trektellen*annual*.pdf` and re-run
`scripts/build_trektellen_annual_dataset.py` — it will be picked up
automatically, with no code changes needed. See that script's docstring
for the merge policy when two files disagree on the same species/year.
