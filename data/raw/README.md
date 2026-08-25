# data/raw/

This folder is where **real, user-downloaded** datasets go. It is empty
by default (only a `.gitkeep` is tracked) - nothing here is committed to
git, because these files may be large and/or license-restricted.

See `DATA_DOWNLOAD_GUIDE.md` at the repository root for exactly what to
download and where to put it. Expected filenames:

```text
data/raw/faa_wildlife_strikes.csv       (required for Task A: damage prediction)
data/raw/trektellen_sites.csv           (required for Task B: activity index)
data/raw/trektellen_counts.csv          (required for Task B: activity index)
data/raw/airports.geojson               (optional)
data/raw/hourly_weather.parquet         (optional)
data/raw/bts_t100_departures.csv        (optional, reserved for future use)
```

Run `python -m birdstrikegeo.cli validate-data --mode real` after adding
files here - it checks which expected files are present and reports any
that are still missing, with instructions, rather than failing with a
confusing stack trace later in the pipeline.

If you just want to see the software work without downloading anything,
use `--mode sample` everywhere instead, which reads from
`data/sample/` (already committed to this repo).
