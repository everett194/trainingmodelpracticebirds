"""
cli.py
------
Command-line entry point:

    python -m birdstrikegeo.cli generate-sample-data
    python -m birdstrikegeo.cli inspect-data [--mode sample|real]
    python -m birdstrikegeo.cli validate-data [--mode sample|real]
    python -m birdstrikegeo.cli prepare-data --mode sample|real
    python -m birdstrikegeo.cli build-geo-features --mode sample|real
    python -m birdstrikegeo.cli train --task damage --mode sample|real
    python -m birdstrikegeo.cli evaluate --task damage --mode sample|real
    python -m birdstrikegeo.cli export-layers --mode sample|real
    python -m birdstrikegeo.cli run-app

Each subcommand is a thin wrapper around the corresponding script in
scripts/ (run as a subprocess with the same Python interpreter), so
`python scripts/train_models.py ...` and
`python -m birdstrikegeo.cli train ...` do exactly the same thing - the
CLI is a convenience layer, not a second implementation.

`--mode real` commands fail gracefully (via each script's own checks,
and scripts/validate_data.py specifically) with an exact list of missing
files and a pointer to DATA_DOWNLOAD_GUIDE.md, rather than crashing.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _run_script(script_name: str, extra_args: list[str]) -> int:
    cmd = [sys.executable, str(SCRIPTS_DIR / script_name), *extra_args]
    result = subprocess.run(cmd)
    return result.returncode


def cmd_generate_sample_data(args: argparse.Namespace) -> int:
    return _run_script("generate_sample_data.py", [])


def cmd_inspect_data(args: argparse.Namespace) -> int:
    return _run_script("inspect_raw_schemas.py", ["--mode", args.mode])


def cmd_validate_data(args: argparse.Namespace) -> int:
    return _run_script("validate_data.py", ["--mode", args.mode])


def cmd_prepare_data(args: argparse.Namespace) -> int:
    return _run_script("prepare_data.py", ["--mode", args.mode])


def cmd_build_geo_features(args: argparse.Namespace) -> int:
    return _run_script("build_geospatial_features.py", ["--mode", args.mode])


def cmd_train(args: argparse.Namespace) -> int:
    return _run_script("train_models.py", ["--task", args.task, "--mode", args.mode])


def cmd_evaluate(args: argparse.Namespace) -> int:
    return _run_script("evaluate_models.py", ["--task", args.task, "--mode", args.mode])


def cmd_export_layers(args: argparse.Namespace) -> int:
    return _run_script("export_arcgis_layers.py", ["--mode", args.mode])


def cmd_run_app(args: argparse.Namespace) -> int:
    print(f"[cli] Starting Flask app (mode={args.mode} is informational only - app.py "
          f"auto-detects sample vs. real data/models at request time). "
          f"Open http://localhost:5001 once it starts.")
    cmd = [sys.executable, str(REPO_ROOT / "app.py")]
    result = subprocess.run(cmd)
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m birdstrikegeo.cli", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("generate-sample-data").set_defaults(func=cmd_generate_sample_data)

    for name, func in [
        ("inspect-data", cmd_inspect_data),
        ("validate-data", cmd_validate_data),
        ("prepare-data", cmd_prepare_data),
        ("build-geo-features", cmd_build_geo_features),
        ("export-layers", cmd_export_layers),
    ]:
        sub = subparsers.add_parser(name)
        sub.add_argument("--mode", choices=["sample", "real"], default="sample")
        sub.set_defaults(func=func)

    for name, func in [("train", cmd_train), ("evaluate", cmd_evaluate)]:
        sub = subparsers.add_parser(name)
        sub.add_argument("--task", choices=["damage"], default="damage")
        sub.add_argument("--mode", choices=["sample", "real"], default="sample")
        sub.set_defaults(func=func)

    run_app = subparsers.add_parser("run-app")
    run_app.add_argument("--mode", choices=["sample", "real"], default="sample")
    run_app.set_defaults(func=cmd_run_app)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
