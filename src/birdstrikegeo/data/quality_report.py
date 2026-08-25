"""
data/quality_report.py
-------------------------
Builds JSON + Markdown data-quality reports. Deliberately generic (a
section-based builder) because FAA, Trektellen, and geospatial-join
quality reports all need different fields, but should all end up as one
consistent, readable document.

Guiding rule (Section 20 of the project spec): never silently drop
records. Every exclusion this project makes (unknown damage target,
unresolved airport, invalid coordinates, ...) must show up as a counted,
explained line in this report.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def sha256_of_file(path: str | Path) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class QualityReportBuilder:
    title: str
    sample_data_flag: bool
    sections: dict = field(default_factory=dict)
    sources: list = field(default_factory=list)

    def add_source(self, name: str, path: str | Path, row_count: int) -> None:
        path = Path(path)
        self.sources.append(
            {
                "name": name,
                "path": str(path),
                "sha256": sha256_of_file(path) if path.exists() else None,
                "row_count": row_count,
            }
        )

    def add_section(self, name: str, content) -> None:
        self.sections[name] = content

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sample_data_flag": self.sample_data_flag,
            "disclaimer": "SAMPLE DATA — NOT A REAL RESULT" if self.sample_data_flag else None,
            "sources": self.sources,
            "sections": self.sections,
        }

    def write(self, json_path: str | Path, markdown_path: str | Path | None = None) -> None:
        json_path = Path(json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

        if markdown_path:
            markdown_path = Path(markdown_path)
            markdown_path.parent.mkdir(parents=True, exist_ok=True)
            markdown_path.write_text(self._to_markdown())

    def _to_markdown(self) -> str:
        lines = [f"# {self.title}", ""]
        if self.sample_data_flag:
            lines += ["> **SAMPLE DATA — NOT A REAL RESULT**", ""]
        lines += [f"Generated: {datetime.now(timezone.utc).isoformat()}", ""]

        lines += ["## Sources", ""]
        for s in self.sources:
            lines += [f"- `{s['path']}` — {s['row_count']} rows — sha256 `{s['sha256']}`"]
        lines += [""]

        for name, content in self.sections.items():
            lines += [f"## {name.replace('_', ' ').title()}", ""]
            lines += [_render_section(content)]
            lines += [""]

        return "\n".join(lines)


def _render_section(content) -> str:
    if isinstance(content, dict):
        return "\n".join(f"- **{k}**: {v}" for k, v in content.items())
    if isinstance(content, list):
        return "\n".join(f"- {row}" for row in content)
    return str(content)


def missingness_by_column(df: pd.DataFrame) -> dict:
    return {col: int(df[col].isna().sum()) for col in df.columns}


def value_counts_section(series: pd.Series, top_n: int = 20) -> dict:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).head(top_n).items()}
