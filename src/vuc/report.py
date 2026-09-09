from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from vuc.frames import format_timestamp


def parse_report_json(text: str) -> dict[str, Any]:
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("model response contains no JSON object") from None
        data = json.loads(stripped[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("report JSON must be an object")
    return data


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: float, duration_s: float) -> float:
    return min(max(value, 0.0), duration_s)


def normalize_report(
    report: dict[str, Any],
    *,
    meta: dict[str, Any],
) -> dict[str, Any]:
    report.setdefault("title", "Video report")
    report.setdefault("one_line_summary", "")
    report.setdefault("sections", [])
    report.setdefault("key_moments", [])
    report["meta"] = meta
    duration_s = _number(meta.get("duration_s"))
    for section in report["sections"]:
        if not isinstance(section, dict):
            continue
        section["start_s"] = round(_clamp(_number(section.get("start_s")), duration_s))
        section["end_s"] = round(_clamp(_number(section.get("end_s")), duration_s))
        citations = []
        for citation in section.get("citations", []):
            if isinstance(citation, str):
                citation = {"claim": citation}
            if not isinstance(citation, dict):
                continue
            start_s = _number(citation.get("start_s", citation.get("timestamp_s")))
            end_s = _number(citation.get("end_s", citation.get("timestamp_s", start_s)), start_s)
            citations.append(
                {
                    "claim": str(citation.get("claim") or citation.get("text") or ""),
                    "start_s": round(_clamp(start_s, duration_s)),
                    "end_s": round(_clamp(max(end_s, start_s), duration_s)),
                }
            )
        section["citations"] = citations
    for moment in report.get("key_moments", []):
        if isinstance(moment, dict):
            timestamp_s = _number(moment.get("timestamp_s", moment.get("start_s")))
            moment["timestamp_s"] = round(_clamp(timestamp_s, duration_s))
    return report


def _citation_timestamp(citation: dict[str, Any]) -> str:
    start = _number(citation.get("start_s"))
    end = _number(citation.get("end_s"), start)
    if round(start) == round(end):
        return f"[{format_timestamp(start)}]"
    return f"[{format_timestamp(start)}–{format_timestamp(end)}]"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [f"# {report.get('title', 'Video report')}", ""]
    summary = str(report.get("one_line_summary") or "")
    if summary:
        lines.extend([summary, ""])
    for section in report.get("sections", []):
        if not isinstance(section, dict):
            continue
        lines.extend([f"## {section.get('title', 'Section')}", ""])
        section_summary = str(section.get("summary") or "")
        if section_summary:
            lines.extend([section_summary, ""])
        for citation in section.get("citations", []):
            claim = citation.get("claim") or citation.get("text") or ""
            lines.append(f"- {_citation_timestamp(citation)} {claim}")
        if section.get("citations"):
            lines.append("")
    lines.extend(["## 핵심 장면", ""])
    for moment in report.get("key_moments", []):
        if not isinstance(moment, dict):
            continue
        timestamp = _number(moment.get("timestamp_s", moment.get("start_s")))
        title = moment.get("title") or moment.get("summary") or "Key moment"
        summary = moment.get("summary") or ""
        suffix = f" — {summary}" if summary and summary != title else ""
        lines.append(f"- [{format_timestamp(timestamp)}] {title}{suffix}")
    if not report.get("key_moments"):
        lines.append("- 없음")
    return "\n".join(lines).rstrip() + "\n"


def write_report(report: dict[str, Any], directory: Path) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / "report.json"
    markdown_path = directory / "report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return markdown_path, json_path
