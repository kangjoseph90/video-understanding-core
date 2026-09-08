from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from vuc.frames import format_timestamp


class CoverageSource(Protocol):
    def citation_coverage(self, start_s: float, end_s: float) -> float: ...

    def evidence_coverage(self, start_s: float, end_s: float, source: str) -> float: ...


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


def _merged_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start_s, end_s in sorted(intervals):
        if end_s <= start_s:
            continue
        if not merged or start_s > merged[-1][1]:
            merged.append([start_s, end_s])
        else:
            merged[-1][1] = max(merged[-1][1], end_s)
    return [(start_s, end_s) for start_s, end_s in merged]


def report_time_coverage(
    report: dict[str, Any], duration_s: float
) -> tuple[float, list[tuple[float, float]]]:
    if duration_s <= 0:
        return 1.0, []
    intervals = []
    sections = report.get("sections")
    if isinstance(sections, list):
        for section in sections:
            if not isinstance(section, dict):
                continue
            start_s = min(duration_s, max(0.0, _number(section.get("start_s"))))
            end_s = min(duration_s, max(0.0, _number(section.get("end_s"))))
            if end_s > start_s:
                intervals.append((start_s, end_s))
    merged = _merged_intervals(intervals)
    covered_s = sum(end_s - start_s for start_s, end_s in merged)
    gaps = []
    cursor = 0.0
    for start_s, end_s in merged:
        if start_s > cursor:
            gaps.append((cursor, start_s))
        cursor = max(cursor, end_s)
    if cursor < duration_s:
        gaps.append((cursor, duration_s))
    return min(1.0, covered_s / duration_s), gaps


def apply_verification(
    report: dict[str, Any],
    coverage_source: CoverageSource,
    *,
    verify_overlap: float,
) -> dict[str, Any]:
    sections = report.get("sections")
    if not isinstance(sections, list):
        report["sections"] = []
        sections = report["sections"]
    for section in sections:
        if not isinstance(section, dict):
            continue
        citations = section.get("citations")
        if not isinstance(citations, list):
            section["citations"] = []
            continue
        normalized = []
        for citation in citations:
            if isinstance(citation, str):
                citation = {"claim": citation}
            if not isinstance(citation, dict):
                continue
            start_s = _number(citation.get("start_s", citation.get("timestamp_s")))
            end_s = _number(citation.get("end_s", citation.get("timestamp_s", start_s)), start_s)
            coverage = coverage_source.citation_coverage(start_s, end_s)
            evidence = citation.get("evidence_span")
            if isinstance(evidence, dict):
                evidence_start_s = _number(evidence.get("start_s"))
                evidence_end_s = _number(evidence.get("end_s"), evidence_start_s)
                evidence_source = str(evidence.get("source") or "")
                evidence_overlap = _interval_overlap_ratio(
                    start_s,
                    end_s,
                    evidence_start_s,
                    evidence_end_s,
                )
                evidence_query_overlap = coverage_source.evidence_coverage(
                    evidence_start_s,
                    evidence_end_s,
                    evidence_source,
                )
                citation["evidence_span"] = {
                    "start_s": round(evidence_start_s),
                    "end_s": round(evidence_end_s),
                    "source": evidence_source,
                }
            else:
                evidence_overlap = 0.0
                evidence_query_overlap = 0.0
            citation.update(
                start_s=round(start_s),
                end_s=round(end_s),
                verification_overlap=round(coverage, 4),
                evidence_overlap=round(evidence_overlap, 4),
                evidence_query_overlap=round(evidence_query_overlap, 4),
                verified=(
                    coverage >= verify_overlap
                    and evidence_overlap >= verify_overlap
                    and evidence_query_overlap >= verify_overlap
                ),
            )
            normalized.append(citation)
        section["citations"] = normalized
    return report


def _interval_overlap_ratio(
    start_s: float,
    end_s: float,
    evidence_start_s: float,
    evidence_end_s: float,
) -> float:
    if end_s < start_s:
        start_s, end_s = end_s, start_s
    if evidence_end_s < evidence_start_s:
        evidence_start_s, evidence_end_s = evidence_end_s, evidence_start_s
    if start_s == end_s:
        return 1.0 if evidence_start_s <= start_s <= evidence_end_s else 0.0
    overlap = max(0.0, min(end_s, evidence_end_s) - max(start_s, evidence_start_s))
    return min(1.0, overlap / (end_s - start_s))


def normalize_report(
    report: dict[str, Any],
    coverage_source: CoverageSource,
    *,
    verify_overlap: float,
    meta: dict[str, Any],
) -> dict[str, Any]:
    report.setdefault("title", "Video report")
    report.setdefault("one_line_summary", "")
    report.setdefault("sections", [])
    report.setdefault("key_moments", [])
    claims = report.setdefault("unverified_claims", [])
    if not isinstance(claims, list):
        claims = [str(claims)]
        report["unverified_claims"] = claims
    report["meta"] = meta
    apply_verification(report, coverage_source, verify_overlap=verify_overlap)
    for section in report["sections"]:
        if not isinstance(section, dict):
            continue
        section["start_s"] = round(_number(section.get("start_s")))
        section["end_s"] = round(_number(section.get("end_s")))
        for citation in section.get("citations", []):
            if citation.get("verified"):
                continue
            claim = str(citation.get("claim") or citation.get("text") or "Unverified citation")
            if claim not in claims:
                claims.append(claim)
    for moment in report.get("key_moments", []):
        if isinstance(moment, dict):
            moment["timestamp_s"] = round(_number(moment.get("timestamp_s", moment.get("start_s"))))
    duration_s = _number(meta.get("duration_s"))
    coverage_ratio, uncovered = report_time_coverage(report, duration_s)
    meta["coverage_ratio"] = round(coverage_ratio, 4)
    meta["uncovered_spans"] = [
        {"start_s": round(start_s), "end_s": round(end_s)} for start_s, end_s in uncovered
    ]
    if coverage_ratio < 0.9 and uncovered:
        gap_text = ", ".join(
            f"[{format_timestamp(start_s)}–{format_timestamp(end_s)}]"
            for start_s, end_s in uncovered
        )
        message = f"보고서 섹션에서 다루지 않은 구간: {gap_text}"
        if message not in claims:
            claims.append(message)
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
            marker = "verified" if citation.get("verified") else "unverified"
            claim = citation.get("claim") or citation.get("text") or ""
            lines.append(f"- {_citation_timestamp(citation)} {claim} ({marker})")
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
    lines.extend(["", "## 검증되지 않은 부분", ""])
    claims = report.get("unverified_claims", [])
    if claims:
        lines.extend(f"- {claim}" for claim in claims)
    else:
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
