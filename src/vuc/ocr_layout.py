"""Frame-local layout and evidence-backed block assembly.

Only boxes observed together can establish reading order. Temporal identities
keep changing neighbours independent; their last/best boxes are never treated
as a synthetic screenshot. No screen-type or vocabulary classifier is used.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from statistics import median

from vuc.ocr import OCRLine, normalize_text, text_key


def neighbours(a: OCRLine, b: OCRLine) -> bool:
    """Stacked, aligned lines of comparable size; never adjacent columns.

    Projected boxes of slanted rows can overlap vertically. Compare centres
    and indentation rather than requiring disjoint axis-aligned boxes.
    """
    ha, hb = max(a.height, 0.001), max(b.height, 0.001)
    if min(ha, hb) / max(ha, hb) < 0.5:
        return False
    dy = (b.top + b.bottom - a.top - a.bottom) / 2
    if not 0.3 * min(ha, hb) - 1e-6 <= dy <= 1.6 * max(ha, hb):
        return False
    wa, wb = max(a.right - a.left, 0.001), max(b.right - b.left, 0.001)
    overlap = max(0.0, min(a.right, b.right) - max(a.left, b.left))
    aligned = min(abs(a.left - b.left), abs(a.right - b.right)) <= 0.6 * max(ha, hb)
    centred = abs((a.left + a.right - b.left - b.right) / 2) <= 0.3 * max(ha, hb)
    hanging_indent = (
        0 <= b.left - a.left <= 1.2 * max(ha, hb) and wa >= 1.5 * wb and b.right <= a.right
    )
    return overlap / min(wa, wb) >= 0.7 and (aligned or centred or hanging_indent)


def frame_links(
    lines: Sequence[OCRLine],
    *,
    compatible: Callable[[OCRLine, OCRLine], bool] | None = None,
) -> list[tuple[int, int]]:
    """Nearest successor in the same column, before temporal grouping."""
    edges = []
    for i, a in enumerate(lines):
        candidates = [j for j, b in enumerate(lines) if i != j and neighbours(a, b)]
        if candidates:
            j = min(candidates, key=lambda j: (lines[j].top + lines[j].bottom, lines[j].left))
            # A different-lived intervening row is a boundary, not permission
            # to skip ahead and join text on its other side.
            if compatible is None or compatible(a, lines[j]):
                edges.append((i, j))
    # A shared wide line below two columns must not connect both columns.
    predecessors: dict[int, int] = {}
    for a, b in edges:
        previous = predecessors.get(b)
        if previous is None or lines[a].bottom > lines[previous].bottom:
            predecessors[b] = a
    return [(a, b) for a, b in edges if predecessors[b] == a]


@dataclass(frozen=True)
class RegionLine(OCRLine):
    """One frame's paragraph, with glyph size separate from region extent."""

    rows: tuple[OCRLine, ...] = ()


def frame_regions(
    lines: Sequence[OCRLine],
    *,
    compatible: Callable[[OCRLine, OCRLine], bool] | None = None,
) -> list[OCRLine]:
    owners = list(range(len(lines)))
    groups = {i: {i} for i in range(len(lines))}
    for a, b in frame_links(lines, compatible=compatible):
        ga, gb = owners[a], owners[b]
        if ga == gb:
            continue
        groups[ga].update(groups.pop(gb))
        for member in groups[ga]:
            owners[member] = ga
    regions = []
    for members in groups.values():
        rows = sorted((lines[i] for i in members), key=lambda row: (row.top + row.bottom, row.left))
        regions.append(region_from_rows(rows))
    return regions


def contains_region(host: OCRLine, candidate: OCRLine) -> bool:
    """Every candidate row must be an observed substring, in reading order.

    Missing rows and changed wrapping are detector topology changes, not new
    content. A changed word or number cannot be justified by fuzzy similarity.
    """
    reference = text_key(host.text)
    rows = candidate.rows if isinstance(candidate, RegionLine) else (candidate,)
    cursor = 0
    for row in rows:
        key = text_key(row.text)
        found = reference.find(key, cursor)
        if found < 0:
            return False
        cursor = found + len(key)
    return True


def region_from_rows(rows: Sequence[OCRLine]) -> OCRLine:
    """Rebuild a measured region after removing already represented rows."""
    if len(rows) == 1:
        return rows[0]
    return RegionLine(
        text=" / ".join(normalize_text(row.text) for row in rows),
        confidence=min(1.0 if row.confidence is None else row.confidence for row in rows),
        top=min(row.top for row in rows),
        bottom=max(row.bottom for row in rows),
        left=min(row.left for row in rows),
        right=max(row.right for row in rows),
        text_height=median(row.height for row in rows),
        rows=tuple(rows),
    )


def visible_regions(
    runs: Sequence[tuple[float, float, OCRLine]],
) -> list[tuple[float, float, OCRLine]]:
    """Assign each observed row one owner during every overlapping interval.

    A detector can alternate a paragraph with its component lines. Independent
    tracks may consequently overlap; a partial paragraph is not extra evidence
    while its complete parent is already present. Ownership is spatial and
    temporal, not global text deduplication (table cells remain independent).
    """
    events: dict[float, list[tuple[int, bool]]] = {}
    for i, (start, end, _) in enumerate(runs):
        events.setdefault(start, []).append((i, True))
        events.setdefault(end, []).append((i, False))
    times = sorted(events)
    active: set[int] = set()
    result: list[tuple[float, float, OCRLine]] = []
    last_output: dict[tuple[int, tuple[int, ...]], int] = {}
    for start, end in zip(times, times[1:], strict=False):
        for i, entering in events[start]:
            if entering:
                active.add(i)
            else:
                active.discard(i)
        claimed: list[OCRLine] = []
        for i in sorted(active, key=lambda i: (-len(text_key(runs[i][2].text)), i)):
            region = runs[i][2]
            rows = region.rows if isinstance(region, RegionLine) else (region,)
            residual = []
            indices = []
            for j, row in enumerate(rows):
                if not any(_covers_row(host, row) for host in claimed):
                    residual.append(row)
                    indices.append(j)
            if not residual:
                continue
            claimed.extend(residual)
            identity = (i, tuple(indices))
            previous = last_output.get(identity)
            if previous is not None and result[previous][1] == start:
                result[previous] = (result[previous][0], end, result[previous][2])
            else:
                last_output[identity] = len(result)
                # Keep the temporally measured height/box when nothing changed.
                measured = region if len(residual) == len(rows) else region_from_rows(residual)
                result.append((start, end, measured))
    return result


def _covers_row(host: OCRLine, row: OCRLine) -> bool:
    key = text_key(row.text)
    if not key or key not in text_key(host.text):
        return False
    height = max(0.001, min(host.bottom - host.top, row.bottom - row.top))
    width = max(0.001, min(host.right - host.left, row.right - row.left))
    return (
        min(host.bottom, row.bottom) - max(host.top, row.top) >= 0.5 * height
        and min(host.right, row.right) - max(host.left, row.left) >= 0.8 * width
    )
