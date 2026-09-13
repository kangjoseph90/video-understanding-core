"""The indexes as the model reads them.

One renderer per index, used both for the files written to the cache and for
the prompt the agent is given, so what is on disk is exactly what was sent.

The audio index and the text index are rendered separately and never
interleaved. They are independent observers of the same video with their own
line formats, and putting them under one heading made the on-screen text read
as a continuation of the transcript.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterable

from vuc.frames import format_span
from vuc.models import Segment, TextCue

AUDIO_INDEX_LABEL = "음성 인덱스 (구간 표기는 [시작-끝], 단위는 초)"
TEXT_INDEX_LABEL = (
    "화면 텍스트 인덱스 ([시작-끝, position, size] 내용, 반복 구간은 ;로 구분, 단위는 초)"
)


def text_location(cue: TextCue) -> str:
    return f"{cue.position}, {cue.size}"


def audio_line(segment: Segment) -> str:
    """One region, one line.

    A transcript is shown with the language it was recognised as; a region with
    no words shows the labels instead.
    """
    parts = []
    if segment.text and segment.language not in {"", "unknown"}:
        parts.append(f"<{segment.language}>")
    parts.extend(f"<{event}>" for event in segment.events)
    if segment.text:
        parts.append(segment.text)
    return f"[{format_span(segment.start, segment.end)}] {' '.join(parts)}"


def render_audio_index(segments: Iterable[Segment]) -> str:
    """The VAD-split timeline in order, minus the regions that say nothing.

    A region where neither a transcript nor a tag came back contributes no
    information, and printing it anyway put an empty row in the index for every
    stretch nothing could be named in. The gap it leaves is visible in the
    timestamps of the lines either side, which is all a reader needed from it.
    """
    return "\n".join(audio_line(segment) for segment in segments if not segment.is_empty)


def text_cue_line(cue: TextCue) -> str:
    return f"[{format_span(cue.start, math.ceil(cue.end))}, {text_location(cue)}] {cue.text}"


def render_text_index(cues: Iterable[TextCue]) -> str:
    """Compact evidence, with every recurrence's actual interval preserved.

    Coarse position and line size give the agent visual context. Repeated text
    shares one row; when its location changes, each interval owns its label.
    """
    ordered = sorted(cues, key=lambda c: (c.start, c.position, c.text))
    groups: dict[str, list[TextCue]] = {}
    for cue in ordered:
        # Preserve punctuation: 0.5, 05, A+B and A-B must stay distinct.
        # ` / ` is our line/block separator, so detector fragmentation must not
        # make the same text a different recurrence. Slashes inside URLs and
        # source text without delimiter spacing remain significant.
        comparable = cue.text.replace(" / ", " ")
        key = re.sub(r"\s+", "", unicodedata.normalize("NFKC", comparable)).casefold()
        groups.setdefault(key, []).append(cue)
    entries: list[tuple[float, str]] = []
    for group in groups.values():
        if len(group) < 2:
            entries.extend((c.start, text_cue_line(c)) for c in group)
            continue
        intervals: list[tuple[float, float, str]] = []
        for cue in group:
            location = text_location(cue)
            if intervals and cue.start <= intervals[-1][1] and location == intervals[-1][2]:
                start, end, _ = intervals[-1]
                intervals[-1] = (start, max(end, cue.end), location)
            else:
                intervals.append((cue.start, cue.end, location))
        if len({location for _, _, location in intervals}) == 1:
            spans = "; ".join(format_span(start, math.ceil(end)) for start, end, _ in intervals)
            spans += f", {intervals[0][2]}"
        else:
            spans = "; ".join(
                f"{format_span(start, math.ceil(end))}, {location}"
                for start, end, location in intervals
            )
        representative = max(
            group,
            key=lambda c: (len(c.text.replace(" / ", " ")), -c.text.count(" / ")),
        )
        entries.append((group[0].start, f"[{spans}] {representative.text}"))
    return "\n".join(line for _, line in sorted(entries, key=lambda x: x[0]))
