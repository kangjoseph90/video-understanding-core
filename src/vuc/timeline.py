"""One timeline, one unit: float seconds from the start of the video.

Sources disagree about how to spell a point in time -- funasr speaks
milliseconds, ffmpeg speaks seconds -- so the conversion happens here, once, at
the ingest boundary. Nothing downstream has to ask what unit a number is in.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass


def from_milliseconds(value: float | int | str) -> float:
    return round(float(value) / 1000.0, 3)


@dataclass(frozen=True, order=True)
class Span:
    """A half-open interval on the video timeline, in seconds."""

    start: float
    end: float

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"span start must not be negative: {self.start}")
        if self.end < self.start:
            raise ValueError(f"span end {self.end} precedes start {self.start}")
        # Coerced so every timestamp leaving this module is a float, whether it
        # arrived as an int literal, a numpy scalar or a string.
        object.__setattr__(self, "start", round(float(self.start), 3))
        object.__setattr__(self, "end", round(float(self.end), 3))

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def overlaps(self, other: Span) -> bool:
        return self.start < other.end and other.start < self.end

    def clamped(self, duration_s: float) -> Span:
        start = min(max(0.0, self.start), duration_s)
        return Span(start, min(max(start, self.end), duration_s))

    def rounded_out(self, duration_s: float) -> Span:
        """Snap to whole seconds outwards, never inwards.

        Rounding is what stops a detector's 0.42s fragments from becoming their
        own index lines, but rounding the usual way would clip the first
        syllable off a region. Floor the start and ceil the end and the region
        can only ever grow, so no speech is lost to tidier numbers -- except at
        the very end, where it stops at the video rather than claiming a second
        that is not there.
        """
        return Span(
            float(math.floor(self.start)),
            float(min(math.ceil(self.end), max(self.end, duration_s))),
        )

    def to_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end}


def merge_spans(spans: Iterable[Span], *, gap_s: float = 0.0) -> list[Span]:
    """Union of spans, joining neighbours separated by at most gap_s."""
    merged: list[Span] = []
    for span in sorted(spans):
        if merged and span.start - merged[-1].end <= gap_s:
            previous = merged.pop()
            merged.append(Span(previous.start, max(previous.end, span.end)))
        else:
            merged.append(span)
    return merged


def complement(spans: Sequence[Span], *, duration_s: float) -> list[Span]:
    """The stretches of the timeline the given spans do not cover."""
    gaps: list[Span] = []
    cursor = 0.0
    for span in merge_spans(spans):
        if span.start > cursor:
            gaps.append(Span(cursor, span.start))
        cursor = max(cursor, span.end)
    if duration_s > cursor:
        gaps.append(Span(cursor, duration_s))
    return gaps


def split_long(spans: Iterable[Span], *, max_s: float) -> list[Span]:
    """Cut spans longer than max_s into equal pieces no longer than max_s.

    A region that swallows ten minutes of a video is useless as an ASR window
    and useless as a citation target, so length is capped here rather than
    trusted to whatever the detector felt like emitting.
    """
    if max_s <= 0:
        raise ValueError("max_s must be positive")
    pieces: list[Span] = []
    for span in spans:
        if span.duration <= max_s:
            pieces.append(span)
            continue
        count = math.ceil(span.duration / max_s)
        step = span.duration / count
        for index in range(count):
            start = round(span.start + index * step, 3)
            end = span.end if index + 1 == count else round(span.start + (index + 1) * step, 3)
            pieces.append(Span(start, end))
    return pieces
