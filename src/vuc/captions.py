"""Channel-provided caption tracks, read from a sidecar and normalised.

A caption track is not a fourth transcriber. It is a piece of text of unknown
provenance that arrived with the video, and nothing here decides what it is --
that judgement needs the VAD split and the OCR rows, and lives in
``caption_fusion``. This module's job is to get the text into a shape that can
be judged: entities unescaped, display markup gone, one timestamp per token.

Two facts about YouTube shaped the whole module.

The first is that a language has either a human track or a machine one, never
both. When someone uploads captions for a language, YouTube stops exposing its
own ASR for it, and ``automatic_captions[lang]`` becomes a format alias of the
same file. Across the field-eval manifest the two came back word for word
identical in every video that had both, and differed only by ``&nbsp;`` before
normalisation. So the caption slot holds at most one real source, and two
tracks agreeing is never evidence of anything.

The second is that automatic tracks carry per-word timestamps and human ones do
not. Rollup captions overlap heavily -- 558 of 577 consecutive events in the
slide lecture -- because an event's duration is how long it stays on screen,
not how long it was spoken. Reading those durations as speech spans is what
made an early measurement report 193.5% overlap with the VAD. The words
underneath carry their own offsets and do not overlap at all, so the fix is to
read the offsets rather than to fold the events.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SIDECAR_SUFFIX = ".subs.json"

MANUAL = "manual"
AUTO = "auto"

_MARKUP = re.compile(r"<[^>]{0,40}>")
_VTT_TIMING = re.compile(
    r"(\d+):(\d+):(\d+)\.(\d+)\s*-->\s*(\d+):(\d+):(\d+)\.(\d+)"
)
_VTT_HEADERS = ("WEBVTT", "Kind:", "Language:", "NOTE", "STYLE", "REGION")

# Caption convention, not speech: sound cues and speaker labels. Bounded at 60
# characters so a parenthetical aside inside a sentence cannot swallow it.
_BRACKETED = re.compile(r"[（(\[【][^)\]）】]{0,60}[)\]）】]")
_SPEAKER_DASH = re.compile(r"^\s*[-–—]\s+")

# Kana and Han are matched one character at a time and must be taken out of the
# general word run first: `\w` covers them, so a single alternation lets it
# swallow a whole unspaced Japanese caption as one token, which leaves nothing
# for the timeline to place or the alignment to compare.
_CJK = "぀-ヿ㐀-䶿一-鿿豈-﫿"
_WORD = re.compile(
    rf"[{_CJK}]|[가-힣]+|[^\W{_CJK}가-힣]+(?:['’][^\W{_CJK}가-힣]+)*"
)


def clean_text(raw: str) -> str:
    """Display markup out, one space between words.

    Skipping this makes identical tracks look different: the automatic alias of
    a human track is the same file with ``&nbsp;`` in it, and comparing the two
    before unescaping reported 0.72 similarity for text that is actually equal.
    """
    text = html.unescape(raw).replace("\xa0", " ")
    text = _MARKUP.sub("", text)
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def strip_conventions(text: str) -> str:
    """Drop what the caption says about the audio rather than of it.

    ``(mid-tempo lighthearted music)``, ``(birds chirp)`` and a leading ``- ``
    are all real rows in the manifest's human tracks. Fused verbatim they
    become transcript. Genuine speech inside brackets is lost with them; in
    captions the bracket is overwhelmingly a sound or speaker annotation.
    """
    return re.sub(r"\s+", " ", _SPEAKER_DASH.sub("", _BRACKETED.sub(" ", text))).strip()


def tokenize(text: str) -> list[str]:
    """Words for Latin and Hangul, single characters for CJK."""
    return _WORD.findall(text)


def normalize(text: str) -> str:
    """Comparison form: no spacing, no punctuation, no case."""
    return re.sub(
        r"[^\w가-힣぀-ヿ一-龯]", "", unicodedata.normalize("NFKC", text)
    ).casefold()


@dataclass(frozen=True)
class CaptionCue:
    """One row of a caption track, at the moment it was spoken or shown.

    Only a start time is kept. An automatic track's rows are single words whose
    end is the next word, and a human track's on-screen duration says nothing
    about how long the line took to say. Spans are derived from neighbours
    where they are needed, which is the only reading that held up in
    measurement.
    """

    start_s: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start_s": round(self.start_s, 3), "text": self.text}


@dataclass(frozen=True)
class CaptionTrack:
    language: str
    kind: str
    source_format: str
    content_sha256: str
    has_word_timing: bool
    cues: tuple[CaptionCue, ...] = ()

    @property
    def text(self) -> str:
        return " ".join(cue.text for cue in self.cues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "kind": self.kind,
            "format": self.source_format,
            "content_sha256": self.content_sha256,
            "has_word_timing": self.has_word_timing,
            "cues": [cue.to_dict() for cue in self.cues],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaptionTrack:
        return cls(
            language=str(data.get("language") or ""),
            kind=str(data.get("kind") or MANUAL),
            source_format=str(data.get("format") or ""),
            content_sha256=str(data.get("content_sha256") or ""),
            has_word_timing=bool(data.get("has_word_timing")),
            cues=tuple(
                CaptionCue(start_s=float(cue["start_s"]), text=str(cue["text"]))
                for cue in data.get("cues") or ()
                if str(cue.get("text") or "").strip()
            ),
        )


def content_hash(cues: tuple[CaptionCue, ...]) -> str:
    """Identity of what a track says, used to fold aliases of one file.

    Two tracks hashing the same are one source, never two that agree.
    """
    body = "|".join(f"{cue.start_s:.2f}{normalize(cue.text)}" for cue in cues)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def parse_json3(payload: dict[str, Any]) -> tuple[tuple[CaptionCue, ...], bool]:
    """YouTube's json3, with per-word offsets used when the track has them.

    Empty ``'\\n'`` events are the rollup mechanism and carry no text. A track
    counts as word-timed when events routinely hold several timed segments;
    a human track puts one whole line in one segment.
    """
    cues: list[CaptionCue] = []
    per_event: list[int] = []
    for event in payload.get("events") or ():
        if not isinstance(event, dict):
            continue
        base = float(event.get("tStartMs") or 0) / 1000.0
        kept = 0
        for segment in event.get("segs") or ():
            if not isinstance(segment, dict):
                continue
            text = clean_text(str(segment.get("utf8") or ""))
            if not text:
                continue
            offset = float(segment.get("tOffsetMs") or 0) / 1000.0
            cues.append(CaptionCue(start_s=base + offset, text=text))
            kept += 1
        per_event.append(kept)
    word_timed = bool(per_event) and sum(1 for n in per_event if n > 1) > len(per_event) * 0.2
    cues.sort(key=lambda cue: cue.start_s)
    return tuple(cues), word_timed


def parse_vtt(payload: str) -> tuple[tuple[CaptionCue, ...], bool]:
    """WebVTT. Never word-timed: the cue is the smallest unit it offers."""
    cues: list[CaptionCue] = []
    for block in payload.split("\n\n"):
        match = _VTT_TIMING.search(block)
        if match is None:
            continue
        hours, minutes, seconds, millis = (int(value) for value in match.groups()[:4])
        body = clean_text(
            " ".join(
                line
                for line in block.split("\n")
                if not _VTT_TIMING.search(line)
                and line.strip()
                and not line.strip().startswith(_VTT_HEADERS)
            )
        )
        if body:
            cues.append(
                CaptionCue(
                    start_s=hours * 3600 + minutes * 60 + seconds + millis / 1000.0,
                    text=body,
                )
            )
    cues.sort(key=lambda cue: cue.start_s)
    return tuple(cues), False


def parse_track(payload: str, source_format: str) -> tuple[tuple[CaptionCue, ...], bool]:
    if source_format == "json3":
        return parse_json3(json.loads(payload))
    return parse_vtt(payload)


def token_stream(
    track: CaptionTrack, *, tail_rate_s: float = 0.06
) -> tuple[tuple[float, str], ...]:
    """One timestamp per token, so tokens can be placed on the VAD timeline.

    A word-timed track already is this stream. A cue-level track is spread
    across its own span, which ends where the next cue begins; the final cue
    gets a span estimated from its length, since nothing follows it.
    """
    rows: list[tuple[float, str]] = []
    for index, cue in enumerate(track.cues):
        tokens = tokenize(strip_conventions(cue.text))
        if not tokens:
            continue
        if track.has_word_timing:
            rows.extend((cue.start_s, token) for token in tokens)
            continue
        following = (
            track.cues[index + 1].start_s
            if index + 1 < len(track.cues)
            else cue.start_s + max(1.0, len(cue.text) * tail_rate_s)
        )
        span = max(following - cue.start_s, 0.2)
        rows.extend(
            (cue.start_s + span * (position + 0.5) / len(tokens), token)
            for position, token in enumerate(tokens)
        )
    rows.sort(key=lambda row: row[0])
    return tuple(rows)


def sidecar_path(video_path: Path) -> Path:
    return video_path.with_suffix(SIDECAR_SUFFIX)


def load_caption_track(video_path: Path) -> CaptionTrack | None:
    """The track for this video, or None when there is nothing usable.

    A sidecar whose ``track`` is null records that the fetch ran and found no
    original-language track -- a real answer, and different from no sidecar at
    all. Both return None here; the distinction is kept in the file so a rerun
    does not go back to the network to learn the same thing.
    """
    sidecar = sidecar_path(video_path)
    if not sidecar.is_file():
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    track = data.get("track")
    if not isinstance(track, dict):
        return None
    parsed = CaptionTrack.from_dict(track)
    return parsed if parsed.cues else None
