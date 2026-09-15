"""Deciding what a caption track is, and letting it correct what we observed.

The obvious design is a priority ladder -- human captions over automatic ones
over ASR -- and it is the wrong one. The source with the highest expected
accuracy is the one with no mechanical check on it at all, so a ladder means
the least verifiable text wins by default. This module does what the rest of
the index already does instead: it treats the track as one more observer and
lets agreement with the others decide.

Two axes do the deciding. A track whose words land inside the VAD's speech is a
transcript of the audio. A track whose lines match what OCR read off the frames
is a copy of the screen text -- the kiwi vlog is a silent video whose captions
overlap speech 1.1% of the time and the text index 89.4% of the time. Language
is checked first, because a translated track keeps perfect timing and would
otherwise pass as a transcript.

The V axis decides before the O axis gets a turn. A lecturer reading their own
slides scores 0.50 on O, high enough to look like screen text, but the track is
still a transcript of speech, and using it to rewrite the OCR rows would be
correcting screen text with spoken paraphrase -- the one thing the text index
has always refused to do.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any

from vuc.captions import (
    CaptionCue,
    CaptionTrack,
    normalize,
    strip_conventions,
    token_stream,
    tokenize,
)
from vuc.models import SPEECH, Segment, TextCue, VideoIndex

SPEECH_TRANSCRIPT = "speech_transcript"
HARDSUB_COPY = "hardsub_copy"
REJECTED_LANGUAGE = "rejected_language"
REJECTED_UNALIGNED = "rejected_unaligned"

# Scripts a declared language must actually be written in. Latin-script
# languages are not separable this way and are left to the V and O axes.
_SCRIPTS = {
    "ko": lambda c: "가" <= c <= "힣",
    "ja": lambda c: "぀" <= c <= "ヿ",
    "zh": lambda c: "一" <= c <= "鿿",
}

_SENTENCE_SPLIT = re.compile(r"[,\.!?·]\s+|\s{2,}")


@dataclass(frozen=True)
class AttributionConfig:
    vad_overlap_min: float = 0.6
    ocr_match_min: float = 0.7
    script_ratio_min: float = 0.05
    ocr_window_s: float = 3.0
    # Below this, the track and the transcript are not describing the same
    # speech and the region keeps what we heard.
    align_ratio_min: float = 0.2
    # A correction fixes a misread; it does not absorb a neighbouring row or
    # drop half of this one.
    scope_ratio_min: float = 0.8
    scope_ratio_max: float = 1.25
    text_match_min: float = 0.55

    def to_dict(self) -> dict[str, Any]:
        return {
            "vad_overlap_min": self.vad_overlap_min,
            "ocr_match_min": self.ocr_match_min,
            "script_ratio_min": self.script_ratio_min,
            "ocr_window_s": self.ocr_window_s,
            "align_ratio_min": self.align_ratio_min,
            "scope_ratio_min": self.scope_ratio_min,
            "scope_ratio_max": self.scope_ratio_max,
            "text_match_min": self.text_match_min,
        }


@dataclass(frozen=True)
class Attribution:
    verdict: str
    language_ok: bool
    script_ratio: float | None
    vad_overlap: float
    ocr_match: float

    @property
    def is_speech_transcript(self) -> bool:
        return self.verdict == SPEECH_TRANSCRIPT

    @property
    def is_hardsub_copy(self) -> bool:
        return self.verdict == HARDSUB_COPY

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "language_ok": self.language_ok,
            "script_ratio": self.script_ratio,
            "vad_overlap": round(self.vad_overlap, 4),
            "ocr_match": round(self.ocr_match, 4),
        }


def script_ratio(text: str, language: str | None) -> float | None:
    """Fraction of the letters that belong to the declared script.

    None when the language has no distinguishing script, which is not a pass
    or a failure -- it means this axis has nothing to say.
    """
    base = str(language or "").lower().replace("_", "-").split("-")[0]
    belongs = _SCRIPTS.get(base)
    if belongs is None:
        return None
    letters = [c for c in text if c.isalpha() or belongs(c)]
    if not letters:
        return 0.0
    return sum(1 for c in letters if belongs(c)) / len(letters)


def vad_overlap(tokens: tuple[tuple[float, str], ...], speech: list[tuple[float, float]]) -> float:
    if not tokens:
        return 0.0
    inside = sum(1 for at, _ in tokens if any(start <= at < end for start, end in speech))
    return inside / len(tokens)


def ocr_match(
    cues: tuple[CaptionCue, ...],
    text_cues: tuple[TextCue, ...],
    *,
    window_s: float,
    head: int = 12,
) -> float:
    """How much of the track was also read off the screen at the same moment."""
    candidates = [
        (cue.start_s, normalize(strip_conventions(cue.text)))
        for cue in cues
        if len(normalize(cue.text)) >= 6
    ]
    if not candidates:
        return 0.0
    observed = [(cue.start, cue.end, normalize(cue.text)) for cue in text_cues]
    hits = 0
    for at, text in candidates:
        for start, end, seen in observed:
            if not seen or end <= at - window_s or start >= at + window_s:
                continue
            if text[:head] in seen or seen[:head] in text:
                hits += 1
                break
    return hits / len(candidates)


def attribute(
    track: CaptionTrack,
    *,
    audio_language: str | None,
    speech: list[tuple[float, float]],
    text_cues: tuple[TextCue, ...],
    config: AttributionConfig,
) -> Attribution:
    ratio = script_ratio(track.text, audio_language or track.language)
    language_ok = ratio is None or ratio >= config.script_ratio_min
    overlap = vad_overlap(token_stream(track), speech)
    match = ocr_match(track.cues, text_cues, window_s=config.ocr_window_s)
    if not language_ok:
        verdict = REJECTED_LANGUAGE
    elif overlap >= config.vad_overlap_min:
        verdict = SPEECH_TRANSCRIPT
    elif match >= config.ocr_match_min:
        verdict = HARDSUB_COPY
    else:
        verdict = REJECTED_UNALIGNED
    return Attribution(
        verdict=verdict,
        language_ok=language_ok,
        script_ratio=None if ratio is None else round(ratio, 4),
        vad_overlap=overlap,
        ocr_match=match,
    )


# ------------------------------------------------------------------ audio


def assign_tokens(
    tokens: tuple[tuple[float, str], ...], regions: list[tuple[float, float]]
) -> tuple[list[list[str]], list[str]]:
    """Each token to the one region holding its timestamp, or to nowhere.

    Exclusive and exhaustive by construction, so a token cannot be counted
    twice or vanish. Tokens outside every speech region are dropped: the VAD
    split is the spine of the index and this pass does not move it, which does
    mean speech the VAD missed stays missed even when the captions have it.
    """
    buckets: list[list[str]] = [[] for _ in regions]
    dropped: list[str] = []
    for at, token in tokens:
        for index, (start, end) in enumerate(regions):
            if start <= at < end:
                buckets[index].append(token)
                break
        else:
            dropped.append(token)
    return buckets, dropped


def fuse_region(asr_text: str, caption_tokens: list[str], *, align_ratio_min: float) -> str:
    """Where the captions reach, the captions win; elsewhere, what we heard.

    An earlier version merged the two with an opcode walk, keeping the ASR's
    tokens wherever the caption had none on the theory that captions abbreviate
    for reading speed. Measured against a Whisper reference that made the index
    worse than leaving it alone -- 6.25% against 5.93% on the talking head --
    because what the captions omit is mostly filler and misrecognition, and
    restoring it undoes the correction. Dropping those tokens instead turned
    the same video into a 17% improvement, and the slide lecture into 41%.

    With them dropped the opcode walk returns the caption verbatim in every one
    of the 72 fused regions measured, so the alignment is kept only as a guard:
    a region whose two accounts do not resemble each other is left alone rather
    than spliced into something neither source said.
    """
    if not caption_tokens:
        return asr_text
    asr_tokens = tokenize(asr_text)
    if not asr_tokens:
        return " ".join(caption_tokens)
    similarity = difflib.SequenceMatcher(
        a=[token.casefold() for token in asr_tokens],
        b=[token.casefold() for token in caption_tokens],
    ).ratio()
    if similarity < align_ratio_min:
        return asr_text
    return " ".join(caption_tokens)


def fuse_audio_index(
    segments: tuple[Segment, ...],
    track: CaptionTrack,
    *,
    config: AttributionConfig,
) -> tuple[tuple[Segment, ...], dict[str, Any]]:
    """Rewrite the speech regions the captions cover. Non-speech is untouched."""
    speech_indexes = [i for i, segment in enumerate(segments) if segment.kind == SPEECH]
    regions = [(segments[i].start, segments[i].end) for i in speech_indexes]
    buckets, dropped = assign_tokens(token_stream(track), regions)

    updated = list(segments)
    stats = {
        "speech_regions": len(speech_indexes),
        "regions_covered": 0,
        "regions_rewritten": 0,
        "regions_guarded": 0,
        "regions_recovered": 0,
        "tokens_total": 0,
        "tokens_dropped": len(dropped),
    }
    stats["tokens_total"] = sum(len(bucket) for bucket in buckets) + len(dropped)
    for position, index in enumerate(speech_indexes):
        tokens = buckets[position]
        if not tokens:
            continue
        stats["regions_covered"] += 1
        original = segments[index]
        fused = fuse_region(original.text, tokens, align_ratio_min=config.align_ratio_min)
        if normalize(fused) == normalize(original.text):
            # Either the guard sent us back to the ASR text or the two agreed;
            # only the first is worth reporting.
            if normalize(" ".join(tokens)) != normalize(original.text):
                stats["regions_guarded"] += 1
            continue
        if not original.text.strip():
            stats["regions_recovered"] += 1
        stats["regions_rewritten"] += 1
        updated[index] = Segment(
            start=original.start,
            end=original.end,
            text=fused,
            language=original.language,
            emotion=original.emotion,
            events=original.events,
            raw_text=original.raw_text,
            kind=original.kind,
        )
    return tuple(updated), stats


# ------------------------------------------------------------------- text


def _candidates(text: str) -> list[str]:
    """A caption line, and the clauses it may have been split into on screen."""
    pieces = [text, *(_SENTENCE_SPLIT.split(text))]
    return [piece.strip() for piece in pieces if len(normalize(piece)) >= 3]


def _caption_spans(
    track: CaptionTrack, *, tail_rate_s: float = 0.06
) -> list[tuple[CaptionCue, float]]:
    """Each caption paired with where it ends, derived the same way tokens are.

    A cue stores only its start, so the line's extent comes from the next one --
    and for the last, from its own length. Without this the overlap test needs
    an arbitrary allowance for how long a line might have been.
    """
    spans: list[tuple[CaptionCue, float]] = []
    for index, cue in enumerate(track.cues):
        following = (
            track.cues[index + 1].start_s
            if index + 1 < len(track.cues)
            else cue.start_s + max(1.0, len(cue.text) * tail_rate_s)
        )
        spans.append((cue, max(following, cue.start_s + 0.2)))
    return spans


def fuse_text_index(
    cues: tuple[TextCue, ...],
    track: CaptionTrack,
    *,
    config: AttributionConfig,
) -> tuple[tuple[TextCue, ...], dict[str, Any]]:
    """Correct what OCR read, using the text it was trying to read.

    This is not the audio transcript rewriting the screen. A track attributed
    as a hardsub copy is the screen text's own source, and it is better
    evidence than a reading of the pixels: OCR returned `영양 균형 행기기`,
    `듬뿐`, `오본` and `시간X` for lines the track spells correctly.

    Rows OCR saw and the captions do not mention are kept -- background text
    like a bowl's `pyrex` is really there. Rows the captions have and OCR never
    saw are not added: they have no position or size, and the text index means
    what OCR saw at that spot on the frame.
    """
    stats = {
        "cues": len(cues),
        "corrected": 0,
        "cosmetic": 0,
        "guarded": 0,
    }
    spans = _caption_spans(track)
    updated: list[TextCue] = []
    for cue in cues:
        best_score = 0.0
        best_text: str | None = None
        for caption, end_s in spans:
            if cue.end <= caption.start_s - config.ocr_window_s:
                continue
            if cue.start >= end_s + config.ocr_window_s:
                continue
            for candidate in _candidates(caption.text):
                score = difflib.SequenceMatcher(
                    None, normalize(cue.text), normalize(candidate)
                ).ratio()
                if score > best_score:
                    best_score, best_text = score, candidate
        if best_text is None or best_score < config.text_match_min or best_text == cue.text:
            updated.append(cue)
            continue
        scope = len(normalize(best_text)) / max(len(normalize(cue.text)), 1)
        if not config.scope_ratio_min <= scope <= config.scope_ratio_max:
            stats["guarded"] += 1
            updated.append(cue)
            continue
        if normalize(best_text) == normalize(cue.text):
            stats["cosmetic"] += 1
        else:
            stats["corrected"] += 1
        updated.append(
            TextCue(
                start=cue.start,
                end=cue.end,
                text=best_text,
                position=cue.position,
                size=cue.size,
            )
        )
    return tuple(updated), stats


# ----------------------------------------------------------------- overlay


@dataclass(frozen=True)
class CaptionOverlay:
    """The index as the model should read it, plus why it reads that way.

    Applied when the prompt is built, not when the index is written. The index
    is what processing the video produced and stays that; a caption track is
    something that arrived alongside it, and folding it into the stored index
    made a track appearing or a threshold moving cost a full rebuild of the
    VAD, the ASR and the OCR -- minutes of model time to redo a correction that
    takes milliseconds.
    """

    segments: tuple[Segment, ...]
    cues: tuple[TextCue, ...]
    summary: dict[str, Any] | None = None

    @property
    def applied(self) -> bool:
        return self.summary is not None


def overlay_captions(
    index: VideoIndex,
    track: CaptionTrack | None,
    *,
    audio_language: str | None,
    config: AttributionConfig,
) -> CaptionOverlay:
    """Judge the track and correct whichever index it belongs to.

    Returns the index untouched when there is no track, when the track is a
    translation or otherwise unaligned, or when anything goes wrong: the
    correction is worth having and worth losing quietly.
    """
    plain = CaptionOverlay(index.audio.segments, index.text.cues)
    if track is None:
        return plain
    try:
        attribution = attribute(
            track,
            audio_language=audio_language,
            speech=[(s.start, s.end) for s in index.audio.segments if s.kind == SPEECH],
            text_cues=index.text.cues,
            config=config,
        )
        segments, cues = index.audio.segments, index.text.cues
        fusion: dict[str, Any] = {}
        if attribution.is_speech_transcript:
            segments, fusion = fuse_audio_index(segments, track, config=config)
        elif attribution.is_hardsub_copy:
            cues, fusion = fuse_text_index(cues, track, config=config)
    except (ValueError, TypeError, KeyError):
        return plain
    return CaptionOverlay(
        segments=segments,
        cues=cues,
        summary={
            "language": track.language,
            "kind": track.kind,
            "format": track.source_format,
            "content_sha256": track.content_sha256,
            "has_word_timing": track.has_word_timing,
            "cues": len(track.cues),
            **attribution.to_dict(),
            "fusion": fusion,
        },
    )
