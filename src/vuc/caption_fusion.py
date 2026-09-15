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
    tokenize,
)
from vuc.models import SPEECH, Segment, TextCue, VideoIndex

SPEECH_TRANSCRIPT = "speech_transcript"
HARDSUB_COPY = "hardsub_copy"
REJECTED_LANGUAGE = "rejected_language"
REJECTED_UNALIGNED = "rejected_unaligned"
MIXED = "mixed"

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
    # A whole-track verdict cannot describe a video that narrates and then goes
    # quiet behind burned-in text; a single cue is too noisy to judge alone.
    window_s: float = 30.0
    window_min_cues: int = 3

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
            "window_s": self.window_s,
            "window_min_cues": self.window_min_cues,
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


# ----------------------------------------------------------------- routing

AUDIO = "audio"
TEXT = "text"
DROP = "drop"


def caption_spans(track: CaptionTrack, *, tail_rate_s: float = 0.06) -> list[tuple[float, float]]:
    """Where each caption line begins and ends.

    A cue stores only its start; the line runs until the next one, and the last
    runs for as long as its own length suggests.
    """
    spans: list[tuple[float, float]] = []
    for index, cue in enumerate(track.cues):
        following = (
            track.cues[index + 1].start_s
            if index + 1 < len(track.cues)
            else cue.start_s + max(1.0, len(cue.text) * tail_rate_s)
        )
        spans.append((cue.start_s, max(following, cue.start_s + 0.2)))
    return spans


def _cue_tokens(track: CaptionTrack) -> list[list[tuple[float, str]]]:
    """Timed tokens per cue, used for overlap measurement only."""
    spans = caption_spans(track)
    rows: list[list[tuple[float, str]]] = []
    for index, cue in enumerate(track.cues):
        tokens = tokenize(strip_conventions(cue.text))
        if not tokens:
            rows.append([])
            continue
        if track.has_word_timing:
            rows.append([(cue.start_s, token) for token in tokens])
            continue
        start, end = spans[index]
        span = end - start
        rows.append(
            [(start + span * (n + 0.5) / len(tokens), token) for n, token in enumerate(tokens)]
        )
    return rows


def _cue_seen_on_screen(
    track: CaptionTrack, text_cues: tuple[TextCue, ...], window_s: float, head: int = 12
) -> list[bool | None]:
    """Per cue: was this line also read off the frames at about this time?

    None for lines too short to judge, so they neither help nor hurt the local
    average.
    """
    observed = [(c.start, c.end, normalize(c.text)) for c in text_cues]
    out: list[bool | None] = []
    for cue in track.cues:
        text = normalize(strip_conventions(cue.text))
        if len(text) < 6:
            out.append(None)
            continue
        out.append(
            any(
                seen
                and end > cue.start_s - window_s
                and start < cue.start_s + window_s
                and (text[:head] in seen or seen[:head] in text)
                for start, end, seen in observed
            )
        )
    return out


def route_cues(
    track: CaptionTrack,
    speech: list[tuple[float, float]],
    text_cues: tuple[TextCue, ...],
    *,
    config: AttributionConfig,
) -> list[str]:
    """Send each line where it belongs, judged over its neighbourhood.

    A whole-track verdict cannot describe a video that narrates for five
    minutes and then goes quiet behind burned-in captions: spliced from the
    talking head and the kiwi vlog, such a track scores V=0.42 and O=0.79 and
    is filed entirely as screen text, losing all nineteen speech corrections
    and overwriting an on-screen name card with the sentence the speaker said.

    One cue on its own is no better -- its overlap is 0 or 1 and means almost
    nothing. The window in between is wide enough to be stable and narrow
    enough to let a video change character halfway through. On the spliced
    case any width from 10s to 90s finds the boundary at exactly the right cue.
    """
    tokens = _cue_tokens(track)
    on_screen = _cue_seen_on_screen(track, text_cues, config.ocr_window_s)
    starts = [cue.start_s for cue in track.cues]
    total = len(track.cues)
    decisions: list[str] = []
    for index in range(total):
        low, high = starts[index] - config.window_s, starts[index] + config.window_s
        near = [n for n in range(total) if low <= starts[n] <= high]
        if len(near) < config.window_min_cues:
            half = max(config.window_min_cues // 2, 1)
            near = list(range(max(0, index - half), min(total, index + half + 1)))
        local = [t for n in near for t in tokens[n]]
        overlap = (
            sum(1 for at, _ in local if any(a <= at < b for a, b in speech)) / len(local)
            if local
            else 0.0
        )
        graded = [on_screen[n] for n in near if on_screen[n] is not None]
        match = (sum(graded) / len(graded)) if graded else 0.0
        if overlap >= config.vad_overlap_min:
            decisions.append(AUDIO)
        elif match >= config.ocr_match_min:
            decisions.append(TEXT)
        else:
            decisions.append(DROP)
    return decisions


# ------------------------------------------------------------------ audio


def assign_cues(
    cues: tuple[CaptionCue, ...], regions: list[tuple[float, float]]
) -> tuple[list[list[CaptionCue]], list[CaptionCue]]:
    """Each line to the one region holding its start, or to nowhere.

    Whole lines rather than loose tokens, because the line is what carries the
    punctuation. Splitting on token timings spread a sentence across two
    regions and the halves came back as bare word lists.

    Placed by where the line starts, which is the one timestamp a track
    actually gives. A line's end is inferred from the next line, so a gap in
    the captions stretches the one before it across the whole gap -- on the
    cooking video that pushed 26 of 143 lines past the end of the region their
    speech was in. Using the start instead placed all but three of them.

    Exclusive and exhaustive, so a line cannot be counted twice or vanish.
    Lines outside every speech region are dropped: the VAD split is the spine
    of the index and this pass does not move it, which does mean speech the VAD
    missed stays missed even when the captions have it.
    """
    buckets: list[list[CaptionCue]] = [[] for _ in regions]
    dropped: list[CaptionCue] = []
    for cue in cues:
        for index, (low, high) in enumerate(regions):
            if low <= cue.start_s < high:
                buckets[index].append(cue)
                break
        else:
            dropped.append(cue)
    return buckets, dropped


def fuse_region(asr_text: str, caption_texts: list[str], *, align_ratio_min: float) -> str:
    """Where the captions reach, the captions win; elsewhere, what we heard.

    An earlier version merged the two with an opcode walk, keeping the ASR's
    tokens wherever the caption had none on the theory that captions abbreviate
    for reading speed. Measured against a Whisper reference that made the index
    worse than leaving it alone -- 6.25% against 5.93% on the talking head --
    because what the captions omit is mostly filler and misrecognition, and
    restoring it undoes the correction. Dropping those tokens instead turned
    the same video into a 17% improvement, and the slide lecture into 41%.

    With them dropped the opcode walk returned the caption verbatim every time,
    so the alignment is kept only as a guard: a region whose two accounts do
    not resemble each other is left alone rather than spliced into something
    neither source said.
    """
    caption = " ".join(text for text in (strip_conventions(t) for t in caption_texts) if text)
    if not caption:
        return asr_text
    asr_tokens = tokenize(asr_text)
    if not asr_tokens:
        return caption
    similarity = difflib.SequenceMatcher(
        a=[token.casefold() for token in asr_tokens],
        b=[token.casefold() for token in tokenize(caption)],
    ).ratio()
    if similarity < align_ratio_min:
        return asr_text
    return caption


def fuse_audio_index(
    segments: tuple[Segment, ...],
    cues: tuple[CaptionCue, ...],
    *,
    align_ratio_min: float,
) -> tuple[tuple[Segment, ...], dict[str, Any]]:
    """Rewrite the speech regions these lines cover. Non-speech is untouched."""
    speech_indexes = [i for i, segment in enumerate(segments) if segment.kind == SPEECH]
    regions = [(segments[i].start, segments[i].end) for i in speech_indexes]
    buckets, dropped = assign_cues(cues, regions)

    updated = list(segments)
    stats = {
        "speech_regions": len(speech_indexes),
        "regions_covered": 0,
        "regions_rewritten": 0,
        "regions_guarded": 0,
        "regions_recovered": 0,
        "cues_placed": sum(len(bucket) for bucket in buckets),
        "cues_dropped": len(dropped),
    }
    for position, index in enumerate(speech_indexes):
        bucket = buckets[position]
        if not bucket:
            continue
        stats["regions_covered"] += 1
        original = segments[index]
        fused = fuse_region(
            original.text, [cue.text for cue in bucket], align_ratio_min=align_ratio_min
        )
        if normalize(fused) == normalize(original.text):
            # Either the guard sent us back to the ASR text or the two agreed;
            # only the first is worth reporting.
            if normalize(" ".join(strip_conventions(c.text) for c in bucket)) != normalize(
                original.text
            ):
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


# -------------------------------------------------------------------- text


def _candidates(text: str) -> list[str]:
    """A caption line, and the clauses it may have been split into on screen."""
    pieces = [text, *(_SENTENCE_SPLIT.split(text))]
    return [piece.strip() for piece in pieces if len(normalize(piece)) >= 3]


def fuse_text_index(
    cues: tuple[TextCue, ...],
    caption_cues: tuple[CaptionCue, ...],
    spans: list[tuple[float, float]],
    *,
    config: AttributionConfig,
) -> tuple[tuple[TextCue, ...], dict[str, Any]]:
    """Correct what OCR read, using the text it was trying to read.

    This is not the audio transcript rewriting the screen. Lines routed here
    were measured against the frames, not the speech, so they are the screen
    text's own source -- better evidence than a reading of the pixels, which
    returned `영양 균형 행기기`, `듬뿐`, `오본` and `시간X` for lines the track
    spells correctly.

    Rows OCR saw and the captions do not mention are kept -- background text
    like a bowl's `pyrex` is really there. Rows the captions have and OCR never
    saw are not added: they have no position or size, and the text index means
    what OCR saw at that spot on the frame.
    """
    stats = {"cues": len(cues), "corrected": 0, "cosmetic": 0, "guarded": 0}
    updated: list[TextCue] = []
    for cue in cues:
        best_score = 0.0
        best_text: str | None = None
        for caption, (start, end) in zip(caption_cues, spans, strict=True):
            if cue.end <= start - config.ocr_window_s or cue.start >= end + config.ocr_window_s:
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


def _verdict(language_ok: bool, counts: dict[str, int]) -> str:
    if not language_ok:
        return REJECTED_LANGUAGE
    if counts[AUDIO] and counts[TEXT]:
        return MIXED
    if counts[AUDIO]:
        return SPEECH_TRANSCRIPT
    if counts[TEXT]:
        return HARDSUB_COPY
    return REJECTED_UNALIGNED


def overlay_captions(
    index: VideoIndex,
    track: CaptionTrack | None,
    *,
    audio_language: str | None,
    config: AttributionConfig,
) -> CaptionOverlay:
    """Gate the track on language, then route each line and let it correct.

    The language check stays whole-track and comes first. A translated track is
    timed to the speech it translates, so every line of it scores as a
    transcript -- routed line by line, an English translation of a Japanese
    vlog would be filed as what was said.
    """
    plain = CaptionOverlay(index.audio.segments, index.text.cues)
    if track is None:
        return plain
    try:
        ratio = script_ratio(track.text, audio_language or track.language)
        language_ok = ratio is None or ratio >= config.script_ratio_min
        spans = caption_spans(track)
        segments, cues = index.audio.segments, index.text.cues
        counts = {AUDIO: 0, TEXT: 0, DROP: len(track.cues)}
        fusion: dict[str, Any] = {}
        if language_ok:
            speech = [(s.start, s.end) for s in segments if s.kind == SPEECH]
            decisions = route_cues(track, speech, cues, config=config)
            counts = {name: decisions.count(name) for name in (AUDIO, TEXT, DROP)}
            rows = list(zip(track.cues, spans, decisions, strict=True))
            audio_rows = [c for c, _, d in rows if d == AUDIO]
            text_rows = [(c, s) for c, s, d in rows if d == TEXT]
            if audio_rows:
                segments, fusion[AUDIO] = fuse_audio_index(
                    segments, tuple(audio_rows), align_ratio_min=config.align_ratio_min
                )
            if text_rows:
                chosen, chosen_spans = zip(*text_rows, strict=True)
                cues, fusion[TEXT] = fuse_text_index(
                    cues, chosen, list(chosen_spans), config=config
                )
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
            "language_ok": language_ok,
            "script_ratio": None if ratio is None else round(ratio, 4),
            "verdict": _verdict(language_ok, counts),
            "routing": counts,
            "fusion": fusion,
        },
    )
