"""What was heard, across the whole timeline -- not just what was said.

VAD splits the video once and every audio model works over that split. Speech
regions go to SenseVoice, non-speech regions go to an event tagger. Neither
half is a gap: ten minutes of music, a laugh, a round of applause and a frying
pan are all things the index should know about, and none of them is speech.

The output is one segment per region, in timeline order. Which model produced
a given line is a detail of how it was obtained, so it is recorded in the
index's metadata and never shown in the line itself.
"""

from __future__ import annotations

import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from vuc.config import AudioEventConfig
from vuc.indexer import Transcriber
from vuc.media import extract_audio_segment
from vuc.models import NON_SPEECH, SPEECH, Segment
from vuc.timeline import Span, split_long

# PANNs speaks AudioSet's 527-label taxonomy, which is an ontology rather than
# a vocabulary: it carries abstract parent nodes, several spellings of the same
# sound, and parenthesised qualifiers. Handing that to an agent verbatim
# produced lines like `<chopping_(food)> <water_tap,_faucet>`. What follows
# turns it into a smaller, steadier set of names.

# Categories, not sounds. `Animal` is the ontology's parent of Dog and Cat, and
# a category name says less than any of its children even when it is right: it
# is compatible with a dog, a whale and a cricket at once. It is also what came
# back, at 0.405, for five seconds of a knife on a chopping board -- with the
# animal-family children all under 0.14 and no animal anywhere in the video.
# The children are kept, so dropping the parent costs nothing that scored.
ABSTRACT_LABELS = frozenset(
    {
        "animal",
        "domestic animals",
        "livestock",
        "wild animals",
        "human sounds",
        "human voice",
        "human group actions",
        "human locomotion",
        "respiratory sounds",
        "digestive",
        "hands",
        "heart sounds",
        "otoacoustic emission",
        "sounds of things",
        "generic impact sounds",
        "surface contact",
        "deformable shell",
        "onomatopoeia",
        "source-ambiguous sounds",
        "specific impact sounds",
        "miscellaneous sources",
        "natural sounds",
        "mechanisms",
        "tools",
        "explosion",
        "wood",
        "glass",
    }
)

# Where the recording was made, not what happened in it. These describe the
# room or the channel and are true of every region in a video at once.
SCENE_LABELS = frozenset(
    {
        "inside",
        "outside",
        "echo",
        "reverberation",
        "background noise",
        "noise",
        "environmental noise",
        "static",
        "mains hum",
        "white noise",
        "pink noise",
        "throbbing",
        "vibration",
        "sound effect",
        "sine wave",
        "harmonic",
        "chirp tone",
        "distortion",
        "silence",
    }
)

# Speech is the VAD split itself, which the index already expresses by which
# list a line is in. A speech label inside a non-speech region is the two
# detectors disagreeing, not an event.
SPEECH_LABELS = frozenset(
    {
        "speech",
        "male speech",
        "female speech",
        "child speech",
        "conversation",
        "narration",
        "speech synthesizer",
        "babbling",
    }
)

# A model saying "I could not tell". Recording it as a finding would invent a
# fact out of an admission that there is none. Emotions were asked for by
# nobody and turned every talking-head line into "<neutral>".
UNKNOWN_LABELS = frozenset(
    {
        "event_unk",
        "event_unknown",
        "emo_unk",
        "emo_unknown",
        "neutral",
        "happy",
        "sad",
        "angry",
        "unknown",
        "withitn",
        "woitn",
    }
)

DISCARDED_LABELS = ABSTRACT_LABELS | SCENE_LABELS | SPEECH_LABELS | UNKNOWN_LABELS

# Families collapsed onto one name, so two labels for one sound read as one
# thing and two detectors that agree can be seen to agree. SenseVoice writes
# BGM and AudioSet writes Music; the index says music.
ALIASES: Mapping[str, str] = {
    "bgm": "music",
    "musical instrument": "music",
    "plucked string instrument": "music",
    "bowed string instrument": "music",
    "wind instrument": "music",
    "brass instrument": "music",
    "percussion": "music",
    "keyboard": "music",
    "guitar": "music",
    "piano": "music",
    "drum kit": "music",
    "singing": "singing",
    "water tap": "running_water",
    "sink": "running_water",
    "bathtub": "running_water",
    "fill": "running_water",
    "stream": "running_water",
    "trickle": "dripping",
    "drip": "dripping",
    "gurgling": "running_water",
    "clapping": "applause",
    "laugh": "laughter",
    "chuckle": "laughter",
    "giggle": "laughter",
    "cry": "crying",
    "sobbing": "crying",
    "hubbub": "crowd",
    "crowd": "crowd",
    "chatter": "crowd",
    "walk": "footsteps",
    "run": "footsteps",
    "breath": "breathing",
    "siren": "alarm",
    "alarm clock": "alarm",
    "telephone dialing": "phone",
    "telephone bell ringing": "phone",
    "ringtone": "phone",
    "rail transport": "train",
    "railroad car": "train",
    "liquid": "water",
    "motor vehicle": "vehicle",
    "car": "vehicle",
    "truck": "vehicle",
    "bus": "vehicle",
    "engine": "vehicle",
    "traffic noise": "vehicle",
    "bird vocalization": "birds",
    "chirp": "birds",
    "bird": "birds",
    "canidae": "dog",
    "rodents": "rodent",
    "roaring cats": "big_cat",
}


def canonical_label(label: str) -> str | None:
    """One AudioSet or SenseVoice label as the index spells it, or None to drop.

    Qualifiers in brackets and every spelling after the first comma are the
    ontology explaining itself, not information: `Chopping (food)` and
    `Water tap, faucet` become `chopping` and, through the alias table,
    `running_water`. Anything with no alias keeps its cleaned-up name rather
    than being dropped, so a correct but unlisted label still reaches the index.
    """
    cleaned = str(label).strip().lower()
    cleaned = re.sub(r"\s*\([^)]*\)", "", cleaned)
    cleaned = cleaned.split(",")[0]
    cleaned = " ".join(cleaned.split())
    if not cleaned or cleaned in DISCARDED_LABELS:
        return None
    return ALIASES.get(cleaned, cleaned).replace(" ", "_")


# SenseVoice's own "there is no speech here" language code.
NO_SPEECH_LANGUAGE = "nospeech"


class AudioEventError(RuntimeError):
    pass


@dataclass(frozen=True)
class Region:
    """One stretch of the VAD-split timeline, and which side of it it is on."""

    span: Span
    kind: str

    @property
    def is_speech(self) -> bool:
        return self.kind == SPEECH


@dataclass(frozen=True)
class EventTag:
    label: str
    confidence: float | None = None


class AudioEventTagger(Protocol):
    name: str

    def tag(self, clip: Path, *, duration_s: float) -> list[EventTag]: ...


def keep_tags(tags: Sequence[EventTag], *, config: AudioEventConfig) -> tuple[str, ...]:
    """Usable labels only, deduplicated, in the order the tagger ranked them.

    Two bars, and a label has to clear both. The absolute one is the usual "is
    this worth anything at all". The relative one is a heuristic: a label has
    to reach a fraction of the tagger's own top score for this clip. It is
    justified by measurement rather than by any account of what the model is
    doing -- six seconds of background music came back as music at 0.83
    trailed by animal at 0.32, snake at 0.25 and squish at 0.22, none of which
    were in the video, and the same cut leaves agreeing labels alone because
    they sit close to the top (biting 0.55, crunch 0.39, chewing 0.25 for one
    mouthful). Scores are per-label and independent, so the gap below the best
    answer is a usable signal here, not a probability mass being divided up.

    A tag with no confidence at all is kept: some models simply do not score
    their output, and saying nothing is not the same as scoring low.
    """
    scored = [tag.confidence for tag in tags if tag.confidence is not None]
    floor = max(scored) * config.relative_floor if scored else 0.0
    kept: list[str] = []
    for tag in tags:
        label = canonical_label(tag.label)
        if label is None or label in kept:
            continue
        if tag.confidence is not None and (
            tag.confidence < config.min_confidence or tag.confidence < floor
        ):
            continue
        kept.append(label)
    return tuple(kept)


@dataclass
class SenseVoiceTagger:
    """Acoustic tags from SenseVoice, without its transcript.

    Available wherever SenseVoice is, so non-speech tagging works with no extra
    download. Its vocabulary is small and it reports no confidence, which is
    why it is the fallback rather than the default.
    """

    transcriber: Transcriber
    name: str = "sensevoice"

    def tag(self, clip: Path, *, duration_s: float) -> list[EventTag]:
        return [
            EventTag(label=event)
            for segment in self.transcriber.transcribe(clip, duration_s)
            for event in segment.events
        ]


PANNS_CHECKPOINT_URL = "https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"
PANNS_LABELS_URL = (
    "http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv"
)


def ensure_panns_assets(data_dir: Path, *, checkpoint: Path) -> None:
    """Fetch the PANNs checkpoint and AudioSet label table if they are absent.

    panns_inference shells out to wget for this, which is not present on a
    stock macOS box and took the whole index run down with it when it was
    missing. Downloading here uses the HTTP client the project already depends
    on, and only ever runs for someone who installed the extra.
    """
    import httpx

    for target, url in (
        (data_dir / "class_labels_indices.csv", PANNS_LABELS_URL),
        (checkpoint, PANNS_CHECKPOINT_URL),
    ):
        if target.exists() and target.stat().st_size > 0:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            with partial.open("wb") as handle:
                for chunk in response.iter_bytes(chunk_size=1 << 20):
                    handle.write(chunk)
        partial.replace(target)


class PannsTagger:
    """AudioSet tagging with PANNs: 527 labels, each with a real probability."""

    name = "panns"

    def __init__(self, config: AudioEventConfig) -> None:
        self.config = config
        data_dir = Path.home() / "panns_data"
        checkpoint = (
            Path(config.model_path) if config.model_path else data_dir / "Cnn14_mAP=0.431.pth"
        )
        # The assets must be on disk before the import: panns_inference reads
        # the AudioSet label table at module scope, so importing it first turns
        # a missing file into an error nothing can recover from.
        ensure_panns_assets(data_dir, checkpoint=checkpoint)
        try:
            import librosa
            from panns_inference import AudioTagging, labels
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise AudioEventError(
                "audio-event dependencies are not installed. Run `uv sync --extra audio-events`."
            ) from exc
        self._librosa = librosa
        self._labels = labels
        self._model = AudioTagging(checkpoint_path=str(checkpoint), device="cpu")

    # One CNN window at the 32kHz rate PANNs expects. Anything shorter cannot
    # be padded into a spectrogram and raises out of torch rather than
    # returning nothing.
    MIN_SAMPLES = 1024

    def tag(self, clip: Path, *, duration_s: float) -> list[EventTag]:
        del duration_s
        waveform, _ = self._librosa.load(str(clip), sr=32000, mono=True)
        if waveform.shape[0] < self.MIN_SAMPLES:
            return []
        clipwise, _ = self._model.inference(waveform[None, :])
        scores = clipwise[0]
        ranked = sorted(range(len(scores)), key=lambda index: float(scores[index]), reverse=True)
        tags: list[EventTag] = []
        for index in ranked[: self.config.top_k]:
            confidence = float(scores[index])
            if confidence < self.config.min_confidence:
                break
            tags.append(EventTag(label=str(self._labels[index]), confidence=round(confidence, 4)))
        return tags


def create_event_tagger(
    config: AudioEventConfig,
    *,
    sensevoice: Transcriber | None = None,
) -> AudioEventTagger | None:
    if config.tagger == "none":
        return None
    if config.tagger == "sensevoice":
        return SenseVoiceTagger(sensevoice) if sensevoice is not None else None
    if config.tagger == "panns":
        return PannsTagger(config)
    raise ValueError(f"unsupported audio event tagger: {config.tagger}")


@dataclass(frozen=True)
class AudioIndexRun:
    segments: tuple[Segment, ...]
    speech_regions: int
    non_speech_regions: int
    transcribed_regions: int
    tagged_regions: int
    processing_s: float


def _speech_segment(
    region: Region,
    transcriber: Transcriber,
    clip: Path,
    *,
    config: AudioEventConfig,
) -> Segment:
    """One line for one speech region: what was said in it.

    SenseVoice returns a sentence or three for a region this size; they are
    joined rather than emitted separately because the region is the unit the
    index is built on, and splitting it would put two lines on the timeline
    where VAD found one stretch of talking.
    """
    parts: list[str] = []
    language = "unknown"
    tags: list[EventTag] = []
    for segment in transcriber.transcribe(clip, region.span.duration):
        tags.extend(EventTag(label=event) for event in segment.events)
        # Over music SenseVoice returns language `nospeech` and a lone full
        # stop. Storing that would put a sentence of punctuation on the
        # timeline and claim a model had heard it.
        text = segment.text.strip()
        if segment.language == NO_SPEECH_LANGUAGE or not any(c.isalnum() for c in text):
            continue
        parts.append(text)
        if language == "unknown" and segment.language != "unknown":
            language = segment.language
    return Segment(
        start=region.span.start,
        end=region.span.end,
        text=" ".join(parts),
        language=language,
        # Tags are the fallback for a speech region that produced no words, so
        # the region still says something rather than rendering as a bare span.
        events=() if parts else keep_tags(tags, config=config),
        kind=SPEECH,
    )


def _non_speech_segments(
    region: Region,
    tagger: AudioEventTagger,
    audio_path: Path,
    clip: Path,
    *,
    config: AudioEventConfig,
) -> list[Segment]:
    """Tag the region in short windows rather than in one go.

    A VAD region is however long the silence was; the tagger returns one set of
    labels for whatever it is handed. Give it twenty-eight seconds of a kitchen
    and it returns the whole kitchen at once, every label around 0.5 to 0.67,
    with no way to tell which second any of them belongs to -- a real tap and a
    cat that was never there, scoring the same. Windowing puts a bound on how
    much can be smeared together and gives each label a time.

    Neighbouring windows that agree are joined again afterwards, so a minute of
    uninterrupted frying stays one line instead of six identical ones.
    """
    windows: list[Segment] = []
    for window in split_long([region.span], max_s=config.window_s):
        extract_audio_segment(audio_path, clip, start_s=window.start, end_s=window.end)
        events = keep_tags(tagger.tag(clip, duration_s=window.duration), config=config)
        # Compared as a set: the labels come back ranked by score, so two
        # windows of the same frying can disagree on the order and nothing
        # else. Requiring the tuple to match left three such pairs unmerged on
        # a single video.
        if windows and set(windows[-1].events) == set(events):
            windows[-1] = replace(windows[-1], end=window.end)
            continue
        windows.append(
            Segment(
                start=window.start,
                end=window.end,
                text="",
                language="unknown",
                events=events,
                kind=NON_SPEECH,
            )
        )
    return windows


def analyze_regions(
    audio_path: Path,
    regions: Sequence[Region],
    *,
    transcriber: Transcriber,
    tagger: AudioEventTagger | None,
    events: AudioEventConfig,
    clip_dir: Path,
) -> AudioIndexRun:
    """Cut each region out once and hand it to the model that side calls for."""
    started = time.monotonic()
    clip_dir.mkdir(parents=True, exist_ok=True)
    for stale in clip_dir.glob("region-*.wav"):
        stale.unlink()

    segments: list[Segment] = []
    transcribed = 0
    tagged = 0
    for index, region in enumerate(regions):
        if not region.is_speech and tagger is None:
            continue
        clip = clip_dir / f"region-{index + 1:05d}.wav"
        try:
            if region.is_speech:
                extract_audio_segment(
                    audio_path, clip, start_s=region.span.start, end_s=region.span.end
                )
                segments.append(_speech_segment(region, transcriber, clip, config=events))
                transcribed += 1
            else:
                # The windows cut their own clips; the region is not an input here.
                segments.extend(
                    _non_speech_segments(region, tagger, audio_path, clip, config=events)
                )
                tagged += 1
        finally:
            clip.unlink(missing_ok=True)

    return AudioIndexRun(
        segments=tuple(sorted(segments, key=lambda item: (item.start, item.end))),
        speech_regions=sum(1 for region in regions if region.is_speech),
        non_speech_regions=sum(1 for region in regions if not region.is_speech),
        transcribed_regions=transcribed,
        tagged_regions=tagged,
        processing_s=round(time.monotonic() - started, 3),
    )


def timeline_regions(speech: Sequence[Span], non_speech: Sequence[Span]) -> list[Region]:
    """The whole timeline, both halves, in order."""
    regions = [Region(span, SPEECH) for span in speech]
    regions.extend(Region(span, NON_SPEECH) for span in non_speech)
    return sorted(regions, key=lambda region: (region.span.start, region.span.end))
