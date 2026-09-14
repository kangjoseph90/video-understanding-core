from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

SPEECH = "speech"
NON_SPEECH = "non_speech"


@dataclass(frozen=True)
class Segment:
    """One region of the VAD-split timeline and what was heard in it.

    ``kind`` records which side of the split produced it, which decides how the
    line is rendered and which model the agent's request for that interval is
    routed to. It is deliberately not shown to the model: the agent is told
    what was heard, not which detector heard it.
    """

    start: float
    end: float
    text: str
    language: str
    emotion: str | None = None
    events: tuple[str, ...] = ()
    raw_text: str = ""
    kind: str = SPEECH

    @property
    def is_speech(self) -> bool:
        return self.kind == SPEECH

    @property
    def is_empty(self) -> bool:
        """Nothing was heard here that anything was willing to name."""
        return not self.text and not self.events

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["events"] = list(self.events)
        return result


@dataclass(frozen=True)
class TextCue:
    """A line of text that was on screen, with where it sat and how big it was.

    No claim is made about what kind of text it is. Whether a line is a
    subtitle, a slide heading, a lower third or a shop sign is a guess, and an
    earlier version of this that guessed from vertical position filed slide
    body text as dialogue. Position and size are measured; the rest is left to
    whoever reads the index.
    """

    start: float
    end: float
    text: str
    position: str
    size: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AudioIndex:
    """What was heard, over the whole timeline.

    The segments are the complete VAD partition -- every second of the video is
    in exactly one of them, including the ones nothing could be said about.
    Those are dropped when the index is rendered, but they are kept here,
    because this is also what tells a later ASR request which side of the split
    an interval falls on.
    """

    segments: tuple[Segment, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"segments": [segment.to_dict() for segment in self.segments]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AudioIndex:
        return cls(
            tuple(
                Segment(
                    start=item["start"],
                    end=item["end"],
                    text=item["text"],
                    language=item["language"],
                    emotion=item.get("emotion"),
                    events=tuple(item.get("events", ())),
                    raw_text=item.get("raw_text", ""),
                    kind=item["kind"],
                )
                for item in data.get("segments", ())
            )
        )


@dataclass(frozen=True)
class TextIndex:
    """What was on screen. An observer of the same video, not of the audio."""

    cues: tuple[TextCue, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"cues": [cue.to_dict() for cue in self.cues]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextIndex:
        return cls(tuple(TextCue(**item) for item in data.get("cues", ())))


@dataclass(frozen=True)
class FrameArtifact:
    path: str
    timestamp_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VisualIndex:
    """The frames the scan kept, and the montages the model is shown."""

    frames: tuple[FrameArtifact, ...] = ()
    montages: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": [frame.to_dict() for frame in self.frames],
            "montages": list(self.montages),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisualIndex:
        return cls(
            tuple(FrameArtifact(**item) for item in data.get("frames", ())),
            tuple(data.get("montages", ())),
        )


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    sha256: str
    duration_s: float
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoIndex:
    """Three separate indexes over one video, plus what the file itself is.

    They are kept apart on purpose. The audio index and the text index are
    independent observers with their own units, their own failure modes and
    their own line formats; folding them into one list made the text look like
    a continuation of the transcript, which it is not. They are stored apart,
    written to separate files, and shown to the model as separate blocks.
    """

    schema_version: int
    video: VideoMetadata
    audio: AudioIndex
    text: TextIndex
    visual: VisualIndex
    created_at: str
    indexer: dict[str, Any] = field(default_factory=dict)
    index_config: dict[str, Any] = field(default_factory=dict)
    # What a channel-provided caption track was judged to be and what it
    # changed. Diagnostic only: it is written here and never rendered, because
    # the agent is told what was heard, not who heard it.
    captions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video": self.video.to_dict(),
            "audio": self.audio.to_dict(),
            "text": self.text.to_dict(),
            "visual": self.visual.to_dict(),
            "created_at": self.created_at,
            "indexer": self.indexer,
            "index_config": self.index_config,
            "captions": self.captions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoIndex:
        return cls(
            schema_version=data["schema_version"],
            video=VideoMetadata(**data["video"]),
            audio=AudioIndex.from_dict(data["audio"]),
            text=TextIndex.from_dict(data["text"]),
            visual=VisualIndex.from_dict(data["visual"]),
            created_at=data["created_at"],
            indexer=data.get("indexer", {}),
            index_config=data.get("index_config", {}),
            captions=data.get("captions", {}),
        )
