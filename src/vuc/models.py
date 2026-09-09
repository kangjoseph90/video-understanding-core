from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str
    language: str
    emotion: str | None = None
    events: tuple[str, ...] = ()
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["events"] = list(self.events)
        return result


@dataclass(frozen=True)
class VideoMetadata:
    path: str
    sha256: str
    duration_s: float
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrameArtifact:
    path: str
    timestamp_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VideoIndex:
    schema_version: int
    video: VideoMetadata
    segments: tuple[Segment, ...]
    frames: tuple[FrameArtifact, ...]
    montages: tuple[str, ...]
    created_at: str
    indexer: dict[str, Any] = field(default_factory=dict)
    frame_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "video": self.video.to_dict(),
            "segments": [segment.to_dict() for segment in self.segments],
            "frames": [frame.to_dict() for frame in self.frames],
            "montages": list(self.montages),
            "created_at": self.created_at,
            "indexer": self.indexer,
            "frame_config": self.frame_config,
        }
