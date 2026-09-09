"""Channel-provided metadata used as a transcription and reporting hint.

A sidecar at <video>.meta.json is picked up automatically, so a video with no
sidecar simply runs without hints. Only the fields that are reliable are used:
the channel name, the title and the human-authored chapter list. Descriptions
are deliberately excluded -- across the field-eval manifest they range from
pure link boilerplate to a series programme that describes other episodes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SIDECAR_SUFFIX = ".meta.json"


@dataclass(frozen=True)
class Chapter:
    start_s: float
    end_s: float
    title: str


@dataclass(frozen=True)
class VideoHints:
    title: str = ""
    channel: str = ""
    language: str | None = None
    chapters: tuple[Chapter, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "channel": self.channel,
            "language": self.language,
            "chapters": [
                {"start_s": c.start_s, "end_s": c.end_s, "title": c.title}
                for c in self.chapters
            ],
        }

    def chapters_at(self, start_s: float, end_s: float) -> tuple[Chapter, ...]:
        return tuple(
            chapter
            for chapter in self.chapters
            if chapter.end_s > start_s and chapter.start_s < end_s
        )

    def prompt_block(self) -> str:
        """Metadata shown to the reporting model, timestamps in seconds."""
        lines = []
        if self.channel:
            lines.append(f"채널: {self.channel}")
        if self.title:
            lines.append(f"제목: {self.title}")
        if self.chapters:
            lines.append("챕터:")
            lines.extend(
                f"[{int(c.start_s)}-{int(c.end_s)}] {c.title}" for c in self.chapters
            )
        if not lines:
            return ""
        return "영상 메타데이터 (채널 제공):\n" + "\n".join(lines)

    def asr_prompt(self, start_s: float, end_s: float) -> str:
        """Whisper initial_prompt: channel - title - chapters covering the chunk.

        Kept short on purpose; a long initial_prompt makes Whisper echo it back
        as transcript text.
        """
        parts = [part for part in (self.channel, self.title) if part]
        chapters = ", ".join(
            chapter.title for chapter in self.chapters_at(start_s, end_s) if chapter.title
        )
        if chapters:
            parts.append(chapters)
        return " - ".join(parts)


def load_video_hints(video_path: Path) -> VideoHints | None:
    sidecar = video_path.with_suffix(SIDECAR_SUFFIX)
    if not sidecar.is_file():
        return None
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return None
    chapters = tuple(
        Chapter(
            start_s=float(chapter.get("start_s") or 0),
            end_s=float(chapter.get("end_s") or 0),
            title=str(chapter.get("title") or "").strip(),
        )
        for chapter in data.get("chapters") or []
        if isinstance(chapter, dict)
    )
    language = data.get("language")
    return VideoHints(
        title=str(data.get("title") or "").strip(),
        channel=str(data.get("channel") or "").strip(),
        language=str(language).strip() if language else None,
        chapters=chapters,
    )
