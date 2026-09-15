"""What the agent's own transcription calls have taught us about this video.

The index is built once from cheap models. The agent then spends real time on
selected intervals with a better one, and until now that work died with the
run: the tool cached its answer under a hash of the exact interval requested,
so a later call five seconds away reused nothing and the next query started
from the same SenseVoice text as the first.

These rows are that work, kept. They live beside the audio rather than inside
the index, because the index is what processing the video produced and this is
something a later pass learned; and they are keyed by time rather than by
region number so a change to the VAD settings reprojects them instead of
orphaning them.

Sentences are stored, not whole answers. A sentence carries the one timestamp
placement needs, and overlapping calls then dedupe against each other
naturally.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vuc.captions import CaptionCue
from vuc.models import SPEECH, Segment


@dataclass(frozen=True)
class TranscriptRow:
    """One sentence a transcription call produced, and what it could hear.

    ``region`` is the VAD region the sentence belongs to and ``clip`` is how
    much of that region the call actually sent to the model. They differ when
    the agent asked for an interval that starts or ends mid-region, and the
    difference is what decides between two accounts of the same speech: a clip
    that covered the whole region heard the run-up to the sentence, and one cut
    short did not.
    """

    start_s: float
    end_s: float
    text: str
    region_start: float
    region_end: float
    clip_start: float
    clip_end: float
    model: str = ""
    recorded_at: str = ""

    @property
    def region(self) -> tuple[float, float]:
        return (round(self.region_start, 3), round(self.region_end, 3))

    @property
    def coverage(self) -> float:
        """Fraction of its region the call actually heard."""
        span = self.region_end - self.region_start
        if span <= 0:
            return 0.0
        overlap = min(self.clip_end, self.region_end) - max(self.clip_start, self.region_start)
        return max(overlap, 0.0) / span

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "text": self.text,
            "region": [round(self.region_start, 3), round(self.region_end, 3)],
            "clip": [round(self.clip_start, 3), round(self.clip_end, 3)],
            "model": self.model,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranscriptRow:
        region = data.get("region") or [0.0, 0.0]
        clip = data.get("clip") or region
        return cls(
            start_s=float(data["start_s"]),
            end_s=float(data["end_s"]),
            text=str(data["text"]),
            region_start=float(region[0]),
            region_end=float(region[1]),
            clip_start=float(clip[0]),
            clip_end=float(clip[1]),
            model=str(data.get("model") or ""),
            recorded_at=str(data.get("recorded_at") or ""),
        )


def append_rows(path: Path, rows: list[TranscriptRow]) -> None:
    """Add what this call heard. Append-only: nothing already written is judged.

    Which of two accounts of the same region to believe is decided when they
    are read, so a call never has to know what came before it.
    """
    if not rows:
        return
    stamp = datetime.now(UTC).isoformat()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            payload = replace(row, recorded_at=row.recorded_at or stamp).to_dict()
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_rows(path: Path) -> tuple[TranscriptRow, ...]:
    """Every sentence worth keeping, one account per region.

    Two calls that both covered a region are not merged. The one that heard
    more of it wins, and a tie goes to the later call, which at least ran
    against the same settings as everything else recent.
    """
    if not path.is_file():
        return ()
    # region -> (coverage, recorded_at) -> the sentences that call produced
    calls: dict[tuple[float, float], dict[tuple[float, str], list[TranscriptRow]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = TranscriptRow.from_dict(json.loads(line))
        except (ValueError, KeyError, TypeError, IndexError):
            continue
        calls.setdefault(row.region, {}).setdefault((row.coverage, row.recorded_at), []).append(row)
    kept = [row for per_region in calls.values() for row in per_region[max(per_region)]]
    return tuple(sorted(kept, key=lambda r: (r.start_s, r.end_s)))


def rows_digest(rows: tuple[TranscriptRow, ...]) -> str:
    """Identity of the accumulated state a run was rendered against.

    The prompt now depends on how much the agent has already learned, so a
    report is only reproducible alongside the state it was written from. Twelve
    hex characters is enough to tell two states apart in a trace.
    """
    body = "|".join(f"{r.start_s:.3f}{r.end_s:.3f}{r.text}" for r in rows)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:12]


def apply_transcriptions(
    segments: tuple[Segment, ...],
    rows: tuple[TranscriptRow, ...],
    *,
    align_ratio_min: float,
) -> tuple[tuple[Segment, ...], dict[str, Any]]:
    """Put the better account of each region into the timeline.

    Applied before the captions, so a channel's own text still wins where it
    reaches. Where it does not, this is what the agent already paid for.
    """
    from vuc.caption_fusion import fuse_audio_index

    stats = {
        "rows": len(rows),
        "digest": rows_digest(rows),
        "regions_rewritten": 0,
        "regions_guarded": 0,
    }
    if not rows:
        return segments, stats
    cues = tuple(CaptionCue(row.start_s, row.text) for row in rows if row.text.strip())
    if not cues:
        return segments, stats
    updated, fused = fuse_audio_index(segments, cues, align_ratio_min=align_ratio_min)
    stats["regions_rewritten"] = fused["regions_rewritten"]
    stats["regions_guarded"] = fused["regions_guarded"]
    stats["rows_placed"] = fused["cues_placed"]
    stats["rows_dropped"] = fused["cues_dropped"]
    return updated, stats


def rows_from_segments(
    lines: list[dict[str, Any]],
    *,
    region: tuple[float, float],
    clip: tuple[float, float],
    model: str,
) -> list[TranscriptRow]:
    """The speech half of one tool call, as rows.

    Non-speech regions are left out. Their labels come from the same tagger the
    index already ran, so recording them would preserve nothing new.
    """
    return [
        TranscriptRow(
            start_s=float(line["start_s"]),
            end_s=float(line["end_s"]),
            text=str(line["text"]).strip(),
            region_start=region[0],
            region_end=region[1],
            clip_start=clip[0],
            clip_end=clip[1],
            model=model,
        )
        for line in lines
        if line.get("kind") == SPEECH and str(line.get("text") or "").strip()
    ]
