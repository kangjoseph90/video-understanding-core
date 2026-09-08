from __future__ import annotations

import hashlib
import json
import math
import time
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from vuc.advanced_asr import AdvancedASRProvider, create_advanced_asr
from vuc.cache import VideoCache
from vuc.config import AppConfig
from vuc.frames import create_montages, extract_sampled_frames
from vuc.media import extract_audio_segment
from vuc.models import Segment, VideoIndex
from vuc.trace import TraceWriter

VIEW_FRAMES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "view_frames",
        "description": "Extract and view timestamped frames from a video interval.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_s": {"type": "number"},
                "end_s": {"type": "number"},
                "fps": {"type": "number", "enum": [0.1, 0.2, 0.5, 1, 2]},
                "resolution": {"type": "integer", "enum": [256, 512, 768]},
            },
            "required": ["start_s", "end_s", "fps", "resolution"],
            "additionalProperties": False,
        },
    },
}

TRANSCRIBE_SEGMENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "transcribe_segment",
        "description": "Transcribe a selected interval with the advanced ASR provider.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_s": {"type": "number"},
                "end_s": {"type": "number"},
            },
            "required": ["start_s", "end_s"],
            "additionalProperties": False,
        },
    },
}

READ_INDEX_SCHEMA = {
    "type": "function",
    "function": {
        "name": "read_index",
        "description": "Read full-quality SenseVoice index segments from a selected interval.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_s": {"type": "number"},
                "end_s": {"type": "number"},
            },
            "required": ["start_s", "end_s"],
            "additionalProperties": False,
        },
    },
}


class ToolError(RuntimeError):
    pass


@dataclass(frozen=True)
class ToolExecution:
    data: dict[str, Any]
    image_paths: tuple[Path, ...] = ()
    asr_elapsed_s: float = 0.0


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if end < start:
            start, end = end, start
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def interval_coverage(
    start_s: float,
    end_s: float,
    intervals: list[tuple[float, float]],
) -> float:
    if end_s < start_s:
        start_s, end_s = end_s, start_s
    merged = merge_intervals(intervals)
    if math.isclose(start_s, end_s):
        return 1.0 if any(start <= start_s <= end for start, end in merged) else 0.0
    overlap = sum(max(0.0, min(end_s, end) - max(start_s, start)) for start, end in merged)
    return min(1.0, overlap / (end_s - start_s))


def normalize_for_cer(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def character_error_rate(reference: str, hypothesis: str) -> float:
    reference_chars = normalize_for_cer(reference)
    hypothesis_chars = normalize_for_cer(hypothesis)
    if not reference_chars:
        return 0.0 if not hypothesis_chars else 1.0
    previous = list(range(len(hypothesis_chars) + 1))
    for row, reference_char in enumerate(reference_chars, start=1):
        current = [row]
        for column, hypothesis_char in enumerate(hypothesis_chars, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_char != hypothesis_char),
                )
            )
        previous = current
    return previous[-1] / len(reference_chars)


def _slice_segment_text(segment: Segment, start_s: float, end_s: float) -> str:
    overlap_start = max(start_s, segment.start)
    overlap_end = min(end_s, segment.end)
    if overlap_end <= overlap_start or not segment.text:
        return ""
    duration = segment.end - segment.start
    if duration <= 0:
        return segment.text
    start_index = math.floor(len(segment.text) * (overlap_start - segment.start) / duration)
    end_index = math.ceil(len(segment.text) * (overlap_end - segment.start) / duration)
    return segment.text[start_index:end_index].strip()


def _segment_dict(segment: Segment) -> dict[str, Any]:
    return segment.to_dict()


class ToolService:
    def __init__(
        self,
        *,
        video_path: Path,
        index: VideoIndex,
        cache: VideoCache,
        config: AppConfig,
        read_index_enabled: bool,
        provider: AdvancedASRProvider | None = None,
    ) -> None:
        self.video_path = video_path
        self.index = index
        self.cache = cache
        self.config = config
        self.read_index_enabled = read_index_enabled
        self._provider = provider
        self.trace = TraceWriter(cache.trace_path)
        self.verified_intervals: list[tuple[float, float]] = []
        self.frame_intervals: list[tuple[float, float]] = []
        self.transcript_intervals: list[tuple[float, float]] = []
        self.asr_processing_s = 0.0
        self.cloud_asr_cost_usd = 0.0
        self.cloud_asr_audio_s = 0.0

    @property
    def provider(self) -> AdvancedASRProvider:
        if self._provider is None:
            self._provider = create_advanced_asr(self.config.advanced_asr)
        return self._provider

    @property
    def provider_name(self) -> str:
        return (
            self._provider.name if self._provider is not None else self.config.advanced_asr.provider
        )

    @property
    def provider_model_name(self) -> str:
        if self._provider is not None:
            return self._provider.model_name
        if self.config.advanced_asr.provider == "local":
            return self.config.advanced_asr.local.model
        return "openai-compatible-stub"

    @property
    def provider_max_segment_s(self) -> float:
        if self._provider is not None:
            configured_limit = (
                self.config.advanced_asr.local.max_segment_s
                if self._provider.name == "local"
                else self.config.advanced_asr.cloud.max_segment_s
            )
            provider_limit = min(self._provider.max_segment_s, configured_limit)
            return min(180.0, provider_limit) if self._provider.name == "local" else provider_limit
        if self.config.advanced_asr.provider == "local":
            return min(180.0, self.config.advanced_asr.local.max_segment_s)
        return self.config.advanced_asr.cloud.max_segment_s

    @property
    def schemas(self) -> list[dict[str, Any]]:
        schemas = [deepcopy(VIEW_FRAMES_SCHEMA), deepcopy(TRANSCRIBE_SEGMENT_SCHEMA)]
        schemas[1]["function"]["description"] += (
            f" Maximum interval for the active provider: {self.provider_max_segment_s:.0f} seconds."
        )
        if self.read_index_enabled:
            schemas.append(deepcopy(READ_INDEX_SCHEMA))
        return schemas

    def _range(self, start_s: Any, end_s: Any) -> tuple[float, float]:
        try:
            start = float(start_s)
            end = float(end_s)
        except (TypeError, ValueError) as exc:
            raise ToolError("start_s and end_s must be numbers") from exc
        if not math.isfinite(start) or not math.isfinite(end):
            raise ToolError("start_s and end_s must be finite")
        if start < 0 or end <= start or end > self.index.video.duration_s + 0.01:
            raise ToolError(
                f"invalid interval [{start}, {end}]; video duration is "
                f"{self.index.video.duration_s:.3f}s"
            )
        return start, min(end, self.index.video.duration_s)

    def _key(self, name: str, arguments: dict[str, Any]) -> str:
        payload = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(f"{name}:{payload}".encode()).hexdigest()[:24]

    def view_frames(
        self,
        start_s: Any,
        end_s: Any,
        fps: Any,
        resolution: Any,
    ) -> ToolExecution:
        start, end = self._range(start_s, end_s)
        try:
            fps_value = float(fps)
            resolution_value = int(resolution)
        except (TypeError, ValueError) as exc:
            raise ToolError("fps and resolution must be numeric") from exc
        if fps_value not in self.config.view_frames.allowed_fps:
            raise ToolError(f"fps must be one of {self.config.view_frames.allowed_fps}")
        if resolution_value not in self.config.view_frames.allowed_resolutions:
            raise ToolError(
                f"resolution must be one of {self.config.view_frames.allowed_resolutions}"
            )
        expected = math.ceil((end - start) * fps_value)
        if expected > self.config.view_frames.max_frames_per_call:
            raise ToolError(
                f"구간을 줄이거나 fps를 낮춰라: 예상 프레임 수 {expected}, "
                f"상한 {self.config.view_frames.max_frames_per_call}"
            )
        arguments = {
            "start_s": start,
            "end_s": end,
            "fps": fps_value,
            "resolution": resolution_value,
        }
        call_dir = self.cache.tool_frames_dir / self._key("view_frames", arguments)
        metadata_path = call_dir / "result.json"
        if metadata_path.exists():
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            images = tuple(Path(path) for path in data["image_paths"])
            self.verified_intervals.append((start, end))
            self.frame_intervals.append((start, end))
            return ToolExecution(data={**data, "cache_hit": True}, image_paths=images)

        frames = extract_sampled_frames(
            self.video_path,
            call_dir / "frames",
            start_s=start,
            end_s=end,
            fps=fps_value,
            resolution=resolution_value,
            jpeg_quality=self.config.frames.jpeg_quality,
        )
        montage_config = replace(
            self.config.frames,
            initial_interval_s=1 / fps_value,
            initial_resolution=resolution_value,
        )
        montages = create_montages(frames, call_dir / "montages", montage_config)
        images = tuple(montages or [Path(frame.path) for frame in frames])
        data = {
            **arguments,
            "frame_count": len(frames),
            "frames": [frame.to_dict() for frame in frames],
            "image_paths": [str(path) for path in images],
            "cache_hit": False,
        }
        self.cache.write_json(metadata_path, data)
        self.verified_intervals.append((start, end))
        self.frame_intervals.append((start, end))
        return ToolExecution(data=data, image_paths=images)

    def _language_hint(self, start_s: float, end_s: float) -> str | None:
        languages = {
            segment.language
            for segment in self.index.segments
            if segment.end > start_s
            and segment.start < end_s
            and segment.language not in {"unknown", "nospeech"}
        }
        return next(iter(languages)) if len(languages) == 1 else None

    def _index_text(self, start_s: float, end_s: float) -> str:
        return " ".join(
            text
            for segment in self.index.segments
            if (text := _slice_segment_text(segment, start_s, end_s))
        )

    def _index_score(self, start_s: float, end_s: float, transcript: str) -> dict[str, Any]:
        index_text = self._index_text(start_s, end_s)
        cer = character_error_rate(transcript, index_text)
        return {
            "index_text": index_text,
            "index_cer": round(cer, 6),
            "index_similarity": round(max(0.0, 1.0 - cer), 6),
            "index_similarity_metric": "1-CER after NFKC/alphanumeric normalization",
        }

    def transcribe_segment(self, start_s: Any, end_s: Any) -> ToolExecution:
        start, end = self._range(start_s, end_s)
        duration = end - start
        if duration > self.provider_max_segment_s + 1e-6:
            raise ToolError(
                f"transcribe_segment interval is {duration:.3f}s; "
                f"{self.provider_name} provider limit is {self.provider_max_segment_s:.0f}s"
            )
        arguments = {
            "start_s": start,
            "end_s": end,
            "provider": self.provider_name,
            "model": self.provider_model_name,
        }
        result_path = self.cache.advanced_asr_dir / f"{self._key('asr', arguments)}.json"
        if result_path.exists():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            data.update(self._index_score(start, end, str(data.get("text") or "")))
            self.cache.write_json(result_path, data)
            self.verified_intervals.append((start, end))
            self.transcript_intervals.append((start, end))
            return ToolExecution(data={**data, "cache_hit": True})

        audio_path = result_path.with_suffix(".wav")
        extract_audio_segment(
            self.cache.audio_path,
            audio_path,
            start_s=start,
            end_s=end,
        )
        language_hint = self._language_hint(start, end)
        wall_started = time.monotonic()
        provider = self.provider
        result = provider.transcribe(
            audio_path,
            audio_duration_s=duration,
            language_hint=language_hint,
        )
        asr_elapsed = time.monotonic() - wall_started
        data = result.to_dict()
        data.update(
            {
                "start_s": start,
                "end_s": end,
                "sentences": [
                    {
                        **sentence.to_dict(),
                        "start_s": sentence.start_s + start,
                        "end_s": sentence.end_s + start,
                    }
                    for sentence in result.sentences
                ],
                "language_hint": language_hint,
                **self._index_score(start, end, result.text),
                "cache_hit": False,
            }
        )
        self.cache.write_json(result_path, data)
        self.verified_intervals.append((start, end))
        self.transcript_intervals.append((start, end))
        self.asr_processing_s += asr_elapsed
        if result.provider == "cloud":
            self.cloud_asr_audio_s += duration
            self.cloud_asr_cost_usd += result.cost_usd or 0.0
        return ToolExecution(data=data, asr_elapsed_s=asr_elapsed)

    def read_index(self, start_s: Any, end_s: Any) -> ToolExecution:
        if not self.read_index_enabled:
            raise ToolError("read_index is disabled because the full index fits the prompt budget")
        start, end = self._range(start_s, end_s)
        segments = [
            _segment_dict(segment)
            for segment in self.index.segments
            if segment.end > start and segment.start < end
        ]
        return ToolExecution(data={"start_s": start, "end_s": end, "segments": segments})

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecution:
        started = time.monotonic()
        try:
            if name == "view_frames":
                execution = self.view_frames(**arguments)
            elif name == "transcribe_segment":
                execution = self.transcribe_segment(**arguments)
            elif name == "read_index":
                execution = self.read_index(**arguments)
            else:
                raise ToolError(f"unknown tool: {name}")
        except (ToolError, OSError, RuntimeError, TypeError) as exc:
            execution = ToolExecution(data={"error": str(exc)})
        self.trace.write(
            step="agent",
            event="tool_call",
            arguments={"name": name, "arguments": arguments},
            result_summary={
                "error": execution.data.get("error"),
                "image_count": len(execution.image_paths),
                "cache_hit": execution.data.get("cache_hit"),
                "provider": execution.data.get("provider"),
                "audio_duration_s": execution.data.get("audio_duration_s"),
                "asr_response_s": round(execution.asr_elapsed_s, 3),
                "cost_usd": execution.data.get("cost_usd"),
            },
            duration_ms=round((time.monotonic() - started) * 1000),
        )
        return execution

    def citation_coverage(self, start_s: float, end_s: float) -> float:
        return interval_coverage(start_s, end_s, self.verified_intervals)

    def evidence_coverage(self, start_s: float, end_s: float, source: str) -> float:
        if source == "view_frames":
            intervals = self.frame_intervals
        elif source == "transcribe_segment":
            intervals = self.transcript_intervals
        else:
            return 0.0
        return interval_coverage(start_s, end_s, intervals)
