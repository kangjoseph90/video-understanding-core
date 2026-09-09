from __future__ import annotations

import hashlib
import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vuc.advanced_asr import AdvancedASRProvider, create_advanced_asr
from vuc.cache import VideoCache
from vuc.config import AppConfig
from vuc.frames import create_montages, extract_sampled_frames
from vuc.media import extract_audio_segment
from vuc.models import VideoIndex
from vuc.trace import TraceWriter

VIEW_FRAMES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "view_frames",
        "description": (
            "Extract timestamped frames from an interval and return n×n montage images. "
            "At most 16 montage images may be returned per call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_s": {"type": "number"},
                "end_s": {"type": "number"},
                "fps": {"type": "number", "exclusiveMinimum": 0},
                "n": {"type": "integer", "minimum": 1},
            },
            "required": ["start_s", "end_s", "fps", "n"],
            "additionalProperties": False,
        },
    },
}

TRANSCRIBE_SEGMENT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "transcribe_segment",
        "description": (
            "Transcribe a selected interval with the advanced ASR provider. "
            "The interval may be at most 60 seconds."
        ),
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


class ToolService:
    def __init__(
        self,
        *,
        video_path: Path,
        index: VideoIndex,
        cache: VideoCache,
        config: AppConfig,
        provider: AdvancedASRProvider | None = None,
    ) -> None:
        self.video_path = video_path
        self.index = index
        self.cache = cache
        self.config = config
        self._provider = provider
        self.trace = TraceWriter(cache.trace_path)
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
            return min(self._provider.max_segment_s, configured_limit)
        return (
            self.config.advanced_asr.local.max_segment_s
            if self.config.advanced_asr.provider == "local"
            else self.config.advanced_asr.cloud.max_segment_s
        )

    @property
    def tool_max_segment_s(self) -> float:
        return min(60.0, self.provider_max_segment_s)

    @property
    def schemas(self) -> list[dict[str, Any]]:
        frames = deepcopy(VIEW_FRAMES_SCHEMA)
        frames["function"]["description"] = (
            "Extract timestamped frames from an interval and return n×n montage images. "
            f"At most {self.config.view_frames.max_montages_per_call} montage images "
            "may be returned per call."
        )
        transcript = deepcopy(TRANSCRIBE_SEGMENT_SCHEMA)
        transcript["function"]["description"] = (
            "Transcribe a selected interval with the advanced ASR provider. "
            f"The interval may be at most {self.tool_max_segment_s:.0f} seconds."
        )
        return [frames, transcript]

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

    def view_frames(self, start_s: Any, end_s: Any, fps: Any, n: Any) -> ToolExecution:
        start, end = self._range(start_s, end_s)
        try:
            fps_value = float(fps)
            n_value = int(n)
            n_number = float(n)
        except (TypeError, ValueError) as exc:
            raise ToolError("fps and n must be numeric") from exc
        if not math.isfinite(fps_value) or fps_value <= 0:
            raise ToolError("fps must be a positive finite number")
        if not math.isfinite(n_number) or n_value < 1 or n_value != n_number:
            raise ToolError("n must be a positive integer")
        frame_count = math.ceil((end - start) * fps_value)
        montage_count = math.ceil(frame_count / (n_value * n_value))
        maximum = self.config.view_frames.max_montages_per_call
        if montage_count > maximum:
            raise ToolError(
                f"increase n or reduce the interval/fps: expected {frame_count} frames and "
                f"{montage_count} montage images; montage limit is {maximum}"
            )
        arguments = {
            "start_s": start,
            "end_s": end,
            "fps": fps_value,
            "n": n_value,
            "resolution": self.config.view_frames.resolution,
        }
        call_dir = self.cache.tool_frames_dir / self._key("view_frames", arguments)
        metadata_path = call_dir / "result.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            images = tuple(Path(path) for path in metadata.pop("image_paths"))
            return ToolExecution(data={**metadata, "cache_hit": True}, image_paths=images)

        frames = extract_sampled_frames(
            self.video_path,
            call_dir / "frames",
            start_s=start,
            end_s=end,
            fps=fps_value,
            resolution=self.config.view_frames.resolution,
            jpeg_quality=self.config.frames.jpeg_quality,
        )
        montages = create_montages(
            frames,
            call_dir / "montages",
            n=n_value,
            jpeg_quality=self.config.frames.jpeg_quality,
        )
        images = tuple(montages)
        data: dict[str, Any] = {
            **arguments,
            "frame_count": len(frames),
            "montage_count": len(montages),
            "cache_hit": False,
        }
        self.cache.write_json(
            metadata_path,
            {**data, "image_paths": [str(path) for path in images]},
        )
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

    def _transcribe(self, start_s: Any, end_s: Any, *, maximum_s: float) -> ToolExecution:
        start, end = self._range(start_s, end_s)
        duration = end - start
        if duration > maximum_s + 1e-6:
            raise ToolError(
                f"transcribe_segment interval is {duration:.3f}s; "
                f"the limit is {maximum_s:.0f}s"
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
            return ToolExecution(data={**data, "cache_hit": True})

        audio_path = result_path.with_suffix(".wav")
        extract_audio_segment(self.cache.audio_path, audio_path, start_s=start, end_s=end)
        language_hint = self._language_hint(start, end)
        wall_started = time.monotonic()
        result = self.provider.transcribe(
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
                "cache_hit": False,
            }
        )
        self.cache.write_json(result_path, data)
        self.asr_processing_s += asr_elapsed
        if result.provider == "cloud":
            self.cloud_asr_audio_s += duration
            self.cloud_asr_cost_usd += result.cost_usd or 0.0
        return ToolExecution(data=data, asr_elapsed_s=asr_elapsed)

    def transcribe_segment(self, start_s: Any, end_s: Any) -> ToolExecution:
        return self._transcribe(start_s, end_s, maximum_s=self.tool_max_segment_s)

    def transcribe_baseline_chunk(self, start_s: Any, end_s: Any) -> ToolExecution:
        return self._transcribe(start_s, end_s, maximum_s=self.provider_max_segment_s)

    def execute(self, name: str, arguments: dict[str, Any]) -> ToolExecution:
        started = time.monotonic()
        try:
            if name == "view_frames":
                execution = self.view_frames(**arguments)
            elif name == "transcribe_segment":
                execution = self.transcribe_segment(**arguments)
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
