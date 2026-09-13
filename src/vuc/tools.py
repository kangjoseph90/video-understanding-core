from __future__ import annotations

import hashlib
import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vuc.advanced_asr import AdvancedASRProvider, ASRResult, create_advanced_asr
from vuc.audio import AudioEventTagger, Region, create_event_tagger, keep_tags
from vuc.cache import VideoCache
from vuc.config import AppConfig
from vuc.frames import (
    create_montages,
    extract_sampled_frames,
    format_span,
    montage_cell_size,
)
from vuc.hints import VideoHints
from vuc.media import extract_audio_segment
from vuc.models import NON_SPEECH, SPEECH, VideoIndex
from vuc.timeline import Span
from vuc.trace import TraceWriter

VIEW_FRAMES_SCHEMA = {
    "type": "function",
    "function": {
        "name": "view_frames",
        "description": (
            "Extract timestamped frames from an interval and return n×n montage images. "
            "The number of returned montages is limited by tool configuration."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_s": {"type": "number"},
                "end_s": {"type": "number"},
                "fps": {"type": "number"},
                "n": {"type": "integer"},
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
            "Re-examine a selected interval and return a more precise account of "
            "what is audible in it. The interval is limited by the configured maximum."
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


# A sliver left over from clipping the VAD split to the requested interval is
# not worth cutting an audio file for, and an ASR handed 80ms of a word will
# happily invent a sentence out of it.
MIN_TOOL_REGION_S = 0.4


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
        trace_path: Path,
        provider: AdvancedASRProvider | None = None,
        tagger: AudioEventTagger | None = None,
        hints: VideoHints | None = None,
    ) -> None:
        self.video_path = video_path
        self.index = index
        self.cache = cache
        self.config = config
        self.hints = hints
        self._provider = provider
        self._tagger = tagger
        self._tagger_ready = tagger is not None
        self.trace = TraceWriter(trace_path)
        self.asr_processing_s = 0.0
        self.cloud_asr_cost_usd = 0.0
        self.cloud_asr_audio_s = 0.0

    @property
    def provider(self) -> AdvancedASRProvider:
        if self._provider is None:
            self._provider = create_advanced_asr(self.config.advanced_asr)
        return self._provider

    @property
    def tagger(self) -> AudioEventTagger | None:
        """Built on first use, and never allowed to take a tool call down.

        The tool is asked for a better account of an interval; if the labeller
        cannot be had, the speech half of that account is still worth
        returning.
        """
        if not self._tagger_ready:
            self._tagger_ready = True
            error: str | None = None
            try:
                self._tagger = create_event_tagger(self.config.audio_events)
            except Exception as exc:  # noqa: BLE001 - degrade, do not abort the call
                self._tagger = None
                error = str(exc)
            if self._tagger is None and self.config.audio_events.tagger != "none":
                # The sensevoice tagger needs a transcriber this service does not
                # own, so asking for it here yields nothing. Say so; a silently
                # untagged non-speech region is indistinguishable from a silent one.
                self.trace.write(
                    step="agent",
                    event="event_tagger_unavailable",
                    arguments={"tagger": self.config.audio_events.tagger},
                    result_summary={"error": error},
                    duration_ms=0,
                )
        return self._tagger

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
        return min(
            self.config.transcribe_segment.max_duration_s,
            self.provider_max_segment_s,
        )

    @property
    def schemas(self) -> list[dict[str, Any]]:
        frames = deepcopy(VIEW_FRAMES_SCHEMA)
        frames["function"]["description"] = (
            "Extract timestamped frames from an interval and return n×n montage images. "
            f"At most {self.config.view_frames.max_montages_per_call} montage images "
            "may be returned per call."
        )
        properties = frames["function"]["parameters"]["properties"]
        properties["fps"]["enum"] = list(self.config.view_frames.fps_options)
        properties["n"]["enum"] = list(self.config.view_frames.grid_options)
        transcript = deepcopy(TRANSCRIBE_SEGMENT_SCHEMA)
        transcript["function"]["description"] = (
            "Re-examine a selected interval and return a more precise account of "
            "what is audible in it, in the same [start-end] form as the index. "
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
        if not math.isfinite(fps_value) or fps_value not in self.config.view_frames.fps_options:
            raise ToolError(f"fps must be one of {self.config.view_frames.fps_options}")
        if (
            not math.isfinite(n_number)
            or n_value != n_number
            or n_value not in self.config.view_frames.grid_options
        ):
            raise ToolError(f"n must be one of {self.config.view_frames.grid_options}")
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
            "montage_width": self.config.montage.width,
            "montage_height": self.config.montage.height,
        }
        call_dir = self.cache.tool_frames_dir / self._key("view_frames", arguments)
        metadata_path = call_dir / "result.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            images = tuple(Path(path) for path in metadata.pop("image_paths"))
            return ToolExecution(data={**metadata, "cache_hit": True}, image_paths=images)

        cell_width, _ = montage_cell_size(
            self.config.montage.width,
            self.config.montage.height,
            n_value,
        )
        frames = extract_sampled_frames(
            self.video_path,
            call_dir / "frames",
            start_s=start,
            end_s=end,
            fps=fps_value,
            resolution=cell_width,
            jpeg_quality=self.config.montage.jpeg_quality,
        )
        montages = create_montages(
            frames,
            call_dir / "montages",
            n=n_value,
            width=self.config.montage.width,
            height=self.config.montage.height,
            jpeg_quality=self.config.montage.jpeg_quality,
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
        """Prefer what the index observed locally, else the declared language.

        baseline_full skips indexing entirely, so without the metadata fallback
        it hands Whisper no language at all and auto-detection can pick the
        wrong one for the whole video.
        """
        languages = {
            segment.language
            for segment in self.index.audio.segments
            if segment.end > start_s
            and segment.start < end_s
            and segment.language not in {"unknown", "nospeech"}
        }
        if len(languages) == 1:
            return next(iter(languages))
        return self.hints.language if self.hints else None

    def _regions(self, span: Span) -> list[Region]:
        """The VAD split, clipped to what was asked for.

        The index already holds that split, so the tool reuses it rather than
        running a second detector that could disagree with the very timeline
        the agent is citing. baseline_full builds no index, so there is nothing
        to reuse and the interval is treated as speech throughout.
        """
        clipped = [
            Region(
                Span(max(span.start, segment.start), min(span.end, segment.end)),
                segment.kind,
            )
            for segment in self.index.audio.segments
            if segment.end > span.start and segment.start < span.end
        ]
        kept = [region for region in clipped if region.span.duration >= MIN_TOOL_REGION_S]
        return kept or [Region(span, SPEECH)]

    def _region_result(
        self,
        region: Region,
        clip: Path,
        *,
        language_hint: str | None,
        prompt: str,
    ) -> tuple[list[dict[str, Any]], ASRResult | None]:
        """One region, answered by whichever model that side of the split calls for."""
        if not region.is_speech:
            tagger = self.tagger
            tags = tagger.tag(clip, duration_s=region.span.duration) if tagger else []
            events = keep_tags(tags, config=self.config.audio_events)
            if not events:
                return [], None
            return [
                {
                    "start_s": region.span.start,
                    "end_s": region.span.end,
                    "kind": NON_SPEECH,
                    "text": "",
                    "events": list(events),
                }
            ], None

        result = self.provider.transcribe(
            clip,
            audio_duration_s=region.span.duration,
            language_hint=language_hint,
            prompt=prompt or None,
        )
        lines = [
            {
                "start_s": round(min(region.span.start + sentence.start_s, region.span.end), 3),
                "end_s": round(min(region.span.start + sentence.end_s, region.span.end), 3),
                "kind": SPEECH,
                "text": sentence.text.strip(),
                "events": [],
            }
            for sentence in result.sentences
            if sentence.text.strip()
        ]
        return lines, result

    @staticmethod
    def _transcript_lines(segments: list[dict[str, Any]]) -> str:
        """The same shape the index uses, so every line the model reads matches."""
        rendered = []
        for segment in segments:
            tags = " ".join(f"<{event}>" for event in segment.get("events") or ())
            body = " ".join(part for part in (tags, segment.get("text") or "") if part)
            span = format_span(segment["start_s"], segment["end_s"])
            rendered.append(f"[{span}] {body}".rstrip())
        return "\n".join(rendered)

    def _transcribe(self, start_s: Any, end_s: Any, *, maximum_s: float) -> ToolExecution:
        start, end = self._range(start_s, end_s)
        duration = end - start
        if duration > maximum_s + 1e-6:
            raise ToolError(
                f"transcribe_segment interval is {duration:.3f}s; the limit is {maximum_s:.0f}s"
            )
        prompt = self.hints.asr_prompt(start, end) if self.hints else ""
        arguments = {
            "start_s": start,
            "end_s": end,
            "provider": self.provider_name,
            "model": self.provider_model_name,
            "tagger": self.config.audio_events.tagger,
            # The prompt steers the transcript, so it belongs in the cache key.
            "prompt": prompt,
        }
        result_path = self.cache.advanced_asr_dir / f"{self._key('asr', arguments)}.json"
        if result_path.exists():
            data = json.loads(result_path.read_text(encoding="utf-8"))
            return ToolExecution(data={**data, "cache_hit": True})

        language_hint = self._language_hint(start, end)
        segments: list[dict[str, Any]] = []
        results: list[ASRResult] = []
        asr_elapsed = 0.0
        speech_s = 0.0
        for index, region in enumerate(self._regions(Span(start, end))):
            clip = result_path.with_suffix(f".{index:03d}.wav")
            extract_audio_segment(
                self.cache.audio_path, clip, start_s=region.span.start, end_s=region.span.end
            )
            try:
                wall_started = time.monotonic()
                lines, result = self._region_result(
                    region, clip, language_hint=language_hint, prompt=prompt
                )
                asr_elapsed += time.monotonic() - wall_started
            finally:
                clip.unlink(missing_ok=True)
            segments.extend(lines)
            if result is not None:
                results.append(result)
                speech_s += region.span.duration

        segments.sort(key=lambda item: (item["start_s"], item["end_s"]))
        provider = results[0].provider if results else self.provider_name
        # None means unknown, not free: summing Nones into 0.0 would report a
        # provider that does not price its output as having cost nothing.
        priced = [item.cost_usd for item in results if item.cost_usd is not None]
        data: dict[str, Any] = {
            "start_s": start,
            "end_s": end,
            "segments": segments,
            "transcript": self._transcript_lines(segments),
            "text": " ".join(item["text"] for item in segments if item["text"]),
            "language": next((item.language for item in results if item.language), language_hint),
            "provider": provider,
            "model": results[0].model if results else self.provider_model_name,
            "audio_duration_s": round(duration, 3),
            "speech_duration_s": round(speech_s, 3),
            "processing_s": round(sum(item.processing_s for item in results), 3),
            "cost_usd": sum(priced) if priced else None,
            "language_hint": language_hint,
            "asr_prompt": prompt,
            "cache_hit": False,
        }
        self.cache.write_json(result_path, data)
        # Model time for the whole call, tagger included: it is excluded from
        # the agent's wall-clock budget for the same reason the ASR time is.
        self.asr_processing_s += asr_elapsed
        if provider == "cloud":
            # Only the speech regions were ever sent to a paid endpoint.
            self.cloud_asr_audio_s += speech_s
            self.cloud_asr_cost_usd += data["cost_usd"] or 0.0
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
