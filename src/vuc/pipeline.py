from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from vuc.cache import VideoCache, new_run_id, sha256_file
from vuc.config import AppConfig
from vuc.frames import create_montages, extract_initial_frames, format_timestamp
from vuc.indexer import SenseVoiceTranscriber, Transcriber
from vuc.media import extract_audio, probe_duration
from vuc.models import FrameArtifact, Segment, VideoIndex, VideoMetadata
from vuc.trace import TraceWriter


def _validate_video(path: Path, config: AppConfig) -> Path:
    video_path = path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"video file not found: {video_path}")
    if video_path.suffix.lower().lstrip(".") not in config.video.allowed_extensions:
        allowed = ", ".join(config.video.allowed_extensions)
        raise ValueError(f"unsupported video extension; expected one of: {allowed}")
    return video_path


def _text_index(segments: list[Segment]) -> str:
    lines = []
    for segment in segments:
        tags = [segment.language, segment.emotion or "", *segment.events]
        tag_text = " ".join(f"<{tag}>" for tag in tags if tag and tag != "unknown")
        lines.append(
            f"[{format_timestamp(segment.start)}–{format_timestamp(segment.end)}] "
            f"{tag_text} {segment.text}".rstrip()
        )
    return "\n".join(lines) + ("\n" if lines else "")


def _load_cached(path: Path) -> VideoIndex:
    data = json.loads(path.read_text(encoding="utf-8"))
    video = VideoMetadata(**data["video"])
    segments = tuple(
        Segment(
            start=item["start"],
            end=item["end"],
            text=item["text"],
            language=item["language"],
            emotion=item.get("emotion") or next(iter(item.get("emotions", [])), None),
            events=tuple(item.get("events", item.get("audio_events", []))),
            raw_text=item.get("raw_text", ""),
        )
        for item in data["segments"]
    )
    frames = tuple(FrameArtifact(**item) for item in data["frames"])
    return VideoIndex(
        schema_version=data["schema_version"],
        video=video,
        segments=segments,
        frames=frames,
        montages=tuple(data["montages"]),
        created_at=data["created_at"],
        indexer=data.get("indexer", {}),
        frame_config=data.get("frame_config", {}),
    )


def _frame_config(config: AppConfig) -> dict[str, float | int]:
    return {
        "interval_s": config.frames.index_interval_s,
        "montage_width": config.montage.width,
        "montage_height": config.montage.height,
        "montage_n": config.frames.index_montage_n,
        "jpeg_quality": config.montage.jpeg_quality,
    }


def index_video(
    path: str | Path,
    config: AppConfig,
    *,
    transcriber: Transcriber | None = None,
    force: bool = False,
    run_id: str | None = None,
) -> tuple[VideoIndex, VideoCache, bool, Path]:
    video_path = _validate_video(Path(path), config)
    video_hash = sha256_file(video_path)
    cache = VideoCache(config.cache.directory, video_hash)
    cache.ensure()
    actual_run_id = run_id or new_run_id("index")
    trace_path = cache.run_dir(actual_run_id) / "trace.jsonl"
    trace = TraceWriter(trace_path)
    if cache.index_json_path.exists() and not force:
        cached = _load_cached(cache.index_json_path)
        if cached.frame_config == _frame_config(config):
            trace.write(
                step="index",
                event="cache_hit",
                arguments={"video_path": str(video_path)},
                result_summary={"video_hash": video_hash},
                duration_ms=0,
            )
            return cached, cache, True, trace_path

    duration_s = probe_duration(video_path)
    metadata = VideoMetadata(
        path=str(video_path),
        sha256=video_hash,
        duration_s=duration_s,
        size_bytes=video_path.stat().st_size,
    )

    def index_audio() -> list[Segment]:
        started = time.monotonic()
        extract_audio(video_path, cache.audio_path)
        active_transcriber = transcriber or SenseVoiceTranscriber(config.indexer, duration_s)
        segments = active_transcriber.transcribe(cache.audio_path)
        trace.write(
            step="index",
            event="sensevoice_index",
            arguments={"model": config.indexer.model, "device": config.indexer.device},
            result_summary={"segments": len(segments), "audio_path": str(cache.audio_path)},
            duration_ms=round((time.monotonic() - started) * 1000),
        )
        return segments

    def index_frames() -> tuple[list[FrameArtifact], list[Path]]:
        started = time.monotonic()
        frames = extract_initial_frames(
            video_path,
            cache.frames_dir,
            duration_s=duration_s,
            frames_config=config.frames,
            montage_config=config.montage,
        )
        montages = create_montages(
            frames,
            cache.montages_dir,
            n=config.frames.index_montage_n,
            width=config.montage.width,
            height=config.montage.height,
            jpeg_quality=config.montage.jpeg_quality,
        )
        trace.write(
            step="index",
            event="initial_frames",
            arguments={
                "interval_s": config.frames.index_interval_s,
                "montage_width": config.montage.width,
                "montage_height": config.montage.height,
                "montage_n": config.frames.index_montage_n,
            },
            result_summary={"frames": len(frames), "montages": len(montages)},
            duration_ms=round((time.monotonic() - started) * 1000),
        )
        return frames, montages

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="vuc-index") as executor:
        audio_future = executor.submit(index_audio)
        frames_future = executor.submit(index_frames)
        segments = audio_future.result()
        frames, montages = frames_future.result()

    index = VideoIndex(
        schema_version=2,
        video=metadata,
        segments=tuple(segments),
        frames=tuple(frames),
        montages=tuple(str(path) for path in montages),
        created_at=datetime.now(UTC).isoformat(),
        indexer={
            "hub": config.indexer.hub,
            "model": config.indexer.model,
            "vad_model": config.indexer.vad_model,
        },
        frame_config=_frame_config(config),
    )
    cache.write_json(cache.index_json_path, index.to_dict())
    cache.index_text_path.write_text(_text_index(segments), encoding="utf-8")
    trace.write(
        step="index",
        event="index_complete",
        arguments={"video_path": str(video_path)},
        result_summary={
            "index_json": str(cache.index_json_path),
            "index_text": str(cache.index_text_path),
        },
        duration_ms=0,
    )
    return index, cache, False, trace_path
