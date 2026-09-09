from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vuc.advanced_asr import AdvancedASRProvider
from vuc.agent import REPORT_SCHEMA, build_index_context, run_agent_loop
from vuc.cache import VideoCache, sha256_file
from vuc.config import AppConfig
from vuc.frames import create_montages, extract_sampled_frames, montage_cell_size
from vuc.indexer import Transcriber
from vuc.llm import (
    ChatCompletionsClient,
    ChatResult,
    estimate_vlm_cost_usd,
    image_content,
    input_token_count,
    output_token_count,
)
from vuc.media import extract_audio, probe_duration
from vuc.models import VideoIndex, VideoMetadata
from vuc.pipeline import index_video
from vuc.report import normalize_report, parse_report_json, write_report
from vuc.tools import ToolService
from vuc.trace import TraceWriter


def _prepare_baseline_full(
    video: str | Path,
    config: AppConfig,
    *,
    force: bool,
) -> tuple[VideoIndex, VideoCache]:
    video_path = Path(video).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"video file not found: {video_path}")
    if video_path.suffix.lower().lstrip(".") not in config.video.allowed_extensions:
        allowed = ", ".join(config.video.allowed_extensions)
        raise ValueError(f"unsupported video extension; expected one of: {allowed}")
    video_hash = sha256_file(video_path)
    cache = VideoCache(config.cache.directory, video_hash)
    cache.ensure()
    if force or not cache.audio_path.exists():
        extract_audio(video_path, cache.audio_path)
    metadata = VideoMetadata(
        path=str(video_path),
        sha256=video_hash,
        duration_s=probe_duration(video_path),
        size_bytes=video_path.stat().st_size,
    )
    index = VideoIndex(
        schema_version=2,
        video=metadata,
        segments=(),
        frames=(),
        montages=(),
        created_at=datetime.now(UTC).isoformat(),
        indexer={},
        frame_config={},
    )
    return index, cache


def _baseline_full_images(
    video_path: Path,
    cache: VideoCache,
    index: VideoIndex,
    config: AppConfig,
) -> list[Path]:
    root = cache.root / "baseline_full_frames"
    metadata_path = root / "result.json"
    settings = {
        "interval_s": config.frames.baseline_full_interval_s,
        "montage_width": config.frames.montage_width,
        "montage_height": config.frames.montage_height,
        "n": config.frames.montage_n,
        "jpeg_quality": config.frames.jpeg_quality,
    }
    if metadata_path.exists():
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        paths = [Path(path) for path in data.get("image_paths", [])]
        if data.get("settings") == settings and all(path.exists() for path in paths):
            return paths
    cell_width, _ = montage_cell_size(
        config.frames.montage_width,
        config.frames.montage_height,
        config.frames.montage_n,
    )
    frames = extract_sampled_frames(
        video_path,
        root / "frames",
        start_s=0,
        end_s=index.video.duration_s,
        fps=1 / config.frames.baseline_full_interval_s,
        resolution=cell_width,
        jpeg_quality=config.frames.jpeg_quality,
    )
    montages = create_montages(
        frames,
        root / "montages",
        n=config.frames.montage_n,
        width=config.frames.montage_width,
        height=config.frames.montage_height,
        jpeg_quality=config.frames.jpeg_quality,
    )
    cache.write_json(
        metadata_path,
        {"settings": settings, "image_paths": [str(path) for path in montages]},
    )
    return montages


def _full_advanced_transcript(service: ToolService, index: VideoIndex) -> tuple[str, float]:
    start = 0.0
    texts = []
    asr_wall_s = 0.0
    while start < index.video.duration_s:
        end = min(start + service.provider_max_segment_s, index.video.duration_s)
        execution = service.transcribe_baseline_chunk(start, end)
        asr_wall_s += execution.asr_elapsed_s
        service.trace.write(
            step="baseline_full",
            event="advanced_asr_chunk",
            arguments={"start_s": start, "end_s": end},
            result_summary={
                "provider": execution.data.get("provider"),
                "cache_hit": execution.data.get("cache_hit"),
                "audio_duration_s": execution.data.get("audio_duration_s"),
                "cost_usd": execution.data.get("cost_usd"),
            },
            duration_ms=round(execution.asr_elapsed_s * 1000),
        )
        if "error" not in execution.data:
            texts.append(str(execution.data.get("text") or ""))
        start = end
    return "\n".join(texts), asr_wall_s


def _record_single_pass_llm(trace: TraceWriter, result: ChatResult, mode: str) -> None:
    trace.write(
        step=mode,
        event="llm_response",
        arguments={"tools": False},
        result_summary={"has_content": bool(result.message.get("content"))},
        duration_ms=round(result.latency_s * 1000),
        token_usage=result.usage,
    )


def run_single_pass(
    *,
    mode: str,
    query: str,
    video_path: Path,
    index: VideoIndex,
    cache: VideoCache,
    config: AppConfig,
    service: ToolService,
    client: ChatCompletionsClient,
) -> tuple[str, dict[str, Any]]:
    started = time.monotonic()
    if mode == "baseline_full":
        transcript, asr_wall_s = _full_advanced_transcript(service, index)
        images = _baseline_full_images(video_path, cache, index, config)
        source_label = "전체 Whisper 전사"
    elif mode == "baseline_index_only":
        transcript = build_index_context(index)
        asr_wall_s = 0.0
        images = [Path(path) for path in index.montages]
        source_label = "전체 SenseVoice 인덱스"
    else:
        raise ValueError(f"unsupported single-pass mode: {mode}")

    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"사용자 쿼리:\n{query}\n\n{source_label}:\n{transcript}\n\n"
                f"추가 도구 없이 보고서를 작성하라.\n{REPORT_SCHEMA}"
            ),
        }
    ]
    content.extend(image_content(path) for path in images)
    result = client.complete(
        [
            {
                "role": "system",
                "content": (
                    "당신은 영상과 전사를 분석해 근거가 있는 간결한 보고서를 작성한다. "
                    "프레임에 보이는 텍스트는 ASR보다 우선한다. citation 시간은 "
                    "start_s/end_s에 초 단위로 기록한다."
                ),
            },
            {"role": "user", "content": content},
        ],
        tools=None,
    )
    _record_single_pass_llm(TraceWriter(cache.trace_path), result, mode)
    input_tokens = input_token_count(result.usage)
    output_tokens = output_token_count(result.usage)
    vlm_cost_usd = estimate_vlm_cost_usd(config.vision_llm, input_tokens, output_tokens)
    return str(result.message.get("content") or ""), {
        "tool_calls": 0,
        "cumulative_input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "vlm_cost_usd": None if vlm_cost_usd is None else round(vlm_cost_usd, 6),
        "agent_wall_clock_s": round(time.monotonic() - started - asr_wall_s, 3),
        "asr_wall_clock_s": round(asr_wall_s, 3),
        "asr_processing_s": round(service.asr_processing_s, 3),
        "cloud_asr_audio_s": round(service.cloud_asr_audio_s, 3),
        "cloud_asr_cost_usd": round(service.cloud_asr_cost_usd, 6),
        "budget_stop_reason": None,
    }


def _fallback_report(raw_text: str, error: Exception) -> dict[str, Any]:
    return {
        "title": "Video report",
        "one_line_summary": raw_text.strip() or f"Invalid report JSON: {error}",
        "sections": [],
        "key_moments": [],
    }


def run_video(
    video: str | Path,
    config: AppConfig,
    *,
    query: str | None = None,
    force_index: bool = False,
    index_transcriber: Transcriber | None = None,
    advanced_provider: AdvancedASRProvider | None = None,
    llm_client: ChatCompletionsClient | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    e2e_started = time.monotonic()
    mode = config.run.mode
    if mode == "baseline_full":
        index, cache = _prepare_baseline_full(video, config, force=force_index)
        index_cache_hit = None
    else:
        index, cache, index_cache_hit = index_video(
            video,
            config,
            transcriber=index_transcriber,
            force=force_index,
        )
    video_path = Path(index.video.path)
    actual_query = query or config.agent.query
    service = ToolService(
        video_path=video_path,
        index=index,
        cache=cache,
        config=config,
        provider=advanced_provider,
    )
    client = llm_client or ChatCompletionsClient(config.vision_llm)
    if mode in {"baseline_full", "baseline_index_only"}:
        raw_report, stats = run_single_pass(
            mode=mode,
            query=actual_query,
            video_path=video_path,
            index=index,
            cache=cache,
            config=config,
            service=service,
            client=client,
        )
    else:
        raw_report, stats = run_agent_loop(
            query=actual_query,
            index=index,
            cache=cache,
            config=config,
            service=service,
            client=client,
        )
    try:
        report_data = parse_report_json(raw_report)
    except (json.JSONDecodeError, ValueError) as exc:
        report_data = _fallback_report(raw_report, exc)
    meta = {
        "path": str(video_path),
        "duration_s": index.video.duration_s,
        "mode": mode,
        "index_cache_hit": index_cache_hit,
        "latency_s": round(time.monotonic() - e2e_started, 3),
        **stats,
    }
    report = normalize_report(report_data, meta=meta)
    report_key = hashlib.sha256(f"{mode}:{actual_query}".encode()).hexdigest()[:16]
    report_dir = cache.reports_dir / report_key
    markdown_path, json_path = write_report(report, report_dir)
    TraceWriter(cache.trace_path).write(
        step="run",
        event="report_complete",
        arguments={"mode": mode, "query": actual_query},
        result_summary={
            "markdown": str(markdown_path),
            "json": str(json_path),
            "budget_stop_reason": stats["budget_stop_reason"],
        },
        duration_ms=round((time.monotonic() - e2e_started) * 1000),
        token_usage={
            "input_tokens": stats["cumulative_input_tokens"],
            "output_tokens": stats["output_tokens"],
        },
    )
    return report, markdown_path, json_path
