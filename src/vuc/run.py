from __future__ import annotations

import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from vuc.advanced_asr import AdvancedASRProvider
from vuc.agent import (
    REPORT_SCHEMA,
    build_index_context,
    run_agent_loop,
    select_evenly,
    system_prompt,
)
from vuc.cache import VideoCache
from vuc.config import AppConfig
from vuc.frames import create_montages, extract_sampled_frames
from vuc.indexer import Transcriber
from vuc.llm import (
    ChatCompletionsClient,
    ChatResult,
    estimate_vlm_cost_usd,
    image_content,
    input_token_count,
    output_token_count,
)
from vuc.models import VideoIndex
from vuc.pipeline import index_video
from vuc.report import normalize_report, parse_report_json, write_report
from vuc.tools import ToolService
from vuc.trace import TraceWriter


def _baseline_images(
    video_path: Path,
    cache: VideoCache,
    index: VideoIndex,
    config: AppConfig,
) -> list[Path]:
    root = cache.root / "baseline_frames"
    metadata_path = root / "result.json"
    if metadata_path.exists():
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        return [Path(path) for path in data["image_paths"]]
    frames = extract_sampled_frames(
        video_path,
        root / "frames",
        start_s=0,
        end_s=index.video.duration_s,
        fps=1 / config.frames.baseline_interval_s,
        resolution=config.frames.baseline_resolution,
        jpeg_quality=config.frames.jpeg_quality,
    )
    montage_config = replace(
        config.frames,
        initial_interval_s=config.frames.baseline_interval_s,
        initial_resolution=config.frames.baseline_resolution,
    )
    montages = create_montages(frames, root / "montages", montage_config)
    images = montages or [Path(frame.path) for frame in frames]
    cache.write_json(metadata_path, {"image_paths": [str(path) for path in images]})
    return images


def _baseline_transcript(service: ToolService, index: VideoIndex) -> tuple[str, float]:
    maximum = service.provider_max_segment_s
    start = 0.0
    texts = []
    asr_wall_s = 0.0
    while start < index.video.duration_s:
        end = min(start + maximum, index.video.duration_s)
        execution = service.execute("transcribe_segment", {"start_s": start, "end_s": end})
        asr_wall_s += execution.asr_elapsed_s
        if "error" not in execution.data:
            texts.append(str(execution.data.get("text") or ""))
        start = end
    return "\n".join(texts), asr_wall_s


def _record_baseline_llm(trace: TraceWriter, result: ChatResult) -> None:
    trace.write(
        step="baseline",
        event="llm_response",
        arguments={"tools": False},
        result_summary={"has_content": bool(result.message.get("content"))},
        duration_ms=round(result.latency_s * 1000),
        token_usage=result.usage,
    )


def run_baseline(
    *,
    query: str,
    video_path: Path,
    index: VideoIndex,
    cache: VideoCache,
    config: AppConfig,
    service: ToolService,
    client: ChatCompletionsClient,
) -> tuple[str, dict[str, Any]]:
    started = time.monotonic()
    transcript, asr_wall_s = _baseline_transcript(service, index)
    if not transcript:
        transcript, _ = build_index_context(index, config.indexer.initial_prompt_tokens)
    images = select_evenly(
        _baseline_images(video_path, cache, index, config),
        config.vision_llm.max_images_per_request,
    )
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"사용자 쿼리:\n{query}\n\n전체 고급 ASR 전사:\n{transcript}\n\n"
                f"추가 도구 없이 보고서를 작성하라.\n{REPORT_SCHEMA}"
            ),
        }
    ]
    content.extend(image_content(path) for path in images)
    result = client.complete(
        [
            {"role": "system", "content": system_prompt(None)},
            {"role": "user", "content": content},
        ],
        tools=None,
    )
    _record_baseline_llm(TraceWriter(cache.trace_path), result)
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
        "index_reliability": None,
        "calibration": [],
        "budget_stop_reason": None,
    }


def _fallback_report(raw_text: str, error: Exception) -> dict[str, Any]:
    return {
        "title": "Video report",
        "one_line_summary": raw_text.strip(),
        "sections": [],
        "key_moments": [],
        "unverified_claims": [f"The model response was not valid report JSON: {error}"],
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
    index, cache, index_cache_hit = index_video(
        video,
        config,
        transcriber=index_transcriber,
        force=force_index,
    )
    video_path = Path(index.video.path)
    actual_query = query or config.agent.query
    _, read_index_enabled = build_index_context(index, config.indexer.initial_prompt_tokens)
    service = ToolService(
        video_path=video_path,
        index=index,
        cache=cache,
        config=config,
        read_index_enabled=read_index_enabled,
        provider=advanced_provider,
    )
    client = llm_client or ChatCompletionsClient(config.vision_llm)
    mode = config.run.mode
    if mode == "baseline":
        raw_report, stats = run_baseline(
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
    report = normalize_report(
        report_data,
        service,
        verify_overlap=config.agent.verify_overlap,
        meta=meta,
    )
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
            "verified_intervals": service.verified_intervals,
            "frame_intervals": service.frame_intervals,
            "transcript_intervals": service.transcript_intervals,
            "budget_stop_reason": stats["budget_stop_reason"],
        },
        duration_ms=round((time.monotonic() - e2e_started) * 1000),
        token_usage={
            "input_tokens": stats["cumulative_input_tokens"],
            "output_tokens": stats["output_tokens"],
        },
    )
    return report, markdown_path, json_path
