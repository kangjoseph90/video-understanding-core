from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vuc.advanced_asr import AdvancedASRProvider
from vuc.agent import REPORT_SCHEMA, build_prompt_body, run_agent_loop
from vuc.audio import AudioEventTagger
from vuc.cache import VideoCache, new_run_id, sha256_file
from vuc.caption_fusion import overlay_captions
from vuc.captions import load_caption_track
from vuc.config import AppConfig
from vuc.frames import create_montages, extract_sampled_frames, montage_cell_size
from vuc.hints import VideoHints, load_video_hints
from vuc.index_text import AUDIO_INDEX_LABEL, render_audio_index, render_text_index
from vuc.indexer import Transcriber
from vuc.llm import (
    ChatCompletionsClient,
    ChatResult,
    LLMError,
    cached_input_token_count,
    estimate_vlm_cost_usd,
    image_content,
    input_token_count,
    output_token_count,
    reasoning_token_count,
)
from vuc.media import extract_audio, probe_duration
from vuc.models import AudioIndex, TextIndex, VideoIndex, VideoMetadata, VisualIndex
from vuc.ocr import OCREngine
from vuc.pipeline import SCHEMA_VERSION, index_video
from vuc.report import normalize_report, parse_report_json, write_report
from vuc.tools import ToolService
from vuc.trace import TraceWriter
from vuc.transcripts import apply_transcriptions, load_rows
from vuc.vad import VADProvider


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
        schema_version=SCHEMA_VERSION,
        video=metadata,
        audio=AudioIndex(),
        text=TextIndex(),
        visual=VisualIndex(),
        created_at=datetime.now(UTC).isoformat(),
    )
    return index, cache


def _baseline_full_images(
    video_path: Path,
    cache: VideoCache,
    index: VideoIndex,
    config: AppConfig,
    trace: TraceWriter,
) -> tuple[list[Path], float]:
    started = time.monotonic()
    root = cache.root / "baseline_full_frames"
    metadata_path = root / "result.json"
    settings = {
        "interval_s": config.frames.baseline_full_interval_s,
        "montage_width": config.montage.width,
        "montage_height": config.montage.height,
        "n": config.frames.baseline_full_montage_n,
        "jpeg_quality": config.montage.jpeg_quality,
    }
    if metadata_path.exists():
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        paths = [Path(path) for path in data.get("image_paths", [])]
        if data.get("settings") == settings and all(path.exists() for path in paths):
            elapsed = time.monotonic() - started
            trace.write(
                step="baseline_full",
                event="full_frames",
                arguments=settings,
                result_summary={"montages": len(paths), "cache_hit": True},
                duration_ms=round(elapsed * 1000),
            )
            return paths, elapsed
    cell_width, _ = montage_cell_size(
        config.montage.width,
        config.montage.height,
        config.frames.baseline_full_montage_n,
    )
    frames = extract_sampled_frames(
        video_path,
        root / "frames",
        start_s=0,
        end_s=index.video.duration_s,
        fps=1 / config.frames.baseline_full_interval_s,
        resolution=cell_width,
        jpeg_quality=config.montage.jpeg_quality,
    )
    montages = create_montages(
        frames,
        root / "montages",
        n=config.frames.baseline_full_montage_n,
        width=config.montage.width,
        height=config.montage.height,
        jpeg_quality=config.montage.jpeg_quality,
    )
    cache.write_json(
        metadata_path,
        {"settings": settings, "image_paths": [str(path) for path in montages]},
    )
    elapsed = time.monotonic() - started
    trace.write(
        step="baseline_full",
        event="full_frames",
        arguments=settings,
        result_summary={
            "frames": len(frames),
            "montages": len(montages),
            "cache_hit": False,
        },
        duration_ms=round(elapsed * 1000),
    )
    return montages, elapsed


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
            texts.append(str(execution.data.get("transcript") or ""))
        start = end
    return "\n".join(text for text in texts if text), asr_wall_s


def _record_single_pass_llm(trace: TraceWriter, result: ChatResult, mode: str) -> None:
    trace.write(
        step=mode,
        event="llm_response",
        arguments={"tools": False},
        result_summary={
            "has_content": bool(result.message.get("content")),
            "attempts": result.attempts,
            "cached_input_tokens": cached_input_token_count(result.usage),
            "reasoning_tokens": reasoning_token_count(result.usage),
        },
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
    hints: VideoHints | None = None,
    rendered_audio: str = "",
    rendered_text: str = "",
    prompt_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    started = time.monotonic()
    frames_wall_s = 0.0
    text_index = ""
    if mode == "baseline_full":
        audio_index, asr_wall_s = _full_advanced_transcript(service, index)
        images, frames_wall_s = _baseline_full_images(
            video_path, cache, index, config, service.trace
        )
        # baseline_full builds no index; the label says where its lines came from.
        audio_label = "전체 Whisper 전사 (구간 표기는 [시작-끝], 단위는 초)"
    elif mode == "baseline_index_only":
        audio_index = rendered_audio
        text_index = rendered_text
        asr_wall_s = 0.0
        images = [Path(path) for path in index.visual.montages]
        audio_label = AUDIO_INDEX_LABEL
    else:
        raise ValueError(f"unsupported single-pass mode: {mode}")

    body = build_prompt_body(
        query,
        index.video.duration_s,
        audio_index=audio_index,
        audio_label=audio_label,
        text_index=text_index,
        hints=hints,
    )
    if prompt_path is not None:
        prompt_path.write_text(body + "\n", encoding="utf-8")
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"{body}\n\n추가 도구 없이 보고서를 작성하라.\n{REPORT_SCHEMA}",
        }
    ]
    content.extend(image_content(path) for path in images)
    result = client.complete(
        [
            {
                "role": "system",
                "content": (
                    "당신은 영상과 전사를 분석해 근거가 있는 간결한 보고서를 작성한다. "
                    # Same framing as the agentic prompt: the sources are
                    # independent observations, and neither outranks the other.
                    "전사와 화면에 보이는 것은 같은 영상에 대한 서로 독립적인 관찰이다. "
                    "어긋나면 그 사실 자체를 근거로 삼아라. "
                    "모든 시각은 초 단위 숫자다. start_s/end_s도 초 단위 숫자로 적는다."
                ),
            },
            {"role": "user", "content": content},
        ],
        tools=None,
    )
    _record_single_pass_llm(service.trace, result, mode)
    input_tokens = input_token_count(result.usage)
    cached_input_tokens = cached_input_token_count(result.usage)
    output_tokens = output_token_count(result.usage)
    vlm_cost_usd = estimate_vlm_cost_usd(
        config.vision_llm, input_tokens, output_tokens, cached_input_tokens
    )
    return str(result.message.get("content") or ""), {
        "tool_calls": 0,
        "cumulative_input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_token_count(result.usage),
        "llm_calls": 1,
        "llm_retries": result.attempts - 1,
        "vlm_cost_usd": None if vlm_cost_usd is None else round(vlm_cost_usd, 6),
        "agent_wall_clock_s": round(time.monotonic() - started - asr_wall_s, 3),
        "vlm_wall_clock_s": round(result.latency_s, 3),
        "frames_wall_clock_s": round(frames_wall_s, 3),
        "tool_wall_clock_s": 0.0,
        "asr_wall_clock_s": round(asr_wall_s, 3),
        "asr_processing_s": round(service.asr_processing_s, 3),
        "cloud_asr_audio_s": round(service.cloud_asr_audio_s, 3),
        "cloud_asr_cost_usd": round(service.cloud_asr_cost_usd, 6),
        "budget_stop_reason": None,
    }


def _index_cold_s(index: VideoIndex, mode: str) -> float | None:
    """Cold indexing cost. None for baseline_full, which never builds an index."""
    if mode == "baseline_full":
        return None
    value = index.indexer.get("cold_total_s")
    return None if value is None else round(float(value), 3)


def _repair_report_json(
    client: ChatCompletionsClient,
    trace: TraceWriter,
    raw_text: str,
    error: Exception,
) -> tuple[dict[str, Any] | None, ChatResult | None]:
    """Ask the model to fix its own malformed report JSON.

    Large multimodal prompts occasionally make the model drop out of the schema
    mid-array, throwing away a run that already paid for ASR, frames and a
    100k-token request. The repair call replays only the broken text, so it
    costs a fraction of the original and never runs when parsing succeeded.
    """
    try:
        result = client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "너는 잘못된 JSON을 고친다. 내용을 새로 만들지 말고 구조만 "
                        "바로잡아 유효한 JSON 객체 하나만 출력한다."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"다음 JSON이 파싱에 실패했다: {error}\n\n"
                        f"{raw_text}\n\n"
                        f"원래 스키마는 다음과 같다.\n{REPORT_SCHEMA}"
                    ),
                },
            ],
            tools=None,
        )
    except LLMError as exc:
        trace.write(
            step="run",
            event="report_repair",
            arguments={"original_error": str(error)},
            result_summary={"repaired": False, "error": str(exc)},
            duration_ms=0,
        )
        return None, None

    try:
        repaired = parse_report_json(str(result.message.get("content") or ""))
    except (json.JSONDecodeError, ValueError) as exc:
        repaired = None
        detail = str(exc)
    else:
        detail = None
    trace.write(
        step="run",
        event="report_repair",
        arguments={"original_error": str(error)},
        result_summary={"repaired": repaired is not None, "error": detail},
        duration_ms=round(result.latency_s * 1000),
        token_usage=result.usage,
    )
    return repaired, result


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
    index_vad: VADProvider | None = None,
    index_event_tagger: AudioEventTagger | None = None,
    index_ocr_engine: OCREngine | None = None,
    advanced_provider: AdvancedASRProvider | None = None,
    event_tagger: AudioEventTagger | None = None,
    llm_client: ChatCompletionsClient | None = None,
    hints: VideoHints | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    e2e_started = time.monotonic()
    mode = config.run.mode
    run_id = new_run_id(mode)
    # A <video>.meta.json sidecar is applied automatically when present.
    if hints is None:
        hints = load_video_hints(Path(video).expanduser().resolve())
    index_started = time.monotonic()
    if mode == "baseline_full":
        index, cache = _prepare_baseline_full(video, config, force=force_index)
        index_cache_hit = None
    else:
        index, cache, index_cache_hit, _ = index_video(
            video,
            config,
            transcriber=index_transcriber,
            vad=index_vad,
            event_tagger=index_event_tagger,
            ocr_engine=index_ocr_engine,
            force=force_index,
            run_id=run_id,
            hints=hints,
        )
    index_wall_s = time.monotonic() - index_started
    run_dir = cache.run_dir(run_id)
    trace_path = run_dir / "trace.jsonl"
    video_path = Path(index.video.path)
    actual_query = query or config.agent.query
    # The stored index is what the video produced; a caption track corrects it
    # only on the way into the prompt. The cached .txt files stay the plain
    # index, and what was actually sent is written beside this run's trace.
    settings = config.captions.attribution()
    # The agent's own transcriptions go on first and the channel's captions
    # over them, so a caption still wins where it reaches and what the agent
    # already paid for fills the rest. Each query leaves the next one a better
    # index than it found.
    learned = load_rows(cache.transcripts_path)
    base, transcript_stats = apply_transcriptions(
        index.audio.segments, learned, align_ratio_min=settings.align_ratio_min
    )
    overlay = overlay_captions(
        replace(index, audio=AudioIndex(base)),
        load_caption_track(video_path) if config.captions.enabled else None,
        audio_language=hints.language if hints else None,
        config=settings,
    )
    rendered_audio = render_audio_index(overlay.segments)
    rendered_text = render_text_index(overlay.cues)
    prompt_path = run_dir / "prompt.txt"
    if transcript_stats["rows"]:
        TraceWriter(trace_path).write(
            step="render",
            event="accumulated_transcripts",
            arguments={"path": str(cache.transcripts_path)},
            result_summary=transcript_stats,
            duration_ms=0,
        )
    if overlay.applied:
        # The verdict and the counts are the only record of why this run's
        # prompt differs from the stored index. They are not in index.json any
        # more, and they never reach the model.
        TraceWriter(trace_path).write(
            step="render",
            event="caption_fusion",
            arguments={
                "language": overlay.summary["language"],
                "kind": overlay.summary["kind"],
            },
            result_summary=overlay.summary,
            duration_ms=0,
        )
    service = ToolService(
        video_path=video_path,
        index=index,
        cache=cache,
        config=config,
        trace_path=trace_path,
        provider=advanced_provider,
        tagger=event_tagger,
        hints=hints,
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
            hints=hints,
            rendered_audio=rendered_audio,
            rendered_text=rendered_text,
            prompt_path=prompt_path,
        )
    else:
        raw_report, stats = run_agent_loop(
            query=actual_query,
            index=index,
            audio_index=rendered_audio,
            text_index=rendered_text,
            config=config,
            service=service,
            client=client,
            hints=hints,
            prompt_path=prompt_path,
        )
    repair_stats: dict[str, Any] = {"report_repaired": False}
    try:
        report_data = parse_report_json(raw_report)
    except (json.JSONDecodeError, ValueError) as exc:
        repaired, repair_result = _repair_report_json(
            client, service.trace, raw_report, exc
        )
        report_data = repaired if repaired is not None else _fallback_report(raw_report, exc)
        repair_stats["report_repaired"] = repaired is not None
        if repair_result is not None:
            repair_input = input_token_count(repair_result.usage)
            repair_cached = cached_input_token_count(repair_result.usage)
            repair_output = output_token_count(repair_result.usage)
            repair_cost = estimate_vlm_cost_usd(
                config.vision_llm, repair_input, repair_output, repair_cached
            )
            stats["cumulative_input_tokens"] += repair_input
            stats["cached_input_tokens"] += repair_cached
            stats["output_tokens"] += repair_output
            stats["reasoning_tokens"] += reasoning_token_count(repair_result.usage)
            stats["llm_calls"] += 1
            stats["vlm_wall_clock_s"] = round(
                stats["vlm_wall_clock_s"] + repair_result.latency_s, 3
            )
            if repair_cost is not None:
                stats["vlm_cost_usd"] = round(
                    (stats["vlm_cost_usd"] or 0.0) + repair_cost, 6
                )
    meta = {
        "path": str(video_path),
        "duration_s": index.video.duration_s,
        "mode": mode,
        "run_id": run_id,
        "trace": str(trace_path),
        "index_cache_hit": index_cache_hit,
        # Which VLM produced this report. Without it, runs from two models are
        # indistinguishable once they land in the same results file.
        "vlm_model": getattr(client, "model", None),
        "vlm_base_url": getattr(client, "base_url", None),
        "hints": None if hints is None else hints.to_dict(),
        # Which accumulated state this report was written from. The prompt
        # improves as the agent transcribes, so the report only reproduces
        # alongside the rows it saw.
        "accumulated_transcripts": transcript_stats["rows"],
        "accumulated_digest": transcript_stats["digest"],
        "latency_s": round(time.monotonic() - e2e_started, 3),
        **repair_stats,
        # Measured this run: 0 when the index cache was reused.
        "index_wall_clock_s": round(index_wall_s, 3),
        # What indexing costs from scratch. Persisted in index.json, so a cached run
        # still reports it and the three modes stay comparable.
        "index_cold_s": _index_cold_s(index, mode),
        **stats,
    }
    report = normalize_report(report_data, meta=meta)
    markdown_path, json_path = write_report(report, run_dir)
    service.trace.write(
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
