from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vuc.cache import VideoCache
from vuc.config import AppConfig
from vuc.frames import format_timestamp
from vuc.llm import ChatCompletionsClient, ChatResult, image_content
from vuc.models import Segment, VideoIndex
from vuc.tools import ToolExecution, ToolService
from vuc.trace import TraceWriter

REPORT_SCHEMA = """
Return only one JSON object with this shape:
{
  "title": "...",
  "one_line_summary": "...",
  "sections": [
    {
      "title": "...", "summary": "...", "start_s": 0, "end_s": 30,
      "citations": [{"claim": "...", "start_s": 0, "end_s": 10}]
    }
  ],
  "key_moments": [{"title": "...", "summary": "...", "timestamp_s": 0}],
  "unverified_claims": ["..."]
}
Do not emit a verified field; it is computed from the trace after your response.
Keep the complete JSON concise enough to fit within the output limit.
""".strip()


@dataclass
class AgentBudget:
    max_tool_calls: int
    max_input_tokens: int
    wall_clock_s: float
    started: float
    excluded_asr_s: float = 0.0
    tool_calls: int = 0
    input_tokens: int = 0

    @classmethod
    def start(cls, config: AppConfig) -> AgentBudget:
        return cls(
            max_tool_calls=config.agent.max_tool_calls,
            max_input_tokens=config.agent.max_input_tokens,
            wall_clock_s=config.agent.wall_clock_s,
            started=time.monotonic(),
        )

    @property
    def effective_elapsed_s(self) -> float:
        return max(0.0, time.monotonic() - self.started - self.excluded_asr_s)

    def reason(self) -> str | None:
        if self.tool_calls >= self.max_tool_calls:
            return "max_tool_calls"
        if self.input_tokens >= self.max_input_tokens:
            return "max_input_tokens"
        if self.effective_elapsed_s >= self.wall_clock_s:
            return "wall_clock"
        return None


def _index_line(segment: Segment) -> str:
    tags = [segment.language, segment.emotion or "", *segment.events]
    tag_text = " ".join(f"<{tag}>" for tag in tags if tag and tag != "unknown")
    return (
        f"[{format_timestamp(segment.start)}–{format_timestamp(segment.end)}] "
        f"{tag_text} {segment.text}"
    ).strip()


def build_index_context(index: VideoIndex, token_budget: int) -> tuple[str, bool]:
    lines = [_index_line(segment) for segment in index.segments]
    full_text = "\n".join(lines)
    estimated_tokens = max(1, len(full_text) // 4)
    if estimated_tokens <= token_budget:
        return full_text, False
    keep = max(2, int(len(lines) * token_budget / estimated_tokens))
    positions = {round(index * (len(lines) - 1) / (keep - 1)) for index in range(keep)}
    sampled = [line for index, line in enumerate(lines) if index in positions]
    return "\n".join(sampled), True


def select_evenly(paths: list[Path], maximum: int) -> list[Path]:
    if len(paths) <= maximum:
        return paths
    if maximum <= 1:
        return [paths[0]]
    positions = {round(index * (len(paths) - 1) / (maximum - 1)) for index in range(maximum)}
    return [path for index, path in enumerate(paths) if index in positions]


def system_prompt(index_reliability: float | None) -> str:
    reliability = "측정 불가" if index_reliability is None else f"{index_reliability:.3f}"
    return f"""당신은 긴 영상을 근거 중심으로 분석하는 에이전트다.

필수 신뢰 규칙:
- 제공된 SenseVoice 인덱스는 저품질 초안이며 언어에 따라 오인식 정도가 크게 다르다.
- 보고서에 인용하거나 핵심 주장의 근거로 삼는 구간은 transcribe_segment로 검증해야 한다.
- 프레임에 보이는 텍스트(슬라이드, 자막, 화면 텍스트)는 ASR보다 우선 신뢰한다.
- 모든 인용은 [mm:ss] 또는 [mm:ss–mm:ss] 형식을 사용한다.

이 영상에서 캘리브레이션한 인덱스 신뢰도: {reliability}
도구를 사용해 중요한 주장과 장면을 확인한 뒤 보고서를 완성하라.
{REPORT_SCHEMA}"""


def _initial_user_message(query: str, index_text: str, images: list[Path]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": f"사용자 쿼리:\n{query}\n\nSenseVoice 초안 인덱스:\n{index_text}",
        }
    ]
    content.extend(image_content(path) for path in images)
    return {"role": "user", "content": content}


def _record_llm(trace: TraceWriter, result: ChatResult, *, finalizing: bool) -> None:
    trace.write(
        step="agent",
        event="llm_response",
        arguments={"finalizing": finalizing},
        result_summary={
            "tool_calls": len(result.message.get("tool_calls") or []),
            "has_content": bool(result.message.get("content")),
        },
        duration_ms=round(result.latency_s * 1000),
        token_usage=result.usage,
    )


def _calibration_ranges(index: VideoIndex, config: AppConfig) -> list[tuple[float, float]]:
    segment_s = config.agent.calibration_segment_s
    count = max(2, min(3, config.agent.calibration_segments))
    generator = random.Random(index.video.sha256)
    bucket_s = index.video.duration_s / count
    starts = []
    for bucket in range(count):
        low = bucket * bucket_s
        high = max(
            low, min((bucket + 1) * bucket_s - segment_s, index.video.duration_s - segment_s)
        )
        starts.append(generator.uniform(low, high))
    return [(start, min(start + segment_s, index.video.duration_s)) for start in starts]


def calibrate(
    service: ToolService,
    index: VideoIndex,
    config: AppConfig,
) -> tuple[float | None, list[dict[str, Any]], float]:
    if not config.agent.calibration_enabled:
        return None, [], 0.0
    results = []
    excluded_s = 0.0
    for start_s, end_s in _calibration_ranges(index, config):
        execution = service.execute("transcribe_segment", {"start_s": start_s, "end_s": end_s})
        excluded_s += execution.asr_elapsed_s
        results.append(execution.data)
    scores = [
        float(result["index_similarity"])
        for result in results
        if "index_similarity" in result and "error" not in result
    ]
    reliability = sum(scores) / len(scores) if scores else None
    return reliability, results, excluded_s


def run_agent_loop(
    *,
    query: str,
    index: VideoIndex,
    cache: VideoCache,
    config: AppConfig,
    service: ToolService,
    client: ChatCompletionsClient,
) -> tuple[str, dict[str, Any]]:
    index_text, _ = build_index_context(index, config.indexer.initial_prompt_tokens)
    initial_images = select_evenly(
        [Path(path) for path in index.montages],
        config.vision_llm.max_images_per_request,
    )
    budget = AgentBudget.start(config)
    reliability, calibration_results, calibration_asr_s = calibrate(service, index, config)
    budget.excluded_asr_s += calibration_asr_s
    trace = TraceWriter(cache.trace_path)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt(reliability)},
        _initial_user_message(
            query,
            f"영상 길이: {index.video.duration_s:.3f}초\n{index_text}",
            initial_images,
        ),
    ]
    final_text = ""
    stop_reason: str | None = None

    while True:
        reason = budget.reason()
        finalizing = reason is not None
        if finalizing:
            stop_reason = reason
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"예산 종료 사유는 {reason}이다. 추가 도구 없이 현재 근거만으로 "
                        "지금 최종 JSON 보고서를 작성하라."
                    ),
                }
            )
        result = client.complete(
            messages,
            tools=None if finalizing else service.schemas,
            tool_choice=None if finalizing else "auto",
        )
        _record_llm(trace, result, finalizing=finalizing)
        budget.input_tokens += int(result.usage.get("prompt_tokens", 0))
        tool_calls = result.message.get("tool_calls") or []
        if finalizing or not tool_calls:
            final_text = str(result.message.get("content") or "")
            break

        assistant_message = {
            "role": "assistant",
            "content": result.message.get("content"),
            "tool_calls": tool_calls,
        }
        messages.append(assistant_message)
        returned_images: list[Path] = []
        for call in tool_calls:
            exhausted = budget.reason()
            if exhausted is not None:
                execution = ToolExecution(data={"error": f"{exhausted} budget exhausted"})
            else:
                function = call.get("function") or {}
                name = str(function.get("name") or "")
                raw_arguments = function.get("arguments") or "{}"
                try:
                    arguments = (
                        raw_arguments
                        if isinstance(raw_arguments, dict)
                        else json.loads(raw_arguments)
                    )
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must be an object")
                except (json.JSONDecodeError, TypeError, ValueError) as exc:
                    arguments = {}
                    execution = ToolExecution(data={"error": f"invalid tool arguments: {exc}"})
                else:
                    execution = service.execute(name, arguments)
                budget.tool_calls += 1
                budget.excluded_asr_s += execution.asr_elapsed_s
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", "unknown"),
                    "content": json.dumps(execution.data, ensure_ascii=False),
                }
            )
            returned_images.extend(execution.image_paths)
        if returned_images:
            content = [
                {
                    "type": "text",
                    "text": "방금 view_frames 도구가 반환한 실제 프레임 이미지다.",
                }
            ]
            content.extend(image_content(path) for path in returned_images)
            messages.append({"role": "user", "content": content})

    stats = {
        "tool_calls": budget.tool_calls,
        "input_tokens": budget.input_tokens,
        "agent_wall_clock_s": round(budget.effective_elapsed_s, 3),
        "asr_wall_clock_s": round(budget.excluded_asr_s, 3),
        "asr_processing_s": round(service.asr_processing_s, 3),
        "cloud_asr_audio_s": round(service.cloud_asr_audio_s, 3),
        "cloud_asr_cost_usd": round(service.cloud_asr_cost_usd, 6),
        "index_reliability": reliability,
        "calibration": calibration_results,
        "budget_stop_reason": stop_reason,
    }
    return final_text, stats
