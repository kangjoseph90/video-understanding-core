from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vuc.config import AppConfig
from vuc.frames import format_span
from vuc.hints import VideoHints
from vuc.llm import (
    ChatCompletionsClient,
    ChatResult,
    cached_input_token_count,
    estimate_vlm_cost_usd,
    image_content,
    input_token_count,
    output_token_count,
    reasoning_token_count,
)
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
  "key_moments": [{"title": "...", "summary": "...", "timestamp_s": 0}]
}
Keep the complete JSON concise enough to fit within the output limit.
""".strip()


@dataclass
class AgentBudget:
    max_tool_calls: int
    wall_clock_s: float
    started: float
    excluded_asr_s: float = 0.0
    tool_calls: int = 0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    llm_calls: int = 0
    llm_retries: int = 0
    vlm_wall_s: float = 0.0
    tool_wall_s: float = 0.0

    @classmethod
    def start(cls, config: AppConfig) -> AgentBudget:
        return cls(
            max_tool_calls=config.agent.max_tool_calls,
            wall_clock_s=config.agent.wall_clock_s,
            started=time.monotonic(),
        )

    @property
    def effective_elapsed_s(self) -> float:
        return max(0.0, time.monotonic() - self.started - self.excluded_asr_s)

    def reason(self) -> str | None:
        if self.tool_calls >= self.max_tool_calls:
            return "max_tool_calls"
        if self.effective_elapsed_s >= self.wall_clock_s:
            return "wall_clock"
        return None


def _index_line(segment: Segment) -> str:
    tags = [segment.language, segment.emotion or "", *segment.events]
    tag_text = " ".join(f"<{tag}>" for tag in tags if tag and tag != "unknown")
    return (
        f"[{format_span(segment.start, segment.end)}] {tag_text} {segment.text}"
    ).strip()


def build_index_context(index: VideoIndex) -> str:
    return "\n".join(_index_line(segment) for segment in index.segments)


def build_prompt_body(
    query: str,
    duration_s: float,
    source_label: str,
    transcript: str,
    hints: VideoHints | None = None,
) -> str:
    """Shared preamble so every mode presents its transcript identically."""
    blocks = [f"사용자 쿼리:\n{query}"]
    metadata = hints.prompt_block() if hints else ""
    if metadata:
        blocks.append(metadata)
    blocks.append(
        f"영상 길이: {int(duration_s)}\n"
        f"{source_label} (구간 표기는 [시작-끝], 단위는 초):\n{transcript}"
    )
    return "\n\n".join(blocks)


def system_prompt() -> str:
    return f"""당신은 긴 영상을 근거 중심으로 분석하는 에이전트다.

분석 규칙:
- 제공된 SenseVoice 인덱스에는 인식 오류가 있을 수 있다.
- SenseVoice 인덱스만으로 충분하면 도구 조회는 필수가 아니다.
- 더 정확한 음성 전사가 필요할 때 transcribe_segment를 사용한다.
- 더 자세한 시각 정보가 필요할 때 해당 시간 구간을 view_frames로 조회한다.
- 프레임에 보이는 텍스트(슬라이드, 자막, 화면 텍스트)는 ASR보다 우선 신뢰한다.
- 모든 시각은 초 단위 숫자다. start_s/end_s도 초 단위 숫자로 적는다.

초기 SenseVoice 인덱스와 타임스탬프 몽타주를 바탕으로 보고서를 작성하라.
필요한 경우에만 도구를 사용하라.
{REPORT_SCHEMA}"""


def _initial_user_message(body: str, images: list[Path]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": body}]
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
            "attempts": result.attempts,
            "cached_input_tokens": cached_input_token_count(result.usage),
            "reasoning_tokens": reasoning_token_count(result.usage),
        },
        duration_ms=round(result.latency_s * 1000),
        token_usage=result.usage,
    )


def run_agent_loop(
    *,
    query: str,
    index: VideoIndex,
    config: AppConfig,
    service: ToolService,
    client: ChatCompletionsClient,
    hints: VideoHints | None = None,
) -> tuple[str, dict[str, Any]]:
    index_text = build_index_context(index)
    initial_images = [Path(path) for path in index.montages]
    budget = AgentBudget.start(config)
    trace = service.trace
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt()},
        _initial_user_message(
            build_prompt_body(
                query,
                index.video.duration_s,
                "전체 SenseVoice 인덱스",
                index_text,
                hints,
            ),
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
                        f"예산 종료 사유는 {reason}이다. 추가 도구 없이 현재 자료만으로 "
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
        budget.input_tokens += input_token_count(result.usage)
        budget.cached_input_tokens += cached_input_token_count(result.usage)
        budget.output_tokens += output_token_count(result.usage)
        budget.reasoning_tokens += reasoning_token_count(result.usage)
        budget.llm_calls += 1
        budget.llm_retries += result.attempts - 1
        budget.vlm_wall_s += result.latency_s
        tool_calls = result.message.get("tool_calls") or []
        if finalizing or not tool_calls:
            final_text = str(result.message.get("content") or "")
            break

        messages.append(
            {
                "role": "assistant",
                "content": result.message.get("content"),
                "tool_calls": tool_calls,
            }
        )
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
                    execution = ToolExecution(data={"error": f"invalid tool arguments: {exc}"})
                else:
                    tool_started = time.monotonic()
                    execution = service.execute(name, arguments)
                    budget.tool_wall_s += time.monotonic() - tool_started
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
                    "text": "방금 view_frames 도구가 반환한 n×n 몽타주 이미지다.",
                }
            ]
            content.extend(image_content(path) for path in returned_images)
            messages.append({"role": "user", "content": content})

    vlm_cost_usd = estimate_vlm_cost_usd(
        config.vision_llm,
        budget.input_tokens,
        budget.output_tokens,
        budget.cached_input_tokens,
    )
    return final_text, {
        "tool_calls": budget.tool_calls,
        "cumulative_input_tokens": budget.input_tokens,
        "cached_input_tokens": budget.cached_input_tokens,
        "output_tokens": budget.output_tokens,
        "reasoning_tokens": budget.reasoning_tokens,
        "llm_calls": budget.llm_calls,
        "llm_retries": budget.llm_retries,
        "vlm_cost_usd": None if vlm_cost_usd is None else round(vlm_cost_usd, 6),
        "agent_wall_clock_s": round(budget.effective_elapsed_s, 3),
        "vlm_wall_clock_s": round(budget.vlm_wall_s, 3),
        "frames_wall_clock_s": 0.0,
        "tool_wall_clock_s": round(budget.tool_wall_s, 3),
        "asr_wall_clock_s": round(budget.excluded_asr_s, 3),
        "asr_processing_s": round(service.asr_processing_s, 3),
        "cloud_asr_audio_s": round(service.cloud_asr_audio_s, 3),
        "cloud_asr_cost_usd": round(service.cloud_asr_cost_usd, 6),
        "budget_stop_reason": stop_reason,
    }
