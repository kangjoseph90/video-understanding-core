from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vuc.config import AppConfig
from vuc.hints import VideoHints
from vuc.index_text import AUDIO_INDEX_LABEL, TEXT_INDEX_LABEL
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
from vuc.models import VideoIndex
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


def build_prompt_body(
    query: str,
    duration_s: float,
    *,
    audio_index: str,
    audio_label: str = AUDIO_INDEX_LABEL,
    text_index: str = "",
    hints: VideoHints | None = None,
) -> str:
    """The prompt's text half, in four named blocks.

    Query, then metadata, then the audio index, then the text index -- each
    labelled, each free-standing. The montages are the fourth component and are
    attached as images beside this. An index that produced nothing is left out
    rather than shown as an empty heading.
    """
    blocks = [f"사용자 쿼리:\n{query}"]
    metadata = hints.prompt_block() if hints else ""
    if metadata:
        blocks.append(metadata)
    blocks.append(f"영상 길이: {int(duration_s)}")
    if audio_index:
        blocks.append(f"{audio_label}:\n{audio_index}")
    if text_index:
        blocks.append(f"{TEXT_INDEX_LABEL}:\n{text_index}")
    return "\n\n".join(blocks)


def system_prompt() -> str:
    return f"""당신은 긴 영상을 근거 중심으로 분석하는 에이전트다.

입력은 메타데이터, 음성 인덱스, 화면 텍스트 인덱스, 몽타주 이미지다.

- 메타데이터는 채널이 제공한 채널명, 제목, 챕터다.
- 음성 인덱스는 영상 전체에서 들린 내용을 시간순으로 적는다. 말은 전사와 언어로,
  말이 아닌 소리는 <태그>로 표시한다. 아무것도 들리지 않은 구간은 줄이 없다.
- 화면 텍스트 인덱스는 화면에서 읽힌 글자다. 형식은 [시작-끝, position, size]이며,
  position은 3×3 위치, size는 줄 높이(small<4%, medium<8%, large≥8%)다. 세미콜론은
  같은 문구가 나타난 서로 떨어진 구간을 구분한다. 위치와 크기는 부가 정보다.
- 몽타주는 선택된 시점의 화면과 시각적 맥락을 보여 주며 각 프레임에 시간이 적혀 있다.

근거 사용 원칙:
- 음성 인덱스와 화면 텍스트 인덱스는 서로 독립적인 관찰이며 별개의 목록이다. 서로
  정정하거나 대체하지 않으므로 함께 검토하고, 충돌하거나 불확실하면 그대로 밝혀라.
- 인덱스에는 인식 오류가 있고 몽타주는 일부 시점만 담는다. 어느 입력에 없다는 이유만으로
  영상에 없었다고 단정하거나, 위치·크기만으로 내용의 종류와 중요도를 판단하지 마라.
- 인덱스만으로 충분하면 도구는 쓰지 않아도 된다. 답에 필요한 음성이 불확실하면
  transcribe_segment를, 화면이 불확실하면 view_frames를 사용한다.
- 모든 시각과 start_s/end_s는 초 단위 숫자로 적는다.

주어진 근거로 사용자 쿼리에 답하는 간결한 보고서를 작성하라.
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
    audio_index: str,
    text_index: str,
    config: AppConfig,
    service: ToolService,
    client: ChatCompletionsClient,
    hints: VideoHints | None = None,
    prompt_path: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    """The index arrives already rendered.

    Rendering is where a caption track and, later, the agent's own accumulated
    transcriptions are applied, so it happens once in the caller rather than
    here -- the tool service still works from the stored index, whose VAD split
    is what an ASR request is routed by.
    """
    initial_images = [Path(path) for path in index.visual.montages]
    budget = AgentBudget.start(config)
    trace = service.trace
    body = build_prompt_body(
        query,
        index.video.duration_s,
        audio_index=audio_index,
        text_index=text_index,
        hints=hints,
    )
    if prompt_path is not None:
        prompt_path.write_text(body + "\n", encoding="utf-8")
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt()},
        _initial_user_message(body, initial_images),
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
