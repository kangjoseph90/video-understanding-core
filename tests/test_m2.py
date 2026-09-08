from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from vuc.advanced_asr import ASRResult, TranscriptSentence
from vuc.agent import AgentBudget, build_index_context, run_agent_loop, system_prompt
from vuc.cache import VideoCache
from vuc.config import load_config
from vuc.llm import ChatResult
from vuc.models import Segment, VideoIndex, VideoMetadata
from vuc.report import apply_verification, normalize_report, report_time_coverage
from vuc.run import route_video
from vuc.tools import ToolError, ToolService, character_error_rate, interval_coverage


class FakeProvider:
    name = "local"
    model_name = "fake-large-v3-turbo"
    max_segment_s = 600

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(
        self,
        audio_path: Path,
        *,
        audio_duration_s: float,
        language_hint: str | None,
    ) -> ASRResult:
        self.calls += 1
        return ASRResult(
            text="verified text",
            sentences=(TranscriptSentence(1, 2, "verified text"),),
            language=language_hint,
            notes=None,
            provider="local",
            model=self.model_name,
            audio_duration_s=audio_duration_s,
            processing_s=0.01,
            cost_usd=None,
        )


def make_service(tmp_path: Path, monkeypatch) -> tuple[ToolService, FakeProvider]:
    project_root = Path(__file__).parents[1]
    config_text = (
        (project_root / "config.yaml")
        .read_text(encoding="utf-8")
        .replace("directory: .vuc-cache", f"directory: {tmp_path / 'cache'}")
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_text, encoding="utf-8")
    config = load_config(config_path)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"video")
    metadata = VideoMetadata(str(video), "a" * 64, 600, video.stat().st_size)
    index = VideoIndex(
        schema_version=1,
        video=metadata,
        segments=(Segment(0, 600, "draft text", "en"),),
        frames=(),
        montages=(),
        created_at="2026-09-08T00:00:00+00:00",
    )
    cache = VideoCache(config.cache.directory, metadata.sha256)
    cache.ensure()
    cache.audio_path.write_bytes(b"wav")
    monkeypatch.setattr(
        "vuc.tools.extract_audio_segment",
        lambda _input, output, **_kwargs: output.write_bytes(b"segment"),
    )
    provider = FakeProvider()
    return (
        ToolService(
            video_path=video,
            index=index,
            cache=cache,
            config=config,
            read_index_enabled=False,
            provider=provider,
        ),
        provider,
    )


def test_local_transcribe_limit_is_three_minutes(tmp_path: Path, monkeypatch) -> None:
    service, _ = make_service(tmp_path, monkeypatch)

    with pytest.raises(ToolError, match="local provider limit is 180s"):
        service.transcribe_segment(0, 180.01)


def test_local_transcribe_limit_cannot_be_configured_above_three_minutes(
    tmp_path: Path, monkeypatch
) -> None:
    service, provider = make_service(tmp_path, monkeypatch)
    provider.max_segment_s = 600
    object.__setattr__(service.config.advanced_asr.local, "max_segment_s", 600)

    assert service.provider_max_segment_s == 180


def test_advanced_asr_notes_absolute_timestamps_and_cache(tmp_path: Path, monkeypatch) -> None:
    service, provider = make_service(tmp_path, monkeypatch)

    first = service.transcribe_segment(10, 20)
    second = service.transcribe_segment(10, 20)

    assert first.data["notes"] is None
    assert first.data["sentences"][0]["start_s"] == 11
    assert first.data["sentences"][0]["end_s"] == 12
    assert first.data["cache_hit"] is False
    assert second.data["cache_hit"] is True
    assert provider.calls == 1


def test_cer_normalizes_spacing_punctuation_and_clips_index_window(
    tmp_path: Path, monkeypatch
) -> None:
    service, _ = make_service(tmp_path, monkeypatch)
    service.index = VideoIndex(
        schema_version=1,
        video=service.index.video,
        segments=(Segment(0, 20, "가나 다라 마바 사아", "ko"),),
        frames=(),
        montages=(),
        created_at=service.index.created_at,
    )

    clipped = service._index_text(10, 20)

    assert "가나" not in clipped
    assert character_error_rate("마바, 사아!", "마바 사아") == 0
    assert character_error_rate("가나다", "가나마") == pytest.approx(1 / 3)


def test_transcribe_trace_records_cost_dimensions(tmp_path: Path, monkeypatch) -> None:
    service, _ = make_service(tmp_path, monkeypatch)

    service.execute("transcribe_segment", {"start_s": 10, "end_s": 20})

    record = json.loads(service.cache.trace_path.read_text(encoding="utf-8").splitlines()[-1])
    summary = record["result_summary"]
    assert summary["provider"] == "local"
    assert summary["audio_duration_s"] == 10
    assert summary["asr_response_s"] >= 0
    assert summary["cost_usd"] is None


def test_verification_uses_union_overlap() -> None:
    class Coverage:
        def citation_coverage(self, start_s: float, end_s: float) -> float:
            return interval_coverage(start_s, end_s, [(0, 4), (3, 8)])

        def evidence_coverage(self, start_s: float, end_s: float, source: str) -> float:
            assert source == "transcribe_segment"
            return interval_coverage(start_s, end_s, [(0, 8)])

    report = {
        "sections": [
            {
                "citations": [
                    {
                        "claim": "covered",
                        "start_s": 0,
                        "end_s": 10,
                        "evidence_span": {
                            "start_s": 0,
                            "end_s": 8,
                            "source": "transcribe_segment",
                        },
                    },
                    {
                        "claim": "not covered",
                        "start_s": 0,
                        "end_s": 11,
                        "evidence_span": {
                            "start_s": 0,
                            "end_s": 8,
                            "source": "transcribe_segment",
                        },
                    },
                ]
            }
        ]
    }

    apply_verification(report, Coverage(), verify_overlap=0.8)

    citations = report["sections"][0]["citations"]
    assert citations[0]["verification_overlap"] == 0.8
    assert citations[0]["evidence_overlap"] == 0.8
    assert citations[0]["verified"] is True
    assert citations[1]["verified"] is False

    normalized = normalize_report(
        report,
        Coverage(),
        verify_overlap=0.8,
        meta={},
    )
    assert "not covered" in normalized["unverified_claims"]


def test_report_coverage_uses_section_union_and_finds_gaps() -> None:
    report = {
        "sections": [
            {"start_s": 0, "end_s": 40},
            {"start_s": 30, "end_s": 60},
            {"start_s": 80, "end_s": 100},
        ]
    }

    ratio, gaps = report_time_coverage(report, 100)

    assert ratio == 0.8
    assert gaps == [(60, 80)]


def test_system_prompt_contains_all_required_reliability_rules() -> None:
    prompt = system_prompt(0.75)

    assert "인덱스는 저품질 초안" in prompt
    assert "transcribe_segment로 검증" in prompt
    assert "프레임에 보이는 텍스트" in prompt
    assert "특정 구간에 대한 시각적 주장" in prompt
    assert "[mm:ss]" in prompt


def test_router_and_asr_excluded_budget() -> None:
    assert route_video(599.999, 600) == "baseline"
    assert route_video(600, 600) == "agentic"
    budget = AgentBudget(12, 200_000, 20, started=time.monotonic() - 100)
    budget.excluded_asr_s = 90

    assert budget.reason() is None


def test_agent_self_check_revises_low_coverage_report(tmp_path: Path, monkeypatch) -> None:
    service, _ = make_service(tmp_path, monkeypatch)
    config = replace(
        service.config,
        agent=replace(service.config.agent, calibration_enabled=False),
    )
    service.config = config

    class CoverageLLM:
        model = "stub"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, **kwargs) -> ChatResult:
            del messages, kwargs
            self.calls += 1
            end_s = 100 if self.calls == 1 else 600
            content = json.dumps(
                {
                    "title": "Coverage",
                    "one_line_summary": "Summary",
                    "sections": [
                        {
                            "title": "Section",
                            "summary": "Summary",
                            "start_s": 0,
                            "end_s": end_s,
                            "citations": [],
                        }
                    ],
                    "key_moments": [],
                    "unverified_claims": [],
                }
            )
            return ChatResult(
                message={"role": "assistant", "content": content},
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                latency_s=0.01,
            )

    client = CoverageLLM()
    raw_report, stats = run_agent_loop(
        query="summary",
        index=service.index,
        cache=service.cache,
        config=config,
        service=service,
        client=client,
    )

    assert client.calls == 2
    assert json.loads(raw_report)["sections"][0]["end_s"] == 600
    assert stats["coverage_self_check"] is True
    assert stats["cumulative_input_tokens"] == 20
    records = [json.loads(line) for line in service.cache.trace_path.read_text().splitlines()]
    assert any(record["event"] == "coverage_self_check" for record in records)


def test_tool_schemas_and_frame_limit_error(tmp_path: Path, monkeypatch) -> None:
    service, _ = make_service(tmp_path, monkeypatch)
    names = [schema["function"]["name"] for schema in service.schemas]

    assert names == ["view_frames", "transcribe_segment"]
    execution = service.execute(
        "view_frames",
        {"start_s": 0, "end_s": 100, "fps": 0.2, "resolution": 512},
    )
    assert "예상 프레임 수 20" in execution.data["error"]

    service.read_index_enabled = True
    assert [schema["function"]["name"] for schema in service.schemas][-1] == "read_index"


def test_large_index_is_uniformly_downsampled() -> None:
    index = VideoIndex(
        schema_version=1,
        video=VideoMetadata("video.mp4", "b" * 64, 100, 1),
        segments=tuple(
            Segment(number, number + 1, "long transcript " * 20, "en") for number in range(100)
        ),
        frames=(),
        montages=(),
        created_at="2026-09-08T00:00:00+00:00",
    )

    context, read_enabled = build_index_context(index, token_budget=100)

    assert read_enabled is True
    assert "[00:00–00:01]" in context
    assert "[01:39–01:40]" in context
