from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from vuc.advanced_asr import ASRResult, TranscriptSentence
from vuc.agent import AgentBudget, build_index_context, run_agent_loop, system_prompt
from vuc.cache import VideoCache
from vuc.config import load_config
from vuc.llm import ChatResult
from vuc.models import FrameArtifact, Segment, VideoIndex, VideoMetadata
from vuc.report import normalize_report
from vuc.run import _full_advanced_transcript
from vuc.tools import ToolError, ToolService


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
            text="accurate text",
            sentences=(TranscriptSentence(1, 2, "accurate text"),),
            language=language_hint,
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
        schema_version=2,
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
            trace_path=tmp_path / "trace.jsonl",
            provider=provider,
        ),
        provider,
    )


def test_transcribe_limit_is_sixty_seconds(tmp_path: Path, monkeypatch) -> None:
    service, _ = make_service(tmp_path, monkeypatch)

    with pytest.raises(ToolError, match="limit is 60s"):
        service.transcribe_segment(0, 60.01)


def test_tool_limit_stays_sixty_when_provider_allows_more(
    tmp_path: Path, monkeypatch
) -> None:
    service, provider = make_service(tmp_path, monkeypatch)
    provider.max_segment_s = 600
    object.__setattr__(service.config.advanced_asr.local, "max_segment_s", 600)

    assert service.provider_max_segment_s == 600
    assert service.tool_max_segment_s == 60


def test_baseline_chunk_uses_provider_limit(tmp_path: Path, monkeypatch) -> None:
    service, provider = make_service(tmp_path, monkeypatch)
    provider.max_segment_s = 180

    result = service.transcribe_baseline_chunk(0, 180)

    assert result.data["audio_duration_s"] == 180
    assert provider.calls == 1


def test_full_baseline_splits_on_provider_limit(tmp_path: Path, monkeypatch) -> None:
    service, provider = make_service(tmp_path, monkeypatch)
    provider.max_segment_s = 180

    transcript, _ = _full_advanced_transcript(service, service.index)

    assert provider.calls == 4
    assert len(transcript.splitlines()) == 4


def test_advanced_asr_absolute_timestamps_and_cache(tmp_path: Path, monkeypatch) -> None:
    service, provider = make_service(tmp_path, monkeypatch)

    first = service.transcribe_segment(10, 20)
    second = service.transcribe_segment(10, 20)

    assert first.data["sentences"][0]["start_s"] == 11
    assert first.data["sentences"][0]["end_s"] == 12
    assert first.data["cache_hit"] is False
    assert second.data["cache_hit"] is True
    assert provider.calls == 1


def test_transcribe_trace_records_cost_dimensions(tmp_path: Path, monkeypatch) -> None:
    service, _ = make_service(tmp_path, monkeypatch)

    service.execute("transcribe_segment", {"start_s": 10, "end_s": 20})

    record = json.loads(service.trace.path.read_text(encoding="utf-8").splitlines()[-1])
    summary = record["result_summary"]
    assert summary["provider"] == "local"
    assert summary["audio_duration_s"] == 10
    assert summary["asr_response_s"] >= 0
    assert summary["cost_usd"] is None


def test_system_prompt_makes_tools_optional() -> None:
    prompt = system_prompt()

    assert "SenseVoice 인덱스만으로 충분하면 도구 조회는 필수가 아니다" in prompt
    assert "transcribe_segment" in prompt
    assert "view_frames" in prompt
    assert "프레임에 보이는 텍스트" in prompt
    assert "[mm:ss]" in prompt


def test_asr_is_excluded_from_agent_budget() -> None:
    budget = AgentBudget(12, 20, started=time.monotonic() - 100)
    budget.excluded_asr_s = 90

    assert budget.reason() is None


def test_agent_can_finish_without_tools_or_forced_retry(tmp_path: Path, monkeypatch) -> None:
    service, _ = make_service(tmp_path, monkeypatch)

    class DirectLLM:
        model = "stub"

        def __init__(self) -> None:
            self.calls = 0

        def complete(self, messages, **kwargs) -> ChatResult:
            del messages, kwargs
            self.calls += 1
            return ChatResult(
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "title": "Direct",
                            "one_line_summary": "Summary",
                            "sections": [],
                            "key_moments": [],
                        }
                    ),
                },
                usage={"prompt_tokens": 10, "completion_tokens": 5},
                latency_s=0.01,
            )

    client = DirectLLM()
    raw_report, stats = run_agent_loop(
        query="summary",
        index=service.index,
        config=service.config,
        service=service,
        client=client,
    )

    assert client.calls == 1
    assert json.loads(raw_report)["title"] == "Direct"
    assert stats["tool_calls"] == 0
    assert stats["cumulative_input_tokens"] == 10


def test_tool_schemas_and_montage_limit(tmp_path: Path, monkeypatch) -> None:
    service, _ = make_service(tmp_path, monkeypatch)
    schemas = service.schemas

    assert [schema["function"]["name"] for schema in schemas] == [
        "view_frames",
        "transcribe_segment",
    ]
    assert schemas[0]["function"]["parameters"]["required"] == [
        "start_s",
        "end_s",
        "fps",
        "n",
    ]
    assert schemas[0]["function"]["parameters"]["properties"]["fps"] == {
        "type": "number",
        "enum": [0.1, 0.2, 0.5, 1.0, 2.0],
    }
    assert schemas[0]["function"]["parameters"]["properties"]["n"] == {
        "type": "integer",
        "enum": [1, 2, 3, 4, 6],
    }
    execution = service.execute(
        "view_frames", {"start_s": 0, "end_s": 100, "fps": 2, "n": 3}
    )
    assert "23 montage images" in execution.data["error"]
    assert "montage limit is 16" in execution.data["error"]


def test_view_frames_uses_agent_selected_grid(tmp_path: Path, monkeypatch) -> None:
    service, _ = make_service(tmp_path, monkeypatch)
    seen: dict[str, int] = {}

    def fake_frames(*_args, **_kwargs):
        return [FrameArtifact(path=f"frame-{i}.jpg", timestamp_s=i) for i in range(20)]

    def fake_montages(frames, _output_dir, *, n, width, height, jpeg_quality):
        del frames, width, height, jpeg_quality
        seen["n"] = n
        return [Path("one.jpg"), Path("two.jpg"), Path("three.jpg")]

    monkeypatch.setattr("vuc.tools.extract_sampled_frames", fake_frames)
    monkeypatch.setattr("vuc.tools.create_montages", fake_montages)

    result = service.view_frames(0, 100, 0.2, 3)

    assert seen["n"] == 3
    assert result.data["frame_count"] == 20
    assert result.data["montage_count"] == 3
    assert len(result.image_paths) == 3


def test_view_frames_rejects_values_outside_enums(tmp_path: Path, monkeypatch) -> None:
    service, _ = make_service(tmp_path, monkeypatch)

    with pytest.raises(ToolError, match="fps must be one of"):
        service.view_frames(0, 10, 0.3, 3)
    with pytest.raises(ToolError, match="n must be one of"):
        service.view_frames(0, 10, 0.5, 5)


def test_full_index_is_never_downsampled() -> None:
    index = VideoIndex(
        schema_version=2,
        video=VideoMetadata("video.mp4", "b" * 64, 100, 1),
        segments=tuple(
            Segment(number, number + 1, f"segment-{number}", "en")
            for number in range(100)
        ),
        frames=(),
        montages=(),
        created_at="2026-09-08T00:00:00+00:00",
    )

    context = build_index_context(index)

    assert "segment-0" in context
    assert "segment-99" in context
    assert len(context.splitlines()) == 100


def test_report_has_only_plain_citation_fields() -> None:
    report = normalize_report(
        {
            "sections": [
                {
                    "start_s": 0,
                    "end_s": 10,
                    "citations": [
                        {
                            "claim": "claim",
                            "start_s": 1,
                            "end_s": 2,
                            "extra": {"source": "model"},
                        }
                    ],
                }
            ]
        },
        meta={"duration_s": 100},
    )

    assert report["sections"][0]["citations"] == [
        {"claim": "claim", "start_s": 1, "end_s": 2}
    ]
    assert report["meta"] == {"duration_s": 100}
