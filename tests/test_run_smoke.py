from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vuc.advanced_asr import ASRResult, TranscriptSentence
from vuc.config import load_config
from vuc.llm import ChatResult
from vuc.models import Segment
from vuc.run import run_video


class StubIndexer:
    def transcribe(self, audio_path: Path) -> list[Segment]:
        return [Segment(0, 4, "draft", "en", emotion="neutral", events=("Speech",))]


class StubAdvancedASR:
    name = "local"
    model_name = "stub"
    max_segment_s = 180

    def transcribe(
        self,
        audio_path: Path,
        *,
        audio_duration_s: float,
        language_hint: str | None,
    ) -> ASRResult:
        return ASRResult(
            text="The video shows a test pattern.",
            sentences=(TranscriptSentence(0, audio_duration_s, "test pattern"),),
            language=language_hint,
            notes=None,
            provider="local",
            model="stub",
            audio_duration_s=audio_duration_s,
            processing_s=0.01,
            cost_usd=None,
        )


class StubLLM:
    model = "stub-vlm"

    def complete(self, messages, **kwargs) -> ChatResult:
        del messages, kwargs
        report = {
            "title": "Smoke report",
            "one_line_summary": "A test pattern.",
            "sections": [
                {
                    "title": "Test",
                    "summary": "A generated sample.",
                    "start_s": 0,
                    "end_s": 4,
                    "citations": [
                        {
                            "claim": "Pattern",
                            "start_s": 0,
                            "end_s": 4,
                            "evidence_span": {
                                "start_s": 0,
                                "end_s": 4,
                                "source": "transcribe_segment",
                            },
                        }
                    ],
                }
            ],
            "key_moments": [{"title": "Start", "summary": "Pattern", "timestamp_s": 0}],
            "unverified_claims": [],
        }
        return ChatResult(
            message={"role": "assistant", "content": json.dumps(report)},
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            latency_s=0.01,
        )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
@pytest.mark.parametrize("mode", ["baseline", "agentic"])
def test_run_explicit_mode_ignores_short_duration(tmp_path: Path, mode: str) -> None:
    video = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=10:duration=4",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000:duration=4",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(video),
        ],
        check=True,
    )
    source = Path(__file__).parents[1] / "config.yaml"
    config_path = tmp_path / "config.yaml"
    config_text = source.read_text(encoding="utf-8")
    config_text = config_text.replace(
        "directory: .vuc-cache", f"directory: {tmp_path / 'cache'}"
    )
    config_text = config_text.replace("mode: agentic", f"mode: {mode}")
    config_path.write_text(config_text, encoding="utf-8")
    config = load_config(config_path)

    report, markdown_path, json_path = run_video(
        video,
        config,
        index_transcriber=StubIndexer(),
        advanced_provider=StubAdvancedASR(),
        llm_client=StubLLM(),
    )

    assert report["meta"]["mode"] == mode
    assert report["meta"]["cumulative_input_tokens"] == 100
    assert report["meta"]["output_tokens"] == 50
    assert report["meta"]["vlm_cost_usd"] is None
    assert report["meta"]["coverage_ratio"] == 1
    assert report["sections"][0]["citations"][0]["verified"] is True
    assert markdown_path.exists()
    assert json_path.exists()
