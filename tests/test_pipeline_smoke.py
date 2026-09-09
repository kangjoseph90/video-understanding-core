from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from vuc.config import load_config
from vuc.models import Segment
from vuc.pipeline import index_video


class StubTranscriber:
    def transcribe(self, audio_path: Path) -> list[Segment]:
        assert audio_path.exists()
        return [
            Segment(
                start=0,
                end=3,
                text="synthetic smoke test",
                language="en",
                emotion="neutral",
                events=("Speech",),
                raw_text="<|en|><|NEUTRAL|><|Speech|>synthetic smoke test",
            )
        ]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_index_video_creates_cache_frames_montage_and_trace(tmp_path: Path) -> None:
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
    source_config = Path(__file__).parents[1] / "config.yaml"
    config_data = (
        source_config.read_text(encoding="utf-8")
        .replace("directory: .vuc-cache", f"directory: {tmp_path / 'cache'}")
        .replace("index_interval_s: 15", "index_interval_s: 1")
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_data, encoding="utf-8")
    config = load_config(config_path)

    index, cache, cache_hit = index_video(video, config, transcriber=StubTranscriber())

    assert not cache_hit
    assert cache.index_json_path.exists()
    assert cache.index_text_path.exists()
    assert cache.trace_path.exists()
    assert len(index.frames) == 4
    assert len(index.montages) == 1
    assert index.frame_config == {
        "interval_s": 1.0,
        "resolution": 256,
        "montage_n": 3,
        "jpeg_quality": 88,
    }
    assert "<Speech>" in cache.index_text_path.read_text(encoding="utf-8")
    trace = [json.loads(line) for line in cache.trace_path.read_text().splitlines()]
    assert {item["event"] for item in trace} == {
        "sensevoice_index",
        "initial_frames",
        "index_complete",
    }

    cached, _, second_cache_hit = index_video(video, config, transcriber=StubTranscriber())
    assert second_cache_hit
    assert cached.video.sha256 == index.video.sha256
