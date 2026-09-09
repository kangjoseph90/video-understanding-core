from __future__ import annotations

import json
from pathlib import Path

from vuc.agent import build_prompt_body
from vuc.hints import Chapter, VideoHints, load_video_hints
from vuc.indexer import SENSEVOICE_LANGUAGES

SIDECAR = {
    "title": "Vegetable Pancake (Yachaejeon: 야채전)",
    "channel": "Maangchi",
    "language": "en",
    "description": "링크와 SNS 홍보뿐인 설명은 힌트로 쓰지 않는다",
    "chapters": [
        {"start_s": 0, "end_s": 67, "title": "Intro"},
        {"start_s": 67, "end_s": 487, "title": "Cooking"},
        {"start_s": 487, "end_s": 603, "title": "Tasting"},
    ],
}


def _write_sidecar(tmp_path: Path) -> Path:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    video.with_suffix(".meta.json").write_text(
        json.dumps(SIDECAR, ensure_ascii=False), encoding="utf-8"
    )
    return video


def test_load_video_hints_reads_sidecar_beside_the_video(tmp_path: Path) -> None:
    hints = load_video_hints(_write_sidecar(tmp_path))

    assert hints is not None
    assert hints.channel == "Maangchi"
    assert hints.language == "en"
    assert len(hints.chapters) == 3


def test_missing_sidecar_means_no_hints(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")

    assert load_video_hints(video) is None


def test_prompt_block_omits_the_description(tmp_path: Path) -> None:
    hints = load_video_hints(_write_sidecar(tmp_path))
    assert hints is not None
    block = hints.prompt_block()

    assert "채널: Maangchi" in block
    assert "[67-487] Cooking" in block
    # Descriptions are unreliable across channels, so they are never a hint.
    assert "SNS 홍보" not in block


def test_asr_prompt_covers_only_the_chapters_touching_the_chunk() -> None:
    hints = VideoHints(
        title="T",
        channel="C",
        chapters=(
            Chapter(0, 67, "Intro"),
            Chapter(67, 487, "Cooking"),
            Chapter(487, 603, "Tasting"),
        ),
    )

    assert hints.asr_prompt(180, 360) == "C - T - Cooking"
    assert hints.asr_prompt(360, 540) == "C - T - Cooking, Tasting"


def test_asr_prompt_falls_back_to_channel_and_title_without_chapters() -> None:
    hints = VideoHints(title="東京Vlog", channel="あかね的日本語教室")

    assert hints.asr_prompt(0, 180) == "あかね的日本語教室 - 東京Vlog"


def test_prompt_body_inserts_metadata_before_the_transcript() -> None:
    hints = VideoHints(title="T", channel="C", chapters=(Chapter(0, 18, "Intro"),))
    body = build_prompt_body("질의", 603.0, "전체 SenseVoice 인덱스", "[0-18] <en> hi", hints)

    assert body.index("영상 메타데이터") < body.index("영상 길이: 603")
    assert body.endswith("[0-18] <en> hi")


def test_prompt_body_without_hints_is_unchanged() -> None:
    body = build_prompt_body("질의", 603.0, "전체 SenseVoice 인덱스", "[0-18] <en> hi")

    assert "영상 메타데이터" not in body


def test_unsupported_language_hint_falls_back_to_auto() -> None:
    # SenseVoice only conditions on its own language table.
    assert "pl" not in SENSEVOICE_LANGUAGES
    assert {"ko", "ja", "en", "auto"} <= SENSEVOICE_LANGUAGES


def test_language_hint_falls_back_to_metadata_when_the_index_is_empty(
    tmp_path: Path, monkeypatch
) -> None:
    from tests.test_m2 import make_service

    service, _ = make_service(tmp_path, monkeypatch)
    # baseline_full builds an index with no segments at all.
    object.__setattr__(service.index, "segments", ())

    assert service._language_hint(0, 180) is None

    service.hints = VideoHints(title="T", channel="C", language="en")
    assert service._language_hint(0, 180) == "en"
