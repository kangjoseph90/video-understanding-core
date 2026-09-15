from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from tests.support import StubOCREngine, StubTagger, StubTranscriber, StubVAD, write_config
from vuc.audio import EventTag
from vuc.config import load_config
from vuc.index_text import render_audio_index
from vuc.models import NON_SPEECH, SPEECH, Segment
from vuc.ocr import OCRLine
from vuc.pipeline import SCHEMA_VERSION, index_video

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")


def make_video(path: Path, *, duration: int = 40) -> Path:
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc2=size=320x180:rate=10:duration={duration}",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:sample_rate=16000:duration={duration}",
            "-shortest",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(path),
        ],
        check=True,
    )
    return path


def build(tmp_path: Path, **kwargs):
    video = make_video(tmp_path / "sample.mp4")
    config = load_config(write_config(tmp_path))
    defaults = {
        "transcriber": StubTranscriber(),
        "vad": StubVAD([(2.4, 9.6), (20.0, 29.0)]),
        "event_tagger": StubTagger(),
        "ocr_engine": StubOCREngine(),
    }
    return video, config, index_video(video, config, **{**defaults, **kwargs})


def test_index_covers_the_whole_timeline_and_caches(tmp_path: Path) -> None:
    video, config, (index, cache, cache_hit, trace_path) = build(tmp_path)

    assert not cache_hit
    assert index.schema_version == SCHEMA_VERSION
    assert cache.index_json_path.exists()
    assert cache.audio_index_path.exists()
    assert cache.text_index_path.exists()

    # VAD split the video; every second of it is on exactly one side, in order.
    assert [segment.kind for segment in index.audio.segments] == [
        NON_SPEECH,
        SPEECH,
        NON_SPEECH,
        SPEECH,
        NON_SPEECH,
    ]
    assert index.audio.segments[0].start == 0.0
    assert index.audio.segments[-1].end == pytest.approx(index.video.duration_s, abs=1.0)
    for previous, following in zip(index.audio.segments, index.audio.segments[1:], strict=False):
        assert previous.end == following.start

    cached, _, second_hit, second_trace = index_video(
        video,
        config,
        transcriber=StubTranscriber(),
        vad=StubVAD([(2.4, 9.6)]),
        event_tagger=StubTagger(),
        ocr_engine=StubOCREngine(),
    )
    assert second_hit
    assert cached.video.sha256 == index.video.sha256
    assert second_trace != trace_path


def test_changing_a_setting_rebuilds_instead_of_serving_the_old_output(
    tmp_path: Path,
) -> None:
    video, _, _ = build(tmp_path)
    widened = load_config(write_config(tmp_path, ("merge_gap_s: 1.5", "merge_gap_s: 8.0")))

    _, _, cache_hit, _ = index_video(
        video,
        widened,
        transcriber=StubTranscriber(),
        vad=StubVAD([(2.4, 9.6)]),
        event_tagger=StubTagger(),
        ocr_engine=StubOCREngine(),
    )

    assert not cache_hit


def test_frames_are_sampled_on_change_with_a_coverage_floor(tmp_path: Path) -> None:
    _, config, (index, _, _, _) = build(tmp_path)

    assert index.visual.frames
    timestamps = [frame.timestamp_s for frame in index.visual.frames]
    assert timestamps == sorted(timestamps)
    edges = [0.0, *timestamps, index.video.duration_s]
    widest = max(b - a for a, b in zip(edges, edges[1:], strict=False))
    assert widest <= config.visual_scan.max_interval_s
    assert len(index.visual.montages) == 1
    # The 3x3 canvas shrinks to fit a partial final group, but the cells keep
    # the size they were laid out at.
    cell = config.montage.width // config.frames.index_montage_n
    with Image.open(index.visual.montages[0]) as montage:
        assert montage.width % cell == 0
        assert montage.width <= config.montage.width


def test_the_two_indexes_are_written_to_separate_files(tmp_path: Path) -> None:
    heading = OCRLine(text="Slide heading", confidence=0.9, top=0.05, bottom=0.2, right=0.5)
    _, _, (index, cache, _, _) = build(tmp_path, ocr_engine=StubOCREngine([heading]))

    assert [cue.text for cue in index.text.cues] == ["Slide heading"]
    assert index.text.cues[0].position == "top left"
    assert index.text.cues[0].size == "large"

    audio_file = cache.audio_index_path.read_text(encoding="utf-8")
    text_file = cache.text_index_path.read_text(encoding="utf-8")
    assert "Slide heading" not in audio_file
    assert text_file.startswith("[0-")
    assert "] Slide heading" in text_file


def test_a_missing_ocr_engine_costs_only_the_on_screen_text(tmp_path: Path) -> None:
    """Every stage degrades rather than taking the index down with it."""
    video = make_video(tmp_path / "sample.mp4")
    config = load_config(write_config(tmp_path, ("engine: rapidocr", "engine: nonexistent")))

    index, cache, _, trace_path = index_video(
        video,
        config,
        transcriber=StubTranscriber(),
        vad=StubVAD([(2.4, 9.6)]),
        event_tagger=StubTagger(),
    )

    assert index.text.cues == ()
    assert index.audio.segments
    assert index.indexer["ocr_engine"] is None
    events = {json.loads(line)["event"] for line in trace_path.read_text().splitlines()}
    assert "ocr_unavailable" in events
    # The file still exists, so "read and found nothing" stays distinguishable
    # from "never ran" only by the trace, never by a missing artifact.
    assert cache.text_index_path.read_text(encoding="utf-8").strip() == ""
    assert cache.audio_index_path.read_text(encoding="utf-8").strip()


def test_trace_records_each_stage(tmp_path: Path) -> None:
    _, _, (_, _, _, trace_path) = build(tmp_path)

    events = [json.loads(line)["event"] for line in trace_path.read_text().splitlines()]
    assert {"vad", "audio_index", "visual_scan", "ocr", "index_complete"} <= set(events)


def test_speech_regions_are_transcribed_and_the_rest_tagged(tmp_path: Path) -> None:
    transcriber = StubTranscriber([Segment(0, 7, "spoken words", "en")])
    tagger = StubTagger()

    _, config, (index, _, _, _) = build(tmp_path, transcriber=transcriber, event_tagger=tagger)

    spoken = [segment for segment in index.audio.segments if segment.kind == SPEECH]
    quiet = [segment for segment in index.audio.segments if segment.kind == NON_SPEECH]
    assert len(transcriber.calls) == len(spoken)
    # One call per window, not per region, and agreeing windows are rejoined.
    assert len(tagger.calls) >= len(quiet)
    assert all(call <= config.audio_events.window_s for call in tagger.calls)
    assert all(segment.text == "spoken words" for segment in spoken)
    assert all(segment.events == ("applause",) for segment in quiet)


def test_a_region_nothing_could_be_named_in_is_not_rendered(tmp_path: Path) -> None:
    """The partition keeps it; the file does not. The timestamps show the gap."""
    _, _, (index, cache, _, _) = build(
        tmp_path, event_tagger=StubTagger([EventTag("EMO_UNKNOWN", 0.9)])
    )

    quiet = [s for s in index.audio.segments if s.kind == NON_SPEECH]
    assert quiet and all(segment.is_empty for segment in quiet)
    rendered = cache.audio_index_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rendered) == len(index.audio.segments) - len(quiet)
    assert rendered == render_audio_index(index.audio.segments).splitlines()


def test_the_sidecar_is_read_by_vuc_index_not_only_by_vuc_run(tmp_path: Path) -> None:
    """A video declaring its language was being indexed without one."""
    video = make_video(tmp_path / "sample.mp4")
    video.with_suffix(".meta.json").write_text(
        json.dumps({"title": "T", "channel": "C", "language": "ja"}), encoding="utf-8"
    )
    config = load_config(write_config(tmp_path))

    index, _, _, _ = index_video(
        video,
        config,
        transcriber=StubTranscriber(),
        vad=StubVAD([(2.4, 9.6)]),
        event_tagger=StubTagger(),
        ocr_engine=StubOCREngine(),
    )

    assert index.indexer["language_hint"] == "ja"
    assert index.index_config["ocr"]["rec_model_path"].endswith("PP-OCRv6_small_rec.onnx")


def test_runtime_ocr_failure_keeps_audio_and_closes_the_worker(tmp_path: Path) -> None:
    from vuc.ocr import OCRError

    class BrokenOCR:
        name = "broken"
        closed = False

        def read_many(self, paths):
            raise OCRError("worker stopped")

        def close(self):
            self.closed = True

    engine = BrokenOCR()
    _, _, (index, cache, _, trace_path) = build(tmp_path, ocr_engine=engine)
    assert index.audio.segments
    assert not index.text.cues
    assert engine.closed
    assert cache.text_index_path.read_text().strip() == ""
    assert not list(cache.text_frames_dir.glob("text-*.jpg"))
    events = [json.loads(row) for row in trace_path.read_text().splitlines()]
    assert any(
        e["event"] == "ocr_failed" and "worker stopped" in e["result_summary"]["error"]
        for e in events
    )


def write_captions(video: Path, *cues: tuple[float, str], kind: str = "manual") -> None:
    from vuc.captions import CaptionCue, content_hash

    rows = tuple(CaptionCue(at, text) for at, text in cues)
    video.with_suffix(".subs.json").write_text(
        json.dumps(
            {
                "audio_language": "en",
                "track": {
                    "language": "en",
                    "kind": kind,
                    "format": "json3",
                    "content_sha256": content_hash(rows),
                    "has_word_timing": True,
                    "cues": [cue.to_dict() for cue in rows],
                },
            }
        ),
        encoding="utf-8",
    )


def render_with_captions(video: Path, index, config) -> tuple[str, object]:
    """What the prompt would carry: the stored index with the track applied."""
    from vuc.caption_fusion import overlay_captions
    from vuc.captions import load_caption_track

    overlay = overlay_captions(
        index,
        load_caption_track(video),
        audio_language="en",
        config=config.captions.attribution(),
    )
    return render_audio_index(overlay.segments), overlay


def index_with_stubs(video: Path, config, text: str = "spoken words"):
    return index_video(
        video,
        config,
        transcriber=StubTranscriber([Segment(0.0, 7.2, text, "en")]),
        vad=StubVAD([(2.4, 9.6)]),
        event_tagger=StubTagger(),
        ocr_engine=StubOCREngine(),
    )


def test_a_caption_track_corrects_the_rendered_index(tmp_path: Path) -> None:
    video = make_video(tmp_path / "sample.mp4")
    write_captions(video, (3.0, "corrected"), (4.0, "spoken"), (5.0, "words"))
    config = load_config(write_config(tmp_path))

    index, _, _, _ = index_with_stubs(video, config)
    rendered, overlay = render_with_captions(video, index, config)

    assert overlay.summary["verdict"] == "speech_transcript"
    assert "corrected spoken words" in rendered


def test_the_stored_index_keeps_what_the_video_produced(tmp_path: Path) -> None:
    """index.json and the cached .txt are the observation; the track is not in them."""
    video = make_video(tmp_path / "sample.mp4")
    write_captions(video, (3.0, "corrected"), (4.0, "spoken"), (5.0, "words"))
    config = load_config(write_config(tmp_path))

    index, cache, _, _ = index_with_stubs(video, config)

    assert "corrected" not in cache.audio_index_path.read_text(encoding="utf-8")
    assert "corrected" not in json.dumps(index.to_dict(), ensure_ascii=False)
    assert "captions" not in index.to_dict()


def test_a_track_appearing_does_not_rebuild_the_index(tmp_path: Path) -> None:
    """The correction is cheap; redoing the VAD, the ASR and the OCR is not."""
    video = make_video(tmp_path / "sample.mp4")
    config = load_config(write_config(tmp_path))
    index_with_stubs(video, config)

    write_captions(video, (3.0, "corrected"), (4.0, "spoken"), (5.0, "words"))
    _, _, cached, _ = index_with_stubs(video, config)

    assert cached is True


def test_a_changed_track_changes_the_prompt_without_touching_the_index(
    tmp_path: Path,
) -> None:
    video = make_video(tmp_path / "sample.mp4")
    write_captions(video, (3.0, "first"), (4.0, "spoken"), (5.0, "words"))
    config = load_config(write_config(tmp_path))
    index, cache, _, _ = index_with_stubs(video, config)
    before = cache.audio_index_path.read_text(encoding="utf-8")

    write_captions(video, (3.0, "second"), (4.0, "spoken"), (5.0, "words"))
    rendered, _ = render_with_captions(video, index, config)

    assert "second spoken words" in rendered
    assert cache.audio_index_path.read_text(encoding="utf-8") == before


def test_no_caption_sidecar_costs_only_the_correction(tmp_path: Path) -> None:
    video = make_video(tmp_path / "sample.mp4")
    config = load_config(write_config(tmp_path))

    index, _, _, _ = index_with_stubs(video, config)
    rendered, overlay = render_with_captions(video, index, config)

    assert overlay.summary is None
    assert "spoken words" in rendered


def test_the_prompt_never_says_where_a_line_came_from(tmp_path: Path) -> None:
    """Provenance lives in the overlay summary and the trace, never in the prompt."""
    from vuc.agent import build_prompt_body

    video = make_video(tmp_path / "sample.mp4")
    write_captions(video, (3.0, "corrected"), (4.0, "spoken"), (5.0, "words"))
    config = load_config(write_config(tmp_path))
    index, _, _, _ = index_with_stubs(video, config)
    rendered, _ = render_with_captions(video, index, config)

    prompt = build_prompt_body(
        "summarise", index.video.duration_s, audio_index=rendered, text_index=""
    )

    assert "corrected" in prompt
    for leak in ("speech_transcript", "hardsub", "caption", "manual", "vad_overlap", "sensevoice"):
        assert leak not in prompt.casefold()


def test_the_run_records_what_it_sent_and_why(tmp_path: Path) -> None:
    """The prompt and the verdict live with the run; the cache keeps the index."""
    from vuc.llm import ChatResult
    from vuc.run import run_video

    class StubVLM:
        model = "stub"

        def complete(self, messages, **kwargs):
            del messages, kwargs
            return ChatResult(
                message={
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "title": "T",
                            "one_line_summary": "S",
                            "sections": [],
                            "key_moments": [],
                        }
                    ),
                },
                usage={"prompt_tokens": 1, "completion_tokens": 1},
                latency_s=0.0,
            )

    video = make_video(tmp_path / "sample.mp4")
    write_captions(video, (3.0, "corrected"), (4.0, "spoken"), (5.0, "words"))
    config = load_config(
        write_config(tmp_path, ("  mode: agentic", "  mode: baseline_index_only"))
    )
    report, _, _ = run_video(
        video,
        config,
        query="q",
        index_transcriber=StubTranscriber([Segment(0.0, 7.2, "spoken words", "en")]),
        index_vad=StubVAD([(2.4, 9.6)]),
        index_event_tagger=StubTagger(),
        index_ocr_engine=StubOCREngine(),
        llm_client=StubVLM(),
    )

    run_dir = Path(report["meta"]["trace"]).parent
    sent = (run_dir / "prompt.txt").read_text(encoding="utf-8")
    assert "corrected spoken words" in sent

    events = [json.loads(line) for line in (run_dir / "trace.jsonl").read_text().splitlines()]
    fusion = next(e for e in events if e["event"] == "caption_fusion")
    assert fusion["result_summary"]["verdict"] == "speech_transcript"
    assert fusion["result_summary"]["fusion"]["regions_rewritten"] == 1
