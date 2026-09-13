"""Building the index: one pass over the audio, one over the frames.

The two halves are independent observers of the same timeline and are treated
as such. The audio half runs VAD first, then routes each region to the model
that suits it. The visual half samples on change rather than on a clock, then
reads whatever text is on the frames it kept. Neither half corrects, outranks
or overwrites the other; they are simply both written down.

Every stage degrades rather than fails. A missing event tagger costs the
non-speech labels, a missing OCR engine costs the on-screen text, and the rest
of the index is still built and still useful.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vuc.audio import (
    AudioEventTagger,
    SenseVoiceTagger,
    analyze_regions,
    create_event_tagger,
    timeline_regions,
)
from vuc.cache import VideoCache, new_run_id, sha256_file
from vuc.config import AppConfig
from vuc.frames import burn_in_timestamp, create_montages, montage_cell_size
from vuc.hints import VideoHints, load_video_hints
from vuc.index_text import render_audio_index, render_text_index
from vuc.indexer import SenseVoiceTranscriber, Transcriber
from vuc.media import MediaError, extract_audio, probe_duration
from vuc.models import (
    AudioIndex,
    Segment,
    TextCue,
    TextIndex,
    VideoIndex,
    VideoMetadata,
    VisualIndex,
)
from vuc.ocr import OCREngine, OCRError, create_ocr_engine, scan_text, text_cues
from vuc.trace import TraceWriter
from vuc.vad import VADProvider, create_vad, non_speech_view, speech_view
from vuc.visual_scan import ScanResult, scan_video

SCHEMA_VERSION = 4


def _validate_video(path: Path, config: AppConfig) -> Path:
    video_path = path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"video file not found: {video_path}")
    if video_path.suffix.lower().lstrip(".") not in config.video.allowed_extensions:
        allowed = ", ".join(config.video.allowed_extensions)
        raise ValueError(f"unsupported video extension; expected one of: {allowed}")
    return video_path


def _load_cached(path: Path) -> VideoIndex:
    return VideoIndex.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _index_config(config: AppConfig, language_hint: str | None) -> dict[str, Any]:
    """Everything that changes what the index says.

    A cached index is only reused when this matches, so turning OCR on or
    widening a VAD merge window rebuilds instead of silently serving the
    previous settings' output.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "language_hint": language_hint,
        "vad": vars(config.vad),
        "audio_events": vars(config.audio_events),
        "visual_scan": vars(config.visual_scan),
        # The resolved settings, not the table they were chosen from: the
        # language picks the recogniser, and a mapping of tuples does not
        # survive a JSON round-trip intact, so comparing it would miss forever.
        "ocr": {
            key: value
            for key, value in vars(config.ocr.for_language(language_hint)).items()
            if key != "rec_by_language"
        },
        "montage_width": config.montage.width,
        "montage_height": config.montage.height,
        "montage_n": config.frames.index_montage_n,
        "jpeg_quality": config.montage.jpeg_quality,
    }


def _build_tagger(
    config: AppConfig,
    trace: TraceWriter,
    *,
    sensevoice: Transcriber,
) -> AudioEventTagger | None:
    """The configured tagger, or the one that is always available.

    PANNs is a 320MB download and an optional extra away; when it cannot be
    had, falling back to SenseVoice's own labels keeps non-speech regions from
    going silent in the index. Losing the whole run over it -- which is what
    used to happen, because panns_inference shells out to wget -- is the one
    outcome that is not acceptable.
    """
    try:
        tagger = create_event_tagger(config.audio_events, sensevoice=sensevoice)
    except Exception as exc:  # noqa: BLE001 - any failure here must degrade, not abort
        trace.write(
            step="index",
            event="event_tagger_unavailable",
            arguments={"tagger": config.audio_events.tagger},
            result_summary={"error": str(exc), "fallback": "sensevoice"},
            duration_ms=0,
        )
        return SenseVoiceTagger(sensevoice)
    return tagger


def _build_ocr_engine(
    config: AppConfig, trace: TraceWriter, *, language: str | None
) -> OCREngine | None:
    if not config.ocr.enabled:
        return None
    try:
        return create_ocr_engine(config.ocr.for_language(language))
    except Exception as exc:  # noqa: BLE001 - the rest of the index is still worth having
        trace.write(
            step="index",
            event="ocr_unavailable",
            arguments={"engine": config.ocr.engine, "language": language},
            result_summary={"error": str(exc)},
            duration_ms=0,
        )
        return None


def write_index_files(cache: VideoCache, index: VideoIndex) -> None:
    """One file per index, written even when empty.

    An empty text_index.txt says the frames were read and showed no text. A
    missing one would be indistinguishable from OCR never having run.
    """
    cache.audio_index_path.write_text(
        render_audio_index(index.audio.segments) + "\n", encoding="utf-8"
    )
    cache.text_index_path.write_text(render_text_index(index.text.cues) + "\n", encoding="utf-8")


def index_video(
    path: str | Path,
    config: AppConfig,
    *,
    transcriber: Transcriber | None = None,
    vad: VADProvider | None = None,
    event_tagger: AudioEventTagger | None = None,
    ocr_engine: OCREngine | None = None,
    force: bool = False,
    run_id: str | None = None,
    hints: VideoHints | None = None,
) -> tuple[VideoIndex, VideoCache, bool, Path]:
    video_path = _validate_video(Path(path), config)
    # A <video>.meta.json sidecar is picked up here rather than only in run_video,
    # which is what `vuc index` was missing: a video declaring `language: ja` was
    # indexed with no language at all, so SenseVoice guessed and the OCR stage
    # loaded the wrong recogniser.
    if hints is None:
        hints = load_video_hints(video_path)
    video_hash = sha256_file(video_path)
    cache = VideoCache(config.cache.directory, video_hash)
    cache.ensure()
    actual_run_id = run_id or new_run_id("index")
    trace_path = cache.run_dir(actual_run_id) / "trace.jsonl"
    trace = TraceWriter(trace_path)
    hint_language = hints.language if hints else None
    index_config = _index_config(config, hint_language)
    if cache.index_json_path.exists() and not force:
        try:
            cached = _load_cached(cache.index_json_path)
        except (KeyError, TypeError, ValueError):
            cached = None
        if cached is not None and cached.index_config == index_config:
            trace.write(
                step="index",
                event="cache_hit",
                arguments={"video_path": str(video_path)},
                result_summary={"video_hash": video_hash},
                duration_ms=0,
            )
            return cached, cache, True, trace_path

    duration_s = probe_duration(video_path)
    metadata = VideoMetadata(
        path=str(video_path),
        sha256=video_hash,
        duration_s=duration_s,
        size_bytes=video_path.stat().st_size,
    )
    timings: dict[str, float] = {}
    counts: dict[str, Any] = {}

    def index_audio() -> list[Segment]:
        started = time.monotonic()
        extract_audio(video_path, cache.audio_path)
        sensevoice = transcriber or SenseVoiceTranscriber(config.indexer, language=hint_language)
        detector = vad or create_vad(config.vad)
        tagger = event_tagger or _build_tagger(config, trace, sensevoice=sensevoice)

        detected = detector.detect(cache.audio_path, duration_s=duration_s)
        speech = speech_view(detected, config=config.vad, duration_s=duration_s)
        non_speech = non_speech_view(speech, config=config.vad, duration_s=duration_s)
        regions = timeline_regions(speech, non_speech)
        trace.write(
            step="index",
            event="vad",
            arguments={"provider": detector.name, "model": config.vad.model},
            result_summary={
                "detected": len(detected),
                "speech_regions": len(speech),
                "non_speech_regions": len(non_speech),
                "speech_s": round(sum(span.duration for span in speech), 3),
            },
            duration_ms=round((time.monotonic() - started) * 1000),
        )

        run = analyze_regions(
            cache.audio_path,
            regions,
            transcriber=sensevoice,
            tagger=tagger,
            events=config.audio_events,
            clip_dir=cache.regions_dir,
        )
        trace.write(
            step="index",
            event="audio_index",
            arguments={
                "model": config.indexer.model,
                "tagger": getattr(tagger, "name", None),
            },
            result_summary={
                "segments": len(run.segments),
                "transcribed_regions": run.transcribed_regions,
                "tagged_regions": run.tagged_regions,
            },
            duration_ms=round(run.processing_s * 1000),
        )
        counts["speech_regions"] = run.speech_regions
        counts["non_speech_regions"] = run.non_speech_regions
        counts["event_tagger"] = getattr(tagger, "name", None)
        timings["audio_s"] = round(time.monotonic() - started, 3)
        return list(run.segments)

    def index_text(scan: ScanResult) -> list[TextCue]:
        """OCR over its own dense scan, read only where the text moved.

        The frames the montage uses are read regardless, and each cut gets a
        candidate further past it than the montage takes -- a quarter second in
        is often still the dissolve, and a motion-blurred caption reads as
        nonsense. Subtitles changing is not treated as a visual event: this
        decides nothing about which frames the model is shown.
        """
        started = time.monotonic()
        ocr_config = config.ocr.for_language(hint_language)
        engine = ocr_engine or _build_ocr_engine(config, trace, language=hint_language)
        counts["ocr_engine"] = None if engine is None else engine.name
        cues: list[TextCue] = []
        try:
            if engine is not None:
                required = [frame.timestamp_s for frame in scan.frames]
                required += [
                    boundary.timestamp_s + config.ocr.boundary_offset_s
                    for boundary in scan.boundaries
                    if boundary.timestamp_s + config.ocr.boundary_offset_s < duration_s
                ]
                observations, sampled = scan_text(
                    video_path,
                    cache.text_frames_dir,
                    engine,
                    config=ocr_config,
                    duration_s=duration_s,
                    required_s=required,
                )
                cues = text_cues(observations, duration_s=duration_s, config=ocr_config)
                trace.write(
                    step="index",
                    event="ocr",
                    arguments={
                        "engine": engine.name,
                        "scan_fps": ocr_config.scan_fps,
                        "change_threshold": ocr_config.change_threshold,
                    },
                    result_summary={
                        "sampled": sampled,
                        "read": len(observations),
                        "verification_reads": sum(item.verification for item in observations),
                        "discovery_reads": sum(item.discovery for item in observations),
                        "crop_reads": sum(len(item.regions) for item in observations),
                        "with_text": sum(1 for item in observations if item.lines),
                        "cues": len(cues),
                    },
                    duration_ms=round((time.monotonic() - started) * 1000),
                )
        except (OCRError, MediaError, OSError) as exc:
            trace.write(
                step="index",
                event="ocr_failed",
                arguments={"engine": getattr(engine, "name", None)},
                result_summary={"error": f"{type(exc).__name__}: {exc}", "cues": 0},
                duration_ms=round((time.monotonic() - started) * 1000),
            )
            cues = []
        finally:
            # The worker holds a model and a pipe; nothing needs either afterwards.
            closer = getattr(engine, "close", None)
            if callable(closer):
                closer()
        timings["ocr_s"] = round(time.monotonic() - started, 3)
        return cues

    def index_visual_and_text() -> tuple[ScanResult, list[Path], list[TextCue]]:
        """One branch: OCR picks its frames using what the scan found."""
        scan, montages = index_visual()
        return scan, montages, index_text(scan)

    def index_visual() -> tuple[ScanResult, list[Path]]:
        started = time.monotonic()
        cell_width, _ = montage_cell_size(
            config.montage.width, config.montage.height, config.frames.index_montage_n
        )
        result = scan_video(
            video_path,
            cache.frames_dir,
            config=config.visual_scan,
            duration_s=duration_s,
            frame_width=cell_width,
        )
        trace.write(
            step="index",
            event="visual_scan",
            arguments={
                "scene_threshold": config.visual_scan.scene_threshold,
                "max_interval_s": config.visual_scan.max_interval_s,
            },
            result_summary={
                "shot_boundaries": len(result.boundaries),
                "sampled": result.sampled,
                "frames": len(result.frames),
            },
            duration_ms=round((time.monotonic() - started) * 1000),
        )
        # The stamp is what lets the model cite a time it can see, and the
        # frames the scan discarded are never shown.
        for frame in result.frames:
            burn_in_timestamp(Path(frame.path), frame.timestamp_s, config.montage.jpeg_quality)
        montages = create_montages(
            [frame.artifact for frame in result.frames],
            cache.montages_dir,
            n=config.frames.index_montage_n,
            width=config.montage.width,
            height=config.montage.height,
            jpeg_quality=config.montage.jpeg_quality,
        )
        timings["visual_s"] = round(time.monotonic() - started, 3)
        return result, montages

    index_started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="vuc-index") as executor:
        audio_future = executor.submit(index_audio)
        visual_future = executor.submit(index_visual_and_text)
        segments = audio_future.result()
        scan, montages, cues = visual_future.result()
    timings["total_s"] = round(time.monotonic() - index_started, 3)

    index = VideoIndex(
        schema_version=SCHEMA_VERSION,
        video=metadata,
        audio=AudioIndex(tuple(segments)),
        text=TextIndex(tuple(cues)),
        visual=VisualIndex(
            frames=tuple(frame.artifact for frame in scan.frames),
            montages=tuple(str(item) for item in montages),
        ),
        created_at=datetime.now(UTC).isoformat(),
        indexer={
            "hub": config.indexer.hub,
            "model": config.indexer.model,
            "vad_model": config.vad.model,
            "language_hint": hint_language,
            **counts,
            # Persisted so later runs that hit the cache can still report the cold cost.
            "cold_audio_s": timings.get("audio_s", 0.0),
            "cold_visual_s": timings.get("visual_s", 0.0),
            "cold_ocr_s": timings.get("ocr_s", 0.0),
            "cold_total_s": timings.get("total_s", 0.0),
        },
        index_config=index_config,
    )
    cache.write_json(cache.index_json_path, index.to_dict())
    write_index_files(cache, index)
    trace.write(
        step="index",
        event="index_complete",
        arguments={"video_path": str(video_path)},
        result_summary={
            "index_json": str(cache.index_json_path),
            "audio_index": str(cache.audio_index_path),
            "text_index": str(cache.text_index_path),
            "segments": len(index.audio.segments),
            "text_cues": len(index.text.cues),
            "cold_audio_s": timings.get("audio_s", 0.0),
            "cold_visual_s": timings.get("visual_s", 0.0),
            "cold_ocr_s": timings.get("ocr_s", 0.0),
        },
        duration_ms=round(timings.get("total_s", 0.0) * 1000),
    )
    return index, cache, False, trace_path
