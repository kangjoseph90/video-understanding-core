from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class CacheConfig:
    directory: Path


@dataclass(frozen=True)
class VideoConfig:
    allowed_extensions: tuple[str, ...]


@dataclass(frozen=True)
class RunConfig:
    mode: str


@dataclass(frozen=True)
class IndexerConfig:
    hub: str
    model: str
    device: str
    cpu_threads: int
    batch_size_s: int


@dataclass(frozen=True)
class VADConfig:
    provider: str
    hub: str
    model: str
    device: str
    cpu_threads: int
    max_single_segment_s: float
    # Silence shorter than this may be swallowed into the speech either side of
    # it; see vad.speech_view for the second, proportional limit.
    merge_gap_s: float
    min_bridge_s: float
    min_region_s: float
    window_max_s: float


@dataclass(frozen=True)
class AudioEventConfig:
    tagger: str
    top_k: int
    min_confidence: float
    # A label must also reach this fraction of the tagger's own top score; see
    # audio.keep_tags for why an absolute bar alone is not enough.
    relative_floor: float
    # Non-speech regions are tagged in windows of at most this long, not whole.
    window_s: float
    model_path: str


@dataclass(frozen=True)
class VisualScanConfig:
    scan_fps: float
    scan_width: int
    scene_threshold: float
    min_interval_s: float
    # The coverage promise: no stretch longer than this goes unsampled, however
    # static the video is.
    max_interval_s: float
    boundary_offset_s: float
    phash_distance: int
    max_frames: int
    workers: int
    # A quarter second past a cut is still inside a dissolve, and a cut into a
    # fade lands on a blank frame. Both are answered the same way: step forward
    # until the picture has settled and is not a single flat colour. Hard cuts
    # settle on the first try and pay nothing.
    settle_step_s: float = 0.25
    settle_max_s: float = 1.5


@dataclass(frozen=True)
class OCRConfig:
    enabled: bool
    engine: str
    # OCR scans on its own clock, not on the visual scan's cuts: text appears
    # and disappears independently of where the camera happens to cut.
    scan_fps: float
    scan_width: int
    workers: int
    # Whole-frame changes are checked on the one-second base clock. Dense
    # rescans are confined to measured, changing text regions.
    change_threshold: float
    # Where to look after a cut. The montage lands at 0.25s to catch the new
    # shot; OCR waits longer, because that quarter second is often still the
    # dissolve and a motion-blurred caption reads as nonsense.
    boundary_offset_s: float
    min_confidence: float
    max_hold_s: float
    # How alike two readings must be to count as one line seen twice.
    same_text_ratio: float
    # How long a line may go unread before it counts as gone rather than
    # missed.
    rejoin_gap_s: float
    det_model_path: str
    rec_model_path: str
    rec_keys_path: str
    # Recognition model per video language. A recogniser only reads the scripts
    # its character dictionary contains, and the shipped one covers Chinese and
    # Latin -- which is why Korean subtitles came back as invented Han
    # characters and Japanese as `扩会计1360月寸`.
    rec_by_language: Mapping[str, tuple[str, str]] = field(default_factory=dict)
    # Scan cheaply, recognise from a legible frame, and periodically verify even
    # a layout whose edge grid did not change (new words can have the same grid).
    recognition_width: int = 1280
    refresh_s: float = 4.0
    singleton_confidence: float = 0.95
    # Minimum block-row height as a fraction of the full frame. Detector
    # fragments are assembled and tracked first; the median track height is
    # compared once so measurement jitter cannot create short-lived leaks.
    min_text_height: float = 0.025
    det_model_config_path: str = ""
    fallback_rec_model_path: str = ""
    fallback_rec_keys_path: str = ""
    # The wheel's old Chinese orientation classifier flips clear Hangul
    # captions. Video text is upright unless explicitly configured otherwise.
    use_angle_cls: bool = False

    def for_language(self, language: str | None) -> OCRConfig:
        tag = str(language or "").lower().replace("_", "-")
        chosen = self.rec_by_language.get(tag) or self.rec_by_language.get(tag.split("-")[0])
        if chosen is None:
            return self
        return replace(
            self,
            rec_model_path=chosen[0],
            rec_keys_path=chosen[1],
            fallback_rec_model_path="",
            fallback_rec_keys_path="",
        )


@dataclass(frozen=True)
class FramesConfig:
    index_montage_n: int
    baseline_full_interval_s: float
    baseline_full_montage_n: int


@dataclass(frozen=True)
class MontageConfig:
    width: int
    height: int
    jpeg_quality: int


@dataclass(frozen=True)
class VisionLLMConfig:
    base_url_env: str
    model_env: str
    api_key_env: str
    input_cost_env: str
    cached_input_cost_env: str
    output_cost_env: str
    timeout_s: float
    max_retries: int
    max_output_tokens: int
    input_cost_per_million_usd: float
    cached_input_cost_per_million_usd: float
    output_cost_per_million_usd: float
    temperature: float | None = None


@dataclass(frozen=True)
class LocalASRConfig:
    model: str
    device: str
    compute_type: str
    cpu_threads: int
    max_segment_s: float


@dataclass(frozen=True)
class CloudASRConfig:
    max_segment_s: float


@dataclass(frozen=True)
class AdvancedASRConfig:
    provider: str
    local: LocalASRConfig
    cloud: CloudASRConfig


@dataclass(frozen=True)
class AgentConfig:
    query: str
    max_tool_calls: int
    wall_clock_s: float


@dataclass(frozen=True)
class ViewFramesConfig:
    fps_options: tuple[float, ...]
    grid_options: tuple[int, ...]
    max_montages_per_call: int


@dataclass(frozen=True)
class TranscribeSegmentConfig:
    max_duration_s: float


@dataclass(frozen=True)
class AppConfig:
    path: Path
    cache: CacheConfig
    run: RunConfig
    video: VideoConfig
    indexer: IndexerConfig
    vad: VADConfig
    audio_events: AudioEventConfig
    visual_scan: VisualScanConfig
    ocr: OCRConfig
    frames: FramesConfig
    montage: MontageConfig
    vision_llm: VisionLLMConfig
    advanced_asr: AdvancedASRConfig
    agent: AgentConfig
    view_frames: ViewFramesConfig
    transcribe_segment: TranscribeSegmentConfig


def _rate(env_name: str) -> float:
    """USD per million tokens from the environment. Unset means unknown (0)."""
    raw = os.environ.get(str(env_name), "").strip()
    if not raw:
        return 0.0
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be a number, got {raw!r}") from exc


def _model_path(value: Any, config_path: Path) -> str:
    """Model paths are written relative to the config, not to the shell's cwd."""
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    return str(path if path.is_absolute() else (config_path.parent / path).resolve())


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"missing or invalid config section: {name}")
    return value


def load_config(path: str | Path, *, dotenv_path: str | Path | None = None) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    env_path = (
        Path(dotenv_path).expanduser().resolve()
        if dotenv_path is not None
        else config_path.parent / ".env"
    )
    load_dotenv(env_path, override=False)
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")

    cache = _section(data, "cache")
    run = _section(data, "run")
    video = _section(data, "video")
    indexer = _section(data, "indexer")
    vad = _section(data, "vad")
    audio_events = _section(data, "audio_events")
    visual_scan = _section(data, "visual_scan")
    ocr = _section(data, "ocr")
    frames = _section(data, "frames")
    montage = _section(data, "montage")
    vision_llm = _section(data, "vision_llm")
    asr = _section(data, "asr")
    advanced_asr = _section(asr, "advanced")
    local_asr = _section(advanced_asr, "local")
    cloud_asr = _section(advanced_asr, "cloud")
    agent = _section(data, "agent")
    tools = _section(data, "tools")
    view_frames = _section(tools, "view_frames")
    transcribe_segment = _section(tools, "transcribe_segment")
    cache_dir = Path(str(cache["directory"])).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = (config_path.parent / cache_dir).resolve()

    result = AppConfig(
        path=config_path,
        cache=CacheConfig(directory=cache_dir),
        run=RunConfig(mode=str(run["mode"])),
        video=VideoConfig(
            allowed_extensions=tuple(
                str(item).lower().lstrip(".") for item in video["allowed_extensions"]
            ),
        ),
        indexer=IndexerConfig(
            hub=str(indexer.get("hub", "ms")),
            model=str(indexer["model"]),
            device=str(indexer["device"]),
            cpu_threads=int(indexer["cpu_threads"]),
            batch_size_s=int(indexer["batch_size_s"]),
        ),
        vad=VADConfig(
            provider=str(vad["provider"]),
            hub=str(vad.get("hub", "ms")),
            model=str(vad["model"]),
            device=str(vad["device"]),
            cpu_threads=int(vad["cpu_threads"]),
            max_single_segment_s=float(vad["max_single_segment_s"]),
            merge_gap_s=float(vad["merge_gap_s"]),
            min_bridge_s=float(vad["min_bridge_s"]),
            min_region_s=float(vad["min_region_s"]),
            window_max_s=float(vad["window_max_s"]),
        ),
        audio_events=AudioEventConfig(
            tagger=str(audio_events["tagger"]),
            top_k=int(audio_events["top_k"]),
            min_confidence=float(audio_events["min_confidence"]),
            relative_floor=float(audio_events["relative_floor"]),
            window_s=float(audio_events["window_s"]),
            model_path=str(audio_events.get("model_path") or ""),
        ),
        visual_scan=VisualScanConfig(
            scan_fps=float(visual_scan["scan_fps"]),
            scan_width=int(visual_scan["scan_width"]),
            scene_threshold=float(visual_scan["scene_threshold"]),
            min_interval_s=float(visual_scan["min_interval_s"]),
            max_interval_s=float(visual_scan["max_interval_s"]),
            boundary_offset_s=float(visual_scan["boundary_offset_s"]),
            phash_distance=int(visual_scan["phash_distance"]),
            max_frames=int(visual_scan["max_frames"]),
            workers=int(visual_scan["workers"]),
            settle_step_s=float(visual_scan.get("settle_step_s", 0.25)),
            settle_max_s=float(visual_scan.get("settle_max_s", 1.5)),
        ),
        ocr=OCRConfig(
            enabled=bool(ocr["enabled"]),
            engine=str(ocr["engine"]),
            scan_fps=float(ocr["scan_fps"]),
            scan_width=int(ocr["scan_width"]),
            workers=int(ocr["workers"]),
            change_threshold=float(ocr["change_threshold"]),
            boundary_offset_s=float(ocr["boundary_offset_s"]),
            min_confidence=float(ocr["min_confidence"]),
            max_hold_s=float(ocr["max_hold_s"]),
            same_text_ratio=float(ocr["same_text_ratio"]),
            rejoin_gap_s=float(ocr["rejoin_gap_s"]),
            det_model_path=_model_path(ocr.get("det_model_path"), config_path),
            rec_model_path=_model_path(ocr.get("rec_model_path"), config_path),
            rec_keys_path=_model_path(ocr.get("rec_keys_path"), config_path),
            recognition_width=int(ocr.get("recognition_width", 1280)),
            refresh_s=float(ocr.get("refresh_s", 4.0)),
            singleton_confidence=float(ocr.get("singleton_confidence", 0.95)),
            min_text_height=float(ocr.get("min_text_height", 0.025)),
            det_model_config_path=_model_path(ocr.get("det_model_config_path"), config_path),
            fallback_rec_model_path=_model_path(ocr.get("fallback_rec_model_path"), config_path),
            fallback_rec_keys_path=_model_path(ocr.get("fallback_rec_keys_path"), config_path),
            use_angle_cls=bool(ocr.get("use_angle_cls", False)),
            rec_by_language={
                str(language).lower(): (
                    _model_path(entry.get("model"), config_path),
                    _model_path(entry.get("keys"), config_path),
                )
                for language, entry in (ocr.get("rec_by_language") or {}).items()
                if isinstance(entry, dict)
            },
        ),
        frames=FramesConfig(
            index_montage_n=int(frames["index_montage_n"]),
            baseline_full_interval_s=float(frames["baseline_full_interval_s"]),
            baseline_full_montage_n=int(frames["baseline_full_montage_n"]),
        ),
        montage=MontageConfig(
            width=int(montage["width"]),
            height=int(montage["height"]),
            jpeg_quality=int(montage["jpeg_quality"]),
        ),
        vision_llm=VisionLLMConfig(
            base_url_env=str(vision_llm["base_url_env"]),
            model_env=str(vision_llm["model_env"]),
            api_key_env=str(vision_llm["api_key_env"]),
            input_cost_env=str(vision_llm["input_cost_env"]),
            cached_input_cost_env=str(vision_llm["cached_input_cost_env"]),
            output_cost_env=str(vision_llm["output_cost_env"]),
            timeout_s=float(vision_llm["timeout_s"]),
            max_retries=int(vision_llm["max_retries"]),
            max_output_tokens=int(vision_llm["max_output_tokens"]),
            input_cost_per_million_usd=_rate(vision_llm["input_cost_env"]),
            cached_input_cost_per_million_usd=_rate(vision_llm["cached_input_cost_env"]),
            output_cost_per_million_usd=_rate(vision_llm["output_cost_env"]),
            temperature=(
                float(vision_llm["temperature"])
                if vision_llm.get("temperature") is not None
                else None
            ),
        ),
        advanced_asr=AdvancedASRConfig(
            provider=str(advanced_asr["provider"]),
            local=LocalASRConfig(
                model=str(local_asr["model"]),
                device=str(local_asr["device"]),
                compute_type=str(local_asr["compute_type"]),
                cpu_threads=int(local_asr["cpu_threads"]),
                max_segment_s=float(local_asr["max_segment_s"]),
            ),
            cloud=CloudASRConfig(
                max_segment_s=float(cloud_asr["max_segment_s"]),
            ),
        ),
        agent=AgentConfig(
            query=str(agent["query"]),
            max_tool_calls=int(agent["max_tool_calls"]),
            wall_clock_s=float(agent["wall_clock_s"]),
        ),
        view_frames=ViewFramesConfig(
            fps_options=tuple(float(value) for value in view_frames["fps_options"]),
            grid_options=tuple(int(value) for value in view_frames["grid_options"]),
            max_montages_per_call=int(view_frames["max_montages_per_call"]),
        ),
        transcribe_segment=TranscribeSegmentConfig(
            max_duration_s=float(transcribe_segment["max_duration_s"]),
        ),
    )
    if result.run.mode not in {"agentic", "baseline_full", "baseline_index_only"}:
        raise ValueError("run.mode must be agentic, baseline_full, or baseline_index_only")
    if result.frames.index_montage_n < 1 or result.frames.baseline_full_montage_n < 1:
        raise ValueError("frame montage grid sizes must be positive")
    if result.vad.window_max_s <= 0 or result.vad.max_single_segment_s <= 0:
        raise ValueError("vad window lengths must be positive")
    if result.vad.merge_gap_s < 0 or result.vad.min_bridge_s < 0:
        raise ValueError("vad merge tolerances must not be negative")
    if result.visual_scan.max_interval_s <= 0 or result.visual_scan.min_interval_s <= 0:
        raise ValueError("visual_scan sampling intervals must be positive")
    if result.visual_scan.min_interval_s > result.visual_scan.max_interval_s:
        raise ValueError("visual_scan.min_interval_s must not exceed max_interval_s")
    if result.audio_events.tagger not in {"panns", "sensevoice", "none"}:
        raise ValueError("audio_events.tagger must be panns, sensevoice, or none")
    if not 0 <= result.audio_events.relative_floor <= 1:
        raise ValueError("audio_events.relative_floor must be between 0 and 1")
    if result.audio_events.window_s <= 0:
        raise ValueError("audio_events.window_s must be positive")
    if result.ocr.scan_fps <= 0 or result.ocr.workers < 1:
        raise ValueError("ocr.scan_fps must be positive and ocr.workers at least 1")
    if min(result.ocr.scan_width, result.ocr.recognition_width) < 32:
        raise ValueError("ocr widths must be at least 32")
    if result.ocr.refresh_s <= 0 or result.ocr.max_hold_s <= 0 or result.ocr.rejoin_gap_s < 0:
        raise ValueError("ocr refresh/hold must be positive and rejoin gap nonnegative")
    for value in (
        result.ocr.min_confidence,
        result.ocr.singleton_confidence,
        result.ocr.same_text_ratio,
        result.ocr.change_threshold,
        result.ocr.min_text_height,
    ):
        if not 0 <= value <= 1:
            raise ValueError("ocr confidence, similarity and change thresholds must be in [0, 1]")
    if result.montage.width < 1 or result.montage.height < 1:
        raise ValueError("montage dimensions must be positive")
    if not result.view_frames.fps_options or any(
        value <= 0 for value in result.view_frames.fps_options
    ):
        raise ValueError("tools.view_frames.fps_options must contain positive values")
    if not result.view_frames.grid_options or any(
        value < 1 for value in result.view_frames.grid_options
    ):
        raise ValueError("tools.view_frames.grid_options must contain positive integers")
    grids = (
        *result.view_frames.grid_options,
        result.frames.index_montage_n,
        result.frames.baseline_full_montage_n,
    )
    if any(result.montage.width % n or result.montage.height % n for n in grids):
        raise ValueError("montage dimensions must be divisible by every configured grid size")
    if result.view_frames.max_montages_per_call < 1:
        raise ValueError("tools.view_frames.max_montages_per_call must be positive")
    if result.transcribe_segment.max_duration_s <= 0:
        raise ValueError("tools.transcribe_segment.max_duration_s must be positive")
    if result.advanced_asr.provider not in {"local", "cloud"}:
        raise ValueError("asr.advanced.provider must be local or cloud")
    if (
        result.vision_llm.input_cost_per_million_usd < 0
        or result.vision_llm.cached_input_cost_per_million_usd < 0
        or result.vision_llm.output_cost_per_million_usd < 0
    ):
        raise ValueError("vision_llm token costs must not be negative")
    if result.vision_llm.temperature is not None and not (
        0.0 <= result.vision_llm.temperature <= 2.0
    ):
        raise ValueError("vision_llm temperature must be between 0.0 and 2.0")
    return result
