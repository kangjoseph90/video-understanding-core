from __future__ import annotations

from dataclasses import dataclass
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
    vad_model: str
    device: str
    cpu_threads: int
    batch_size_s: int
    max_segment_s: int
    merge_vad: bool
    merge_length_s: int


@dataclass(frozen=True)
class FramesConfig:
    index_interval_s: float
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
    timeout_s: float
    max_retries: int
    max_output_tokens: int
    input_cost_per_million_usd: float
    output_cost_per_million_usd: float


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
    frames: FramesConfig
    montage: MontageConfig
    vision_llm: VisionLLMConfig
    advanced_asr: AdvancedASRConfig
    agent: AgentConfig
    view_frames: ViewFramesConfig
    transcribe_segment: TranscribeSegmentConfig


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
            vad_model=str(indexer["vad_model"]),
            device=str(indexer["device"]),
            cpu_threads=int(indexer["cpu_threads"]),
            batch_size_s=int(indexer["batch_size_s"]),
            max_segment_s=int(indexer["max_segment_s"]),
            merge_vad=bool(indexer["merge_vad"]),
            merge_length_s=int(indexer["merge_length_s"]),
        ),
        frames=FramesConfig(
            index_interval_s=float(frames["index_interval_s"]),
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
            timeout_s=float(vision_llm["timeout_s"]),
            max_retries=int(vision_llm["max_retries"]),
            max_output_tokens=int(vision_llm["max_output_tokens"]),
            input_cost_per_million_usd=float(vision_llm["input_cost_per_million_usd"]),
            output_cost_per_million_usd=float(vision_llm["output_cost_per_million_usd"]),
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
        raise ValueError(
            "run.mode must be agentic, baseline_full, or baseline_index_only"
        )
    if result.frames.index_montage_n < 1 or result.frames.baseline_full_montage_n < 1:
        raise ValueError("frame montage grid sizes must be positive")
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
    if any(
        result.montage.width % n or result.montage.height % n
        for n in grids
    ):
        raise ValueError("montage dimensions must be divisible by every configured grid size")
    if result.view_frames.max_montages_per_call < 1:
        raise ValueError("tools.view_frames.max_montages_per_call must be positive")
    if result.transcribe_segment.max_duration_s <= 0:
        raise ValueError("tools.transcribe_segment.max_duration_s must be positive")
    if result.advanced_asr.provider not in {"local", "cloud"}:
        raise ValueError("asr.advanced.provider must be local or cloud")
    if (
        result.vision_llm.input_cost_per_million_usd < 0
        or result.vision_llm.output_cost_per_million_usd < 0
    ):
        raise ValueError("vision_llm token costs must not be negative")
    return result
