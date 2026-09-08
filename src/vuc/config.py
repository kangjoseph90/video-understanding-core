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
    short_threshold_s: float
    allowed_extensions: tuple[str, ...]


@dataclass(frozen=True)
class IndexerConfig:
    backend: str
    hub: str
    model: str
    vad_model: str
    device: str
    cpu_threads: int
    batch_size_s: int
    max_segment_s: int
    merge_vad: bool
    merge_length_s: int
    initial_prompt_tokens: int


@dataclass(frozen=True)
class FramesConfig:
    initial_interval_s: float
    initial_resolution: int
    baseline_interval_s: float
    baseline_resolution: int
    montage_enabled: bool
    montage_grid: str
    jpeg_quality: int

    @property
    def montage_shape(self) -> tuple[int, int]:
        try:
            columns, rows = (int(part) for part in self.montage_grid.lower().split("x", 1))
        except (TypeError, ValueError) as exc:
            raise ValueError("frames.montage_grid must look like '3x3'") from exc
        if columns < 1 or rows < 1:
            raise ValueError("frames.montage_grid values must be positive")
        return columns, rows


@dataclass(frozen=True)
class VisionLLMConfig:
    base_url_env: str
    model_env: str
    api_key_env: str
    max_images_per_request: int
    max_image_resolution: int
    timeout_s: float
    max_retries: int
    max_output_tokens: int
    input_cost_per_million_usd: float
    output_cost_per_million_usd: float


@dataclass(frozen=True)
class LocalASRConfig:
    backend: str
    model: str
    device: str
    compute_type: str
    cpu_threads: int
    max_segment_s: float


@dataclass(frozen=True)
class CloudASRConfig:
    base_url_env: str
    model_env: str
    api_key_env: str
    timeout_s: float
    max_retries: int
    concurrency: int
    max_segment_s: float
    cost_per_minute_usd: float


@dataclass(frozen=True)
class AdvancedASRConfig:
    provider: str
    local: LocalASRConfig
    cloud: CloudASRConfig


@dataclass(frozen=True)
class AgentConfig:
    query: str
    max_tool_calls: int
    max_input_tokens: int
    wall_clock_s: float
    calibration_enabled: bool
    calibration_segments: int
    calibration_segment_s: float
    verify_overlap: float


@dataclass(frozen=True)
class ViewFramesConfig:
    allowed_fps: tuple[float, ...]
    allowed_resolutions: tuple[int, ...]
    max_frames_per_call: int


@dataclass(frozen=True)
class AppConfig:
    path: Path
    cache: CacheConfig
    video: VideoConfig
    indexer: IndexerConfig
    frames: FramesConfig
    vision_llm: VisionLLMConfig
    advanced_asr: AdvancedASRConfig
    agent: AgentConfig
    view_frames: ViewFramesConfig
    raw: dict[str, Any]


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
    video = _section(data, "video")
    indexer = _section(data, "indexer")
    frames = _section(data, "frames")
    vision_llm = _section(data, "vision_llm")
    asr = _section(data, "asr")
    advanced_asr = _section(asr, "advanced")
    local_asr = _section(advanced_asr, "local")
    cloud_asr = _section(advanced_asr, "cloud")
    agent = _section(data, "agent")
    tools = _section(data, "tools")
    view_frames = _section(tools, "view_frames")
    cache_dir = Path(str(cache["directory"])).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = (config_path.parent / cache_dir).resolve()

    result = AppConfig(
        path=config_path,
        cache=CacheConfig(directory=cache_dir),
        video=VideoConfig(
            short_threshold_s=float(video["short_threshold_s"]),
            allowed_extensions=tuple(
                str(item).lower().lstrip(".") for item in video["allowed_extensions"]
            ),
        ),
        indexer=IndexerConfig(
            backend=str(indexer["backend"]),
            hub=str(indexer.get("hub", "ms")),
            model=str(indexer["model"]),
            vad_model=str(indexer["vad_model"]),
            device=str(indexer["device"]),
            cpu_threads=int(indexer["cpu_threads"]),
            batch_size_s=int(indexer["batch_size_s"]),
            max_segment_s=int(indexer["max_segment_s"]),
            merge_vad=bool(indexer["merge_vad"]),
            merge_length_s=int(indexer["merge_length_s"]),
            initial_prompt_tokens=int(indexer["initial_prompt_tokens"]),
        ),
        frames=FramesConfig(
            initial_interval_s=float(frames["initial_interval_s"]),
            initial_resolution=int(frames["initial_resolution"]),
            baseline_interval_s=float(frames["baseline_interval_s"]),
            baseline_resolution=int(frames["baseline_resolution"]),
            montage_enabled=bool(frames["montage_enabled"]),
            montage_grid=str(frames["montage_grid"]),
            jpeg_quality=int(frames["jpeg_quality"]),
        ),
        vision_llm=VisionLLMConfig(
            base_url_env=str(vision_llm["base_url_env"]),
            model_env=str(vision_llm["model_env"]),
            api_key_env=str(vision_llm["api_key_env"]),
            max_images_per_request=int(vision_llm["max_images_per_request"]),
            max_image_resolution=int(vision_llm["max_image_resolution"]),
            timeout_s=float(vision_llm["timeout_s"]),
            max_retries=int(vision_llm["max_retries"]),
            max_output_tokens=int(vision_llm["max_output_tokens"]),
            input_cost_per_million_usd=float(vision_llm["input_cost_per_million_usd"]),
            output_cost_per_million_usd=float(vision_llm["output_cost_per_million_usd"]),
        ),
        advanced_asr=AdvancedASRConfig(
            provider=str(advanced_asr["provider"]),
            local=LocalASRConfig(
                backend=str(local_asr["backend"]),
                model=str(local_asr["model"]),
                device=str(local_asr["device"]),
                compute_type=str(local_asr["compute_type"]),
                cpu_threads=int(local_asr["cpu_threads"]),
                max_segment_s=float(local_asr["max_segment_s"]),
            ),
            cloud=CloudASRConfig(
                base_url_env=str(cloud_asr["base_url_env"]),
                model_env=str(cloud_asr["model_env"]),
                api_key_env=str(cloud_asr["api_key_env"]),
                timeout_s=float(cloud_asr["timeout_s"]),
                max_retries=int(cloud_asr["max_retries"]),
                concurrency=int(cloud_asr["concurrency"]),
                max_segment_s=float(cloud_asr["max_segment_s"]),
                cost_per_minute_usd=float(cloud_asr["cost_per_minute_usd"]),
            ),
        ),
        agent=AgentConfig(
            query=str(agent["query"]),
            max_tool_calls=int(agent["max_tool_calls"]),
            max_input_tokens=int(agent["max_input_tokens"]),
            wall_clock_s=float(agent["wall_clock_s"]),
            calibration_enabled=bool(agent["calibration_enabled"]),
            calibration_segments=int(agent["calibration_segments"]),
            calibration_segment_s=float(agent["calibration_segment_s"]),
            verify_overlap=float(agent["verify_overlap"]),
        ),
        view_frames=ViewFramesConfig(
            allowed_fps=tuple(float(value) for value in view_frames["allowed_fps"]),
            allowed_resolutions=tuple(int(value) for value in view_frames["allowed_resolutions"]),
            max_frames_per_call=int(view_frames["max_frames_per_call"]),
        ),
        raw=data,
    )
    _ = result.frames.montage_shape
    if result.advanced_asr.provider not in {"local", "cloud"}:
        raise ValueError("asr.advanced.provider must be local or cloud")
    if not 0 <= result.agent.verify_overlap <= 1:
        raise ValueError("agent.verify_overlap must be between 0 and 1")
    if (
        result.vision_llm.input_cost_per_million_usd < 0
        or result.vision_llm.output_cost_per_million_usd < 0
    ):
        raise ValueError("vision_llm token costs must not be negative")
    return result
