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
class AppConfig:
    path: Path
    cache: CacheConfig
    video: VideoConfig
    indexer: IndexerConfig
    frames: FramesConfig
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
        raw=data,
    )
    _ = result.frames.montage_shape
    return result
