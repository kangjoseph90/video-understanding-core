from __future__ import annotations

import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from vuc.config import FramesConfig
from vuc.media import MediaError, require_binary
from vuc.models import FrameArtifact


def format_timestamp(timestamp_s: float) -> str:
    total_seconds = max(0, int(timestamp_s))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _burn_in(image_path: Path, timestamp_s: float, quality: int) -> None:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    label = format_timestamp(timestamp_s)
    font = _font(max(14, image.width // 22))
    box = draw.textbbox((0, 0), label, font=font, stroke_width=1)
    padding = max(4, image.width // 80)
    width = box[2] - box[0] + padding * 2
    height = box[3] - box[1] + padding * 2
    y = image.height - height - padding
    draw.rounded_rectangle(
        (padding, y, padding + width, y + height),
        radius=padding,
        fill=(0, 0, 0),
    )
    draw.text(
        (padding * 2, y + padding),
        label,
        fill=(255, 255, 255),
        font=font,
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    image.save(image_path, "JPEG", quality=quality, optimize=True)


def extract_initial_frames(
    video_path: Path,
    output_dir: Path,
    *,
    duration_s: float,
    config: FramesConfig,
) -> list[FrameArtifact]:
    return extract_sampled_frames(
        video_path,
        output_dir,
        start_s=0,
        end_s=duration_s,
        fps=1 / config.index_interval_s,
        resolution=config.index_resolution,
        jpeg_quality=config.jpeg_quality,
    )


def extract_sampled_frames(
    video_path: Path,
    output_dir: Path,
    *,
    start_s: float,
    end_s: float,
    fps: float,
    resolution: int,
    jpeg_quality: int,
) -> list[FrameArtifact]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in output_dir.glob("frame-*.jpg"):
        old_frame.unlink()
    scale = f"scale=w='if(gte(iw,ih),{resolution},-2)':h='if(gte(iw,ih),-2,{resolution})'"
    command = [
        require_binary("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-i",
        str(video_path),
        "-t",
        f"{end_s - start_s:.3f}",
        "-vf",
        f"fps=fps={fps}:start_time=0,{scale}",
        "-pix_fmt",
        "yuvj420p",
        "-q:v",
        "3",
        str(output_dir / "frame-%06d.jpg"),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise MediaError(f"frame extraction failed: {completed.stderr.strip()}")

    paths = sorted(output_dir.glob("frame-*.jpg"))
    artifacts: list[FrameArtifact] = []
    for index, path in enumerate(paths):
        timestamp_s = min(start_s + index / fps, end_s)
        _burn_in(path, timestamp_s, jpeg_quality)
        artifacts.append(FrameArtifact(path=str(path), timestamp_s=timestamp_s))
    return artifacts


def create_montages(
    frames: list[FrameArtifact],
    output_dir: Path,
    *,
    n: int,
    jpeg_quality: int,
) -> list[Path]:
    if not frames:
        return []
    if n < 1:
        raise ValueError("montage grid size must be positive")
    group_size = n * n
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_montage in output_dir.glob("montage-*.jpg"):
        old_montage.unlink()

    montages: list[Path] = []
    for group_index in range(math.ceil(len(frames) / group_size)):
        group = frames[group_index * group_size : (group_index + 1) * group_size]
        with Image.open(group[0].path) as first:
            tile_width, tile_height = first.size
        canvas = Image.new("RGB", (tile_width * n, tile_height * n), color=(18, 18, 18))
        for cell_index, artifact in enumerate(group):
            with Image.open(artifact.path) as frame:
                tile = frame.convert("RGB")
            x = (cell_index % n) * tile_width
            y = (cell_index // n) * tile_height
            canvas.paste(tile, (x, y))
        montage_path = output_dir / f"montage-{group_index + 1:04d}.jpg"
        canvas.save(montage_path, "JPEG", quality=jpeg_quality, optimize=True)
        montages.append(montage_path)
    return montages
