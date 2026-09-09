from __future__ import annotations

import math
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from vuc.config import FramesConfig, MontageConfig
from vuc.media import MediaError, require_binary
from vuc.models import FrameArtifact


def format_timestamp(timestamp_s: float) -> str:
    total_seconds = max(0, int(timestamp_s))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


def format_instant(timestamp_s: float) -> str:
    """Every timestamp shown to the model is a bare second count."""
    return str(max(0, int(timestamp_s)))


def format_span(start_s: float, end_s: float) -> str:
    """Interval as bare second counts, matching format_instant."""
    return f"{max(0, int(start_s))}-{max(0, int(end_s))}"


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
    label = format_instant(timestamp_s)
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
    frames_config: FramesConfig,
    montage_config: MontageConfig,
) -> list[FrameArtifact]:
    cell_width, _ = montage_cell_size(
        montage_config.width,
        montage_config.height,
        frames_config.index_montage_n,
    )
    return extract_sampled_frames(
        video_path,
        output_dir,
        start_s=0,
        end_s=duration_s,
        fps=1 / frames_config.index_interval_s,
        resolution=cell_width,
        jpeg_quality=montage_config.jpeg_quality,
    )


def montage_cell_size(width: int, height: int, n: int) -> tuple[int, int]:
    if n < 1:
        raise ValueError("montage grid size must be positive")
    if width % n or height % n:
        raise ValueError("montage dimensions must be divisible by grid size")
    return width // n, height // n


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
    width: int,
    height: int,
    jpeg_quality: int,
) -> list[Path]:
    if not frames:
        return []
    if n < 1:
        raise ValueError("montage grid size must be positive")
    tile_width, tile_height = montage_cell_size(width, height, n)
    group_size = n * n
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_montage in output_dir.glob("montage-*.jpg"):
        old_montage.unlink()

    montages: list[Path] = []
    for group_index in range(math.ceil(len(frames) / group_size)):
        group = frames[group_index * group_size : (group_index + 1) * group_size]
        # Keep the requested cell size, but do not pay for empty cells in the final group.
        actual_n = math.ceil(math.sqrt(len(group)))
        canvas = Image.new(
            "RGB", (tile_width * actual_n, tile_height * actual_n), color=(18, 18, 18)
        )
        for cell_index, artifact in enumerate(group):
            with Image.open(artifact.path) as frame:
                tile = ImageOps.contain(
                    frame.convert("RGB"),
                    (tile_width, tile_height),
                    method=Image.Resampling.LANCZOS,
                )
            x = (cell_index % actual_n) * tile_width + (tile_width - tile.width) // 2
            y = (cell_index // actual_n) * tile_height + (tile_height - tile.height) // 2
            canvas.paste(tile, (x, y))
        montage_path = output_dir / f"montage-{group_index + 1:04d}.jpg"
        canvas.save(montage_path, "JPEG", quality=jpeg_quality, optimize=True)
        montages.append(montage_path)
    return montages
