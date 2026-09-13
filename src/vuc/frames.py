from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

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


def burn_in_timestamp(image_path: Path, timestamp_s: float, quality: int) -> None:
    """Stamp the second count onto the frame: it is how the model cites time."""
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


def extract_plain_frames(
    video_path: Path,
    output_dir: Path,
    *,
    prefix: str,
    fps: float,
    width: int,
    duration_s: float,
    indices: list[int] | None = None,
    first_center_s: float | None = None,
) -> list[tuple[float, Path]]:
    """A flat sampling of the video, unmarked, for something else to read.

    No timestamp is burned on: these frames are input to OCR, and stamping them
    first put our own second counter into the text index as on-screen text.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = None if indices is None else sorted(set(indices))
    if selected == []:
        return []
    # Align a dense clock with the centres of the coarse clock. FFmpeg's
    # nearest rounding otherwise samples 1fps around n+.5 but 4fps around
    # n+.125, changing even the supposedly unchanged base OCR frames.
    shift = 0.0 if first_center_s is None else first_center_s - 0.5 / fps
    filters = (
        f"fps={fps}"
        if first_center_s is None
        else f"setpts=PTS-({shift})/TB,fps={fps}:start_time=0"
    )
    if selected is not None:
        if selected[0] < 0:
            raise ValueError("frame indices must be nonnegative")
        runs: list[list[int]] = []
        for index in selected:
            if runs and index == runs[-1][1] + 1:
                runs[-1][1] = index
            else:
                runs.append([index, index])
        # A flat a+b+c+... expression hits libavutil's recursion limit on a
        # long video. A balanced tree has logarithmic depth.
        terms = [f"between(n,{a},{b})" for a, b in runs]
        while len(terms) > 1:
            terms = [
                f"({terms[i]}+{terms[i + 1]})" if i + 1 < len(terms) else terms[i]
                for i in range(0, len(terms), 2)
            ]
        filters += f",select='{terms[0]}'"
    filters += f",scale=w='min(iw,{width})':h=-2"
    command = [
        require_binary("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-fps_mode",
        "vfr",
        "-q:v",
        "3",
        str(output_dir / f"{prefix}-%06d.jpg"),
    ]
    # A long video's selection can exceed the OS argument-length limit.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".ffilter") as script:
        script.write(filters)
        script.flush()
        command[-1:-1] = ["-/filter:v", script.name]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise MediaError(f"frame extraction failed: {completed.stderr.strip()}")
    paths = sorted(output_dir.glob(f"{prefix}-*.jpg"))
    if selected is not None and len(paths) != len(selected):
        raise MediaError(f"requested {len(selected)} frames, extracted {len(paths)}")
    numbers = range(len(paths)) if selected is None else selected
    return [
        (min(round(index / fps + (first_center_s or 0), 3), duration_s), path)
        for index, path in zip(numbers, paths, strict=True)
    ]


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
        burn_in_timestamp(path, timestamp_s, jpeg_quality)
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
