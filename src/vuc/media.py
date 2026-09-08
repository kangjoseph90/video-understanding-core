from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path


class MediaError(RuntimeError):
    pass


def require_binary(name: str) -> str:
    binary = shutil.which(name)
    if binary is None:
        raise MediaError(f"required executable not found on PATH: {name}")
    return binary


def probe_duration(video_path: Path) -> float:
    command = [
        require_binary("ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise MediaError(f"ffprobe failed: {completed.stderr.strip()}")
    try:
        return float(json.loads(completed.stdout)["format"]["duration"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MediaError("ffprobe returned no valid duration") from exc


def extract_audio(video_path: Path, audio_path: Path) -> None:
    command = [
        require_binary("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(audio_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise MediaError(f"audio extraction failed: {completed.stderr.strip()}")

