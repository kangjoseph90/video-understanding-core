from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def new_run_id(label: str) -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{timestamp}-{label}-{uuid.uuid4().hex[:8]}"


class VideoCache:
    def __init__(self, root: Path, video_hash: str) -> None:
        self.root = root / video_hash
        self.audio_path = self.root / "audio.wav"
        self.index_json_path = self.root / "index.json"
        self.index_text_path = self.root / "index.txt"
        self.frames_dir = self.root / "frames"
        self.montages_dir = self.root / "montages"
        self.tool_frames_dir = self.root / "tool_frames"
        self.advanced_asr_dir = self.root / "advanced_asr"
        self.runs_dir = self.root / "runs"

    def ensure(self) -> None:
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.montages_dir.mkdir(parents=True, exist_ok=True)
        self.tool_frames_dir.mkdir(parents=True, exist_ok=True)
        self.advanced_asr_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        directory = self.runs_dir / run_id
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def write_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
