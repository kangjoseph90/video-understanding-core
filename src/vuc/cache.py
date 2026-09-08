from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class VideoCache:
    def __init__(self, root: Path, video_hash: str) -> None:
        self.root = root / video_hash
        self.audio_path = self.root / "audio.wav"
        self.index_json_path = self.root / "index.json"
        self.index_text_path = self.root / "index.txt"
        self.frames_dir = self.root / "frames"
        self.montages_dir = self.root / "montages"
        self.trace_path = self.root / "trace.jsonl"

    def ensure(self) -> None:
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.montages_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, path: Path, value: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
