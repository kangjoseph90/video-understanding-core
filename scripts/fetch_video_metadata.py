"""Save a compact metadata sidecar next to each downloaded video.

yt-dlp's own JSON is hundreds of kilobytes of format lists; the pipeline only
needs what can be used as a transcription and reporting hint. The sidecar lives
at <video>.meta.json so run_video() can pick it up without being told where to
look.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import yaml

FIELDS = ("title", "channel", "uploader", "description", "duration", "language")


def _sidecar(info: dict[str, Any], *, video_id: str, language: str | None) -> dict[str, Any]:
    chapters = [
        {
            "start_s": int(chapter.get("start_time") or 0),
            "end_s": int(chapter.get("end_time") or 0),
            "title": str(chapter.get("title") or "").strip(),
        }
        for chapter in (info.get("chapters") or [])
    ]
    return {
        "video_id": video_id,
        "title": str(info.get("title") or "").strip(),
        "channel": str(info.get("channel") or info.get("uploader") or "").strip(),
        "description": str(info.get("description") or "").strip(),
        "duration_s": info.get("duration"),
        # yt-dlp reports the declared audio language, which is usually absent;
        # fall back to the manifest so the ASR language hint is still available.
        "language": info.get("language") or language,
        "chapters": chapters,
        "webpage_url": info.get("webpage_url"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="eval/youtube-field-eval.yaml")
    parser.add_argument("--video-dir", default="eval/videos")
    parser.add_argument(
        "--info-dir",
        help="reuse already-downloaded yt-dlp JSON instead of refetching",
    )
    args = parser.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    video_dir = Path(args.video_dir)
    info_dir = Path(args.info_dir) if args.info_dir else None

    for entry in manifest["videos"]:
        video_id = entry["id"]
        video = video_dir / entry["file"]
        if not video.is_file():
            print(f"!! no video file for {video_id}: {video}")
            continue

        raw: dict[str, Any] | None = None
        if info_dir:
            candidate = info_dir / f"{video.stem}.json"
            if candidate.exists():
                raw = json.loads(candidate.read_text(encoding="utf-8"))
        if raw is None:
            completed = subprocess.run(
                [
                    "yt-dlp",
                    "--no-playlist",
                    "--skip-download",
                    "--dump-single-json",
                    f"https://www.youtube.com/watch?v={video_id}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            raw = json.loads(completed.stdout)

        sidecar = _sidecar(raw, video_id=video_id, language=entry.get("language"))
        out = video.with_suffix(".meta.json")
        out.write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"{out.name}: {sidecar['language']} | {len(sidecar['chapters'])} chapters | "
            f"{len(sidecar['description'])} desc chars"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
