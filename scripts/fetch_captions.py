"""Save the channel-provided caption track next to each video.

Deliberately avoids yt-dlp's ``--write-sub`` / ``--write-auto-sub`` flags. When
a language has a human track, asking for the automatic one returns the human
one, so the convenience flags cannot tell the two slots apart -- across the
field-eval manifest they handed back byte-identical files and made one source
look like two that agreed. The subtitle dictionaries in the info JSON name the
real URLs, so this reads those directly.

Only the original-language track is taken. A translated track keeps the audio's
timing and would otherwise pass every alignment check while saying something
the video never said.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vuc.captions import AUTO, MANUAL, content_hash, parse_track, sidecar_path  # noqa: E402

PREFERRED_FORMATS = ("json3", "vtt")


def _pick(tracks: dict[str, Any], language: str) -> list[dict[str, Any]] | None:
    """The track for this language, trying the bare tag before the regional one."""
    tag = language.lower().replace("_", "-")
    for key in (tag, tag.split("-")[0]):
        for available, formats in tracks.items():
            if str(available).lower() == key and formats:
                return formats
    return None


def _fetch(formats: list[dict[str, Any]]) -> tuple[str, str] | None:
    for wanted in PREFERRED_FORMATS:
        for entry in formats:
            if str(entry.get("ext")) != wanted or not entry.get("url"):
                continue
            with urllib.request.urlopen(entry["url"], timeout=60) as response:
                return response.read().decode("utf-8"), wanted
    return None


def build_sidecar(info: dict[str, Any], language: str) -> dict[str, Any]:
    """Manual first, automatic only when no human track exists for the language."""
    manual = _pick(info.get("subtitles") or {}, language)
    automatic = _pick(info.get("automatic_captions") or {}, language)
    chosen, kind = (manual, MANUAL) if manual else (automatic, AUTO)

    track: dict[str, Any] | None = None
    if chosen:
        fetched = _fetch(chosen)
        if fetched is not None:
            payload, source_format = fetched
            cues, word_timed = parse_track(payload, source_format)
            if cues:
                track = {
                    "language": language,
                    "kind": kind,
                    "format": source_format,
                    "content_sha256": content_hash(cues),
                    "has_word_timing": word_timed,
                    "cues": [cue.to_dict() for cue in cues],
                }
    return {
        "video_id": info.get("id"),
        "fetched_at": datetime.now(UTC).isoformat(),
        "audio_language": language,
        # null records that the fetch ran and found nothing usable, which is a
        # different fact from the sidecar being absent.
        "track": track,
    }


def _info(video_id: str, info_dir: Path | None, stem: str) -> dict[str, Any]:
    if info_dir:
        candidate = info_dir / f"{stem}.json"
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
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
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="eval/youtube-field-eval.yaml")
    parser.add_argument("--video-dir", default="eval/videos")
    parser.add_argument("--info-dir", help="reuse already-downloaded yt-dlp JSON")
    args = parser.parse_args()

    manifest = yaml.safe_load(Path(args.manifest).read_text(encoding="utf-8"))
    video_dir = Path(args.video_dir)
    info_dir = Path(args.info_dir) if args.info_dir else None

    for entry in manifest["videos"]:
        video = video_dir / entry["file"]
        if not video.is_file():
            print(f"!! no video file for {entry['id']}: {video}")
            continue
        language = entry.get("language")
        if not language:
            meta = video.with_suffix(".meta.json")
            if meta.is_file():
                language = json.loads(meta.read_text(encoding="utf-8")).get("language")
        if not language:
            print(f"!! no audio language for {entry['id']}; skipping")
            continue

        sidecar = build_sidecar(_info(entry["id"], info_dir, video.stem), str(language))
        out = sidecar_path(video)
        out.write_text(
            json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        track = sidecar["track"]
        if track is None:
            print(f"{out.name}: no {language} track")
        else:
            print(
                f"{out.name}: {track['kind']}/{track['format']} "
                f"{len(track['cues'])} cues word_timing={track['has_word_timing']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
