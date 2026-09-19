"""Run an indexing sweep over all videos in a field-eval manifest.

Downloads any missing manifest videos via yt-dlp, runs index_video() on each,
and prints timing and statistics.
"""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from vuc.config import load_config
from vuc.pipeline import index_video

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

print = functools.partial(print, flush=True)


def download_video(url: str, output_path: Path) -> bool:
    """Download video at <=720p using yt-dlp."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-warnings",
        "-f",
        "bv*[height<=720]+ba/b[height<=720]/b",
        "--merge-output-format",
        "mp4",
        "-o",
        str(output_path),
        url,
    ]
    print(f"--> Downloading {output_path.name} from {url}...")
    result = subprocess.run(command, check=False)
    return result.returncode == 0 and output_path.is_file()


def run_sweep(
    manifest_path: Path,
    video_dir: Path,
    config_path: Path,
    *,
    force: bool = False,
    out_path: Path | None = None,
) -> int:
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    config = load_config(config_path)
    entries = manifest.get("videos", [])
    print(f"=== Indexing Sweep: {len(entries)} videos ===")
    print(f"Manifest: {manifest_path}")
    print(f"Config:   {config_path}")
    print(f"Storage:  {video_dir}\n")

    results: list[dict[str, Any]] = []
    failures = 0

    for i, entry in enumerate(entries, 1):
        video_file = video_dir / entry["file"]
        title = entry.get("title", entry["id"])
        url = entry.get("url")
        print(f"[{i}/{len(entries)}] {entry['category']} - {title}")

        if not video_file.is_file():
            if url:
                success = download_video(url, video_file)
                if not success:
                    print(f"!! Failed to download {url}")
                    failures += 1
                    continue
            else:
                print(f"!! File missing and no URL for {entry['id']}: {video_file}")
                failures += 1
                continue

        start_time = time.monotonic()
        try:
            index, cache, hit, trace_path = index_video(
                video_file,
                config,
                force=force,
            )
            elapsed = time.monotonic() - start_time
            indexer_meta = index.indexer
            speech_segs = sum(1 for s in index.audio.segments if s.is_speech)
            non_speech_segs = sum(1 for s in index.audio.segments if not s.is_speech)
            text_cues = len(index.text.cues)
            montages = len(index.visual.montages)

            row = {
                "id": entry["id"],
                "category": entry["category"],
                "file": entry["file"],
                "duration_s": index.video.duration_s,
                "cache_hit": hit,
                "elapsed_s": round(elapsed, 2),
                "cold_total_s": indexer_meta.get("cold_total_s"),
                "cold_audio_s": indexer_meta.get("cold_audio_s"),
                "cold_visual_s": indexer_meta.get("cold_visual_s"),
                "cold_ocr_s": indexer_meta.get("cold_ocr_s"),
                "speech_segments": speech_segs,
                "non_speech_segments": non_speech_segs,
                "text_cues": text_cues,
                "montages": montages,
                "index_json": str(cache.index_json_path),
            }
            results.append(row)

            status = "CACHE HIT" if hit else f"INDEXED in {elapsed:.1f}s"
            print(
                f"    {status} | audio={speech_segs}sp/{non_speech_segs}nsp, "
                f"cues={text_cues}, montages={montages}"
            )
            if not hit:
                print(
                    f"    cold timings: total={row['cold_total_s']}s "
                    f"(audio={row['cold_audio_s']}s, visual={row['cold_visual_s']}s, "
                    f"ocr={row['cold_ocr_s']}s)"
                )

        except Exception as exc:
            failures += 1
            print(f"!! Indexing failed for {video_file.name}: {exc}")
            import traceback
            traceback.print_exc()

        print()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(results)} rows to {out_path}")

    # Summary table
    print("\n" + "=" * 80)
    header = (
        f"{'Category':<20} {'Duration':<9} {'Status':<10} "
        f"{'Speech':<8} {'OCR':<6} {'Cold Total':<10}"
    )
    print(header)
    print("-" * 80)
    for r in results:
        status_str = "Hit" if r["cache_hit"] else f"{r['elapsed_s']}s"
        cold_str = f"{r['cold_total_s']}s" if r["cold_total_s"] else "-"
        print(
            f"{r['category']:<22} {r['duration_s']:>6.1f}s  {status_str:<12} "
            f"{r['speech_segments']:>4} segs {r['text_cues']:>4} cues {cold_str:>8}"
        )
    print("=" * 80)

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an indexing sweep over manifest videos.")
    parser.add_argument("--manifest", default="eval/youtube-field-eval.yaml")
    parser.add_argument("--video-dir", default="eval/videos")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="eval/index-sweep-results.jsonl")
    parser.add_argument("--force", action="store_true", help="Force re-indexing even if cached")
    args = parser.parse_args()

    return run_sweep(
        Path(args.manifest),
        Path(args.video_dir),
        Path(args.config),
        force=args.force,
        out_path=Path(args.out),
    )


if __name__ == "__main__":
    raise SystemExit(main())
