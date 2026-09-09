"""Run the three explicit modes over every video in a field-eval manifest.

Each video runs baseline_index_only -> agentic -> baseline_full so the later
modes reuse the index, montage and ASR caches the earlier ones built. Results
land in the per-video cache under runs/<run-id>/ and a summary row per run is
appended to the output JSONL as soon as that run finishes, so an interrupted
sweep keeps everything it already measured.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from vuc.config import load_config
from vuc.run import run_video

MODES = ("baseline_index_only", "agentic", "baseline_full")

SUMMARY_KEYS = (
    "duration_s",
    "latency_s",
    "index_wall_clock_s",
    "index_cold_s",
    "frames_wall_clock_s",
    "asr_wall_clock_s",
    "tool_wall_clock_s",
    "vlm_wall_clock_s",
    "tool_calls",
    "cumulative_input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "llm_calls",
    "llm_retries",
    "vlm_cost_usd",
    "index_cache_hit",
    "budget_stop_reason",
    "run_id",
)


def _videos(manifest: Path, video_dir: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    return [{**entry, "path": video_dir / entry["file"]} for entry in data["videos"]]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="eval/youtube-field-eval.yaml")
    parser.add_argument("--video-dir", default="eval/videos")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default="eval/field-eval-results.jsonl")
    parser.add_argument(
        "--videos",
        nargs="*",
        help="explicit video files, overriding manifest lookup",
    )
    args = parser.parse_args()

    base = load_config(args.config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.videos:
        targets = [(Path(path).name, Path(path)) for path in args.videos]
    else:
        targets = []
        for entry in _videos(Path(args.manifest), Path(args.video_dir)):
            if not entry["path"].is_file():
                print(f"!! missing video for {entry['id']}: {entry['path']}")
                continue
            targets.append((entry["category"], entry["path"]))

    print(f"{len(targets)} videos x {len(MODES)} modes = {len(targets) * len(MODES)} runs\n")
    failures = 0
    for label, video in targets:
        for mode in MODES:
            config = replace(base, run=replace(base.run, mode=mode))
            started = time.monotonic()
            print(f"--> {label} / {mode} ({video.name})", flush=True)
            try:
                report, markdown_path, _ = run_video(video, config)
            except Exception as exc:  # noqa: BLE001 - one bad run must not stop the sweep
                failures += 1
                print(f"    FAILED after {time.monotonic() - started:.1f}s: {exc}")
                traceback.print_exc()
                row = {
                    "label": label,
                    "video": video.name,
                    "mode": mode,
                    "error": str(exc),
                }
            else:
                meta = report["meta"]
                citations = [
                    citation
                    for section in report.get("sections", [])
                    for citation in section.get("citations", [])
                ]
                row = {
                    "label": label,
                    "video": video.name,
                    "mode": mode,
                    "title": report.get("title"),
                    "sections": len(report.get("sections", [])),
                    "citations": len(citations),
                    "markdown": str(markdown_path),
                    **{key: meta.get(key) for key in SUMMARY_KEYS},
                }
                print(
                    f"    done {meta['latency_s']:.1f}s  "
                    f"in={meta['cumulative_input_tokens']} "
                    f"cached={meta['cached_input_tokens']} "
                    f"out={meta['output_tokens']} "
                    f"tools={meta['tool_calls']} "
                    f"cost=${meta['vlm_cost_usd']}",
                    flush=True,
                )
            with out_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nwrote {out_path} ({failures} failed)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
