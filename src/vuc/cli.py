from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import IO, Any

from vuc.config import load_config
from vuc.pipeline import index_video
from vuc.run import run_video


@contextmanager
def result_channel() -> Iterator[IO[str]]:
    """Keep third-party chatter out of the CLI's JSON.

    funasr, panns_inference and their dependencies print banners on import and
    a progress bar on every forward pass, some of it from C. `vuc index | jq`
    is unusable if any of that lands on stdout, so the real stdout is set aside
    for the result and file descriptor 1 is pointed at stderr for the run.
    """
    channel = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    previous, sys.stdout = sys.stdout, sys.stderr
    try:
        yield channel
    finally:
        sys.stdout = previous
        channel.flush()
        channel.close()


def emit(channel: IO[str], payload: dict[str, Any]) -> None:
    channel.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vuc", description="Video Understanding Core")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="build the local video index")
    index_parser.add_argument("video", type=Path)
    index_parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    index_parser.add_argument("--force", action="store_true", help="ignore a cached index")

    run_parser = subparsers.add_parser("run", help="generate a video report")
    run_parser.add_argument("video", type=Path)
    run_parser.add_argument("--query")
    run_parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    run_parser.add_argument("--force-index", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        with result_channel() as channel:
            if args.command == "index":
                config = load_config(args.config)
                index, cache, cache_hit, trace_path = index_video(
                    args.video, config, force=args.force
                )
                emit(
                    channel,
                    {
                        "cache_hit": cache_hit,
                        "video_hash": index.video.sha256,
                        "duration_s": index.video.duration_s,
                        "segments": len(index.audio.segments),
                        "text_cues": len(index.text.cues),
                        "frames": len(index.visual.frames),
                        "montages": len(index.visual.montages),
                        "index_json": str(cache.index_json_path),
                        "audio_index": str(cache.audio_index_path),
                        "text_index": str(cache.text_index_path),
                        "trace": str(trace_path),
                    },
                )
                return 0
            if args.command == "run":
                config = load_config(args.config)
                report, markdown_path, json_path = run_video(
                    args.video,
                    config,
                    query=args.query,
                    force_index=args.force_index,
                )
                emit(
                    channel,
                    {
                        "mode": report["meta"]["mode"],
                        "markdown": str(markdown_path),
                        "json": str(json_path),
                        "trace": report["meta"]["trace"],
                        "meta": report["meta"],
                    },
                )
                return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
