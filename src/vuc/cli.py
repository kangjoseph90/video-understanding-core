from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vuc.config import load_config
from vuc.pipeline import index_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vuc", description="Video Understanding Core")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="build the local video index")
    index_parser.add_argument("video", type=Path)
    index_parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    index_parser.add_argument("--force", action="store_true", help="ignore a cached index")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            config = load_config(args.config)
            index, cache, cache_hit = index_video(args.video, config, force=args.force)
            print(
                json.dumps(
                    {
                        "cache_hit": cache_hit,
                        "video_hash": index.video.sha256,
                        "duration_s": index.video.duration_s,
                        "segments": len(index.segments),
                        "frames": len(index.frames),
                        "montages": len(index.montages),
                        "index_json": str(cache.index_json_path),
                        "index_text": str(cache.index_text_path),
                        "trace": str(cache.trace_path),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return 0
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
