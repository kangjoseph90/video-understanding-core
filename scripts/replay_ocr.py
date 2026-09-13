"""Replay saved OCR measurements without loading models or updating an index cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

from vuc.config import load_config
from vuc.index_text import render_text_index
from vuc.ocr import Observation, OCRLine, text_cues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    parser.add_argument("--duration", type=float, required=True, help="Video duration in seconds")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True, help="A new directory")
    args = parser.parse_args(argv)
    if args.duration <= 0:
        parser.error("--duration must be positive")
    if args.output_dir.exists():
        parser.error("--output-dir must not already exist; existing results are never overwritten")
    raw = args.observations.read_bytes()
    observations = [
        Observation(
            item["timestamp_s"],
            tuple(OCRLine(**row) for row in item["lines"]),
            verification=item.get("verification", False),
            discovery=item.get("discovery", False),
            regions=tuple(tuple(box) for box in item.get("regions", ())),
        )
        for item in (json.loads(row) for row in raw.splitlines() if row.strip())
    ]
    config = load_config(args.config).ocr
    began = time.perf_counter()
    cues = text_cues(observations, duration_s=args.duration, config=config)
    elapsed = time.perf_counter() - began
    rendered = render_text_index(cues)
    metrics = {
        "observations_sha256": hashlib.sha256(raw).hexdigest(),
        "observations": len(observations),
        "cues": len(cues),
        "rendered_rows": len(rendered.splitlines()),
        "postprocess_seconds": elapsed,
        "ocr_config": asdict(config),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "text_index.txt").write_text(rendered, encoding="utf-8")
    (args.output_dir / "text_cues.json").write_text(
        json.dumps([asdict(cue) for cue in cues], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"{len(cues)} cues, {len(rendered.splitlines())} rows, {elapsed:.3f}s → {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
