"""RapidOCR in a process of its own.

onnxruntime and torch cannot both be unloaded from one interpreter on macOS:
at shutdown they abort with `recursive_mutex lock failed` and the process dies
with SIGABRT. The index was already written by then, so nothing was lost --
but `vuc index` exited 134, which to anything reading an exit code is a failed
run. Indexing needs both libraries at once, so the only real fix is to keep
them apart.

The protocol is one JSON request per line in, one JSON response per line out.
A request carries a batch of frames rather than one, because the reading is
thread-scalable and the round trip is not: eight threads take a frame from
139ms to 65ms. Real stdout is handed to the response channel and file
descriptor 1 is pointed at stderr, so anything the OCR stack decides to print
-- including from C -- cannot land in the middle of a reply.
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from functools import partial
from pathlib import Path


def main() -> int:
    channel = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def reply(payload: dict[str, object]) -> None:
        channel.write(json.dumps(payload, ensure_ascii=False) + "\n")
        channel.flush()

    from vuc.config import OCRConfig
    from vuc.ocr import RapidOCREngine

    try:
        received = json.loads(sys.stdin.readline())
        crop_workers = max(1, int(received.get("crop_workers", received.get("workers", 1))))
        settings = {key: value for key, value in received.items() if key != "crop_workers"}
        workers = max(1, int(settings.get("workers", 1)))
        threads = max(1, (os.cpu_count() or 1) // workers)
        engine = RapidOCREngine(OCRConfig(**settings), threads=threads)
    except Exception as exc:  # noqa: BLE001 - the parent decides what a failure costs
        reply({"error": f"{type(exc).__name__}: {exc}"})
        return 1
    reply({"ready": True})

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vuc-ocr")
    crop_engine = None
    crop_pool = None
    for line in sys.stdin:
        request = line.strip()
        if not request:
            continue
        try:
            payload = json.loads(request)
            paths = [Path(item) for item in payload["paths"]]
            cropped = bool(payload.get("cropped", False))
            selected_engine = engine
            selected_pool = pool
            if cropped and settings.get("device") in {"dml", "directml"}:
                if crop_engine is None:
                    cpu_settings = {**settings, "device": "cpu", "workers": crop_workers}
                    crop_threads = max(1, (os.cpu_count() or 1) // crop_workers)
                    crop_engine = RapidOCREngine(
                        OCRConfig(**cpu_settings),
                        threads=crop_threads,
                    )
                    crop_pool = ThreadPoolExecutor(
                        max_workers=crop_workers,
                        thread_name_prefix="vuc-ocr-crop",
                    )
                selected_engine = crop_engine
                assert crop_pool is not None
                selected_pool = crop_pool
            reader = partial(selected_engine.read, cropped=cropped)
            batch = list(selected_pool.map(reader, paths))
        except Exception as exc:  # noqa: BLE001 - one bad batch is not fatal
            reply({"error": f"{type(exc).__name__}: {exc}"})
            continue
        reply({"frames": [[asdict(item) for item in lines] for lines in batch]})
    pool.shutdown(wait=False)
    if crop_pool is not None:
        crop_pool.shutdown(wait=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
