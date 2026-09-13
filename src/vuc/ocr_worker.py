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
        settings = json.loads(sys.stdin.readline())
        workers = max(1, int(settings.get("workers", 1)))
        threads = max(1, (os.cpu_count() or 1) // workers)
        engine = RapidOCREngine(OCRConfig(**settings), threads=threads)
    except Exception as exc:  # noqa: BLE001 - the parent decides what a failure costs
        reply({"error": f"{type(exc).__name__}: {exc}"})
        return 1
    reply({"ready": True})

    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="vuc-ocr")
    for line in sys.stdin:
        request = line.strip()
        if not request:
            continue
        try:
            paths = [Path(item) for item in json.loads(request)["paths"]]
            batch = list(pool.map(engine.read, paths))
        except Exception as exc:  # noqa: BLE001 - one bad batch is not fatal
            reply({"error": f"{type(exc).__name__}: {exc}"})
            continue
        reply({"frames": [[asdict(item) for item in lines] for lines in batch]})
    pool.shutdown(wait=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
