from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def write(
        self,
        *,
        step: str,
        event: str,
        arguments: dict[str, Any],
        result_summary: dict[str, Any],
        duration_ms: int,
    ) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "step": step,
            "event": event,
            "arguments": arguments,
            "result_summary": result_summary,
            "token_usage": None,
            "duration_ms": duration_ms,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
