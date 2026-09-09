#!/usr/bin/env python3
"""Probe image and tool-calling support of an OpenAI-compatible vision LLM."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from itertools import cycle, islice
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv

TIMESTAMP_PATTERN = re.compile(r"(?<!\d)(\d{1,3}):(\d{2})(?!\d)")
VIEW_FRAMES_TOOL = {
    "type": "function",
    "function": {
        "name": "view_frames",
        "description": "View timestamped frames from a selected video interval.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_s": {"type": "number"},
                "end_s": {"type": "number"},
                "fps": {"type": "number", "exclusiveMinimum": 0},
                "n": {"type": "integer", "minimum": 1},
            },
            "required": ["start_s", "end_s", "fps", "n"],
            "additionalProperties": False,
        },
    },
}


def image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def extract_timestamps(text: str) -> list[str]:
    return [f"{int(minutes):02d}:{seconds}" for minutes, seconds in TIMESTAMP_PATTERN.findall(text)]


def positional_accuracy(actual: list[str], expected: list[str]) -> dict[str, Any]:
    correct = sum(
        actual[index] == value for index, value in enumerate(expected) if index < len(actual)
    )
    return {
        "expected": expected,
        "actual": actual,
        "correct": correct,
        "total": len(expected),
        "accuracy": correct / len(expected) if expected else 0.0,
    }


class VisionClient:
    def __init__(self, base_url: str, model: str, api_key: str, timeout_s: float) -> None:
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.client = httpx.Client(
            timeout=timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def request(
        self,
        *,
        prompt: str,
        images: list[Path],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], float]:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": image_data_url(path)}} for path in images
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": 512,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        started = time.monotonic()
        response = self.client.post(self.url, json=payload)
        latency_s = time.monotonic() - started
        response.raise_for_status()
        return response.json(), latency_s


def message_from(response: dict[str, Any]) -> dict[str, Any]:
    return response["choices"][0]["message"]


def usage_from(response: dict[str, Any]) -> dict[str, Any]:
    usage = response.get("usage")
    return usage if isinstance(usage, dict) else {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    load_dotenv(args.env, override=False)
    values = {key: os.environ.get(key, "") for key in ("VLM_BASE_URL", "VLM_MODEL", "VLM_API_KEY")}
    missing = [key for key, value in values.items() if not value]
    if missing:
        raise RuntimeError(f"missing environment variables: {', '.join(missing)}")

    index = json.loads(args.index_json.read_text(encoding="utf-8"))
    montage_paths = [Path(path) for path in index["montages"]]
    if not montage_paths:
        raise RuntimeError("index has no montage images")
    first_montage = montage_paths[0]
    expected = [
        f"{int(frame['timestamp_s']) // 60:02d}:{int(frame['timestamp_s']) % 60:02d}"
        for frame in index["frames"][:9]
    ]
    client = VisionClient(
        values["VLM_BASE_URL"],
        values["VLM_MODEL"],
        values["VLM_API_KEY"],
        args.timeout,
    )
    results: dict[str, Any] = {
        "model": values["VLM_MODEL"],
        "montage": str(first_montage),
        "timestamp_reading": {},
        "tool_calling": {},
        "image_count_limits": [],
    }

    try:
        response, latency = client.request(
            prompt=(
                "이 3x3 몽타주의 각 셀 좌하단 타임스탬프를 왼쪽 위부터 행 우선 순서로 "
                "나열하세요. 설명 없이 MM:SS 값만 쉼표로 구분하세요."
            ),
            images=[first_montage],
        )
        message = message_from(response)
        answer = str(message.get("content") or "")
        results["timestamp_reading"] = {
            "success": True,
            **positional_accuracy(extract_timestamps(answer), expected),
            "answer": answer,
            "latency_s": round(latency, 3),
            "usage": usage_from(response),
        }
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        results["timestamp_reading"] = {"success": False, "error": str(exc)}

    try:
        response, latency = client.request(
            prompt=(
                "이미지를 확인한 뒤 더 자세한 프레임 검사가 필요하므로 view_frames 도구를 "
                "0초부터 30초까지, 0.1fps, 3x3 몽타주로 호출하세요."
            ),
            images=[first_montage],
            tools=[VIEW_FRAMES_TOOL],
            tool_choice={"type": "function", "function": {"name": "view_frames"}},
        )
        message = message_from(response)
        tool_calls = message.get("tool_calls") or []
        results["tool_calling"] = {
            "success": bool(tool_calls),
            "tool_names": [call.get("function", {}).get("name") for call in tool_calls],
            "arguments": [call.get("function", {}).get("arguments") for call in tool_calls],
            "latency_s": round(latency, 3),
            "usage": usage_from(response),
        }
    except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        results["tool_calling"] = {"success": False, "error": str(exc)}

    for image_count in (1, 4, 8, 16):
        images = list(islice(cycle(montage_paths), image_count))
        record: dict[str, Any] = {"image_count": image_count}
        try:
            response, latency = client.request(
                prompt=(
                    f"첨부된 이미지 {image_count}장을 모두 확인하세요. 각 이미지를 확인했다면 "
                    "설명 없이 OK라고만 답하세요."
                ),
                images=images,
            )
            record.update(
                success=True,
                latency_s=round(latency, 3),
                answer=str(message_from(response).get("content") or ""),
                usage=usage_from(response),
            )
        except httpx.HTTPStatusError as exc:
            record.update(
                success=False,
                status_code=exc.response.status_code,
                error=exc.response.text[:1000],
            )
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            record.update(success=False, error=str(exc))
        results["image_count_limits"].append(record)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("index_json", type=Path)
    parser.add_argument("--env", type=Path, default=Path(".env"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".vuc-cache/measurements/vision-spike.json"),
    )
    parser.add_argument("--timeout", type=float, default=180)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        results = run(args)
    except (OSError, RuntimeError, httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
