from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol

from vuc.config import IndexerConfig
from vuc.models import Segment

TAG_PATTERN = re.compile(r"<\|([^|]+)\|>")
RICH_PREFIX_PATTERN = re.compile(r"(?:<\|[^|]+\|>)+")
LANGUAGES = {"zh", "en", "yue", "ja", "ko", "nospeech"}
EMOTIONS = {"happy", "sad", "angry", "neutral", "emo_unk", "emo_unknown"}
NON_EVENTS = LANGUAGES | EMOTIONS | {"withitn", "woitn"}


class Transcriber(Protocol):
    def transcribe(self, audio_path: Path) -> list[Segment]: ...


def parse_rich_text(raw_text: str) -> tuple[str, str, tuple[str, ...], tuple[str, ...]]:
    tags = tuple(match.group(1) for match in TAG_PATTERN.finditer(raw_text))
    normalized = tuple(tag.lower() for tag in tags)
    language = next((tag for tag in normalized if tag in LANGUAGES), "unknown")
    emotions = tuple(tag for tag in normalized if tag in EMOTIONS)
    audio_events = tuple(
        original
        for original, lower in zip(tags, normalized, strict=True)
        if lower not in NON_EVENTS
    )
    text = TAG_PATTERN.sub("", raw_text).strip()
    return text, language, emotions, audio_events


def _as_items(result: Any) -> Iterable[dict[str, Any]]:
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                yield item
    elif isinstance(result, dict):
        yield result


def _split_rich_segments(text: str) -> list[str]:
    matches = list(RICH_PREFIX_PATTERN.finditer(text))
    return [
        text[match.start() : matches[index + 1].start()].strip()
        if index + 1 < len(matches)
        else text[match.start() :].strip()
        for index, match in enumerate(matches)
    ]


def normalize_funasr_result(result: Any, duration_s: float) -> list[Segment]:
    segments: list[Segment] = []
    for item in _as_items(result):
        sentence_info = item.get("sentence_info")
        if isinstance(sentence_info, list) and sentence_info:
            rich_segments = _split_rich_segments(str(item.get("text") or ""))
            for index, sentence in enumerate(sentence_info):
                if not isinstance(sentence, dict):
                    continue
                rich_text = rich_segments[index] if index < len(rich_segments) else ""
                raw = str(rich_text or sentence.get("raw_text") or sentence.get("text") or "")
                text, language, emotions, events = parse_rich_text(raw)
                segments.append(
                    Segment(
                        start=float(sentence.get("start", 0)) / 1000,
                        end=float(sentence.get("end", 0)) / 1000,
                        text=text,
                        language=language,
                        emotions=emotions,
                        audio_events=events,
                        raw_text=raw,
                    )
                )
            continue

        raw = str(item.get("raw_text") or item.get("text") or "")
        text, language, emotions, events = parse_rich_text(raw)
        segments.append(
            Segment(
                start=0.0,
                end=duration_s,
                text=text,
                language=language,
                emotions=emotions,
                audio_events=events,
                raw_text=raw,
            )
        )
    return segments


class SenseVoiceTranscriber:
    def __init__(self, config: IndexerConfig, duration_s: float) -> None:
        self.config = config
        self.duration_s = duration_s
        try:
            from funasr import AutoModel
        except ImportError as exc:
            raise RuntimeError(
                "SenseVoice dependencies are not installed. Run `uv sync --extra sensevoice`."
            ) from exc

        self._model = AutoModel(
            model=config.model,
            hub=config.hub,
            vad_model=config.vad_model,
            vad_kwargs={"max_single_segment_time": config.max_segment_s * 1000},
            device=config.device,
            ncpu=config.cpu_threads,
            disable_update=True,
        )

    def transcribe(self, audio_path: Path) -> list[Segment]:
        result = self._model.generate(
            input=str(audio_path),
            cache={},
            language="auto",
            use_itn=True,
            batch_size_s=self.config.batch_size_s,
            merge_vad=self.config.merge_vad,
            merge_length_s=self.config.merge_length_s,
            sentence_timestamp=True,
            return_raw_text=True,
        )
        return normalize_funasr_result(result, self.duration_s)
