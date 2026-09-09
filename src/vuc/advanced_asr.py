from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from vuc.config import AdvancedASRConfig


class AdvancedASRError(RuntimeError):
    """An advanced ASR provider could not complete a transcription."""


@dataclass(frozen=True)
class TranscriptSentence:
    start_s: float
    end_s: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ASRResult:
    text: str
    sentences: tuple[TranscriptSentence, ...]
    language: str | None
    provider: str
    model: str
    audio_duration_s: float
    processing_s: float
    cost_usd: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "sentences": [sentence.to_dict() for sentence in self.sentences],
            "language": self.language,
            "provider": self.provider,
            "model": self.model,
            "audio_duration_s": self.audio_duration_s,
            "processing_s": self.processing_s,
            "cost_usd": self.cost_usd,
        }


class AdvancedASRProvider(Protocol):
    name: str
    model_name: str
    max_segment_s: float

    def transcribe(
        self,
        audio_path: Path,
        *,
        audio_duration_s: float,
        language_hint: str | None,
    ) -> ASRResult: ...


class FasterWhisperProvider:
    name = "local"

    def __init__(self, config: AdvancedASRConfig) -> None:
        local = config.local
        self.model_name = local.model
        self.max_segment_s = local.max_segment_s
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise AdvancedASRError(
                "local advanced ASR dependencies are missing; run `uv sync --extra advanced-asr`"
            ) from exc
        self._model = WhisperModel(
            local.model,
            device=local.device,
            compute_type=local.compute_type,
            cpu_threads=local.cpu_threads,
        )

    def transcribe(
        self,
        audio_path: Path,
        *,
        audio_duration_s: float,
        language_hint: str | None,
    ) -> ASRResult:
        started = time.monotonic()
        segments, info = self._model.transcribe(
            str(audio_path),
            language=language_hint,
            vad_filter=False,
            beam_size=5,
        )
        sentences = tuple(
            TranscriptSentence(
                start_s=float(segment.start),
                end_s=float(segment.end),
                text=segment.text.strip(),
            )
            for segment in segments
            if segment.text.strip()
        )
        processing_s = time.monotonic() - started
        return ASRResult(
            text=" ".join(sentence.text for sentence in sentences),
            sentences=sentences,
            language=getattr(info, "language", language_hint),
            provider=self.name,
            model=self.model_name,
            audio_duration_s=audio_duration_s,
            processing_s=processing_s,
            cost_usd=None,
        )


class CloudASRStubProvider:
    name = "cloud"

    def __init__(self, config: AdvancedASRConfig) -> None:
        self.model_name = "openai-compatible-stub"
        self.max_segment_s = config.cloud.max_segment_s

    def transcribe(
        self,
        audio_path: Path,
        *,
        audio_duration_s: float,
        language_hint: str | None,
    ) -> ASRResult:
        del audio_path, audio_duration_s, language_hint
        raise AdvancedASRError(
            "cloud advanced ASR is configured but its M2 implementation is a stub; "
            "switch asr.advanced.provider to local"
        )


def create_advanced_asr(config: AdvancedASRConfig) -> AdvancedASRProvider:
    if config.provider == "local":
        return FasterWhisperProvider(config)
    if config.provider == "cloud":
        return CloudASRStubProvider(config)
    raise ValueError(f"unsupported advanced ASR provider: {config.provider}")
