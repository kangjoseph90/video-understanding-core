"""Stand-ins for the models the index calls, so tests stay hermetic and fast.

The real components here are a 300MB AudioSet checkpoint, an ONNX OCR bundle
and two funasr models. Loading any of them in a unit test trades seconds for
nothing: what is under test is the routing and the rendering, not whether
somebody else's network trained correctly.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from vuc.audio import EventTag
from vuc.models import Segment
from vuc.ocr import OCRLine
from vuc.timeline import Span


class StubVAD:
    """Returns whatever speech spans the test says the detector found."""

    name = "stub"

    def __init__(self, spans: Sequence[tuple[float, float]] = ()) -> None:
        self.spans = [Span(start, end) for start, end in spans]
        self.calls = 0

    def detect(self, audio_path: Path, *, duration_s: float) -> list[Span]:
        del audio_path, duration_s
        self.calls += 1
        return list(self.spans)


class StubTranscriber:
    """Records every clip it is handed, so callers can assert on where it ran."""

    def __init__(self, segments: Sequence[Segment] | None = None) -> None:
        self.segments = list(segments) if segments is not None else None
        self.calls: list[float] = []

    def transcribe(self, audio_path: Path, duration_s: float) -> list[Segment]:
        assert audio_path.exists()
        self.calls.append(duration_s)
        if self.segments is not None:
            return list(self.segments)
        return [Segment(0.0, duration_s, "spoken words", "en")]


class StubTagger:
    name = "stub_tagger"

    def __init__(self, tags: Sequence[EventTag] = ()) -> None:
        self.tags = list(tags) or [EventTag("Applause", 0.8)]
        self.calls: list[float] = []

    def tag(self, clip: Path, *, duration_s: float) -> list[EventTag]:
        assert clip.exists()
        self.calls.append(duration_s)
        return list(self.tags)


class StubOCREngine:
    name = "stub_ocr"

    def __init__(self, lines: Sequence[OCRLine] = ()) -> None:
        self.lines = list(lines)
        self.reads: list[str] = []

    def read_many(self, image_paths: Sequence[Path]) -> list[list[OCRLine]]:
        self.reads.extend(path.name for path in image_paths)
        return [list(self.lines) for _ in image_paths]


def write_config(tmp_path: Path, *replacements: tuple[str, str]) -> Path:
    """The shipped config with the cache redirected, plus any test overrides."""
    text = (Path(__file__).parents[1] / "config.yaml").read_text(encoding="utf-8")
    text = text.replace("directory: .vuc-cache", f"directory: {tmp_path / 'cache'}")
    for old, new in replacements:
        assert old in text, f"config.yaml no longer contains {old!r}"
        text = text.replace(old, new)
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path
