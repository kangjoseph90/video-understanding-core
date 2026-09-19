"""The worker handshake, exercised for real.

A field mismatch between the parent and the worker is invisible from the
inside: the engine fails to start, the pipeline degrades as designed, and the
index is written with an empty text half. That is exactly what happened when
the worker read one setting with `pop` -- every video indexed cleanly and
silently had no on-screen text at all. Nothing short of starting the process
catches it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from vuc.config import load_config
from vuc.ocr import IsolatedOCREngine, OCRError

pytest.importorskip("rapidocr_onnxruntime")


def test_the_worker_starts_and_reads_a_frame(tmp_path: Path) -> None:
    config = load_config(Path(__file__).parents[1] / "config.yaml").ocr
    if not Path(config.rec_model_path or "/nonexistent").exists():
        pytest.skip("run scripts/fetch_ocr_models.py first")

    image = Image.new("RGB", (480, 120), color=(255, 255, 255))
    ImageDraw.Draw(image).text((20, 40), "HELLO", fill=(0, 0, 0))
    frame = tmp_path / "frame.jpg"
    image.save(frame)

    engine = IsolatedOCREngine(config)
    try:
        read = engine.read_many([frame, frame])
        cropped = engine.read_many([frame, frame], cropped=True)
    finally:
        engine.close()

    assert len(read) == 2
    assert len(cropped) == 2
    assert any("HELLO" in line.text.upper() for line in read[0])
    assert any("HELLO" in line.text.upper() for line in cropped[0])


def test_a_worker_that_cannot_start_says_so(tmp_path: Path) -> None:
    from dataclasses import replace

    config = load_config(Path(__file__).parents[1] / "config.yaml").ocr
    broken = replace(config, rec_model_path=str(tmp_path / "missing.onnx"))

    with pytest.raises(OCRError, match="OCR worker failed to start"):
        IsolatedOCREngine(broken)
