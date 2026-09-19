from __future__ import annotations

from pathlib import Path

from vuc.config import OCRConfig
from vuc.index_text import render_text_index
from vuc.ocr import (
    Observation,
    OCRLine,
    normalize_text,
    ocr_candidates,
    rapidocr_options,
    same_text,
    signature_changed,
    text_cues,
)

CONFIG = OCRConfig(
    enabled=True,
    engine="rapidocr",
    scan_fps=1.0,
    scan_width=448,
    workers=2,
    change_threshold=0.06,
    boundary_offset_s=0.75,
    min_confidence=0.5,
    max_hold_s=30.0,
    same_text_ratio=0.8,
    rejoin_gap_s=4.0,
    det_model_path="",
    rec_model_path="",
    rec_keys_path="",
    singleton_confidence=0.85,
    min_text_height=0,
)


def line(text: str, *, top=0.9, bottom=0.95, left=0.3, right=0.7, confidence=0.9) -> OCRLine:
    return OCRLine(text=text, confidence=confidence, top=top, bottom=bottom, left=left, right=right)


def with_confirmations(observations: list[Observation]) -> list[Observation]:
    """Give layout/identity fixtures two distinct measurements per appearance."""
    return sorted(
        [
            sample
            for o in observations
            for sample in ((o, Observation(o.timestamp_s + 0.25, o.lines)) if o.lines else (o,))
        ],
        key=lambda o: o.timestamp_s,
    )


def test_position_is_a_coarse_ninth_of_the_frame() -> None:
    assert line("a", top=0.90, bottom=0.95, left=0.30, right=0.70).position == "bottom center"
    assert line("a", top=0.02, bottom=0.08, left=0.01, right=0.20).position == "top left"
    assert line("a", top=0.40, bottom=0.50, left=0.80, right=0.99).position == "middle right"


def test_size_comes_from_measured_height_only() -> None:
    assert line("a", top=0.90, bottom=0.93).size == "small"
    assert line("a", top=0.40, bottom=0.46).size == "medium"
    assert line("a", top=0.10, bottom=0.50).size == "large"


def test_a_bottom_line_is_not_labelled_a_subtitle() -> None:
    """Where text sat is measured; what kind of text it is would be a guess."""
    cues = text_cues(
        with_confirmations([Observation(4.0, (line("Hello"),))]), duration_s=60.0, config=CONFIG
    )

    assert cues[0].position == "bottom center"
    fields = cues[0].to_dict()
    assert set(fields) == {"start", "end", "text", "position", "size"}
    assert "subtitle" not in str(fields)


def test_the_same_line_on_consecutive_frames_is_one_cue() -> None:
    observations = [
        Observation(0.0, (line("What are they used for?", top=0.05, bottom=0.11),)),
        Observation(4.0, (line("What are they used for?", top=0.05, bottom=0.11),)),
        Observation(8.0, ()),
    ]

    cues = text_cues(observations, duration_s=60.0, config=CONFIG)

    assert len(cues) == 1
    assert (cues[0].start, cues[0].end) == (0.0, 8.0)


def test_a_cue_is_not_held_across_an_unobserved_stretch() -> None:
    """Nothing was looked at in between, so nothing may be claimed about it."""
    observations = [Observation(0.0, (line("Slide one"),)), Observation(400.0, ())]

    cues = text_cues(with_confirmations(observations), duration_s=500.0, config=CONFIG)

    assert cues[0].end == 30.25


def test_two_lines_in_different_places_are_two_cues() -> None:
    observations = [
        Observation(
            10.0,
            (
                line("Chapter 3", top=0.04, bottom=0.14, left=0.05, right=0.4),
                line("subtitle text", top=0.90, bottom=0.95),
            ),
        )
    ]

    cues = text_cues(with_confirmations(observations), duration_s=20.0, config=CONFIG)

    assert {(cue.text, cue.position, cue.size) for cue in cues} == {
        ("Chapter 3", "top left", "large"),
        ("subtitle text", "bottom center", "medium"),
    }


def test_readings_the_engine_was_unsure_of_are_discarded() -> None:
    observations = [
        Observation(
            0.0,
            (line("solid", confidence=0.9), line("guess", confidence=0.1)),
        )
    ]

    cues = text_cues(with_confirmations(observations), duration_s=10.0, config=CONFIG)

    assert [cue.text for cue in cues] == ["solid"]


def test_rapidocr_is_configured_with_flat_names_only() -> None:
    """The dotted spelling is accepted and then ignored, so it must not be used."""
    config = OCRConfig(
        enabled=True,
        engine="rapidocr",
        scan_fps=1.0,
        scan_width=448,
        workers=2,
        change_threshold=0.06,
        boundary_offset_s=0.75,
        min_confidence=0.5,
        max_hold_s=30.0,
        same_text_ratio=0.8,
        rejoin_gap_s=4.0,
        det_model_path="",
        rec_model_path="/models/korean_rec.onnx",
        rec_keys_path="/models/korean_dict.txt",
    )

    assert rapidocr_options(config) == {
        "rec_model_path": "/models/korean_rec.onnx",
        "rec_keys_path": "/models/korean_dict.txt",
    }


def test_rapidocr_options_sets_device_acceleration() -> None:
    from dataclasses import replace

    config_dml = replace(CONFIG, device="dml", rec_model_path="/models/rec.onnx")
    assert rapidocr_options(config_dml).get("use_dml") is True

    config_cuda = replace(CONFIG, device="cuda", rec_model_path="/models/rec.onnx")
    assert rapidocr_options(config_cuda).get("use_cuda") is True


def test_whitespace_is_collapsed_before_lines_are_compared() -> None:
    assert normalize_text("  What   are\nthey used for? ") == "What are they used for?"


def test_the_text_index_stands_alone() -> None:
    """Its own list, its own format. Nothing about the audio appears in it."""
    cues = text_cues(
        with_confirmations(
            [
                Observation(
                    0.0,
                    (line("Whatarethey usedfor?", top=0.04, bottom=0.09, left=0.02, right=0.30),),
                )
            ]
        ),
        duration_s=21.0,
        config=CONFIG,
    )

    assert render_text_index(cues) == "[0-21, top left, medium] Whatarethey usedfor?"


def test_engines_read_a_batch_at_a_time(tmp_path: Path) -> None:
    """The worker round trip is the expensive part, so frames go over in batches."""

    class Recorder:
        name = "recorder"

        def __init__(self) -> None:
            self.batches: list[int] = []

        def read_many(self, image_paths):
            self.batches.append(len(image_paths))
            return [[line("text")] for _ in image_paths]

    engine = Recorder()
    paths = [tmp_path / f"f{index}.jpg" for index in range(5)]

    assert len(engine.read_many(paths)) == 5
    assert engine.batches == [5]


def test_boxes_sharing_a_span_and_a_cell_become_one_entry() -> None:
    """A slide of headlines is one thing on screen, not twelve."""
    observations = [
        Observation(
            14.0,
            (
                line("New York City Schools Ban", top=0.04, bottom=0.09, left=0.02, right=0.30),
                line("ChatGPT Amid Cheating Worries", top=0.10, bottom=0.15, left=0.02, right=0.30),
                line("-CNET", top=0.05, bottom=0.09, left=0.40, right=0.55),
            ),
        ),
        Observation(25.0, ()),
    ]

    cues = text_cues(with_confirmations(observations), duration_s=60.0, config=CONFIG)

    assert [(cue.position, cue.text) for cue in cues] == [
        ("top center", "-CNET"),
        ("top left", "New York City Schools Ban / ChatGPT Amid Cheating Worries"),
    ]


def test_joined_text_keeps_reading_order() -> None:
    """Top to bottom, then left to right -- not whatever order the engine emitted."""
    observations = [
        Observation(
            0.0,
            (
                line("second", top=0.10, bottom=0.15, left=0.02, right=0.30),
                line("first", top=0.04, bottom=0.09, left=0.02, right=0.30),
            ),
        ),
        Observation(5.0, ()),
    ]

    cues = text_cues(with_confirmations(observations), duration_s=10.0, config=CONFIG)

    assert cues[0].text == "first / second"


def test_different_cells_stay_apart() -> None:
    observations = [
        Observation(
            0.0,
            (
                line("heading", top=0.04, bottom=0.20, left=0.02, right=0.30),
                line("caption", top=0.90, bottom=0.95, left=0.35, right=0.65),
            ),
        ),
        Observation(5.0, ()),
    ]

    cues = text_cues(with_confirmations(observations), duration_s=10.0, config=CONFIG)

    assert {(c.position, c.size, c.text) for c in cues} == {
        ("top left", "large", "heading"),
        ("bottom center", "medium", "caption"),
    }


def test_a_logo_read_slightly_differently_is_still_one_cue() -> None:
    """Eight minutes of one unchanging logo was six entries with six time ranges."""
    spellings = ["open culture", "open.cuiture", "opencuiture", "open cultune"]
    observations = [
        Observation(float(index * 20), (line(text, top=0.04, bottom=0.09, left=0.02, right=0.3),))
        for index, text in enumerate(spellings)
    ]
    observations.append(Observation(80.0, ()))

    cues = text_cues(observations, duration_s=120.0, config=CONFIG)

    assert len(cues) == 1
    assert (cues[0].start, cues[0].end) == (0.0, 80.0)


def test_different_text_in_the_same_place_is_not_merged() -> None:
    observations = [
        Observation(0.0, (line("INSPIRATION", top=0.4, bottom=0.55),)),
        Observation(20.0, (line("BARRIERS", top=0.4, bottom=0.55),)),
        Observation(40.0, ()),
    ]

    cues = text_cues(with_confirmations(observations), duration_s=60.0, config=CONFIG)

    assert sorted(cue.text for cue in cues) == ["BARRIERS", "INSPIRATION"]


def test_short_strings_are_never_matched_loosely() -> None:
    """At three characters everything resembles everything."""
    assert not same_text("CC", "GO", ratio=0.8)
    assert not same_text("TED", "TEA", ratio=0.8)
    assert same_text("CC", "cc", ratio=0.8)


def frame_with(tmp_path: Path, name: str, *, text: str) -> Path:
    """A frame with a caption-sized band of text across the lower third."""
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (448, 252), color=(240, 240, 240))
    if text:
        draw = ImageDraw.Draw(image)
        for row in range(6):
            draw.text((10, 170 + row * 12), text * 6, fill=(0, 0, 0))
    path = tmp_path / name
    image.save(path)
    return path


def test_changed_frames_and_periodic_verification_are_read(tmp_path: Path) -> None:
    """Verify periodically: equal edge grids do not prove equal text."""
    frames = [
        (0.0, frame_with(tmp_path, "a.jpg", text="")),
        (1.0, frame_with(tmp_path, "b.jpg", text="HELLO THERE")),
        (2.0, frame_with(tmp_path, "c.jpg", text="HELLO THERE")),
        (3.0, frame_with(tmp_path, "d.jpg", text="HELLO THERE")),
        (4.0, frame_with(tmp_path, "e.jpg", text="HELLO THERE")),
        (5.0, frame_with(tmp_path, "f.jpg", text="HELLO THERE")),
        (6.0, frame_with(tmp_path, "g.jpg", text="")),
    ]

    picked = ocr_candidates(frames, config=CONFIG)

    assert picked == [0, 1, 5, 6]


def test_frames_something_else_asked_for_are_read_regardless(tmp_path: Path) -> None:
    frames = [(float(index), frame_with(tmp_path, f"{index}.jpg", text="")) for index in range(5)]

    picked = ocr_candidates(frames, config=CONFIG, required_s=[3.0])

    assert 3 in picked


def test_a_required_moment_snaps_to_the_nearest_sampled_frame(tmp_path: Path) -> None:
    frames = [(float(index), frame_with(tmp_path, f"{index}.jpg", text="")) for index in range(5)]

    picked = ocr_candidates(frames, config=CONFIG, required_s=[2.7])

    assert 3 in picked


def test_an_unchanged_grid_is_not_a_change() -> None:
    signature = (1, 2, 3, 4, 5, 6, 7, 8)
    assert not signature_changed(signature, signature, threshold=0.06)
    assert signature_changed((), signature, threshold=0.06)


def test_a_line_missed_by_one_reading_is_not_gone() -> None:
    """A hand passing over a logo does not end it."""
    logo = line("open culture", top=0.04, bottom=0.09, left=0.02, right=0.3)
    observations = [
        Observation(0.0, (logo,)),
        Observation(2.0, ()),
        Observation(4.0, (logo,)),
        Observation(40.0, ()),
    ]

    cues = text_cues(observations, duration_s=60.0, config=CONFIG)

    assert len(cues) == 1
    assert cues[0].start == 0.0


def test_a_line_really_gone_does_end() -> None:
    logo = line("open culture", top=0.04, bottom=0.09, left=0.02, right=0.3)
    observations = [
        Observation(0.0, (logo,)),
        Observation(10.0, ()),
        Observation(20.0, (logo,)),
        Observation(30.0, ()),
    ]

    cues = text_cues(with_confirmations(observations), duration_s=60.0, config=CONFIG)

    assert len(cues) == 2
