"""Evidence, geometry and temporal invariants independent of video vocabulary."""

import math
from dataclasses import replace

import pytest

from tests.test_ocr_index import CONFIG, line
from vuc.ocr import Observation, polygon_text_height, text_cues
from vuc.ocr_layout import frame_links, frame_regions, region_from_rows, visible_regions
from vuc.ocr_tracking import _present


def test_rotated_text_measures_glyph_thickness_not_bounding_height():
    angle = math.radians(20)
    box = [
        (x * math.cos(angle) - y * math.sin(angle), x * math.sin(angle) + y * math.cos(angle))
        for x, y in ((0, 0), (200, 0), (200, 20), (0, 20))
    ]
    assert max(y for _, y in box) - min(y for _, y in box) > 80
    assert polygon_text_height(box, 1000) == pytest.approx(0.02)
    vertical = [(0, 0), (20, 0), (20, 200), (0, 200)]
    assert polygon_text_height(vertical, 1000) == pytest.approx(0.02)


def test_tilt_cannot_lift_small_text_over_the_track_floor():
    small = replace(line("Printed label", top=0.2, bottom=0.29), text_height=0.018)
    obs = [Observation(t, (small,)) for t in range(3)]
    assert text_cues(obs, duration_s=3, config=replace(CONFIG, min_text_height=0.025)) == []


def test_region_height_describes_glyphs_instead_of_the_paragraph_extent():
    rows = [
        line(text, top=y, bottom=y + 0.03)
        for text, y in (("First row", 0.1), ("Second row", 0.14), ("Third row", 0.18))
    ]
    block = region_from_rows(rows)
    assert block.bottom - block.top == pytest.approx(0.11)
    assert block.height == pytest.approx(0.03)
    assert block.size == "small"


def test_weak_row_is_not_hidden_by_confident_neighbours():
    block = region_from_rows([line("First", confidence=0.99), line("Uncertain", confidence=0.72)])
    assert block.confidence == 0.72


def test_layout_does_not_jump_over_an_incompatible_intervening_row():
    rows = [line(str(i), top=0.1 + i * 0.035, bottom=0.15 + i * 0.035) for i in range(3)]
    assert frame_links(rows, compatible=lambda a, b: a is rows[0] and b is rows[2]) == []


def test_paragraph_spacing_and_columns_are_boundaries():
    rows = [
        line("First", top=0.1, bottom=0.13, left=0.1, right=0.4),
        line("Second paragraph", top=0.165, bottom=0.195, left=0.1, right=0.4),
        line("Other column", top=0.1, bottom=0.13, left=0.6, right=0.9),
    ]
    assert len(frame_regions(rows)) == 3


def test_grid_boundary_does_not_split_an_adjacent_title():
    rows = (
        line("Creative", top=0.24, bottom=0.30, left=0.6, right=0.9),
        line("Commons", top=0.315, bottom=0.375, left=0.6, right=0.9),
        line("Licenses", top=0.39, bottom=0.45, left=0.6, right=0.9),
    )
    cues = text_cues(
        [Observation(0, rows), Observation(1, rows), Observation(2, ())],
        duration_s=3,
        config=CONFIG,
    )
    assert [c.text for c in cues] == ["Creative / Commons / Licenses"]


def test_nonconcurrent_lines_cannot_make_a_synthetic_paragraph():
    first = line("First statement", top=0.2, bottom=0.24)
    second = line("Second statement", top=0.25, bottom=0.29)
    obs = [Observation(t, (first if t < 2 else second,)) for t in range(4)]
    cues = text_cues(obs, duration_s=4, config=CONFIG)
    assert {c.text for c in cues} == {first.text, second.text}


def test_long_lived_label_is_independent_of_changing_neighbours():
    label = line("Persistent label", top=0.1, bottom=0.14)
    obs = [
        Observation(t, (label, line(text, top=0.15, bottom=0.19)))
        for t, text in enumerate(
            ["First statement"] * 2 + ["Second statement"] * 2 + ["Third statement"] * 2
        )
    ]
    cues = text_cues(obs, duration_s=6, config=CONFIG)
    assert [(c.start, c.end) for c in cues if c.text == label.text] == [(0, 6)]
    assert sum(label.text in c.text for c in cues) == 1


def test_horizontal_join_ignores_interleaved_column_traversal():
    rows = (
        line("B", top=0.590, bottom=0.618, left=0.236, right=0.251),
        line("The United States", top=0.586, bottom=0.618, left=0.267, right=0.401),
        line("Type message", top=0.572, bottom=0.607, left=0.421, right=0.691),
    )
    for ordered in (rows, rows[::-1]):
        assert {r.text for r in _present(ordered, CONFIG)} == {
            "B The United States",
            "Type message",
        }


def test_one_noisy_extension_cannot_disqualify_the_repeated_clean_reading():
    clean = line("A clear statement", confidence=0.99, left=0.1, right=0.4)
    noisy = line("A clear statement stray", confidence=0.91, left=0.1, right=0.5)
    obs = [Observation(t, (noisy if t == 3 else clean,)) for t in range(7)]
    cues = text_cues(obs, duration_s=7, config=CONFIG)
    assert [c.text for c in cues] == [clean.text]


def test_persistent_logo_cannot_turn_into_a_future_heading():
    logo = line("ACME", top=0.1, bottom=0.15, left=0.05, right=0.12, confidence=0.99)
    heading = line(
        "ACME A newly published heading",
        top=0.1,
        bottom=0.15,
        left=0.05,
        right=0.8,
        confidence=0.99,
    )
    obs = [Observation(t, (logo if t < 5 else heading,)) for t in range(8)]
    cues = text_cues(obs, duration_s=8, config=CONFIG)
    assert [(c.start, c.end, c.text) for c in cues] == [(0, 5, logo.text), (5, 8, heading.text)]


def test_sustained_crop_ends_the_complete_reading_at_the_first_crop():
    full = line("The whole sentence remains readable", left=0.1, right=0.8, confidence=0.99)
    crop = line("sentence remains readable", left=0.3, right=0.8, confidence=0.99)
    obs = [Observation(t, (full if t < 3 else crop,)) for t in range(10)]
    cues = text_cues(obs, duration_s=10, config=CONFIG)
    assert [(c.start, c.end) for c in cues if c.text == full.text] == [(0, 3)]


@pytest.mark.parametrize(
    "words",
    [("네", "아", "응"), ("Yes", "Wait", "Ready"), ("是", "好", "完"), ("はい", "次へ", "終了")],
)
def test_fast_captions_can_use_a_stable_slot_in_every_language(words):
    obs = [Observation(t, (line(word, confidence=0.99),)) for t, word in enumerate(words)]
    cues = text_cues(obs, duration_s=3, config=CONFIG)
    assert [c.text for c in cues] == list(words)


@pytest.mark.parametrize("text", ["二", "山", "111", "Newラッ", "Welcome"])
def test_confident_isolated_readings_still_need_temporal_support(text):
    obs = [Observation(0, (line(text, confidence=0.999),)), Observation(1, ())]
    assert text_cues(obs, duration_s=2, config=CONFIG) == []


def test_other_locations_do_not_confirm_a_singleton():
    obs = [
        Observation(
            t,
            (
                line(
                    "Wrong" if t == 1 else "Other",
                    confidence=0.99,
                    top=0.1 if t == 1 else 0.8,
                    bottom=0.15 if t == 1 else 0.85,
                ),
            ),
        )
        for t in range(3)
    ]
    cues = text_cues(obs, duration_s=3, config=CONFIG)
    assert all(c.text != "Wrong" for c in cues)


def test_duplicate_timestamp_is_not_a_second_measurement():
    o = Observation(0, (line("One reading", confidence=0.99),))
    assert text_cues([o, o], duration_s=2, config=CONFIG) == []


def test_a_missed_row_can_use_two_confirmed_neighbours_in_its_paragraph():
    rows = tuple(
        line(text, top=0.2 + i * 0.05, bottom=0.24 + i * 0.05, confidence=0.99)
        for i, text in enumerate(["Sugar 30g", "Garlic 20g", "Oil 21g"])
    )
    obs = [Observation(0, rows), Observation(3, (rows[0], rows[2])), Observation(4, ())]
    cues = text_cues(obs, duration_s=5, config=CONFIG)
    assert any("Garlic 20g" in c.text for c in cues)
    # Unconfirmed adjacent guesses alone are not corroboration.
    assert text_cues([Observation(0, rows), Observation(1, ())], duration_s=2, config=CONFIG) == []


def test_repeated_legible_text_does_not_require_singleton_confidence():
    obs = [Observation(t, (line("Ingredient quantity 20g", confidence=0.90),)) for t in range(3)]
    config = replace(CONFIG, singleton_confidence=0.95)
    assert [c.text for c in text_cues(obs, duration_s=3, config=config)] == [
        "Ingredient quantity 20g"
    ]


def test_overlapping_region_and_its_rows_have_one_owner():
    a = line("First row", top=0.1, bottom=0.14)
    b = line("Second row", top=0.15, bottom=0.19)
    parent = region_from_rows([a, b])
    result = visible_regions([(0, 5, parent), (1, 4, b)])
    assert [(s, e, r.text) for s, e, r in result] == [(0, 5, parent.text)]
    other_column = replace(b, left=0.8, right=0.99)
    result = visible_regions([(0, 5, parent), (1, 4, other_column)])
    assert any(r.text == b.text for _, _, r in result)
