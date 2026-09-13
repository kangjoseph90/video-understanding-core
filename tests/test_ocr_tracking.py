from dataclasses import replace

import pytest

from tests.test_ocr_index import CONFIG, line, with_confirmations
from vuc.index_text import render_text_index
from vuc.models import TextCue
from vuc.ocr import Observation, same_text, text_cues


def test_consensus_does_not_freeze_the_first_bad_reading():
    observations = [Observation(0, (line("open cuiture", confidence=0.75),))]
    observations += [Observation(t, (line("open culture", confidence=0.94),)) for t in (1, 2, 3)]
    observations += [Observation(4, ())]
    cues = text_cues(observations, duration_s=5, config=CONFIG)
    assert len(cues) == 1
    assert cues[0].text == "open culture"
    assert (cues[0].start, cues[0].end) == (0, 4)


def test_best_reading_beats_a_frequently_repeated_systematic_error():
    wrong = line("Dyeöng", confidence=0.953)
    correct = line("Dyeong", confidence=0.997)
    observations = [Observation(t, (wrong,)) for t in range(513)]
    observations += [Observation(t, (correct,)) for t in range(513, 522)]
    observations.append(Observation(522, ()))

    cues = text_cues(observations, duration_s=523, config=CONFIG)

    assert [(cue.start, cue.end, cue.text) for cue in cues] == [(0, 522, "Dyeong")]


def test_cleaner_full_width_reading_beats_a_noisy_suffix():
    noisy = line("천천히 돌려 회오리 모양을 만들어줍니다으", confidence=0.953)
    clean = line("천천히 돌려 회오리 모양을 만들어줍니다", confidence=0.997)
    observations = [Observation(0, (noisy,)), Observation(1, (clean,)), Observation(2, ())]

    cues = text_cues(observations, duration_s=3, config=CONFIG)

    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (0, 2, "천천히 돌려 회오리 모양을 만들어줍니다")
    ]


def test_narrow_high_confidence_crop_does_not_replace_the_complete_reading():
    complete = line("Creative Commons Licenses", left=0.10, right=0.70, confidence=0.98)
    crop = line("Commons Licenses", left=0.35, right=0.70, confidence=0.999)
    observations = [Observation(t, (complete,)) for t in (0, 1, 2)]
    observations += [Observation(3, (crop,)), Observation(4, ())]

    cues = text_cues(observations, duration_s=5, config=CONFIG)

    assert [(cue.start, cue.end, cue.text) for cue in cues] == [(0, 4, "Creative Commons Licenses")]


def test_position_cell_boundary_does_not_split_a_track():
    observations = [
        Observation(0, (line("a title", left=0.27, right=0.37),)),
        Observation(1, (line("a title", left=0.29, right=0.39),)),
        Observation(2, ()),
    ]
    assert len(text_cues(observations, duration_s=3, config=CONFIG)) == 1


def test_slightly_overlapping_detector_fragments_form_one_row():
    fragments = (
        line("Commons Attribution", top=0.75, bottom=0.774, left=0.54, right=0.69),
        line("License (CC", top=0.75, bottom=0.778, left=0.689, right=0.802),
        line("BY", top=0.75, bottom=0.776, left=0.80, right=0.84),
    )
    cues = text_cues(
        [Observation(0, fragments), Observation(1, fragments), Observation(2, ())],
        duration_s=3,
        config=replace(CONFIG, min_text_height=0.025),
    )
    assert [cue.text for cue in cues] == ["Commons Attribution License (CC BY"]


def test_one_tall_fragment_does_not_lift_an_entire_tiny_row_over_the_floor():
    fragments = (
        line("Library", top=0.04, bottom=0.060, left=0.10, right=0.18),
        line("Google Scholar", top=0.04, bottom=0.061, left=0.19, right=0.28),
        line("NEBIS", top=0.04, bottom=0.060, left=0.29, right=0.32),
        line("SLSP-ALMA", top=0.037, bottom=0.064, left=0.33, right=0.41),
    )
    cues = text_cues(
        [
            Observation(0, fragments),
            Observation(1, fragments),
            Observation(
                2,
                (
                    line(
                        "Google Scholar NEBIS SLSP-ALMA",
                        top=0.037,
                        bottom=0.064,
                        left=0.19,
                        right=0.41,
                    ),
                ),
            ),
            Observation(3, ()),
        ],
        duration_s=4,
        config=replace(CONFIG, min_text_height=0.025),
    )
    assert cues == []


def test_repeated_labels_in_neighbouring_controls_do_not_form_one_row():
    labels = (
        line("Rightslink", top=0.04, bottom=0.06, left=0.28, right=0.32),
        line("E Classical", top=0.04, bottom=0.06, left=0.33, right=0.38),
        line("Rightslink", top=0.04, bottom=0.06, left=0.39, right=0.45),
        line("Rightslink", top=0.04, bottom=0.06, left=0.46, right=0.52),
    )
    cues = text_cues(
        [Observation(0, labels), Observation(1, labels), Observation(2, ())],
        duration_s=3,
        config=replace(CONFIG, min_text_height=0.018),
    )
    assert sum(cue.text == "Rightslink" for cue in cues) == 2
    assert all("Rightslink Rightslink" not in cue.text for cue in cues)


def test_side_by_side_columns_do_not_form_a_block():
    left = line(
        "GPT-4 clearly has the capability",
        top=0.21,
        bottom=0.26,
        left=0.22,
        right=0.45,
    )
    right = line(
        "Original equation: 6(-2g-1)=-(13g+2)",
        top=0.21,
        bottom=0.25,
        left=0.48,
        right=0.72,
    )
    cues = text_cues(
        [Observation(0, (left, right)), Observation(1, (left, right)), Observation(2, ())],
        duration_s=3,
        config=CONFIG,
    )
    assert {cue.text for cue in cues} == {left.text, right.text}


def test_scrolled_wrapped_line_can_overlap_its_parent_representative():
    first = line(
        "Keep working hard in",
        top=0.53,
        bottom=0.56,
        left=0.345,
        right=0.92,
    )
    continuation = line(
        "your classes.",
        top=0.54,
        bottom=0.565,
        left=0.358,
        right=0.434,
    )
    cues = text_cues(
        [Observation(0, (first, continuation)), Observation(1, (first, continuation))],
        duration_s=2,
        config=CONFIG,
    )
    assert [cue.text for cue in cues] == ["Keep working hard in / your classes."]


def test_continuously_visible_unique_text_can_move_across_the_screen():
    observations = [
        Observation(0, (line("moving label", left=0.05, right=0.25),)),
        Observation(1, (line("moving label", left=0.75, right=0.95),)),
        Observation(2, ()),
    ]
    cues = text_cues(observations, duration_s=3, config=CONFIG)
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [(0, 2, "moving label")]


def test_missed_text_that_reappears_far_away_starts_a_new_track():
    observations = [
        Observation(0, (line("Licenses", top=0.58, bottom=0.68, left=0.37, right=0.63),)),
        Observation(1, ()),
        Observation(2, (line("Licenses", top=0.37, bottom=0.46, left=0.67, right=0.86),)),
        Observation(3, (line("Licenses", top=0.37, bottom=0.46, left=0.67, right=0.86),)),
        Observation(4, ()),
    ]
    cues = text_cues(with_confirmations(observations), duration_s=5, config=CONFIG)
    assert [(cue.start, cue.end) for cue in cues] == [(0, 1), (2, 4)]


def test_tiny_remainder_does_not_extend_the_complete_reading():
    full = line("Commons Attribution License (CC BY", left=0.54, right=0.84)
    fragment = line("BY", left=0.80, right=0.84, confidence=0.99)
    observations = [Observation(t, (full,)) for t in (0, 1, 2)]
    observations += [Observation(t, (fragment,)) for t in (3, 4, 5)]
    observations += [Observation(6, ())]
    cues = text_cues(observations, duration_s=7, config=CONFIG)
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (0, 3, "Commons Attribution License (CC BY")
    ]


def test_partial_track_covered_by_a_colocated_full_reading_is_suppressed():
    full = line("terms of the Creative Commons Attribution License", left=0.34, right=0.84)
    prefix = line("terms of the", left=0.34, right=0.46)
    suffix = line("Creative Commons Attribution License", left=0.54, right=0.84)
    observations = [
        Observation(0, (full,)),
        Observation(1, (prefix, suffix)),
        Observation(2, (prefix, suffix)),
        Observation(3, (full,)),
        Observation(4, ()),
    ]
    cues = text_cues(observations, duration_s=5, config=CONFIG)
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (0, 4, "terms of the Creative Commons Attribution License")
    ]


def test_scrolling_streamed_text_keeps_only_the_completed_line():
    short = line(
        "1. Maintain good grades: Your GPA is important",
        top=0.68,
        bottom=0.71,
        left=0.345,
        right=0.72,
        confidence=0.99,
    )
    complete = line(
        "1. Maintain good grades: Your GPA is important, so keep working hard",
        top=0.53,
        bottom=0.56,
        left=0.345,
        right=0.92,
        confidence=0.99,
    )
    observations = [
        Observation(0, (short,)),
        Observation(1, (complete,)),
        Observation(2, (complete,)),
        Observation(3, ()),
    ]
    cues = text_cues(observations, duration_s=4, config=CONFIG)
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [(1, 3, complete.text)]


def test_mature_track_absorbs_a_low_confidence_noisy_suffix():
    clean = line("FRA", left=0.83, right=0.95, confidence=0.99)
    noisy = line("画口FRA", left=0.83, right=0.95, confidence=0.76)
    observations = [Observation(t, (clean,)) for t in (0, 1, 2)]
    observations += [Observation(t, (noisy,)) for t in (3, 4, 5)]
    observations += [Observation(6, (clean,)), Observation(7, ())]
    cues = text_cues(observations, duration_s=8, config=CONFIG)
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [(0, 7, "FRA")]


def test_mature_track_does_not_absorb_a_confident_number_change():
    old = line("Chapter 10", confidence=0.99)
    new = line("Chapter 100", confidence=0.99)
    observations = [Observation(t, (old,)) for t in (0, 1, 2)]
    observations += [Observation(3, (new,)), Observation(4, ())]
    cues = text_cues(observations, duration_s=5, config=CONFIG)
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (0, 3, "Chapter 10"),
        (3, 4, "Chapter 100"),
    ]


def test_one_lower_confidence_digit_suffix_is_treated_as_a_transient_probe():
    config = replace(CONFIG, singleton_confidence=0.95)
    readings = (
        ("천천히 돌려 회오리 모양을 만들어줍니다으", 0.953),
        ("천천히 돌려 회오리 모양을 만들어줍니다", 0.997),
        ("천천히 돌려 회오리 모양을 만들어줍니다0", 0.960),
        ("천천히 돌려 회오리 모양을 만들어줍니다으", 0.946),
    )
    observations = [
        Observation(timestamp, (line(text, confidence=confidence),))
        for timestamp, (text, confidence) in enumerate(readings)
    ]
    observations.append(Observation(4, ()))

    cues = text_cues(observations, duration_s=5, config=config)

    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (0, 4, "천천히 돌려 회오리 모양을 만들어줍니다")
    ]


def test_repeated_digit_suffix_is_committed_from_its_first_observation():
    base = line("Episode", confidence=0.99)
    numbered = line("Episode 2", confidence=0.98)
    observations = [Observation(t, (base,)) for t in (0, 1)]
    observations += [Observation(t, (numbered,)) for t in (2, 3)]
    observations.append(Observation(4, ()))

    cues = text_cues(observations, duration_s=5, config=CONFIG)

    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (0, 2, "Episode"),
        (2, 4, "Episode 2"),
    ]


def test_two_equal_labels_in_the_same_cell_are_independent():
    observations = [
        Observation(
            0,
            (
                line("Open", top=0.05, bottom=0.08, left=0.01, right=0.1),
                line("Open", top=0.22, bottom=0.25, left=0.01, right=0.1),
            ),
        ),
        Observation(1, (line("Open", top=0.22, bottom=0.25, left=0.01, right=0.1),)),
        Observation(2, ()),
    ]
    cues = text_cues(with_confirmations(observations), duration_s=3, config=CONFIG)
    assert sorted((c.start, c.end) for c in cues) == [(0, 1), (0, 2)]


@pytest.mark.parametrize(
    "left,right",
    [
        ("Add 10 grams", "Add 20 grams"),
        ("0.5", "05"),
        ("Value -20", "Value +20"),
        ("料金は1360円", "料金は1380円"),
    ],
)
def test_fuzzy_dedup_never_erases_a_changed_quantity(left, right):
    assert not same_text(left, right, ratio=0.8)


def test_confident_near_identical_sentences_remain_separate():
    observations = [
        Observation(0, (line("We should approve this request", confidence=0.99),)),
        Observation(1, (line("We should reject this request", confidence=0.99),)),
        Observation(2, ()),
    ]
    cues = text_cues(with_confirmations(observations), duration_s=3, config=CONFIG)
    assert len(cues) == 2
    assert [(c.start, c.end) for c in cues] == [(0, 1), (1, 2)]


def test_equal_text_cannot_rejoin_across_a_long_unobserved_gap():
    observations = [Observation(0, (line("Same title"),)), Observation(100, (line("Same title"),))]
    cues = text_cues(with_confirmations(observations), duration_s=102, config=CONFIG)
    assert [(c.start, c.end) for c in cues] == [(0, 30.25), (100, 102)]


def test_low_confidence_singleton_is_kept_in_observations_but_not_printed():
    observation = Observation(
        0, (line("uncertain", confidence=0.78), line("yes", top=0.1, bottom=0.15, confidence=0.99))
    )
    cues = text_cues(
        [observation, Observation(0.25, (observation.lines[1],)), Observation(1, ())],
        duration_s=2,
        config=CONFIG,
    )
    assert [c.text for c in cues] == ["yes"]
    assert len(observation.lines) == 2


@pytest.mark.parametrize("text", ["3", "B", "to"])
def test_tiny_ascii_singleton_fragments_are_not_printed(text):
    fragment = line(text, confidence=1.0)
    assert (
        text_cues(
            [Observation(0, (fragment,)), Observation(1, ())],
            duration_s=2,
            config=CONFIG,
        )
        == []
    )


def test_persistent_text_does_not_republish_with_every_caption():
    observations = []
    for t, caption in enumerate(["First caption", "Next caption", "Last caption"]):
        observations.append(
            Observation(t, (line("Permanent heading", top=0.05, bottom=0.1), line(caption)))
        )
    observations.append(Observation(3, ()))
    cues = text_cues(observations, duration_s=4, config=CONFIG)
    assert len([c for c in cues if c.text == "Permanent heading"]) == 1
    assert len(cues) == 4


def test_repeated_stacked_title_lines_form_blocks_before_compaction():
    def title(timestamp):
        return Observation(
            timestamp,
            (
                line("Creative", top=0.16, bottom=0.25, left=0.68, right=0.85),
                line("Commons", top=0.27, bottom=0.36, left=0.66, right=0.87),
                line("Licenses", top=0.37, bottom=0.46, left=0.67, right=0.86),
            ),
        )

    observations = []
    for start in (0, 10, 20):
        observations.extend((title(start), title(start + 1), Observation(start + 2, ())))
    cues = text_cues(observations, duration_s=23, config=CONFIG)
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (0, 2, "Creative / Commons / Licenses"),
        (10, 12, "Creative / Commons / Licenses"),
        (20, 22, "Creative / Commons / Licenses"),
    ]
    assert render_text_index(cues) == (
        "[0-2; 10-12; 20-22, top right, large] Creative / Commons / Licenses"
    )


def test_block_grouping_ignores_position_and_size_labels():
    upper = line("Upper line", top=0.27, bottom=0.32, left=0.05, right=0.28)
    lower = line("Lower line", top=0.33, bottom=0.42, left=0.05, right=0.28)
    assert (upper.position, upper.size) == ("top left", "medium")
    assert (lower.position, lower.size) == ("middle left", "large")

    cues = text_cues(
        [Observation(0, (upper, lower)), Observation(1, (upper, lower)), Observation(2, ())],
        duration_s=3,
        config=CONFIG,
    )

    assert len(cues) == 1
    assert cues[0].text == "Upper line / Lower line"


def test_visual_agreement_can_join_confident_hangul_glyph_noise():
    a = replace(line("요리비책", confidence=0.99), appearance="f" * 64)
    b = replace(line("묘리비책", confidence=0.97), appearance="f" * 63 + "e")
    cues = text_cues(
        [Observation(0, (a,)), Observation(1, (b,)), Observation(2, (a,)), Observation(3, ())],
        duration_s=4,
        config=CONFIG,
    )
    assert len(cues) == 1
    assert cues[0].text == "요리비책"


def test_recurring_output_preserves_gaps_and_punctuation():
    cues = [TextCue(t, t + 2, "Channel title", "top left", "small") for t in (0, 10, 20)]
    output = render_text_index(cues)
    assert output == "[0-2; 10-12; 20-22, top left, small] Channel title"
    assert "[0-22]" not in output
    assert len(cues) == 3
    quantities = [
        TextCue(t, t + 1, text, "top left", "small") for t, text in enumerate(["0.5", "05", "0.5"])
    ]
    assert render_text_index(quantities).splitlines() == [
        "[0-1; 2-3, top left, small] 0.5",
        "[1-2, top left, small] 05",
    ]


def test_two_recurrences_are_compacted():
    cues = [TextCue(t, t + 2, "Repeated title", "top left", "small") for t in (0, 10)]
    assert render_text_index(cues) == "[0-2; 10-12, top left, small] Repeated title"


def test_recurring_output_ignores_its_own_block_separator():
    cues = [
        TextCue(0, 2, "terms of the Creative Commons", "bottom center", "small"),
        TextCue(2, 4, "terms of the / Creative Commons", "bottom center", "small"),
        TextCue(4, 6, "terms of the Creative Commons", "bottom center", "small"),
    ]
    assert render_text_index(cues) == ("[0-6, bottom center, small] terms of the Creative Commons")


def test_regional_language_tags_select_the_expected_dictionary():
    config = replace(
        CONFIG,
        rec_by_language={
            "zh": ("chinese.onnx", "chinese.txt"),
            "ko": ("korean.onnx", "korean.txt"),
        },
        fallback_rec_model_path="fallback.onnx",
    )
    assert config.for_language("zh-Hant-TW").rec_model_path == "chinese.onnx"
    assert config.for_language("ko_KR").rec_model_path == "korean.onnx"
    assert config.for_language("ko").fallback_rec_model_path == ""
    assert config.for_language(None).fallback_rec_model_path == "fallback.onnx"


def test_slanted_lines_are_read_left_to_right():
    from vuc.ocr import reading_order

    left = line("登机口", top=0.81, bottom=0.86, left=0.10, right=0.35)
    right = line("GATES CLOSE", top=0.80, bottom=0.85, left=0.38, right=0.90)
    assert [x.text for x in reading_order([right, left])] == ["登机口", "GATES CLOSE"]


def test_tiny_text_is_discarded_even_when_confident_and_repeated():
    config = replace(CONFIG, min_text_height=0.025)
    tiny = line("small but real", top=0.1, bottom=0.11, confidence=0.99)
    assert (
        text_cues([Observation(0, (tiny,)), Observation(1, ())], duration_s=2, config=config) == []
    )
    cues = text_cues(
        [Observation(0, (tiny,)), Observation(1, (tiny,)), Observation(2, ())],
        duration_s=3,
        config=config,
    )
    assert cues == []


def test_text_at_the_size_floor_is_discarded_without_boundary_flapping():
    config = replace(CONFIG, min_text_height=0.025)
    at_floor = line("10:54", top=0.95, bottom=0.975, confidence=1.0)
    below_floor = line("10:54", top=0.95, bottom=0.972, confidence=1.0)
    observations = [
        Observation(0, (at_floor,)),
        Observation(1, (below_floor,)),
        Observation(2, (at_floor,)),
    ]
    assert text_cues(observations, duration_s=3, config=config) == []


def test_long_lived_below_floor_text_cannot_leak_from_one_tall_measurement():
    config = replace(CONFIG, min_text_height=0.025)
    observations = [
        Observation(
            timestamp,
            (
                line(
                    "persistent browser label",
                    top=0.1,
                    bottom=0.127 if timestamp == 5 else 0.12,
                    confidence=0.99,
                ),
            ),
        )
        for timestamp in range(10)
    ]
    observations.append(Observation(10, ()))
    assert text_cues(observations, duration_s=11, config=config) == []


def test_long_lived_above_floor_text_is_one_track_despite_one_short_measurement():
    config = replace(CONFIG, min_text_height=0.025)
    observations = [
        Observation(
            timestamp,
            (
                line(
                    "persistent useful label",
                    top=0.1,
                    bottom=0.12 if timestamp == 5 else 0.127,
                    confidence=0.99,
                ),
            ),
        )
        for timestamp in range(10)
    ]
    observations.append(Observation(10, ()))
    cues = text_cues(observations, duration_s=11, config=config)
    assert [(cue.start, cue.end, cue.text) for cue in cues] == [(0, 10, "persistent useful label")]


def test_repeated_element_shares_size_evidence_across_broken_tracks():
    config = replace(CONFIG, min_text_height=0.025, rejoin_gap_s=1)
    below = line(
        "Display Capture",
        top=0.60,
        bottom=0.622,
        left=0.28,
        right=0.355,
        confidence=0.99,
    )
    above = line(
        "Display Capture",
        top=0.60,
        bottom=0.627,
        left=0.28,
        right=0.355,
        confidence=0.99,
    )
    observations = [Observation(t, (below,)) for t in range(5)]
    observations += [Observation(6, ()), Observation(8, (above,)), Observation(9, (above,))]
    observations.append(Observation(10, ()))
    assert text_cues(observations, duration_s=11, config=config) == []


def test_repeated_large_text_shares_size_evidence_across_a_coordinate_boundary():
    config = replace(CONFIG, min_text_height=0.025, rejoin_gap_s=1)
    complete = line(
        "Creative Commons Licenses",
        top=0.10,
        bottom=0.1653,
        left=0.139,
        right=0.739,
        confidence=0.99,
    )
    fragments = (
        line(
            "Creative Commons",
            top=0.10,
            bottom=0.1236,
            left=0.094,
            right=0.45,
            confidence=0.99,
        ),
        line(
            "Licenses",
            top=0.10,
            bottom=0.1236,
            left=0.455,
            right=0.694,
            confidence=0.99,
        ),
    )
    observations = [Observation(t, (complete,)) for t in range(41, 84)]
    observations.append(Observation(96, ()))
    observations += [Observation(t, fragments) for t in range(158, 252)]
    observations.append(Observation(252, ()))

    cues = text_cues(observations, duration_s=253, config=config)

    assert [(cue.start, cue.end, cue.text) for cue in cues] == [
        (41, 96, "Creative Commons Licenses"),
        (158, 252, "Creative Commons Licenses"),
    ]


def test_same_text_at_a_different_width_does_not_share_size_evidence():
    config = replace(CONFIG, min_text_height=0.025, rejoin_gap_s=1)
    large = line(
        "Channel name",
        top=0.10,
        bottom=0.16,
        left=0.10,
        right=0.60,
        confidence=0.99,
    )
    small = line(
        "Channel name",
        top=0.80,
        bottom=0.82,
        left=0.76,
        right=0.90,
        confidence=0.99,
    )
    observations = [Observation(t, (large,)) for t in (0, 1)]
    observations += [Observation(2, ()), Observation(5, (small,)), Observation(6, (small,))]
    observations.append(Observation(7, ()))

    cues = text_cues(observations, duration_s=8, config=config)

    assert [(cue.start, cue.end, cue.text) for cue in cues] == [(0, 2, "Channel name")]


def test_few_tall_partial_reads_do_not_lift_a_complete_tiny_label():
    config = replace(CONFIG, min_text_height=0.025)
    complete = line(
        "taiwan ten cafe",
        top=0.09,
        bottom=0.114,
        left=0.76,
        right=0.89,
        confidence=0.99,
    )
    observations = [Observation(t, (complete,)) for t in range(6)]
    observations += [
        Observation(6, (line("taiwan ten", top=0.09, bottom=0.13, left=0.76, right=0.86),)),
        Observation(7, (line("taiwan", top=0.09, bottom=0.13, left=0.76, right=0.82),)),
        Observation(8, ()),
    ]
    assert text_cues(observations, duration_s=9, config=config) == []


def test_size_filter_runs_before_lines_can_form_a_tall_block():
    config = replace(CONFIG, min_text_height=0.025)
    tiny_lines = (
        line("first row", top=0.1, bottom=0.11),
        line("second row", top=0.12, bottom=0.13),
        line("third row", top=0.14, bottom=0.15),
    )
    observations = [Observation(t, tiny_lines) for t in (0, 1, 2)]
    assert text_cues(observations, duration_s=3, config=config) == []
    assert len(observations[0].lines) == 3


def test_small_paragraph_continuation_is_supported_by_the_row_above():
    config = replace(CONFIG, min_text_height=0.025)
    first = line(
        "Keep working hard in",
        top=0.59,
        bottom=0.62,
        left=0.345,
        right=0.92,
    )
    continuation = line(
        "your classes.",
        top=0.627,
        bottom=0.651,
        left=0.358,
        right=0.434,
    )
    observations = [
        Observation(0, (first, continuation)),
        Observation(1, (first, continuation)),
        Observation(2, ()),
    ]
    cues = text_cues(observations, duration_s=3, config=config)
    assert [cue.text for cue in cues] == ["Keep working hard in / your classes."]


def test_continuation_first_seen_one_sample_late_joins_its_paragraph():
    first = line(
        "Explore your interests in your community",
        top=0.59,
        bottom=0.62,
        left=0.345,
        right=0.92,
    )
    continuation = line(
        "to discover your passions.",
        top=0.627,
        bottom=0.653,
        left=0.358,
        right=0.62,
    )
    observations = [
        Observation(0, (first,)),
        Observation(1, (first, continuation)),
        Observation(2, (first, continuation)),
        Observation(3, ()),
    ]
    cues = text_cues(observations, duration_s=4, config=CONFIG)
    assert [cue.text for cue in cues] == [
        "Explore your interests in your community / to discover your passions."
    ]


def test_small_ui_list_item_is_not_a_paragraph_continuation():
    config = replace(CONFIG, min_text_height=0.025)
    microphone = line(
        "Microphone",
        top=0.60,
        bottom=0.627,
        left=0.280,
        right=0.345,
    )
    display_capture = line(
        "Display Capture",
        top=0.636,
        bottom=0.658,
        left=0.278,
        right=0.353,
    )
    observations = [
        Observation(0, (microphone, display_capture)),
        Observation(1, (microphone, display_capture)),
        Observation(2, ()),
    ]
    cues = text_cues(observations, duration_s=3, config=config)
    assert [cue.text for cue in cues] == ["Microphone"]


def test_low_confidence_extreme_vertical_crop_is_not_published():
    rotated_garbage = line(
        "MasnVsco", top=0.142, bottom=0.254, left=0.891, right=0.903, confidence=0.78
    )
    observations = [Observation(t, (rotated_garbage,)) for t in (0, 1, 2)]
    assert text_cues(observations, duration_s=3, config=CONFIG) == []


def test_confident_vertical_text_is_preserved():
    vertical = line("縦書き", top=0.4, bottom=0.8, left=0.1, right=0.14, confidence=0.99)
    cues = text_cues(
        [Observation(0, (vertical,)), Observation(1, (vertical,)), Observation(2, ())],
        duration_s=3,
        config=CONFIG,
    )
    assert [cue.text for cue in cues] == ["縦書き"]


def test_legible_short_caption_survives_size_filter():
    caption = line("네", top=0.8, bottom=0.85, confidence=0.99)
    cues = text_cues(
        [Observation(0, (caption,)), Observation(0.25, (caption,)), Observation(1, ())],
        duration_s=2,
        config=replace(CONFIG, min_text_height=0.025),
    )
    assert [c.text for c in cues] == ["네"]


def test_recurring_text_keeps_each_location_and_size():
    cues = [TextCue(t, t + 1, "Label", "top left", "small") for t in (0, 10, 20)]
    cues += [TextCue(30, 31, "Label", "bottom right", "large")]
    expected = (
        "[0-1, top left, small; 10-11, top left, small; 20-21, top left, small; "
        "30-31, bottom right, large] Label"
    )
    assert render_text_index(cues).splitlines() == [
        expected,
    ]
