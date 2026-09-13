from __future__ import annotations

import pytest

from vuc.config import VADConfig
from vuc.timeline import Span
from vuc.vad import non_speech_view, normalize_fsmn_result, speech_view


def make_config(**overrides: float | str | int) -> VADConfig:
    base = {
        "provider": "fsmn",
        "hub": "hf",
        "model": "fsmn-vad",
        "device": "cpu",
        "cpu_threads": 1,
        "max_single_segment_s": 30.0,
        "merge_gap_s": 1.5,
        "min_bridge_s": 1.0,
        "min_region_s": 2.0,
        "window_max_s": 30.0,
    }
    return VADConfig(**{**base, **overrides})


def test_funasr_milliseconds_become_seconds() -> None:
    spans = normalize_fsmn_result([{"key": "clip", "value": [[1230, 4560]]}], 60.0)

    assert spans == [Span(1.23, 4.56)]


def test_rounding_only_ever_widens_a_region() -> None:
    """Tidier numbers must never cost a syllable at either end."""
    detected = [Span(4.4, 9.1), Span(20.9, 24.2)]

    regions = speech_view(detected, config=make_config(), duration_s=60.0)

    assert regions == [Span(4.0, 10.0), Span(20.0, 25.0)]
    for original in detected:
        assert any(
            region.start <= original.start and region.end >= original.end
            for region in regions
        )


def test_short_silence_between_utterances_is_swallowed() -> None:
    regions = speech_view(
        [Span(0.0, 12.0), Span(13.2, 25.0)], config=make_config(), duration_s=60.0
    )

    assert regions == [Span(0.0, 25.0)]


def test_long_pause_stays_a_pause() -> None:
    regions = speech_view(
        [Span(0.0, 12.0), Span(20.0, 25.0)], config=make_config(), duration_s=60.0
    )

    assert regions == [Span(0.0, 12.0), Span(20.0, 25.0)]


def test_brief_detections_are_not_bridged_across_a_wide_gap() -> None:
    """Two coughs 2.4s apart are two coughs, not one four-second utterance."""
    config = make_config(merge_gap_s=5.0, min_bridge_s=0.5)

    regions = speech_view([Span(0.1, 0.6), Span(3.0, 3.4)], config=config, duration_s=60.0)

    assert regions == [Span(0.0, 1.0), Span(3.0, 4.0)]


def test_the_same_gap_is_bridged_between_long_utterances() -> None:
    config = make_config(merge_gap_s=5.0, min_bridge_s=0.5, window_max_s=60.0)

    regions = speech_view([Span(0.0, 20.0), Span(21.4, 40.0)], config=config, duration_s=60.0)

    assert regions == [Span(0.0, 40.0)]


def test_no_region_outgrows_the_window_limit() -> None:
    config = make_config(window_max_s=20.0)

    regions = speech_view([Span(0.0, 95.0)], config=config, duration_s=100.0)

    assert all(region.duration <= 20.0 for region in regions)
    assert regions[0].start == 0.0
    assert regions[-1].end == 95.0


def test_the_two_views_partition_the_whole_timeline() -> None:
    """Every second is on exactly one side: no overlap, no hole, no leftover."""
    config = make_config()
    speech = speech_view(
        [Span(3.2, 18.0), Span(40.0, 44.0), Span(80.0, 96.0)],
        config=config,
        duration_s=120.0,
    )
    non_speech = non_speech_view(speech, config=config, duration_s=120.0)

    ordered = sorted([*speech, *non_speech])
    assert ordered[0].start == 0.0
    assert ordered[-1].end == 120.0
    for previous, following in zip(ordered, ordered[1:], strict=False):
        assert previous.end == following.start
    assert sum(span.duration for span in ordered) == pytest.approx(120.0)


def test_silence_is_cut_into_workable_pieces() -> None:
    config = make_config(window_max_s=30.0)

    non_speech = non_speech_view([Span(0.0, 5.0)], config=config, duration_s=200.0)

    assert non_speech[0].start == 5.0
    assert non_speech[-1].end == 200.0
    assert all(span.duration <= 30.0 for span in non_speech)


def test_a_silent_video_is_all_one_side() -> None:
    config = make_config()

    speech = speech_view([], config=config, duration_s=45.0)
    non_speech = non_speech_view(speech, config=config, duration_s=45.0)

    assert speech == []
    assert sum(span.duration for span in non_speech) == pytest.approx(45.0)


def test_a_rounding_remainder_at_the_tail_is_not_a_region() -> None:
    """A 13ms leftover is arithmetic, not silence anything can be said about."""
    config = make_config(window_max_s=200.0)

    speech = speech_view([Span(0.0, 119.6)], config=config, duration_s=120.013)
    non_speech = non_speech_view(speech, config=config, duration_s=120.013)

    assert speech[-1].end == 120.013
    assert non_speech == []


def test_a_one_second_pause_does_not_become_an_empty_line() -> None:
    """Nothing can be said about a second of room tone, so it joins the speech."""
    config = make_config(window_max_s=200.0)

    speech = speech_view(
        [Span(10.0, 20.0), Span(21.2, 30.0)], config=config, duration_s=60.0
    )
    non_speech = non_speech_view(speech, config=config, duration_s=60.0)

    assert speech == [Span(10.0, 30.0)]
    assert non_speech == [Span(0.0, 10.0), Span(30.0, 60.0)]


def test_absorbing_a_pause_never_leaves_a_hole() -> None:
    config = make_config(window_max_s=200.0)
    detected = [Span(0.6, 9.0), Span(10.2, 18.0), Span(25.0, 40.0), Span(41.0, 49.4)]

    speech = speech_view(detected, config=config, duration_s=50.0)
    non_speech = non_speech_view(speech, config=config, duration_s=50.0)

    ordered = sorted([*speech, *non_speech])
    assert ordered[0].start == 0.0
    assert ordered[-1].end == 50.0
    for previous, following in zip(ordered, ordered[1:], strict=False):
        assert previous.end == following.start
    assert all(span.duration >= config.min_region_s for span in non_speech)


def test_a_region_never_claims_time_past_the_end_of_the_video() -> None:
    config = make_config()

    speech = speech_view([Span(110.0, 120.01)], config=config, duration_s=120.013)

    assert speech[-1].end == 120.013


def test_a_short_utterance_is_kept_even_though_silence_that_short_is_not() -> None:
    """"yes" is a second long. Length cannot tell it from a mouth noise."""
    config = make_config()

    speech = speech_view(
        [Span(100.0, 130.0), Span(508.06, 508.95)], config=config, duration_s=600.0
    )

    assert speech == [Span(100.0, 130.0), Span(508.0, 509.0)]
