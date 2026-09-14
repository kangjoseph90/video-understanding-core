from __future__ import annotations

from vuc.caption_fusion import (
    HARDSUB_COPY,
    REJECTED_LANGUAGE,
    REJECTED_UNALIGNED,
    SPEECH_TRANSCRIPT,
    AttributionConfig,
    assign_tokens,
    attribute,
    fuse_audio_index,
    fuse_region,
    fuse_text_index,
    script_ratio,
)
from vuc.captions import CaptionCue, CaptionTrack
from vuc.models import NON_SPEECH, SPEECH, Segment, TextCue

CONFIG = AttributionConfig()


def track(*cues: tuple[float, str], word_timing: bool = True, language: str = "en") -> CaptionTrack:
    return CaptionTrack(
        language=language,
        kind="manual",
        source_format="json3",
        content_sha256="hash",
        has_word_timing=word_timing,
        cues=tuple(CaptionCue(at, text) for at, text in cues),
    )


def speech(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text, language="en", kind=SPEECH)


def cue(start: float, end: float, text: str) -> TextCue:
    return TextCue(start=start, end=end, text=text, position="bottom center", size="medium")


# ------------------------------------------------------------------- axes


def test_script_ratio_separates_declared_language_from_actual_script() -> None:
    assert script_ratio("안녕하세요 여러분", "ko") == 1.0
    assert script_ratio("Hello everyone", "ko") == 0.0
    assert script_ratio("こんにちは", "ja") == 1.0


def test_latin_languages_have_no_script_signal_and_say_so() -> None:
    """None is not a pass or a fail; this axis simply cannot separate en from fr."""
    assert script_ratio("Hello everyone", "en") is None
    assert script_ratio("Bonjour", "fr") is None


def test_a_track_in_the_wrong_script_is_rejected_before_timing_is_considered() -> None:
    english = track((1.0, "Hello"), (2.0, "everyone"), language="ko")

    result = attribute(
        english,
        audio_language="ko",
        speech=[(0.0, 10.0)],
        text_cues=(),
        config=CONFIG,
    )

    assert result.verdict == REJECTED_LANGUAGE
    assert result.script_ratio == 0.0


def test_words_landing_in_speech_make_it_a_transcript() -> None:
    result = attribute(
        track((1.0, "hello"), (2.0, "everyone")),
        audio_language="en",
        speech=[(0.0, 10.0)],
        text_cues=(),
        config=CONFIG,
    )

    assert result.verdict == SPEECH_TRANSCRIPT
    assert result.vad_overlap == 1.0


def test_words_matching_the_screen_but_not_the_speech_make_it_a_hardsub_copy() -> None:
    text = "오늘도 새벽부터 편집하다가"
    result = attribute(
        track((30.0, text), word_timing=False, language="ko"),
        audio_language="ko",
        speech=[(0.0, 5.0)],
        text_cues=(cue(30.0, 34.0, text),),
        config=CONFIG,
    )

    assert result.verdict == HARDSUB_COPY
    assert result.vad_overlap == 0.0
    assert result.ocr_match == 1.0


def test_a_lecturer_reading_their_slides_stays_a_transcript() -> None:
    """The V axis decides first.

    A track that is both spoken and on screen must not be allowed to rewrite the
    OCR rows: the speaker paraphrases the slide, and correcting screen text with
    spoken paraphrase is the one thing the text index refuses to do. Measured at
    O = 0.501 on the slide lecture.
    """
    shared = "creative commons licenses"
    result = attribute(
        track((1.0, shared), word_timing=False),
        audio_language="en",
        speech=[(0.0, 10.0)],
        text_cues=(cue(0.0, 8.0, shared),),
        config=CONFIG,
    )

    assert result.vad_overlap >= CONFIG.vad_overlap_min
    assert result.ocr_match >= CONFIG.ocr_match_min
    assert result.verdict == SPEECH_TRANSCRIPT


def test_a_track_matching_neither_observer_is_dropped() -> None:
    result = attribute(
        track((500.0, "buy"), (501.0, "my"), (502.0, "merch")),
        audio_language="en",
        speech=[(0.0, 10.0)],
        text_cues=(),
        config=CONFIG,
    )

    assert result.verdict == REJECTED_UNALIGNED


# --------------------------------------------------------------- assignment


def test_every_token_lands_in_exactly_one_region_or_none() -> None:
    regions = [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]
    tokens = ((9.9, "a"), (10.0, "b"), (10.1, "c"), (25.0, "d"))

    buckets, dropped = assign_tokens(tokens, regions)

    assert [len(bucket) for bucket in buckets] == [1, 2, 1]
    assert dropped == []
    assert sum(len(bucket) for bucket in buckets) + len(dropped) == len(tokens)


def test_tokens_outside_every_speech_region_are_dropped_not_duplicated() -> None:
    """The VAD split is the spine; this pass does not move it."""
    tokens = ((-1.0, "before"), (5.0, "in"), (99.0, "after"))

    buckets, dropped = assign_tokens(tokens, [(0.0, 10.0)])

    assert buckets == [["in"]]
    assert dropped == ["before", "after"]


def test_a_cue_spanning_a_boundary_splits_rather_than_repeating() -> None:
    regions = [(0.0, 10.0), (10.0, 20.0)]
    tokens = ((8.0, "one"), (9.5, "two"), (11.0, "three"), (12.0, "four"))

    buckets, dropped = assign_tokens(tokens, regions)

    assert buckets == [["one", "two"], ["three", "four"]]
    assert dropped == []


# ------------------------------------------------------------------- audio


def test_where_the_captions_reach_the_captions_win() -> None:
    fused = fuse_region(
        "I'm Maya Rarick at Central Trovva",
        ["I'm", "Maja", "Drabczyk", "at", "Centrum", "Cyfrowe"],
        align_ratio_min=0.2,
    )

    assert fused == "I'm Maja Drabczyk at Centrum Cyfrowe"


def test_tokens_the_captions_omit_are_not_restored() -> None:
    """Restoring them measured worse than leaving the index alone entirely."""
    fused = fuse_region("um so I think uh yes", ["so", "I", "think", "yes"], align_ratio_min=0.2)

    assert fused == "so I think yes"


def test_a_region_the_captions_do_not_reach_keeps_what_was_heard() -> None:
    assert fuse_region("what was heard", [], align_ratio_min=0.2) == "what was heard"


def test_an_empty_transcript_is_recovered_from_the_captions() -> None:
    assert fuse_region("", ["recovered", "words"], align_ratio_min=0.2) == "recovered words"


def test_two_accounts_that_do_not_resemble_each_other_are_left_alone() -> None:
    """Better an uncorrected region than one spliced into what neither said."""
    fused = fuse_region(
        "the quick brown fox jumps",
        ["completely", "unrelated", "advertising", "copy"],
        align_ratio_min=0.2,
    )

    assert fused == "the quick brown fox jumps"


def test_non_speech_regions_are_never_touched() -> None:
    segments = (
        speech(0.0, 5.0, "wrong words"),
        Segment(5.0, 10.0, "", "unknown", events=("music",), kind=NON_SPEECH),
    )

    fused, stats = fuse_audio_index(segments, track((1.0, "right"), (2.0, "words")), config=CONFIG)

    assert fused[1] == segments[1]
    assert fused[0].text == "right words"
    assert stats["speech_regions"] == 1


def test_fusion_preserves_every_field_but_the_text() -> None:
    original = Segment(0.0, 5.0, "old", "ko", emotion="happy", events=("laugh",), raw_text="raw")
    fused, _ = fuse_audio_index((original,), track((1.0, "old"), (2.0, "new")), config=CONFIG)

    assert fused[0].text == "old new"
    assert fused[0].language == "ko"
    assert fused[0].emotion == "happy"
    assert fused[0].events == ("laugh",)
    assert fused[0].raw_text == "raw"
    assert (fused[0].start, fused[0].end, fused[0].kind) == (0.0, 5.0, SPEECH)


def test_a_recovered_empty_region_is_counted_as_such() -> None:
    fused, stats = fuse_audio_index(
        (speech(0.0, 5.0, ""),), track((1.0, "found"), (2.0, "speech")), config=CONFIG
    )

    assert fused[0].text == "found speech"
    assert stats["regions_recovered"] == 1
    assert stats["regions_rewritten"] == 1


def test_the_guard_is_reported_rather_than_silently_skipping() -> None:
    segments = (speech(0.0, 5.0, "the quick brown fox jumps over"),)
    unrelated = track((1.0, "completely"), (2.0, "unrelated"), (3.0, "advertising"))

    fused, stats = fuse_audio_index(segments, unrelated, config=CONFIG)

    assert fused == segments
    assert stats["regions_guarded"] == 1
    assert stats["regions_rewritten"] == 0


# -------------------------------------------------------------------- text


def test_a_misread_line_is_corrected_from_the_track() -> None:
    cues = (cue(54.0, 58.0, "영양 균형 행기기 1일차,오늘은 점심 도시락부터"),)
    corrected, stats = fuse_text_index(
        cues,
        track((54.0, "영양 균형 챙기기 1일차, 오늘은 점심 도시락부터"), word_timing=False),
        config=CONFIG,
    )

    assert corrected[0].text == "영양 균형 챙기기 1일차, 오늘은 점심 도시락부터"
    assert stats["corrected"] == 1


def test_position_and_size_come_from_ocr_because_the_track_has_none() -> None:
    cues = (TextCue(10.0, 14.0, "듬뿐 올려주고", "top left", "large"),)
    corrected, _ = fuse_text_index(
        cues, track((10.0, "듬뿍 올려주고"), word_timing=False), config=CONFIG
    )

    assert corrected[0].text == "듬뿍 올려주고"
    assert corrected[0].position == "top left"
    assert corrected[0].size == "large"


def test_a_correction_may_not_absorb_a_neighbouring_row() -> None:
    """Measured failure: the row took the previous line's text and duplicated it."""
    cues = (cue(40.0, 44.0, "한국인 3명 중 1명이 영양불균형이라고 해서"),)
    long_cue = track(
        (40.0, "마침 제스프리 캠페인을 보는데 한국인 3명 중 1명이 영양불균형이라고 해서"),
        word_timing=False,
    )

    corrected, stats = fuse_text_index(cues, long_cue, config=CONFIG)

    assert corrected == cues
    assert stats["guarded"] == 1


def test_a_correction_may_not_drop_half_the_row() -> None:
    joined = "영양불균형으로 고민이신 분들! / 저처럼 키위 하나를 더해 보는 것도 추천!"
    cues = (cue(588.0, 592.0, joined),)
    short_cue = track((588.0, "저처럼 키위 하나를 더해 보는 것도 추천!"), word_timing=False)

    corrected, stats = fuse_text_index(cues, short_cue, config=CONFIG)

    assert corrected == cues
    assert stats["guarded"] == 1


def test_text_only_ocr_saw_is_kept() -> None:
    """Background text really is on the frame; the track not mentioning it is not evidence."""
    cues = (cue(60.0, 68.0, "pyrex"),)

    corrected, stats = fuse_text_index(
        cues, track((60.0, "완전히 다른 자막 문장입니다"), word_timing=False), config=CONFIG
    )

    assert corrected == cues
    assert stats["corrected"] == 0


def test_lines_only_the_track_has_are_not_added() -> None:
    """They have no position or size, and OCR never saw them."""
    cues = (cue(10.0, 14.0, "seen on screen"),)

    corrected, _ = fuse_text_index(
        cues,
        track((10.0, "seen on screen"), (200.0, "never on screen"), word_timing=False),
        config=CONFIG,
    )

    assert len(corrected) == 1
    assert corrected[0].text == "seen on screen"
