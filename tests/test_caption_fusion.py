from __future__ import annotations

from dataclasses import replace

from vuc.caption_fusion import (
    AUDIO,
    DROP,
    MIXED,
    REJECTED_LANGUAGE,
    REJECTED_UNALIGNED,
    TEXT,
    AttributionConfig,
    assign_cues,
    caption_spans,
    fuse_audio_index,
    fuse_region,
    fuse_text_index,
    overlay_captions,
    route_cues,
    script_ratio,
)
from vuc.captions import CaptionCue, CaptionTrack
from vuc.models import (
    NON_SPEECH,
    SPEECH,
    AudioIndex,
    Segment,
    TextCue,
    TextIndex,
    VideoIndex,
    VideoMetadata,
    VisualIndex,
)
from vuc.pipeline import SCHEMA_VERSION

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




def make_index(segments=(), cues=(), duration=600.0) -> VideoIndex:
    return VideoIndex(
        schema_version=SCHEMA_VERSION,
        video=VideoMetadata(path="/tmp/v.mp4", sha256="h", duration_s=duration, size_bytes=1),
        audio=AudioIndex(tuple(segments)),
        text=TextIndex(tuple(cues)),
        visual=VisualIndex(),
        created_at="now",
    )


def overlay(segments, cues, track, *, language=None, config=CONFIG):
    return overlay_captions(
        make_index(segments, cues), track, audio_language=language, config=config
    )


def place(track):
    return track.cues, caption_spans(track)


# ------------------------------------------------------------------- axes


def test_script_ratio_separates_declared_language_from_actual_script() -> None:
    assert script_ratio("안녕하세요 여러분", "ko") == 1.0
    assert script_ratio("Hello everyone", "ko") == 0.0
    assert script_ratio("こんにちは", "ja") == 1.0


def test_latin_languages_have_no_script_signal_and_say_so() -> None:
    """None is not a pass or a fail; this axis simply cannot separate en from fr."""
    assert script_ratio("Hello everyone", "en") is None
    assert script_ratio("Bonjour", "fr") is None


def test_a_track_in_the_wrong_script_is_rejected_before_anything_is_routed() -> None:
    english = track((1.0, "Hello"), (2.0, "everyone"), language="ko")

    result = overlay((speech(0.0, 10.0, "x"),), (), english, language="ko")

    assert result.summary["verdict"] == REJECTED_LANGUAGE
    assert result.summary["script_ratio"] == 0.0
    assert result.summary["routing"] == {AUDIO: 0, TEXT: 0, DROP: 2}


def test_a_translated_track_is_stopped_by_the_language_gate_not_by_routing() -> None:
    """It is timed to the speech it translates, so every line looks like one."""
    japanese_audio = (speech(0.0, 30.0, "何か"),)
    english = track((1.0, "Hello everyone this is Akane"), language="ja", word_timing=False)

    result = overlay(japanese_audio, (), english, language="ja")

    assert result.summary["verdict"] == REJECTED_LANGUAGE
    assert result.segments == japanese_audio


# ---------------------------------------------------------------- routing


def test_lines_over_speech_are_routed_to_the_transcript() -> None:
    decisions = route_cues(
        track((1.0, "hello"), (2.0, "everyone")), [(0.0, 10.0)], (), config=CONFIG
    )

    assert decisions == [AUDIO, AUDIO]


def test_lines_matching_the_screen_but_not_the_speech_are_routed_to_the_text_index() -> None:
    text = "오늘도 새벽부터 편집하다가"
    decisions = route_cues(
        track((30.0, text), word_timing=False, language="ko"),
        [(0.0, 5.0)],
        (cue(30.0, 34.0, text),),
        config=CONFIG,
    )

    assert decisions == [TEXT]


def test_lines_matching_neither_observer_are_dropped() -> None:
    decisions = route_cues(
        track((500.0, "buy"), (501.0, "my"), (502.0, "merch")), [(0.0, 10.0)], (), config=CONFIG
    )

    assert decisions == [DROP, DROP, DROP]


def test_a_lecturer_reading_their_slides_still_routes_to_the_transcript() -> None:
    """Speech wins the line even when the same words are on the frame.

    Correcting screen text with spoken paraphrase is the one thing the text
    index refuses to do, and a slide lecture scores 0.50 on the OCR axis.
    """
    shared = "creative commons licenses"
    decisions = route_cues(
        track((1.0, shared), word_timing=False),
        [(0.0, 10.0)],
        (cue(0.0, 8.0, shared),),
        config=CONFIG,
    )

    assert decisions == [AUDIO]


def test_one_video_may_change_character_halfway_through() -> None:
    """A whole-track verdict cannot describe this; the window can."""
    narrated = tuple((float(t), "spoken words here") for t in range(5, 60, 5))
    silent = tuple((float(t), "화면에 적힌 문장입니다") for t in range(200, 260, 5))
    mixed = track(*narrated, *silent, word_timing=False, language="ko")
    screen = tuple(
        cue(float(t), float(t) + 4, "화면에 적힌 문장입니다") for t in range(200, 260, 5)
    )

    decisions = route_cues(mixed, [(0.0, 100.0)], screen, config=CONFIG)

    assert set(decisions[: len(narrated)]) == {AUDIO}
    assert set(decisions[len(narrated) :]) == {TEXT}


def test_the_window_is_not_delicate() -> None:
    narrated = tuple((float(t), "spoken words here") for t in range(5, 60, 5))
    silent = tuple((float(t), "화면에 적힌 문장입니다") for t in range(200, 260, 5))
    mixed = track(*narrated, *silent, word_timing=False, language="ko")
    screen = tuple(
        cue(float(t), float(t) + 4, "화면에 적힌 문장입니다") for t in range(200, 260, 5)
    )

    for window_s in (10.0, 20.0, 30.0, 45.0, 60.0):
        decisions = route_cues(
            mixed, [(0.0, 100.0)], screen, config=replace(CONFIG, window_s=window_s)
        )
        assert decisions.count(AUDIO) == len(narrated), window_s
        assert decisions.count(TEXT) == len(silent), window_s


# --------------------------------------------------------------- placement


def test_a_line_is_placed_where_it_starts() -> None:
    """The start is the one timestamp a track actually gives.

    The end is inferred from the next line, so a gap in the captions stretches
    the line before it across the whole gap and its middle lands in silence.
    """
    written = track((0.0, "one two three"), (10.0, "four"), word_timing=False)
    buckets, dropped = assign_cues(written.cues, [(0.0, 5.0), (5.0, 20.0)])

    assert [[c.text for c in b] for b in buckets] == [["one two three"], ["four"]]
    assert dropped == []


def test_a_gap_in_the_captions_does_not_drag_the_line_before_it() -> None:
    sparse = track((10.0, "spoken here"), (9999.0, "much later"), word_timing=False)

    buckets, dropped = assign_cues(sparse.cues, [(0.0, 50.0)])

    assert [c.text for c in buckets[0]] == ["spoken here"]
    assert [c.text for c in dropped] == ["much later"]


def test_every_line_lands_in_exactly_one_region_or_none() -> None:
    lines = (CaptionCue(1.0, "a"), CaptionCue(12.0, "b"), CaptionCue(99.0, "c"))
    buckets, dropped = assign_cues(lines, [(0.0, 10.0), (10.0, 20.0)])

    assert [[c.text for c in b] for b in buckets] == [["a"], ["b"]]
    assert [c.text for c in dropped] == ["c"]
    assert sum(len(b) for b in buckets) + len(dropped) == len(lines)


def test_a_line_is_never_split_across_two_regions() -> None:
    """Splitting on token timings returned the halves as bare word lists."""
    straddling = (CaptionCue(8.0, "a sentence that runs over the boundary"),)
    buckets, dropped = assign_cues(straddling, [(0.0, 10.0), (10.0, 20.0)])

    assert [len(b) for b in buckets] == [1, 0]
    assert dropped == []


# ------------------------------------------------------------------- audio


def test_where_the_captions_reach_the_captions_win() -> None:
    fused = fuse_region(
        "I'm Maya Rarick at Central Trovva",
        ["I'm Maja Drabczyk at Centrum Cyfrowe"],
        align_ratio_min=0.2,
    )

    assert fused == "I'm Maja Drabczyk at Centrum Cyfrowe"


def test_punctuation_survives_the_correction() -> None:
    """Rejoining tokens dropped every comma the caption and the ASR both had."""
    fused = fuse_region(
        "Hi everybody, spring has come.",
        ["Hi, everybody, spring has come."],
        align_ratio_min=0.2,
    )

    assert fused == "Hi, everybody, spring has come."


def test_sound_and_speaker_annotations_do_not_become_transcript() -> None:
    fused = fuse_region(
        "Hi everybody spring has come",
        ["- Hi, everybody, spring has come.", "(birds chirp)"],
        align_ratio_min=0.2,
    )

    assert fused == "Hi, everybody, spring has come."


def test_tokens_the_captions_omit_are_not_restored() -> None:
    """Restoring them measured worse than leaving the index alone entirely."""
    fused = fuse_region("um so I think uh yes", ["so I think yes"], align_ratio_min=0.2)

    assert fused == "so I think yes"


def test_a_region_the_captions_do_not_reach_keeps_what_was_heard() -> None:
    assert fuse_region("what was heard", [], align_ratio_min=0.2) == "what was heard"


def test_an_empty_transcript_is_recovered_from_the_captions() -> None:
    assert fuse_region("", ["recovered words"], align_ratio_min=0.2) == "recovered words"


def test_two_accounts_that_do_not_resemble_each_other_are_left_alone() -> None:
    """Better an uncorrected region than one spliced into what neither said."""
    fused = fuse_region(
        "the quick brown fox jumps",
        ["completely unrelated advertising copy"],
        align_ratio_min=0.2,
    )

    assert fused == "the quick brown fox jumps"


def test_non_speech_regions_are_never_touched() -> None:
    segments = (
        speech(0.0, 5.0, "wrong words"),
        Segment(5.0, 10.0, "", "unknown", events=("music",), kind=NON_SPEECH),
    )
    lines, _ = place(track((1.0, "right"), (2.0, "words")))

    fused, stats = fuse_audio_index(segments, lines, config=CONFIG)

    assert fused[1] == segments[1]
    assert fused[0].text == "right words"
    assert stats["speech_regions"] == 1


def test_fusion_preserves_every_field_but_the_text() -> None:
    original = Segment(0.0, 5.0, "old", "ko", emotion="happy", events=("laugh",), raw_text="raw")
    lines, _ = place(track((1.0, "old"), (2.0, "new")))

    fused, _ = fuse_audio_index((original,), lines, config=CONFIG)

    assert fused[0].text == "old new"
    assert fused[0].language == "ko"
    assert fused[0].emotion == "happy"
    assert fused[0].events == ("laugh",)
    assert fused[0].raw_text == "raw"
    assert (fused[0].start, fused[0].end, fused[0].kind) == (0.0, 5.0, SPEECH)


def test_a_recovered_empty_region_is_counted_as_such() -> None:
    lines, _ = place(track((1.0, "found"), (2.0, "speech")))

    fused, stats = fuse_audio_index((speech(0.0, 5.0, ""),), lines, config=CONFIG)

    assert fused[0].text == "found speech"
    assert stats["regions_recovered"] == 1
    assert stats["regions_rewritten"] == 1


def test_the_guard_is_reported_rather_than_silently_skipping() -> None:
    segments = (speech(0.0, 5.0, "the quick brown fox jumps over"),)
    lines, _ = place(track((1.0, "completely"), (2.0, "unrelated"), (3.0, "advertising")))

    fused, stats = fuse_audio_index(segments, lines, config=CONFIG)

    assert fused == segments
    assert stats["regions_guarded"] == 1
    assert stats["regions_rewritten"] == 0


# -------------------------------------------------------------------- text
def test_a_misread_line_is_corrected_from_the_track() -> None:
    cues = (cue(54.0, 58.0, "영양 균형 행기기 1일차,오늘은 점심 도시락부터"),)
    fixed = "영양 균형 챙기기 1일차, 오늘은 점심 도시락부터"
    lines, spans = place(track((54.0, fixed), word_timing=False))
    corrected, stats = fuse_text_index(cues, lines, spans, config=CONFIG)

    assert corrected[0].text == "영양 균형 챙기기 1일차, 오늘은 점심 도시락부터"
    assert stats["corrected"] == 1


def test_position_and_size_come_from_ocr_because_the_track_has_none() -> None:
    cues = (TextCue(10.0, 14.0, "듬뿐 올려주고", "top left", "large"),)
    lines, spans = place(track((10.0, "듬뿍 올려주고"), word_timing=False))
    corrected, _ = fuse_text_index(cues, lines, spans, config=CONFIG)

    assert corrected[0].text == "듬뿍 올려주고"
    assert corrected[0].position == "top left"
    assert corrected[0].size == "large"


def test_a_correction_may_not_absorb_a_neighbouring_row() -> None:
    """Measured failure: the row took the previous line's text and duplicated it."""
    cues = (cue(40.0, 44.0, "한국인 3명 중 1명이 영양불균형이라고 해서"),)
    lines, spans = place(
        track(
            (40.0, "마침 제스프리 캠페인을 보는데 한국인 3명 중 1명이 영양불균형이라고 해서"),
            word_timing=False,
        )
    )

    corrected, stats = fuse_text_index(cues, lines, spans, config=CONFIG)

    assert corrected == cues
    assert stats["guarded"] == 1


def test_a_correction_may_not_drop_half_the_row() -> None:
    joined = "영양불균형으로 고민이신 분들! / 저처럼 키위 하나를 더해 보는 것도 추천!"
    cues = (cue(588.0, 592.0, joined),)
    half = "저처럼 키위 하나를 더해 보는 것도 추천!"
    lines, spans = place(track((588.0, half), word_timing=False))

    corrected, stats = fuse_text_index(cues, lines, spans, config=CONFIG)

    assert corrected == cues
    assert stats["guarded"] == 1


def test_text_only_ocr_saw_is_kept() -> None:
    """Background text really is on the frame; the track not mentioning it is not evidence."""
    cues = (cue(60.0, 68.0, "pyrex"),)

    lines, spans = place(track((60.0, "완전히 다른 자막 문장입니다"), word_timing=False))
    corrected, stats = fuse_text_index(cues, lines, spans, config=CONFIG)

    assert corrected == cues
    assert stats["corrected"] == 0


def test_lines_only_the_track_has_are_not_added() -> None:
    """They have no position or size, and OCR never saw them."""
    cues = (cue(10.0, 14.0, "seen on screen"),)

    lines, spans = place(
        track((10.0, "seen on screen"), (200.0, "never on screen"), word_timing=False)
    )
    corrected, _ = fuse_text_index(cues, lines, spans, config=CONFIG)

    assert len(corrected) == 1
    assert corrected[0].text == "seen on screen"


# ----------------------------------------------------------------- overlay


def test_a_video_that_narrates_then_goes_quiet_feeds_both_indexes() -> None:
    """The case a whole-track verdict files entirely as one or the other."""
    # A plausible mis-transcription of the same eleven lines, so the two
    # accounts resemble each other and the alignment guard does not fire.
    segments = (
        speech(0.0, 60.0, " ".join(["spokn words here"] * 11)),
        Segment(60.0, 260.0, "", "unknown", events=("music",), kind=NON_SPEECH),
    )
    # The OCR reading differs from the track only past the matching head, which
    # is the shape a real misread takes: recognisable line, one wrong syllable.
    screen = tuple(
        cue(float(t), float(t) + 4, "화면에 적힌 문장입니다 오늘도 펀집하다가")
        for t in range(200, 260, 5)
    )
    narrated = tuple((float(t), "spoken words here") for t in range(5, 60, 5))
    silent = tuple(
        (float(t), "화면에 적힌 문장입니다 오늘도 편집하다가") for t in range(200, 260, 5)
    )
    mixed = track(*narrated, *silent, word_timing=False, language="ko")

    result = overlay(segments, screen, mixed, language="ko")

    assert result.summary["verdict"] == MIXED
    assert result.summary["routing"][AUDIO] == len(narrated)
    assert result.summary["routing"][TEXT] == len(silent)
    assert "spokn" not in result.segments[0].text
    assert result.segments[0].text.startswith("spoken words here")
    assert result.segments[1] == segments[1]
    assert any(before.text != after.text for before, after in zip(screen, result.cues, strict=True))


def test_a_track_nothing_can_be_done_with_leaves_the_index_alone() -> None:
    segments = (speech(0.0, 10.0, "heard"),)
    result = overlay(segments, (), track((500.0, "buy"), (501.0, "merch")))

    assert result.summary["verdict"] == REJECTED_UNALIGNED
    assert result.segments == segments


def test_no_track_is_not_a_failure() -> None:
    segments = (speech(0.0, 10.0, "heard"),)
    result = overlay(segments, (), None)

    assert result.summary is None
    assert result.applied is False
    assert result.segments == segments
