from __future__ import annotations

import json
from pathlib import Path

from vuc.captions import (
    CaptionCue,
    CaptionTrack,
    clean_text,
    content_hash,
    load_caption_track,
    parse_json3,
    parse_vtt,
    strip_conventions,
    token_stream,
    tokenize,
)


def json3(*events: dict) -> dict:
    return {"events": list(events)}


def event(start_ms: int, *segs: tuple[str, int | None]) -> dict:
    return {
        "tStartMs": start_ms,
        "segs": [{"utf8": text, "tOffsetMs": offset} for text, offset in segs],
    }


def test_entities_and_nbsp_are_removed_before_anything_compares_tracks() -> None:
    """The automatic alias of a human track differs only by these."""
    assert clean_text("head&nbsp; of policy") == "head of policy"
    assert clean_text("a\xa0b") == "a b"
    assert clean_text("Tom &amp; Jerry") == "Tom & Jerry"


def test_display_markup_and_line_breaks_do_not_survive() -> None:
    assert clean_text("<c.colorE5E5E5>hello</c>") == "hello"
    assert clean_text("<00:00:01.234>word") == "word"
    assert clean_text("first\nsecond") == "first second"


def test_sound_and_speaker_annotations_are_not_speech() -> None:
    assert strip_conventions("(mid-tempo lighthearted music)") == ""
    assert strip_conventions("- Hi, everybody") == "Hi, everybody"
    assert strip_conventions("I wanna flip it (laughs)") == "I wanna flip it"
    assert strip_conventions("【注意】これ") == "これ"


def test_a_long_parenthetical_is_left_alone() -> None:
    """The bound exists so a bracket cannot swallow a whole sentence."""
    long_aside = "(" + "x" * 80 + ")"

    assert strip_conventions(f"before {long_aside} after") == f"before {long_aside} after"


def test_word_timed_tracks_are_recognised_and_filler_events_dropped() -> None:
    payload = json3(
        event(3920, ("hello", None), ("everyone", 400)),
        {"tStartMs": 5190, "segs": [{"utf8": "\n"}]},
        event(5200, ("today", None), ("we", 400), ("talk", 800)),
    )

    cues, word_timed = parse_json3(payload)

    assert word_timed is True
    assert [cue.text for cue in cues] == ["hello", "everyone", "today", "we", "talk"]
    assert cues[1].start_s == 4.32


def test_one_segment_per_event_is_not_word_timing() -> None:
    payload = json3(
        event(1000, ("a whole caption line", None)),
        event(5000, ("another whole line", None)),
    )

    cues, word_timed = parse_json3(payload)

    assert word_timed is False
    assert len(cues) == 2


def test_vtt_cues_are_parsed_and_headers_ignored() -> None:
    payload = (
        "WEBVTT\nKind: captions\nLanguage: en\n\n"
        "00:00:01.500 --> 00:00:03.000\nfirst line\n\n"
        "00:00:04.000 --> 00:00:06.000\nsecond line\n"
    )

    cues, word_timed = parse_vtt(payload)

    assert word_timed is False
    assert [(cue.start_s, cue.text) for cue in cues] == [(1.5, "first line"), (4.0, "second line")]


def test_word_timed_tokens_keep_their_own_timestamps() -> None:
    track = CaptionTrack(
        language="en",
        kind="auto",
        source_format="json3",
        content_sha256="x",
        has_word_timing=True,
        cues=(CaptionCue(1.0, "hello"), CaptionCue(2.5, "everyone")),
    )

    assert token_stream(track) == ((1.0, "hello"), (2.5, "everyone"))


def test_cue_level_tokens_are_spread_across_the_cue_not_stacked_on_its_start() -> None:
    """A human track offers no word times, so the span to the next cue is used."""
    track = CaptionTrack(
        language="en",
        kind="manual",
        source_format="json3",
        content_sha256="x",
        has_word_timing=False,
        cues=(CaptionCue(0.0, "one two three four"), CaptionCue(4.0, "five")),
    )

    stream = token_stream(track)

    assert [token for _, token in stream] == ["one", "two", "three", "four", "five"]
    times = [at for at, _ in stream[:4]]
    assert times == [0.5, 1.5, 2.5, 3.5]


def test_the_last_cue_gets_a_span_estimated_from_its_length() -> None:
    track = CaptionTrack(
        language="en",
        kind="manual",
        source_format="json3",
        content_sha256="x",
        has_word_timing=False,
        cues=(CaptionCue(10.0, "final words here"),),
    )

    stream = token_stream(track)

    assert [token for _, token in stream] == ["final", "words", "here"]
    assert all(at > 10.0 for at, _ in stream)


def test_conventions_are_stripped_before_tokens_reach_the_timeline() -> None:
    track = CaptionTrack(
        language="en",
        kind="manual",
        source_format="json3",
        content_sha256="x",
        has_word_timing=True,
        cues=(CaptionCue(1.0, "(birds chirp)"), CaptionCue(2.0, "- real speech")),
    )

    assert [token for _, token in token_stream(track)] == ["real", "speech"]


def test_the_same_track_in_two_formats_hashes_the_same_so_aliases_fold() -> None:
    """This is how the automatic alias of a human track is recognised as one file."""
    from_json3, _ = parse_json3(json3(event(1000, ("head&nbsp; of policy", None))))
    from_vtt, _ = parse_vtt(
        "WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nhead of policy\n"
    )

    assert content_hash(from_json3) == content_hash(from_vtt)
    assert content_hash(from_json3) != content_hash((CaptionCue(1.0, "different"),))


def test_cjk_tokenises_per_character_so_unspaced_text_can_be_placed() -> None:
    """`\\w` covers Han and kana, so they must be taken out of the word run."""
    assert tokenize("안녕하세요 여러분") == ["안녕하세요", "여러분"]
    assert tokenize("東京にある") == ["東", "京", "に", "あ", "る"]
    assert tokenize("don't stop") == ["don't", "stop"]
    assert tokenize("café") == ["café"]


def test_no_sidecar_is_not_an_error() -> None:
    assert load_caption_track(Path("/nonexistent/video.mp4")) is None


def test_a_null_track_reads_as_no_track(tmp_path: Path) -> None:
    """The fetch ran and found nothing; that is an answer, not a failure."""
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    (tmp_path / "clip.subs.json").write_text(
        json.dumps({"audio_language": "ja", "track": None}), encoding="utf-8"
    )

    assert load_caption_track(video) is None


def test_a_malformed_sidecar_degrades_rather_than_raising(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    (tmp_path / "clip.subs.json").write_text("{not json", encoding="utf-8")

    assert load_caption_track(video) is None


def test_a_real_sidecar_round_trips(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    track = CaptionTrack(
        language="ko",
        kind="manual",
        source_format="json3",
        content_sha256="abc",
        has_word_timing=False,
        cues=(CaptionCue(1.0, "안녕하세요"),),
    )
    (tmp_path / "clip.subs.json").write_text(
        json.dumps({"track": track.to_dict()}, ensure_ascii=False), encoding="utf-8"
    )

    loaded = load_caption_track(video)

    assert loaded == track
