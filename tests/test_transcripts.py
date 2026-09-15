from __future__ import annotations

from pathlib import Path

from vuc.models import NON_SPEECH, SPEECH, Segment
from vuc.transcripts import (
    TranscriptRow,
    append_rows,
    apply_transcriptions,
    load_rows,
    rows_from_segments,
)


def row(start: float, text: str, *, region=(0.0, 30.0), clip=None, at="") -> TranscriptRow:
    clip = clip or region
    return TranscriptRow(
        start_s=start,
        end_s=start + 2.0,
        text=text,
        region_start=region[0],
        region_end=region[1],
        clip_start=clip[0],
        clip_end=clip[1],
        model="whisper",
        recorded_at=at,
    )


def speech(start: float, end: float, text: str) -> Segment:
    return Segment(start=start, end=end, text=text, language="en", kind=SPEECH)


def test_nothing_recorded_yet_is_not_an_error(tmp_path: Path) -> None:
    assert load_rows(tmp_path / "missing.jsonl") == ()


def test_rows_survive_a_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "asr.jsonl"
    append_rows(path, [row(1.0, "first"), row(4.0, "second")])

    assert [r.text for r in load_rows(path)] == ["first", "second"]


def test_a_later_call_appends_rather_than_replacing_the_file(tmp_path: Path) -> None:
    path = tmp_path / "asr.jsonl"
    append_rows(path, [row(1.0, "first", region=(0.0, 10.0))])
    append_rows(path, [row(20.0, "later", region=(15.0, 25.0))])

    assert [r.text for r in load_rows(path)] == ["first", "later"]


def test_the_call_that_heard_more_of_a_region_wins(tmp_path: Path) -> None:
    """A clip cut short by the requested interval missed the run-up."""
    path = tmp_path / "asr.jsonl"
    append_rows(path, [row(5.0, "clipped", region=(0.0, 30.0), clip=(20.0, 30.0), at="2026-01-02")])
    append_rows(path, [row(5.0, "whole", region=(0.0, 30.0), clip=(0.0, 30.0), at="2026-01-01")])

    kept = load_rows(path)

    assert [r.text for r in kept] == ["whole"]


def test_equal_coverage_goes_to_the_later_call(tmp_path: Path) -> None:
    path = tmp_path / "asr.jsonl"
    append_rows(path, [row(5.0, "older", at="2026-01-01")])
    append_rows(path, [row(5.0, "newer", at="2026-01-02")])

    assert [r.text for r in load_rows(path)] == ["newer"]


def test_every_sentence_of_the_winning_call_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "asr.jsonl"
    append_rows(path, [row(1.0, "one", at="2026-01-02"), row(5.0, "two", at="2026-01-02")])
    append_rows(path, [row(1.0, "stale", at="2026-01-01")])

    assert [r.text for r in load_rows(path)] == ["one", "two"]


def test_a_corrupt_line_does_not_lose_the_rest(tmp_path: Path) -> None:
    path = tmp_path / "asr.jsonl"
    append_rows(path, [row(1.0, "good", region=(0.0, 10.0))])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("{not json\n")
    append_rows(path, [row(20.0, "also good", region=(15.0, 25.0))])

    assert [r.text for r in load_rows(path)] == ["good", "also good"]


def test_only_speech_is_recorded() -> None:
    """Non-speech labels come from the tagger the index already ran."""
    lines = [
        {"start_s": 1.0, "end_s": 3.0, "kind": SPEECH, "text": "words"},
        {"start_s": 3.0, "end_s": 5.0, "kind": NON_SPEECH, "text": "", "events": ["music"]},
        {"start_s": 5.0, "end_s": 6.0, "kind": SPEECH, "text": "   "},
    ]

    rows = rows_from_segments(lines, region=(0.0, 10.0), clip=(0.0, 10.0), model="m")

    assert [r.text for r in rows] == ["words"]


def test_accumulated_work_replaces_what_the_index_heard() -> None:
    segments = (speech(0.0, 30.0, "Maya Rarick at Central Trovva"),)

    updated, stats = apply_transcriptions(
        segments, (row(2.0, "Maja Drabczyk at Centrum Cyfrowe"),), align_ratio_min=0.2
    )

    assert updated[0].text == "Maja Drabczyk at Centrum Cyfrowe"
    assert stats["regions_rewritten"] == 1


def test_regions_nothing_was_recorded_for_are_untouched() -> None:
    segments = (speech(0.0, 30.0, "heard"), speech(30.0, 60.0, "also heard"))

    updated, _ = apply_transcriptions(segments, (row(2.0, "rewritten"),), align_ratio_min=0.2)

    assert updated[1] == segments[1]


def test_no_accumulation_leaves_the_timeline_alone() -> None:
    segments = (speech(0.0, 30.0, "heard"),)

    updated, stats = apply_transcriptions(segments, (), align_ratio_min=0.2)

    assert updated == segments
    assert stats["rows"] == 0
