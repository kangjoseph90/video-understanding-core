from __future__ import annotations

import pytest

from vuc.timeline import Span, complement, from_milliseconds, merge_spans, split_long


def test_every_timestamp_leaves_as_a_float() -> None:
    span = Span(3, 9)

    assert isinstance(span.start, float)
    assert isinstance(span.end, float)
    assert span.to_dict() == {"start": 3.0, "end": 9.0}


def test_milliseconds_are_converted_at_the_boundary() -> None:
    assert from_milliseconds(1500) == 1.5
    assert from_milliseconds("250") == 0.25


def test_a_span_cannot_run_backwards() -> None:
    with pytest.raises(ValueError, match="precedes start"):
        Span(10, 4)


def test_rounding_out_never_narrows() -> None:
    assert Span(4.4, 9.1).rounded_out(60) == Span(4.0, 10.0)


def test_rounding_out_stops_at_the_end_of_the_video() -> None:
    """Widening is fine; claiming a second the video does not have is not."""
    assert Span(50.0, 59.9).rounded_out(59.9) == Span(50.0, 59.9)


def test_merge_joins_only_within_the_gap() -> None:
    spans = [Span(0, 5), Span(6, 10), Span(20, 25)]

    assert merge_spans(spans, gap_s=1) == [Span(0, 10), Span(20, 25)]
    assert merge_spans(spans) == spans


def test_complement_covers_the_tail_and_the_head() -> None:
    gaps = complement([Span(10, 20)], duration_s=30)

    assert gaps == [Span(0, 10), Span(20, 30)]


def test_split_preserves_the_outer_edges() -> None:
    pieces = split_long([Span(0, 70)], max_s=30)

    assert len(pieces) == 3
    assert pieces[0].start == 0.0
    assert pieces[-1].end == 70.0
    assert all(piece.duration <= 30 for piece in pieces)
