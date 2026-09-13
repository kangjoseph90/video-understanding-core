from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from tests.support import StubTagger, StubTranscriber
from vuc.audio import (
    EventTag,
    Region,
    SenseVoiceTagger,
    analyze_regions,
    canonical_label,
    create_event_tagger,
    keep_tags,
    timeline_regions,
)
from vuc.config import AudioEventConfig
from vuc.index_text import render_audio_index
from vuc.models import NON_SPEECH, SPEECH, AudioIndex, Segment
from vuc.timeline import Span

EVENTS = AudioEventConfig(
    tagger="stub",
    top_k=4,
    min_confidence=0.2,
    relative_floor=0.4,
    window_s=10.0,
    model_path="",
)


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is required")
    path = tmp_path / "audio.wav"
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=60",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(path),
        ],
        check=True,
    )
    return path


def run(audio: Path, tmp_path: Path, regions, *, transcriber=None, tagger=None):
    return analyze_regions(
        audio,
        regions,
        transcriber=transcriber or StubTranscriber(),
        tagger=tagger,
        events=EVENTS,
        clip_dir=tmp_path / "regions",
    )


def test_transcription_only_ever_runs_inside_speech_regions(audio: Path, tmp_path: Path) -> None:
    transcriber = StubTranscriber()
    regions = timeline_regions([Span(0, 10)], [Span(10, 30)])

    run(audio, tmp_path, regions, transcriber=transcriber, tagger=StubTagger())

    assert transcriber.calls == [10.0]


def test_non_speech_regions_still_produce_something(audio: Path, tmp_path: Path) -> None:
    """Silence is an observation, not an absence of one."""
    tagger = StubTagger([EventTag("Sizzle", 0.7), EventTag("Frying (food)", 0.5)])
    regions = timeline_regions([Span(0, 10)], [Span(10, 30)])

    result = run(audio, tmp_path, regions, tagger=tagger)

    assert tagger.calls == [10.0, 10.0]
    quiet = result.segments[1]
    assert quiet.kind == NON_SPEECH
    assert quiet.events == ("sizzle", "frying")


def test_every_region_yields_exactly_one_segment_in_order(audio: Path, tmp_path: Path) -> None:
    regions = timeline_regions([Span(0, 10), Span(20, 30)], [Span(10, 20), Span(30, 40)])

    result = run(audio, tmp_path, regions, tagger=StubTagger())

    assert [(item.start, item.end, item.kind) for item in result.segments] == [
        (0.0, 10.0, SPEECH),
        (10.0, 20.0, NON_SPEECH),
        (20.0, 30.0, SPEECH),
        (30.0, 40.0, NON_SPEECH),
    ]


def test_several_sentences_in_one_region_stay_one_line(audio: Path, tmp_path: Path) -> None:
    transcriber = StubTranscriber(
        [
            Segment(0.0, 4.0, "first sentence.", "en"),
            Segment(4.0, 9.0, "second sentence.", "en"),
        ]
    )

    result = run(audio, tmp_path, timeline_regions([Span(0, 10)], []), transcriber=transcriber)

    assert len(result.segments) == 1
    assert result.segments[0].text == "first sentence. second sentence."


def test_a_lone_full_stop_over_music_is_not_a_transcript(audio: Path, tmp_path: Path) -> None:
    """SenseVoice answers `nospeech` and a period; storing it would invent speech."""
    transcriber = StubTranscriber([Segment(0.0, 10.0, ".", "nospeech", events=("BGM",))])

    result = run(audio, tmp_path, timeline_regions([Span(0, 10)], []), transcriber=transcriber)

    assert result.segments[0].text == ""
    assert result.segments[0].events == ("music",)


def test_regions_are_tagged_when_no_tagger_is_available(audio: Path, tmp_path: Path) -> None:
    """Without a tagger the non-speech half is skipped, and nothing else breaks."""
    regions = timeline_regions([Span(0, 10)], [Span(10, 30)])

    result = run(audio, tmp_path, regions, tagger=None)

    assert [item.kind for item in result.segments] == [SPEECH]
    assert result.non_speech_regions == 1
    assert result.tagged_regions == 0


def test_uninformative_labels_are_dropped() -> None:
    tags = [
        EventTag("Speech", 0.9),
        EventTag("EMO_UNKNOWN", 0.9),
        EventTag("NEUTRAL", 0.9),
        EventTag("Silence", 0.9),
        EventTag("Applause", 0.9),
    ]

    assert keep_tags(tags, config=EVENTS) == ("applause",)


def test_the_raw_taxonomy_is_not_what_the_agent_sees() -> None:
    """Bracketed qualifiers and alternate spellings are the ontology, not information."""
    assert canonical_label("Chopping (food)") == "chopping"
    assert canonical_label("Frying (food)") == "frying"
    assert canonical_label("Chewing, mastication") == "chewing"
    assert canonical_label("Dishes, pots, and pans") == "dishes"
    assert canonical_label("Crumpling, crinkling") == "crumpling"


def test_one_name_per_sound_however_it_was_spelled() -> None:
    assert canonical_label("Water tap, faucet") == "running_water"
    assert canonical_label("Sink (filling or washing)") == "running_water"
    assert canonical_label("Hubbub, speech noise, speech babble") == "crowd"
    assert canonical_label("Walk, footsteps") == "footsteps"
    assert canonical_label("Musical instrument") == "music"
    assert canonical_label("BGM") == "music"


def test_a_category_is_not_a_sound() -> None:
    """`Animal` is the parent of Dog and Cat; it fires when the tagger cannot tell."""
    assert canonical_label("Animal") is None
    assert canonical_label("Domestic animals, pets") is None
    assert canonical_label("Sounds of things") is None
    # The leaves underneath it still come through.
    assert canonical_label("Cat") == "cat"
    assert canonical_label("Meow") == "meow"


def test_the_room_is_not_an_event() -> None:
    assert canonical_label("Inside, small room") is None
    assert canonical_label("Reverberation") is None
    assert canonical_label("Background noise") is None


def test_an_unlisted_label_keeps_its_cleaned_up_name() -> None:
    """Nothing real is lost for want of an alias."""
    assert canonical_label("Sizzle") == "sizzle"
    assert canonical_label("Zipper (clothing)") == "zipper"


def test_the_tail_below_the_taggers_best_answer_is_guesswork() -> None:
    """Measured on six seconds of background music; none of the rest was there."""
    tags = [
        EventTag("Music", 0.834),
        EventTag("Animal", 0.316),
        EventTag("Snake", 0.247),
        EventTag("Squish", 0.221),
    ]

    assert keep_tags(tags, config=EVENTS) == ("music",)


def test_labels_that_agree_with_the_top_one_all_survive() -> None:
    """Three names for one mouthful, all well above the floor."""
    tags = [
        EventTag("Biting", 0.549),
        EventTag("Crunch", 0.389),
        EventTag("Chewing, mastication", 0.254),
        EventTag("Inside, small room", 0.131),
    ]

    assert keep_tags(tags, config=EVENTS) == ("biting", "crunch", "chewing")


def test_a_lone_confident_label_is_not_penalised_by_the_relative_floor() -> None:
    assert keep_tags([EventTag("Chopping (food)", 0.321)], config=EVENTS) == ("chopping",)


def test_low_confidence_labels_are_dropped_but_unscored_ones_are_kept() -> None:
    tags = [EventTag("Applause", 0.05), EventTag("Laughter", None)]

    assert keep_tags(tags, config=EVENTS) == ("laughter",)


def test_different_detectors_agree_on_one_spelling() -> None:
    assert keep_tags([EventTag("BGM")], config=EVENTS) == ("music",)
    assert keep_tags([EventTag("Music")], config=EVENTS) == ("music",)
    assert keep_tags([EventTag("Clapping")], config=EVENTS) == ("applause",)


def test_sensevoice_is_the_fallback_tagger_and_needs_no_download(tmp_path: Path) -> None:
    tagger = create_event_tagger(
        AudioEventConfig(
            tagger="sensevoice",
            top_k=4,
            min_confidence=0.2,
            relative_floor=0.4,
            window_s=10.0,
            model_path="",
        ),
        sensevoice=StubTranscriber([Segment(0, 5, "", "nospeech", events=("Applause",))]),
    )
    clip = tmp_path / "clip.wav"
    clip.write_bytes(b"wav")

    assert isinstance(tagger, SenseVoiceTagger)
    assert tagger.tag(clip, duration_s=5.0) == [EventTag("Applause")]


def test_the_rendered_index_does_not_say_which_model_spoke() -> None:
    """One line per region, in order. Where a line came from is not the agent's problem."""
    segments = [
        Segment(0, 21, "choose between two or three licenses", "en", kind=SPEECH),
        Segment(21, 60, "", "unknown", events=("sizzle", "frying"), kind=NON_SPEECH),
        Segment(60, 75, "다시 이어서 설명드리면", "ko", kind=SPEECH),
    ]

    assert render_audio_index(segments) == (
        "[0-21] <en> choose between two or three licenses\n"
        "[21-60] <sizzle> <frying>\n"
        "[60-75] <ko> 다시 이어서 설명드리면"
    )


def test_a_region_that_yielded_nothing_is_left_out() -> None:
    """An empty row says nothing the neighbouring timestamps do not already show."""
    segments = [
        Segment(0, 10, "hello", "en", kind=SPEECH),
        Segment(10, 20, "", "unknown", kind=NON_SPEECH),
        Segment(20, 30, "goodbye", "en", kind=SPEECH),
    ]

    assert render_audio_index(segments).splitlines() == [
        "[0-10] <en> hello",
        "[20-30] <en> goodbye",
    ]


def test_the_partition_keeps_what_the_rendering_drops() -> None:
    """The ASR tool routes off these, so the empty regions have to survive."""
    segments = [
        Segment(0, 10, "hello", "en", kind=SPEECH),
        Segment(10, 20, "", "unknown", kind=NON_SPEECH),
    ]
    index = AudioIndex(tuple(segments))

    assert len(index.segments) == 2
    assert len(render_audio_index(index.segments).splitlines()) == 1
    assert AudioIndex.from_dict(index.to_dict()) == index


def test_region_kind_is_recorded_but_never_rendered() -> None:
    region = Region(Span(0, 10), NON_SPEECH)
    assert not region.is_speech
    assert "non_speech" not in render_audio_index([Segment(0, 10, "", "unknown", kind=NON_SPEECH)])


def test_a_long_silence_is_tagged_in_windows_not_whole(audio: Path, tmp_path: Path) -> None:
    """The region is however long the silence was; the tagger gets 10s at a time."""
    tagger = StubTagger()
    regions = timeline_regions([], [Span(0, 28)])

    run(audio, tmp_path, regions, tagger=tagger)

    # Equal windows rather than two full ones and a runt.
    assert len(tagger.calls) == 3
    assert all(call <= 10.0 for call in tagger.calls)
    assert sum(tagger.calls) == pytest.approx(28.0)


def test_windows_that_agree_are_rejoined(audio: Path, tmp_path: Path) -> None:
    """A minute of uninterrupted frying is one line, not six identical ones."""
    result = run(audio, tmp_path, timeline_regions([], [Span(0, 28)]), tagger=StubTagger())

    assert [(s.start, s.end, s.events) for s in result.segments] == [
        (0.0, 28.0, ("applause",))
    ]


def test_windows_that_differ_each_get_their_own_line(audio: Path, tmp_path: Path) -> None:
    class Changing:
        name = "changing"

        def __init__(self) -> None:
            self.seen = 0

        def tag(self, clip: Path, *, duration_s: float) -> list[EventTag]:
            del clip, duration_s
            self.seen += 1
            return [EventTag("Sizzle" if self.seen < 3 else "Applause", 0.8)]

    result = run(audio, tmp_path, timeline_regions([], [Span(0, 28)]), tagger=Changing())

    assert [(s.start, s.end, s.events) for s in result.segments] == [
        (0.0, 18.667, ("sizzle",)),
        (18.667, 28.0, ("applause",)),
    ]
