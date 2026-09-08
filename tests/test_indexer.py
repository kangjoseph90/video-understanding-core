from vuc.indexer import normalize_funasr_result, parse_rich_text


def test_parse_rich_text_preserves_semantic_tags() -> None:
    text, language, emotions, events = parse_rich_text(
        "<|ko|><|HAPPY|><|Speech|><|BGM|><|withitn|>안녕하세요"
    )

    assert text == "안녕하세요"
    assert language == "ko"
    assert emotions == ("happy",)
    assert events == ("Speech", "BGM")


def test_normalize_sentence_info_uses_absolute_vad_timestamps() -> None:
    result = [
        {
            "text": (
                "<|en|><|NEUTRAL|><|Speech|>Hello world "
                "<|ko|><|HAPPY|><|BGM|>안녕하세요"
            ),
            "sentence_info": [
                {
                    "start": 1250,
                    "end": 3750,
                    "text": "Hello world",
                },
                {
                    "start": 4000,
                    "end": 5500,
                    "text": "안녕하세요",
                },
            ]
        }
    ]

    segments = normalize_funasr_result(result, duration_s=10)

    assert len(segments) == 2
    assert segments[0].start == 1.25
    assert segments[0].end == 3.75
    assert segments[0].language == "en"
    assert segments[0].audio_events == ("Speech",)
    assert segments[1].language == "ko"
    assert segments[1].emotions == ("happy",)
    assert segments[1].audio_events == ("BGM",)
