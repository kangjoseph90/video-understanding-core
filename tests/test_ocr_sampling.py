import shutil
import subprocess
from dataclasses import replace

import pytest
from PIL import Image, ImageDraw, ImageFont

from tests.test_ocr_index import CONFIG, line
from vuc.frames import extract_ocr_frames, extract_plain_frames
from vuc.ocr import Observation, ocr_candidates, text_cues
from vuc.ocr_tracking import verification_targets

DENSE = replace(CONFIG, scan_fps=4, singleton_confidence=0.95, min_text_height=0.025)


def samples(tmp_path, texts):
    result = []
    for i, text in enumerate(texts):
        image = Image.new("RGB", (448, 252), "white")
        ImageDraw.Draw(image).text(
            (160, 210), text, font=ImageFont.load_default(size=14), fill="black"
        )
        path = tmp_path / f"{i}.png"
        image.save(path)
        result.append((i / 4, path))
    return result


def test_whole_frame_scan_does_not_discover_new_subsecond_text(tmp_path):
    frames = samples(tmp_path, ["", "", "Next step", "Next step", "", ""])
    picked = ocr_candidates(frames, config=DENSE)
    assert 2 not in picked  # Only measured regions may get extra OCR.


def test_a_single_frame_flash_does_not_trigger_extra_ocr(tmp_path):
    frames = samples(tmp_path, ["", "", "Next step", "", "", ""])
    assert 2 not in ocr_candidates(frames, config=DENSE)


def test_static_video_is_not_read_four_times_per_second(tmp_path):
    frames = samples(tmp_path, ["Same heading"] * 33)
    assert ocr_candidates(frames, config=DENSE) == [0, 16, 32]


@pytest.mark.parametrize("text", ["다음 단계로", "Next step", "开始下一步", "次のステップ"])
def test_neighbour_confirms_a_short_caption_without_publishing_probe_noise(text):
    caption = line(text, confidence=0.93)
    normal = [Observation(0, ()), Observation(1, (caption,)), Observation(2, ())]
    assert set(verification_targets(normal, duration_s=3, config=DENSE)) == {1}
    assert text_cues(normal, duration_s=3, config=DENSE) == []
    verified = [
        Observation(
            t,
            (
                replace(caption, confidence=0.99),
                line("Unrelated noise", top=0.1, bottom=0.16, confidence=0.999),
            ),
            verification=True,
        )
        for t in (0.75, 1.25)
    ]
    cues = text_cues(normal + verified, duration_s=3, config=DENSE)
    assert [(c.text, c.start, c.end) for c in cues] == [(text, 1, 2)]


@pytest.mark.parametrize(
    "probe",
    [
        line("Different text", confidence=0.99),
        line("Next step", top=0.1, bottom=0.15, confidence=0.99),
        line("Next step", confidence=0.94),
        line("Next step", confidence=0.2),
    ],
)
def test_verification_requires_same_reading_location_and_a_clear_observation(probe):
    obs = [
        Observation(1, (line("Next step", confidence=0.93),)),
        Observation(1.25, (probe,), verification=True),
        Observation(2, ()),
    ]
    assert text_cues(obs, duration_s=3, config=DENSE) == []


@pytest.mark.parametrize("text", ["二", "山", "国", "11", "네"])
def test_verification_does_not_bypass_isolated_short_glyph_policy(text):
    row = line(text, confidence=0.99)
    obs = [Observation(1, (row,)), Observation(1.25, (row,), verification=True), Observation(2, ())]
    assert verification_targets(obs, duration_s=3, config=DENSE) == {}
    assert text_cues(obs, duration_s=3, config=DENSE) == []


def test_probe_does_not_close_or_extend_other_tracks():
    row = line("Persistent title", confidence=0.99)
    normal = [Observation(0, (row,)), Observation(2, (row,)), Observation(4, ())]
    probes = [Observation(t, (), verification=True) for t in (0.25, 1.75, 2.25)]
    assert text_cues(normal + probes, duration_s=5, config=DENSE) == text_cues(
        normal, duration_s=5, config=DENSE
    )


def test_duplicate_or_distant_probe_is_not_confirmation():
    row = line("Next step", confidence=0.99)
    for timestamp in (1, 3):
        obs = [
            Observation(1, (row,)),
            Observation(timestamp, (row,), verification=True),
            Observation(4, ()),
        ]
        assert text_cues(obs, duration_s=5, config=DENSE) == []


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_selected_decode_has_the_same_frames_and_timestamps(tmp_path):
    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=128x72:rate=20:duration=2",
            "-y",
            str(video),
        ],
        check=True,
    )
    full = extract_plain_frames(
        video, tmp_path / "full", prefix="frame", fps=4, width=128, duration_s=2
    )
    chosen = extract_plain_frames(
        video,
        tmp_path / "chosen",
        prefix="frame",
        fps=4,
        width=128,
        duration_s=2,
        indices=[0, 2, 3, 6, 7],
    )
    assert [t for t, _ in chosen] == [0, 0.5, 0.75, 1.5, 1.75]
    for index, (_, path) in zip([0, 2, 3, 6, 7], chosen, strict=True):
        assert Image.open(path).tobytes() == Image.open(full[index][1]).tobytes()


def test_extra_samples_do_not_create_a_text_slot_from_different_predictions():
    obs = [
        Observation(t, (line(text, confidence=0.99),), discovery=True)
        for t, text in [(0.5, "First guess"), (0.75, "Other guess"), (1.0, "Third guess")]
    ]
    assert text_cues(obs, duration_s=2, config=DENSE) == []


def test_extra_samples_need_exact_repeat_and_one_clear_reading():
    caption = line("Step 21", confidence=0.90)
    low = [Observation(t, (caption,), discovery=True) for t in (0.5, 0.75)]
    assert text_cues(low, duration_s=2, config=DENSE) == []
    clear = Observation(0.75, (replace(caption, confidence=0.99),), discovery=True)
    assert [c.text for c in text_cues([low[0], clear], duration_s=2, config=DENSE)] == ["Step 21"]
    changed = replace(clear, lines=(replace(caption, text="Step 22", confidence=0.99),))
    assert text_cues([low[0], changed], duration_s=2, config=DENSE) == []


def test_extra_glyphs_do_not_attach_to_a_confirmed_caption():
    caption = line("Start here", confidence=0.99)
    glyph = line("三", left=0.25, right=0.29, confidence=0.99)
    obs = [Observation(t, (caption, glyph), discovery=True) for t in (0.5, 0.75)]
    assert [c.text for c in text_cues(obs, duration_s=2, config=DENSE)] == ["Start here"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_many_disjoint_frames_do_not_exceed_ffmpeg_expression_depth(tmp_path):
    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x36:rate=10:duration=2",
            "-y",
            str(video),
        ],
        check=True,
    )
    selected = list(range(0, 400, 2))
    frames = extract_plain_frames(
        video,
        tmp_path / "chosen",
        prefix="frame",
        fps=200,
        width=64,
        duration_s=2,
        indices=selected,
    )
    assert [t for t, _ in frames] == [i / 200 for i in selected]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_scan_runs_ocr_only_on_selected_high_resolution_frames(tmp_path):
    from tests.support import StubOCREngine
    from vuc.ocr import scan_text

    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=white:size=448x252:rate=20:duration=8",
            "-y",
            str(video),
        ],
        check=True,
    )
    engine = StubOCREngine([line("Persistent text", confidence=0.99)])
    obs, sampled = scan_text(video, tmp_path / "scan", engine, config=DENSE, duration_s=8)
    assert sampled == 31
    assert len(engine.reads) == 2
    assert [o.timestamp_s for o in obs] == [0.5, 4.5]
    assert not list((tmp_path / "scan").glob("*.jpg"))
    assert (tmp_path / "scan/observations.jsonl").exists()


def test_three_exact_readings_can_verify_without_lowering_confidence_threshold():
    caption = line("Readable caption", confidence=0.89)
    two = [Observation(t, (caption,), discovery=True) for t in (1, 1.5)]
    assert text_cues(two, duration_s=2, config=DENSE) == []
    confirmed = Observation(1.25, (replace(caption, confidence=0.94),), verification=True)
    assert [c.text for c in text_cues([*two, confirmed], duration_s=2, config=DENSE)] == [
        "Readable caption"
    ]
    wrong = replace(confirmed, lines=(replace(caption, text="Different caption"),))
    assert text_cues([*two, wrong], duration_s=2, config=DENSE) == []
    assert text_cues([*two, two[0]], duration_s=2, config=DENSE) == []


def test_verification_follows_measured_motion_instead_of_the_old_best_box():
    first = line("Moving caption", top=0.232, bottom=0.319, confidence=0.86)
    moved = replace(first, top=0.256, bottom=0.343, confidence=0.85)
    between = replace(first, top=0.238, bottom=0.321, confidence=0.94)
    obs = [
        Observation(1, (first,), discovery=True),
        Observation(1.25, (between,), verification=True),
        Observation(1.5, (moved,), discovery=True),
        Observation(2, ()),
    ]
    assert [c.text for c in text_cues(obs, duration_s=3, config=DENSE)] == ["Moving caption"]


def test_a_subsecond_caption_does_not_render_as_a_zero_second_interval():
    from vuc.index_text import render_text_index
    from vuc.models import TextCue

    cues = [
        TextCue(1.25, 1.75, "Brief caption", "bottom center", "medium"),
        TextCue(3.25, 3.75, "Brief caption", "bottom center", "medium"),
    ]
    assert render_text_index(cues) == "[1-2; 3-4, bottom center, medium] Brief caption"


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_dense_clock_preserves_the_coarse_sample_centres(tmp_path):
    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=128x72:rate=20:duration=2",
            "-y",
            str(video),
        ],
        check=True,
    )
    coarse = extract_plain_frames(
        video, tmp_path / "coarse", prefix="f", fps=1, width=128, duration_s=2, first_center_s=0.5
    )
    dense = extract_plain_frames(
        video, tmp_path / "dense", prefix="f", fps=4, width=128, duration_s=2, first_center_s=0.5
    )
    for timestamp, path in coarse:
        matching = next(p for t, p in dense if t == timestamp)
        assert Image.open(path).tobytes() == Image.open(matching).tobytes()


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_joint_ocr_decode_preserves_both_independent_outputs(tmp_path):
    video = tmp_path / "source.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=128x72:rate=20:duration=2",
            "-y",
            str(video),
        ],
        check=True,
    )
    expected_dense = extract_plain_frames(
        video,
        tmp_path / "expected-dense",
        prefix="f",
        fps=4,
        width=96,
        duration_s=2,
        first_center_s=0.5,
    )
    expected_base = extract_plain_frames(
        video,
        tmp_path / "expected-base",
        prefix="f",
        fps=1,
        width=128,
        duration_s=2,
        first_center_s=0.5,
    )
    dense, base = extract_ocr_frames(
        video,
        tmp_path / "joint",
        scan_fps=4,
        scan_width=96,
        recognition_width=128,
        duration_s=2,
    )
    assert [timestamp for timestamp, _ in dense] == [timestamp for timestamp, _ in expected_dense]
    assert [timestamp for timestamp, _ in base] == [timestamp for timestamp, _ in expected_base]
    for actual, expected in zip(dense, expected_dense, strict=True):
        assert Image.open(actual[1]).tobytes() == Image.open(expected[1]).tobytes()
    for actual, expected in zip(base, expected_base, strict=True):
        assert Image.open(actual[1]).tobytes() == Image.open(expected[1]).tobytes()


def test_verification_targets_only_the_unsupported_row_in_a_frame():
    heading = line("Stable heading", top=0.1, bottom=0.16, confidence=0.99)
    caption = line("Short caption", confidence=0.93)
    observations = [
        Observation(0, (heading,)),
        Observation(1, (heading, caption)),
        Observation(2, (heading,)),
        Observation(3, ()),
    ]
    targets = verification_targets(observations, duration_s=4, config=DENSE)
    assert [[row.text for row in rows] for rows in targets.values()] == [["Short caption"]]
