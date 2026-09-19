from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from vuc.config import VisualScanConfig
from vuc.visual_scan import (
    Candidate,
    ScanFrame,
    ShotBoundary,
    _extract_one,
    _extract_settle_frames,
    _settled_frame,
    dedupe,
    edge_density,
    is_flat,
    perceptual_hash,
    sample_candidates,
)


def make_config(**overrides: float | int) -> VisualScanConfig:
    base = {
        "scan_fps": 2.0,
        "scan_width": 320,
        "scene_threshold": 0.3,
        "min_interval_s": 2.0,
        "max_interval_s": 30.0,
        "boundary_offset_s": 0.25,
        "phash_distance": 12,
        "max_frames": 120,
        "workers": 2,
    }
    return VisualScanConfig(**{**base, **overrides})


def gaps(values: list[float], duration_s: float) -> list[float]:
    edges = [0.0, *values, duration_s]
    return [round(b - a, 3) for a, b in zip(edges, edges[1:], strict=False)]


def test_a_motionless_video_is_still_sampled_every_interval() -> None:
    """Nothing changes for ten minutes, and the coverage promise still holds."""
    candidates = sample_candidates([], config=make_config(), duration_s=600.0)

    timestamps = [candidate.timestamp_s for candidate in candidates]
    assert timestamps[0] == 0.0
    assert max(gaps(timestamps, 600.0)) <= 30.0


def test_filler_lands_inside_the_gaps_the_cuts_leave() -> None:
    """Cuts first; filler then spreads through whatever gap is left over.

    The interval is a bound on how long the video may go unwatched, not a clock
    to sample on, so nothing is expected at 30s or 60s merely because those are
    round numbers. Here 11.25 to 72.25 is 61 seconds and needs two frames to
    fall under the bound, and they are spread evenly rather than snapped to a
    grid that would leave one of them a second away from the cut at 72.25.
    """
    boundaries = [ShotBoundary(4.0, 0.8), ShotBoundary(11.0, 0.6), ShotBoundary(72.0, 0.9)]

    candidates = sample_candidates(boundaries, config=make_config(), duration_s=120.0)

    timestamps = [candidate.timestamp_s for candidate in candidates]
    assert [c.timestamp_s for c in candidates if c.at_shot_boundary] == [4.25, 11.25, 72.25]
    assert timestamps == sorted(timestamps)
    assert max(gaps(timestamps, 120.0)) <= 30.0
    assert timestamps == [0.0, 4.25, 11.25, 31.583, 51.917, 72.25, 96.125]


def test_a_cut_is_never_dropped_for_sitting_near_a_round_number() -> None:
    """The old grid claimed 30.0 and then refused the cut 0.25s away from it."""
    boundaries = [ShotBoundary(29.75, 0.9), ShotBoundary(59.75, 0.9)]

    candidates = sample_candidates(boundaries, config=make_config(), duration_s=120.0)

    assert [c.timestamp_s for c in candidates if c.at_shot_boundary] == [30.0, 60.0]


def test_a_burst_of_cuts_is_thinned_to_the_minimum_spacing() -> None:
    boundaries = [ShotBoundary(value, 0.5) for value in (10.0, 10.5, 11.0, 11.5, 13.0)]

    candidates = sample_candidates(boundaries, config=make_config(), duration_s=60.0)

    cuts = [candidate.timestamp_s for candidate in candidates if candidate.at_shot_boundary]
    assert cuts == [10.25, 13.25]


def test_the_frame_budget_only_ever_takes_back_cuts() -> None:
    """Coverage is the promise; cuts are the extras, so cuts are what give way.

    The budget also has to survive the thinning: the filler a thinned timeline
    needs is reserved up front, so cutting back never pushes the total over.
    """
    boundaries = [ShotBoundary(float(value), value / 100) for value in range(2, 60, 3)]
    config = make_config(max_frames=6, max_interval_s=20.0)

    candidates = sample_candidates(boundaries, config=config, duration_s=60.0)

    assert len(candidates) <= 6
    # The cuts that survived are the strongest ones; here the score rises with
    # the timestamp, so the late cuts are the ones that stay.
    cuts = [candidate.timestamp_s for candidate in candidates if candidate.at_shot_boundary]
    assert cuts == [53.25, 56.25, 59.25]
    timestamps = [candidate.timestamp_s for candidate in candidates]
    assert max(gaps(timestamps, 60.0)) <= 20.0


def test_a_flat_frame_and_a_gradient_do_not_share_a_hash() -> None:
    """A plain difference hash answers "no" everywhere on both of these."""
    flat = Image.new("L", (64, 64), color=128)
    gradient = Image.linear_gradient("L").resize((64, 64))

    assert perceptual_hash(flat) != perceptual_hash(gradient)


def test_text_raises_edge_density_above_a_photograph() -> None:
    plain = Image.new("RGB", (64, 64), color=(120, 120, 120))
    striped = Image.new("RGB", (64, 64), color=(255, 255, 255))
    for x in range(0, 64, 4):
        for y in range(64):
            striped.putpixel((x, y), (0, 0, 0))

    assert edge_density(striped) > edge_density(plain)


def frame(timestamp: float, phash: str, *, cut: bool = False) -> ScanFrame:
    return ScanFrame(timestamp, f"{timestamp}.jpg", phash, 0.1, cut)


def test_identical_frames_collapse_but_never_past_the_interval() -> None:
    same = "0" * 48
    frames = [frame(value, same) for value in (0.0, 30.0, 60.0, 90.0)]

    kept = dedupe(frames, config=make_config(), duration_s=120.0)

    assert [item.timestamp_s for item in kept] == [0.0, 30.0, 60.0, 90.0]


def test_a_cut_close_to_a_grid_point_does_not_open_a_hole() -> None:
    """Keeping the cut must not let the grid point next to it be dropped."""
    same = "0" * 48
    frames = [frame(0.0, same), frame(29.0, same, cut=True), frame(30.0, same), frame(60.0, same)]

    kept = dedupe(frames, config=make_config(), duration_s=60.0)

    timestamps = [item.timestamp_s for item in kept]
    assert timestamps == [0.0, 29.0, 30.0]
    assert max(gaps(timestamps, 60.0)) <= 30.0


def test_near_duplicates_inside_the_interval_are_dropped() -> None:
    same = "0" * 48
    other = "f" * 48
    frames = [frame(0.0, same), frame(2.0, same), frame(4.0, other), frame(6.0, other)]

    kept = dedupe(frames, config=make_config(), duration_s=8.0)

    assert [item.timestamp_s for item in kept] == [0.0, 4.0]


def test_a_cut_is_kept_even_when_the_two_shots_look_alike() -> None:
    same = "0" * 48
    frames = [frame(0.0, same), frame(2.0, same, cut=True)]

    kept = dedupe(frames, config=make_config(), duration_s=10.0)

    assert [item.timestamp_s for item in kept] == [0.0, 2.0]


def test_scan_frames_become_montage_artifacts(tmp_path: Path) -> None:
    path = tmp_path / "scan.jpg"
    Image.new("RGB", (16, 16)).save(path)
    artifact = ScanFrame(12.0, str(path), "0" * 48, 0.1, False).artifact

    assert artifact.timestamp_s == 12.0
    assert artifact.path == str(path)


def test_candidates_carry_whether_they_came_from_a_cut() -> None:
    assert Candidate(4.25, True, 0.9).at_shot_boundary
    assert not Candidate(0.0, False, float("inf")).at_shot_boundary


def test_a_blank_frame_is_not_content() -> None:
    assert is_flat(Image.new("RGB", (64, 64), color=(0, 0, 0)))
    assert is_flat(Image.new("RGB", (64, 64), color=(255, 255, 255)))
    assert is_flat(Image.new("RGB", (64, 64), color=(90, 90, 90)))


def test_a_picture_is_content_however_dim() -> None:
    """Low key is not blank: a dim shot still has structure to look at."""
    dim = Image.new("RGB", (64, 64), color=(20, 20, 20))
    for x in range(0, 64, 4):
        for y in range(64):
            dim.putpixel((x, y), (120, 120, 120))

    assert not is_flat(dim)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg required")
def test_batched_settle_decode_matches_individual_seeks(tmp_path: Path) -> None:
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
    config = make_config(settle_step_s=0.25, settle_max_s=1.0)
    batch = _extract_settle_frames(
        video,
        tmp_path / "batch.jpg",
        start_s=0.25,
        config=config,
        width=96,
        duration_s=2,
    )
    assert [timestamp for timestamp, _ in batch] == [0.25, 0.5, 0.75, 1.0]
    for index, (timestamp, actual_path) in enumerate(batch):
        expected_path = tmp_path / f"single-{index}.jpg"
        assert _extract_one(
            video, expected_path, timestamp_s=timestamp, width=96, hwaccel="none"
        )
        with Image.open(actual_path) as actual, Image.open(expected_path) as expected:
            assert perceptual_hash(actual) == perceptual_hash(expected)
            assert is_flat(actual) == is_flat(expected)


class _Reel:
    """A stand-in video: a timestamp maps to the picture showing at it."""

    def __init__(self, pictures: dict[float, Image.Image]) -> None:
        self.pictures = pictures
        self.asked: list[float] = []
        self.batches = 0

    def extract_settle_frames(
        self,
        _video: Path,
        output: Path,
        *,
        start_s: float,
        config: VisualScanConfig,
        width: int,
        duration_s: float,
    ) -> list[tuple[float, Path]]:
        del width
        self.batches += 1
        result = []
        for index in range(math.ceil(config.settle_max_s / config.settle_step_s)):
            timestamp_s = round(start_s + index * config.settle_step_s, 3)
            if timestamp_s >= duration_s:
                break
            self.asked.append(timestamp_s)
            nearest = min(self.pictures, key=lambda moment: abs(moment - timestamp_s))
            path = output.with_name(f"{output.stem}-probe-{index:02d}{output.suffix}")
            self.pictures[nearest].save(path)
            result.append((timestamp_s, path))
        return result


def _noise(seed: int) -> Image.Image:
    image = Image.new("RGB", (64, 64))
    # Blocky and pseudo-random on purpose: a smooth ramp survives the hash's
    # 8x8 downscale looking like every other smooth ramp, so two "different"
    # pictures built that way compare equal and prove nothing.
    state = seed * 7919 + 13
    for block_x in range(8):
        for block_y in range(8):
            state = (state * 1103515245 + 12345) % (1 << 31)
            shade = 20 + (state >> 16) % 216
            for x in range(block_x * 8, block_x * 8 + 8):
                for y in range(block_y * 8, block_y * 8 + 8):
                    image.putpixel((x, y), (shade, (shade * 3) % 256, (shade * 7) % 256))
    return image


def _settle(monkeypatch, reel: _Reel, tmp_path: Path, **overrides: float | int):
    monkeypatch.setattr("vuc.visual_scan._extract_settle_frames", reel.extract_settle_frames)
    return _settled_frame(
        Path("video.mp4"),
        tmp_path / "scan.jpg",
        candidate=Candidate(10.0, True, 0.9),
        config=make_config(**overrides),
        width=320,
        duration_s=60.0,
    )


def test_a_hard_cut_is_taken_where_it_was_asked_for(monkeypatch, tmp_path: Path) -> None:
    """The new shot is already there, so settling confirms it and stops."""
    shot = _noise(3)
    reel = _Reel({10.0: shot, 10.25: shot, 10.5: shot, 10.75: shot})

    taken = _settle(monkeypatch, reel, tmp_path)

    assert taken is not None and taken[1] == 10.0
    assert reel.batches == 1
    assert reel.asked[:2] == [10.0, 10.25]


def test_a_dissolve_is_followed_until_the_picture_stops_changing(
    monkeypatch, tmp_path: Path
) -> None:
    """A quarter second into a two-second dissolve is still both shots at once."""
    settled = _noise(7)
    reel = _Reel({10.0: _noise(1), 10.25: _noise(2), 10.5: settled, 10.75: settled})

    taken = _settle(monkeypatch, reel, tmp_path)

    assert taken is not None and taken[1] == 10.5


def test_a_cut_into_a_fade_steps_past_the_blank(monkeypatch, tmp_path: Path) -> None:
    black = Image.new("RGB", (64, 64), color=(0, 0, 0))
    picture = _noise(5)
    reel = _Reel({10.0: black, 10.25: black, 10.5: picture, 10.75: picture})

    taken = _settle(monkeypatch, reel, tmp_path)

    assert taken is not None and taken[1] == 10.5


def test_a_frame_that_never_settles_is_still_used(monkeypatch, tmp_path: Path) -> None:
    """Give up on the search, not on the frame: coverage outranks tidiness."""
    reel = _Reel({10.0 + step * 0.25: _noise(step + 1) for step in range(10)})

    taken = _settle(monkeypatch, reel, tmp_path)

    assert taken is not None and taken[1] == 10.0
    assert max(reel.asked) <= 10.0 + 1.5


def test_a_stretch_of_nothing_but_blank_yields_no_frame(monkeypatch, tmp_path: Path) -> None:
    black = Image.new("RGB", (64, 64), color=(0, 0, 0))
    reel = _Reel({10.0 + step * 0.25: black for step in range(10)})

    assert _settle(monkeypatch, reel, tmp_path) is None
