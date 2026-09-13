"""A cheap full-video visual pass that decides where to look.

Fixed 15-second sampling has two failure modes at once: it spends frames on a
static slide that has not changed in four minutes, and it misses a caption card
that was on screen for three seconds. This pass finds the cuts first, samples
around them, and drops frames visually identical to the one before -- so a
lecture yields a handful of slide frames and a fast-cut vlog yields many,
without either being configured by hand.

Change-driven sampling alone would leave a motionless talking head unsampled
for ten minutes, so a coarse grid runs underneath it and is never thinned away:
whatever else happens, no stretch longer than max_interval_s goes unseen.

Everything here is ffmpeg and Pillow. No model runs, so the whole scan costs
roughly one decode of the video.
"""

from __future__ import annotations

import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageFilter

from vuc.config import VisualScanConfig
from vuc.media import MediaError, require_binary
from vuc.models import FrameArtifact

SCENE_LINE = re.compile(r"pts_time:(?P<time>[0-9.]+)")
SCORE_LINE = re.compile(r"lavfi\.scene_score=(?P<score>[0-9.]+)")


@dataclass(frozen=True)
class ShotBoundary:
    timestamp_s: float
    score: float


@dataclass(frozen=True)
class Candidate:
    timestamp_s: float
    at_shot_boundary: bool
    score: float


@dataclass(frozen=True)
class ScanFrame:
    """One sampled frame and the cheap signals read off it."""

    timestamp_s: float
    path: str
    phash: str
    edge_density: float
    at_shot_boundary: bool

    @property
    def artifact(self) -> FrameArtifact:
        return FrameArtifact(path=self.path, timestamp_s=self.timestamp_s)


@dataclass(frozen=True)
class ScanResult:
    frames: tuple[ScanFrame, ...]
    boundaries: tuple[ShotBoundary, ...]
    sampled: int


def detect_shots(
    video_path: Path,
    *,
    config: VisualScanConfig,
    duration_s: float,
) -> list[ShotBoundary]:
    """Scene-change timestamps from one decimated, downscaled ffmpeg pass."""
    scan_filter = (
        f"fps={config.scan_fps},scale={config.scan_width}:-2,"
        f"select='gt(scene,{config.scene_threshold})',metadata=print:file=-"
    )
    command = [
        require_binary("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        scan_filter,
        "-f",
        "null",
        "-",
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise MediaError(f"shot detection failed: {completed.stderr.strip()}")

    boundaries: list[ShotBoundary] = []
    pending: float | None = None
    for line in completed.stdout.splitlines():
        time_match = SCENE_LINE.search(line)
        if time_match:
            pending = float(time_match.group("time"))
            continue
        score_match = SCORE_LINE.search(line)
        if score_match and pending is not None:
            timestamp = min(pending, duration_s)
            boundaries.append(ShotBoundary(round(timestamp, 3), float(score_match.group("score"))))
            pending = None
    return boundaries


def sample_candidates(
    boundaries: list[ShotBoundary],
    *,
    config: VisualScanConfig,
    duration_s: float,
) -> list[Candidate]:
    """Every cut, then just enough filler to close any gap longer than the interval.

    The sampling is event-driven; `max_interval_s` is a bound on how long the
    video may go unwatched, not a clock to sample on. An earlier version read it
    as a clock -- it laid frames at 0s, 30s, 60s and so on and then admitted a
    cut only where one of those fixed slots left room. That inverted the
    priority: a moment when nothing happened outranked a moment when the picture
    changed, and most of the cuts it refused were refused for sitting near a
    round number, which is not a reason. Filler is now placed inside whatever
    gaps remain and spread evenly through them, so a densely cut video needs
    none at all.
    """
    spacing = max(config.min_interval_s, 0.0)
    # The first shot begins at the start of the video and no detector reports a
    # boundary there, so the opening is an event in its own right.
    events: list[Candidate] = [Candidate(0.0, False, float("inf"))]

    for boundary in sorted(boundaries, key=lambda item: item.timestamp_s):
        # Land just inside the new shot rather than on the dissolve itself.
        moment = round(boundary.timestamp_s + config.boundary_offset_s, 3)
        if not 0 <= moment < duration_s:
            continue
        # Spacing is measured against the moments already accepted -- cuts and
        # the opening -- and never against a position chosen by the clock.
        if moment - events[-1].timestamp_s < spacing:
            continue
        events.append(Candidate(moment, True, boundary.score))

    # Thinning and filling are not independent: dropping a cut can reopen a gap
    # that then has to be filled again, so the budget has to be settled by
    # measuring the finished timeline rather than by reserving for a worst case
    # that a densely cut video never reaches. Each pass strictly removes cuts,
    # so this ends -- normally on the second pass.
    picked = events + _coverage_filler(events, config=config, duration_s=duration_s)
    while config.max_frames and len(picked) > config.max_frames:
        surplus = len(picked) - config.max_frames
        weakest = sorted(
            (candidate for candidate in events if candidate.at_shot_boundary),
            key=lambda item: item.score,
        )[:surplus]
        if not weakest:
            # Only the coverage promise is left; it outranks the budget.
            break
        dropped = {id(candidate) for candidate in weakest}
        events = [candidate for candidate in events if id(candidate) not in dropped]
        picked = events + _coverage_filler(events, config=config, duration_s=duration_s)
    return sorted(picked, key=lambda item: item.timestamp_s)


def _coverage_filler(
    events: list[Candidate],
    *,
    config: VisualScanConfig,
    duration_s: float,
) -> list[Candidate]:
    """Frames spread through whatever gap is longer than the interval allows."""
    filler: list[Candidate] = []
    edges = [candidate.timestamp_s for candidate in events] + [duration_s]
    for start, end in zip(edges, edges[1:], strict=False):
        gap = end - start
        if gap <= config.max_interval_s:
            continue
        needed = math.ceil(gap / config.max_interval_s) - 1
        step = gap / (needed + 1)
        filler += [
            # Filler carries the coverage promise, so the budget cannot take it.
            Candidate(round(start + step * (index + 1), 3), False, float("inf"))
            for index in range(needed)
        ]
    return filler


def _extract_one(
    video_path: Path,
    output_path: Path,
    *,
    timestamp_s: float,
    width: int,
) -> Path | None:
    command = [
        require_binary("ffmpeg"),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{timestamp_s:.3f}",
        "-i",
        str(video_path),
        "-frames:v",
        "1",
        "-vf",
        f"scale=w='min(iw,{width})':h=-2",
        "-q:v",
        "3",
        str(output_path),
    ]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0 or not output_path.exists():
        return None
    return output_path


def is_flat(image: Image.Image) -> bool:
    """Whether the frame is effectively one colour.

    A cut into a fade is a real cut, so the detector reports it and the sample
    lands on the blank. The tile is then worse than absent: it costs a ninth of
    a montage and invites "the video cuts to black" from a reader who cannot
    know it was a quarter-second of a dissolve.
    """
    values = image.convert("L").resize((160, 90), Image.Resampling.BILINEAR).tobytes()
    mean = sum(values) / len(values)
    deviation = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
    return deviation < 20.0 or mean < 12.0 or mean > 243.0


def perceptual_hash(image: Image.Image) -> str:
    """192-bit hash: brightness pattern plus differences along both axes.

    A difference hash alone is blind in a way that matters here. Its bits ask
    "is this pixel brighter than the next one", so every smoothly increasing
    image answers no everywhere -- a flat grey frame and a gradient get the same
    hash and one of them is silently deduplicated away. Comparing each cell
    against the frame mean fixes that, and keeping both difference directions
    keeps texture and layout in the signature.
    """
    grey = image.convert("L")
    cells = grey.resize((8, 8), Image.Resampling.LANCZOS).tobytes()
    mean = sum(cells) / len(cells)
    average = 0
    for value in cells:
        average = (average << 1) | int(value > mean)

    wide = grey.resize((9, 8), Image.Resampling.LANCZOS).tobytes()
    horizontal = 0
    for row in range(8):
        for column in range(8):
            horizontal = (horizontal << 1) | int(
                wide[row * 9 + column] > wide[row * 9 + column + 1]
            )

    tall = grey.resize((8, 9), Image.Resampling.LANCZOS).tobytes()
    vertical = 0
    for row in range(8):
        for column in range(8):
            vertical = (vertical << 1) | int(tall[row * 8 + column] > tall[(row + 1) * 8 + column])
    return f"{average:016x}{horizontal:016x}{vertical:016x}"


def hamming(left: str, right: str) -> int:
    return bin(int(left, 16) ^ int(right, 16)).count("1")


def edge_density(image: Image.Image, *, threshold: int = 48) -> float:
    """Share of pixels sitting on a strong edge.

    Text -- slides, burned-in subtitles, UI chrome -- produces far more edge
    pixels than photographic content, so this is a usable free prior for
    "something here is worth paying OCR for".
    """
    if image.width < 2 or image.height < 2:
        return 0.0
    histogram = image.convert("L").filter(ImageFilter.FIND_EDGES).histogram()
    total = sum(histogram)
    return 0.0 if total == 0 else sum(histogram[threshold:]) / total


def _analyse(path: Path, candidate: Candidate, timestamp_s: float | None = None) -> ScanFrame:
    with Image.open(path) as source:
        image = source.convert("RGB")
        return ScanFrame(
            timestamp_s=candidate.timestamp_s if timestamp_s is None else timestamp_s,
            path=str(path),
            phash=perceptual_hash(image),
            edge_density=edge_density(image),
            at_shot_boundary=candidate.at_shot_boundary,
        )


def _settled_frame(
    video_path: Path,
    output_path: Path,
    *,
    candidate: Candidate,
    config: VisualScanConfig,
    width: int,
    duration_s: float,
) -> tuple[Path, float] | None:
    """The candidate's frame, stepped forward past a transition or a blank.

    Two failures share one answer. A cut offset of a quarter second is often
    still inside a dissolve, so the tile shows one shot ghosted over the next;
    and a cut into a fade lands on a flat frame that says nothing. Both end as
    soon as the picture stops changing and has some content in it, so a hard cut
    settles on the first attempt and costs one extra decode to confirm it.
    """
    moment = candidate.timestamp_s
    limit = moment + config.settle_max_s
    previous: tuple[float, str] | None = None
    first_usable: tuple[float, str] | None = None
    chosen: tuple[float, str] | None = None
    on_disk: tuple[float, str] | None = None

    while moment < duration_s:
        if _extract_one(video_path, output_path, timestamp_s=moment, width=width) is None:
            break
        with Image.open(output_path) as source:
            image = source.convert("RGB")
            flat, digest = is_flat(image), perceptual_hash(image)
        on_disk = (moment, digest)
        if flat:
            # A blank is never the answer, and it breaks the run of readings
            # that a settled picture would have to be part of.
            previous = None
        else:
            if first_usable is None:
                first_usable = (moment, digest)
            if previous is not None and hamming(previous[1], digest) <= config.phash_distance:
                # Two readings in a row agree, so the picture has stopped
                # changing. The earlier one is as close to the cut as it gets.
                chosen = previous
                break
            previous = (moment, digest)
        moment = round(moment + config.settle_step_s, 3)
        if moment >= limit:
            break

    # Give up on the search, not on the frame: coverage outranks tidiness.
    chosen = chosen or first_usable
    if chosen is None or on_disk is None:
        return None
    # The search usually ends one step past the frame it settled on. That step
    # is by definition the same picture, so it can stand in for it and save a
    # decode; anything else has to be fetched again.
    if on_disk[0] != chosen[0] and hamming(on_disk[1], chosen[1]) > config.phash_distance:
        _extract_one(video_path, output_path, timestamp_s=chosen[0], width=width)
    return output_path, chosen[0]


def dedupe(
    frames: list[ScanFrame], *, config: VisualScanConfig, duration_s: float
) -> list[ScanFrame]:
    """Drop frames indistinguishable from the last one kept.

    Two frames are never dropped. A cut is information regardless of how
    similar the shots either side of it look, and a frame is kept whenever
    dropping it would leave a stretch longer than max_interval_s unrepresented
    -- which is what keeps the coverage promise true for a static talking head,
    where every frame in the video is a near-duplicate of the one before.
    """
    kept: list[ScanFrame] = []
    for index, frame in enumerate(frames):
        if not kept:
            kept.append(frame)
            continue
        if frame.at_shot_boundary or hamming(frame.phash, kept[-1].phash) > config.phash_distance:
            kept.append(frame)
            continue
        following = frames[index + 1].timestamp_s if index + 1 < len(frames) else duration_s
        if following - kept[-1].timestamp_s > config.max_interval_s:
            kept.append(frame)
    return kept


def scan_video(
    video_path: Path,
    output_dir: Path,
    *,
    config: VisualScanConfig,
    duration_s: float,
    frame_width: int,
) -> ScanResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("scan-*.jpg", "frame-*.jpg"):
        for stale in output_dir.glob(pattern):
            stale.unlink()

    boundaries = detect_shots(video_path, config=config, duration_s=duration_s)
    candidates = sample_candidates(boundaries, config=config, duration_s=duration_s)

    digits = max(4, len(str(len(candidates))))
    with ThreadPoolExecutor(max_workers=config.workers, thread_name_prefix="vuc-scan") as pool:
        sampled = list(
            pool.map(
                lambda item: _settled_frame(
                    video_path,
                    output_dir / f"scan-{item[0] + 1:0{digits}d}.jpg",
                    candidate=item[1],
                    config=config,
                    width=frame_width,
                    duration_s=duration_s,
                ),
                list(enumerate(candidates)),
            )
        )

    analysed = [
        _analyse(taken[0], candidate, taken[1])
        for taken, candidate in zip(sampled, candidates, strict=True)
        if taken is not None
    ]
    kept = dedupe(analysed, config=config, duration_s=duration_s)
    keep_paths = {frame.path for frame in kept}
    for frame in analysed:
        if frame.path not in keep_paths:
            Path(frame.path).unlink(missing_ok=True)

    return ScanResult(
        frames=tuple(kept),
        boundaries=tuple(boundaries),
        sampled=len(analysed),
    )
