"""Rescan only measured, changing text slots; never search a new screen area."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from PIL import Image

from vuc.config import OCRConfig
from vuc.ocr import Observation, OCREngine, OCRLine, crop_fingerprint, same_text, text_key

Box = tuple[float, float, float, float]  # left, top, right, bottom


@dataclass(frozen=True)
class Region:
    box: Box
    rows: tuple[OCRLine, ...] = ()


def confidence(line: OCRLine) -> float:
    return 1.0 if line.confidence is None else line.confidence


def padded_box(lines: Sequence[OCRLine]) -> Box:
    h = max(line.height for line in lines)
    return (
        max(0.0, min(line.left for line in lines) - 2 * h),
        max(0.0, min(line.top for line in lines) - 0.5 * h),
        min(1.0, max(line.right for line in lines) + 2 * h),
        min(1.0, max(line.bottom for line in lines) + 0.5 * h),
    )


def same_slot(a: OCRLine, b: OCRLine) -> bool:
    h = max(a.height, b.height, 0.001)
    overlap = min(a.right, b.right) - max(a.left, b.left)
    width = max(0.001, min(a.right - a.left, b.right - b.left))
    return (
        min(a.height, b.height) >= 0.67 * h
        and abs(a.top + a.bottom - b.top - b.bottom) / 2 <= 0.35 * h
        and overlap >= 0.65 * width
    )


def changing_regions(
    observations: Sequence[Observation], *, config: OCRConfig
) -> list[tuple[float, float, Region]]:
    """A slot must show different legible readings at two nearby base samples.

    Stable logos do not qualify merely because the camera or their box moved.
    Slots expire after the rejoin gap; no video-long subtitle zone is invented.
    """
    anchors = [
        (o.timestamp_s, line)
        for o in observations
        if not o.discovery and not o.verification
        for line in o.lines
        if line.height >= config.min_text_height
        and len(text_key(line.text)) >= 3
        and confidence(line) >= config.min_confidence
    ]
    anchors.sort(key=lambda item: item[0])
    windows = []
    for i, (end, line) in enumerate(anchors):
        for j in range(i - 1, -1, -1):
            start, old = anchors[j]
            if end - start > config.rejoin_gap_s:
                break
            if (
                end > start
                and same_slot(old, line)
                and max(confidence(old), confidence(line)) >= config.singleton_confidence
                and not same_text(old.text, line.text, ratio=config.same_text_ratio)
            ):
                windows.append((start, end, Region(padded_box((old, line)), (old, line))))
                break
    return windows


def in_changing_region(
    timestamp: float, row: OCRLine, windows: Sequence[tuple[float, float, Region]]
) -> bool:
    return any(
        start <= timestamp <= end and any(same_slot(row, anchor) for anchor in region.rows)
        for start, end, region in windows
    )


def region_candidates(
    frames: Sequence[tuple[float, Path]],
    observations: Sequence[Observation],
    *,
    config: OCRConfig,
) -> dict[int, list[Region]]:
    windows = changing_regions(observations, config=config)
    base_times = {o.timestamp_s for o in observations}
    plans: dict[int, list[Region]] = defaultdict(list)

    # Adjacent slots reuse thumbnails, but a long video must not keep every
    # decoded image in memory. The temporal routing windows are local.
    @lru_cache(maxsize=64)
    def image_at(index: int) -> Image.Image:
        with Image.open(frames[index][1]) as source:
            return source.convert("L")

    for start, end, region in windows:
        box, rows = region.box, region.rows
        previous = None
        last_read = -float("inf")
        for index, (timestamp, _) in enumerate(frames):
            if not start <= timestamp <= end:
                continue
            image = image_at(index)
            pixels = (
                box[0] * image.width,
                box[1] * image.height,
                box[2] * image.width,
                box[3] * image.height,
            )
            fingerprint = int(crop_fingerprint(image, pixels), 16)
            if previous is None or timestamp in base_times:
                previous = fingerprint
                continue
            if timestamp - last_read >= 0.5 and (previous ^ fingerprint).bit_count() / 256 >= 0.12:
                add_region(plans, index, box, rows=rows)
                previous = fingerprint
                last_read = timestamp
    return dict(plans)


def add_region(
    plans: dict[int, list[Region]], index: int, box: Box, *, rows: tuple[OCRLine, ...] = ()
) -> None:
    regions = plans.setdefault(index, [])
    # Same-row jitter can produce overlapping crops. Read their union once;
    # preserve separate columns and rows, plus the actual routing evidence.
    pending = list(regions)
    kept: list[Region] = []
    while pending:
        old = pending.pop()
        a, b, c, d = old.box
        contains = (a <= box[0] and b <= box[1] and c >= box[2] and d >= box[3]) or (
            box[0] <= a and box[1] <= b and box[2] >= c and box[3] >= d
        )
        aligned = min(d, box[3]) - max(b, box[1]) >= 0.8 * min(d - b, box[3] - box[1]) and min(
            c, box[2]
        ) - max(a, box[0]) >= 0.6 * min(c - a, box[2] - box[0])
        if contains or aligned:
            box = (min(a, box[0]), min(b, box[1]), max(c, box[2]), max(d, box[3]))
            rows = tuple(dict.fromkeys((*rows, *old.rows)))
            pending.extend(kept)
            kept.clear()
        else:
            kept.append(old)
    regions[:] = [*kept, Region(box, rows)]


def read_regions(
    frames: Sequence[tuple[float, Path]],
    paths: dict[int, Path],
    plans: dict[int, list[Region]],
    engine: OCREngine,
    output_dir: Path,
    *,
    verification: bool = False,
) -> list[Observation]:
    jobs = []
    try:
        for index, boxes in sorted(plans.items()):
            with Image.open(paths[index]) as source:
                for region in boxes:
                    box = region.box
                    bounds = (
                        int(box[0] * source.width),
                        int(box[1] * source.height),
                        int(box[2] * source.width),
                        int(box[3] * source.height),
                    )
                    crop = source.crop(bounds)
                    path = output_dir / f"text-crop-{len(jobs):06d}.png"
                    crop.save(path)
                    jobs.append((index, path, bounds, source.size, region.rows))
        read = engine.read_many([job[1] for job in jobs], cropped=True) if jobs else []
    finally:
        for path in output_dir.glob("text-crop-*.png"):
            path.unlink(missing_ok=True)
    lines: dict[int, list[OCRLine]] = defaultdict(list)
    for (index, _, (x, y, right, bottom), (w, h), references), readings in zip(
        jobs, read, strict=True
    ):
        cw, ch = right - x, bottom - y
        for line in readings:
            # A crop must not publish a truncated sentence as complete text.
            margin = max(2, 0.15 * line.height * ch)
            if (
                (x > 0 and line.left * cw < margin)
                or (right < w and (1 - line.right) * cw < margin)
                or (y > 0 and line.top * ch < margin)
                or (bottom < h and (1 - line.bottom) * ch < margin)
            ):
                continue
            measured = replace(
                line,
                left=(x + line.left * cw) / w,
                right=(x + line.right * cw) / w,
                top=(y + line.top * ch) / h,
                bottom=(y + line.bottom * ch) / h,
                text_height=line.height * ch / h,
            )
            # Padding supplies recognition context, not permission to discover
            # another row or a smaller background label inside the crop.
            if not any(same_slot(measured, row) for row in references):
                continue
            lines[index].append(measured)
    return [
        Observation(
            frames[i][0],
            tuple(lines[i]),
            verification=verification,
            discovery=not verification,
            regions=tuple(region.box for region in plans[i]),
        )
        for i in sorted(plans)
    ]
