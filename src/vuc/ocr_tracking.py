"""Temporal text identities backed by frame-local layout evidence.

A track owns a measured line, not a screen cell. Text that lives for minutes is
kept independently of nearby changing text. Frame-local geometry establishes
block relationships; temporal coexistence decides which remain valid for the
output. No dictionary corrections or generated text are used.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from difflib import SequenceMatcher
from functools import cached_property
from statistics import median

from vuc.config import OCRConfig
from vuc.models import TextCue
from vuc.ocr import Observation, OCRLine, normalize_text, reading_order, same_text, text_key
from vuc.ocr_layout import RegionLine, contains_region, frame_regions, visible_regions

SizeSample = tuple[float, float, float]
SizeEvidence = dict[str, list[SizeSample]]
NUMBER_PATTERN = re.compile(r"[+-]?\d+(?:[.,:/-]\d+)*")


def _confidence(line: OCRLine) -> float:
    return 1.0 if line.confidence is None else line.confidence


def _number_tokens(text: str) -> list[str]:
    return NUMBER_PATTERN.findall(normalize_text(text))


def _is_single_digit_extension(base: str, extended: str) -> bool:
    """Whether `extended` adds one digit at an edge of a nonnumeric reading."""

    base_key, extended_key = text_key(base), text_key(extended)
    if len(base_key) < 4 or _number_tokens(base) or len(extended_key) != len(base_key) + 1:
        return False
    if extended_key.startswith(base_key):
        extra = extended_key[-1]
    elif extended_key.endswith(base_key):
        extra = extended_key[0]
    else:
        return False
    return extra.isdigit()


def _height(line: OCRLine) -> float:
    return line.height


def _area(line: OCRLine) -> float:
    return max(line.right - line.left, 0.001) * max(line.bottom - line.top, 0.001)


def _novel_numbers(candidate: str, reference: str) -> bool:
    """A crop can omit a value, but cannot invent or change one."""
    remaining = iter(_number_tokens(reference))
    return any(not any(old == number for old in remaining) for number in _number_tokens(candidate))


def _same_size_span(line: OCRLine, sample: SizeSample) -> bool:
    """Whether a raw reading measures the same horizontal rendering.

    Size evidence must survive small detector shifts without merging a small
    logo with a larger occurrence of the same words. Compare continuous spans
    instead of quantising coordinates: adjacent 5% buckets have a hard boundary
    even when their boxes differ by only a pixel.
    """

    left, right, _ = sample
    line_width = max(line.right - line.left, 0.001)
    sample_width = max(right - left, 0.001)
    width_similarity = min(line_width, sample_width) / max(line_width, sample_width)
    line_center = (line.left + line.right) / 2
    sample_center = (left + right) / 2
    return width_similarity >= 0.67 and abs(line_center - sample_center) <= max(
        0.05, 0.25 * max(line_width, sample_width)
    )


def spatial_overlap(a: OCRLine, b: OCRLine) -> float:
    """Line overlap, tolerant of jitter and partially cropped text."""
    ha, hb = max(a.bottom - a.top, 0.001), max(b.bottom - b.top, 0.001)
    wa, wb = max(a.right - a.left, 0.001), max(b.right - b.left, 0.001)
    y = max(0.0, min(a.bottom, b.bottom) - max(a.top, b.top)) / min(ha, hb)
    x = max(0.0, min(a.right, b.right) - max(a.left, b.left)) / min(wa, wb)
    if max(ha, hb) / min(ha, hb) > 2.5:
        return 0.0
    return min(y, x)


@dataclass
class ReadingEvidence:
    best: OCRLine
    observations: int = 1


@dataclass
class Track:
    start: float
    last: float
    box: OCRLine
    missed: float | None = None
    hits: int = 0
    heights: list[float] = field(default_factory=list)
    readings: dict[str, ReadingEvidence] = field(default_factory=dict)
    pending_numeric: tuple[float, OCRLine] | None = None
    cropped_since: float | None = None

    samples: dict[float, OCRLine] = field(default_factory=dict)

    @cached_property
    def representative(self) -> OCRLine:
        # Consecutive video frames are correlated measurements, so repetition
        # establishes the track but does not outvote its clearest OCR reading.
        # Observation count only breaks an exact confidence tie.
        readings = list(self.readings.values())
        complete = [
            reading
            for reading in readings
            if not any(
                len(text_key(reading.best.text)) < 0.8 * len(text_key(other.best.text))
                and other.observations >= 2
                and contains_region(other.best, reading.best)
                and _area(other.best) > 1.2 * _area(reading.best)
                for other in readings
                if other is not reading
            )
        ]
        best = max(
            complete or readings,
            key=lambda reading: (
                _confidence(reading.best),
                reading.observations,
                len(reading.best.text),
            ),
        ).best
        # Detector boxes fluctuate around the size floor. Use one robust height
        # for the whole track so a long-lived tiny element cannot leak out as a
        # one-frame cue merely because one box happened to measure taller.
        height = median(self.heights) if self.heights else best.height
        return replace(best, text_height=height)

    def observe(self, timestamp: float, line: OCRLine) -> None:
        if self.readings:
            best = self.representative
            cropped = (
                text_key(line.text) != text_key(best.text)
                and contains_region(best, line)
                and _area(line) < 0.8 * _area(best)
            )
            if cropped:
                if self.cropped_since is None:
                    self.cropped_since = timestamp
            else:
                self.cropped_since = None
        self.__dict__.pop("representative", None)
        self.samples[timestamp] = line
        self.last, self.box, self.missed = timestamp, line, None
        self.hits += 1
        self.heights.append(line.height)
        key = text_key(line.text)
        canonical_key = None
        for old_key, old_reading in self.readings.items():
            coverage = min(len(key), len(old_key)) / max(len(key), len(old_key), 1)
            established_or_substantial = self.hits > 3 or (key in old_key and coverage >= 0.6)
            line_width = max(line.right - line.left, 0.001)
            old_width = max(old_reading.best.right - old_reading.best.left, 0.001)
            narrower_crop = len(key) < len(old_key) and (
                line_width < 0.8 * old_width or _area(line) < 0.8 * _area(old_reading.best)
            )
            if (
                established_or_substantial
                and (
                    contains_region(old_reading.best, line)
                    or contains_region(line, old_reading.best)
                )
                and not _novel_numbers(line.text, old_reading.best.text)
                and not (
                    len(key) > 1.25 * len(old_key) and _area(line) > 1.2 * _area(old_reading.best)
                )
                and (
                    (contains_region(old_reading.best, line) and coverage < 0.85)
                    or narrower_crop
                    or _confidence(line) <= _confidence(old_reading.best)
                )
            ):
                canonical_key = old_key
                break
        if canonical_key is not None:
            self.readings[canonical_key].observations += 1
            return
        if key not in self.readings:
            self.readings[key] = ReadingEvidence(line)
        else:
            reading = self.readings[key]
            reading.observations += 1
            if _confidence(line) > _confidence(reading.best):
                reading.best = line

    def defer_numeric(self, timestamp: float, line: OCRLine) -> None:
        """Hold one uncertain digit extension until another sample confirms it."""

        self.last, self.missed = timestamp, None
        self.pending_numeric = (timestamp, line)


def _numeric_probe_matches(track: Track, line: OCRLine) -> bool:
    """Allow one lower-quality digit extension without committing a text change."""

    best = track.representative
    if _is_single_digit_extension(best.text, line.text):
        if track.pending_numeric is not None:
            return text_key(track.pending_numeric[1].text) == text_key(line.text)
        return track.hits >= 2 and _confidence(line) < _confidence(best)
    # A cleaner base reading immediately after one numeric singleton is also a
    # correction, not two separate elements.
    return (
        track.hits == 1
        and _is_single_digit_extension(line.text, best.text)
        and _confidence(line) > _confidence(best)
    )


def _recentred(line: OCRLine, at: OCRLine) -> OCRLine:
    """`line`'s measurements, translated to the centre of `at`."""
    height, width = line.bottom - line.top, line.right - line.left
    y, x = (at.top + at.bottom) / 2, (at.left + at.right) / 2
    moved = replace(
        line,
        top=max(0.0, y - height / 2),
        bottom=min(1.0, y + height / 2),
        left=max(0.0, x - width / 2),
        right=min(1.0, x + width / 2),
    )
    if isinstance(line, RegionLine):
        dx, dy = x - (line.left + line.right) / 2, y - (line.top + line.bottom) / 2
        moved = replace(
            moved,
            rows=tuple(
                replace(
                    row,
                    left=row.left + dx,
                    right=row.right + dx,
                    top=row.top + dy,
                    bottom=row.bottom + dy,
                )
                for row in line.rows
            ),
        )
    return moved


def _match_score(track: Track, line: OCRLine, config: OCRConfig) -> float:
    overlap = spatial_overlap(track.box, line)
    # A zoom or cut changes the measurable rendering. Do not carry its size,
    # position and neighbouring paragraph across unrelated screen layouts.
    old_height = max(_height(track.box), 0.001)
    new_height = max(_height(line), 0.001)
    if min(old_height, new_height) / max(old_height, new_height) < 0.6:
        return 0.0
    # Track readable text moving with an object or a small camera pan. A grid
    # boundary or a one-line vertical shift is not a new appearance. The
    # reading is compared the way the rest of this module compares readings:
    # demanding an exact match here meant one misread glyph in a scrolling
    # chat panel -- `argue for or` read once as `argue tor or` -- started a
    # second track for a sentence that had never left the screen.
    if overlap < 0.45 and same_text(
        track.representative.text, line.text, ratio=config.same_text_ratio
    ):
        a = track.box
        dx = abs((a.left + a.right - line.left - line.right) / 2)
        dy = abs((a.top + a.bottom - line.top - line.bottom) / 2)
        if len(text_key(line.text)) >= 4 and max(dx, dy) <= 0.12:
            overlap = 0.45
    if overlap < 0.45:
        return 0.0
    best = track.representative
    a, b = text_key(best.text), text_key(line.text)
    coverage = min(len(a), len(b)) / max(len(a), len(b), 1)
    contained = (
        coverage >= 0.6
        and min(len(a), len(b)) >= 2
        and (contains_region(best, line) or contains_region(line, best))
        and overlap >= 0.8
    )
    if contained:
        old_numbers = _number_tokens(best.text)
        new_numbers = _number_tokens(line.text)
        # Keep quantity changes separate. The sole provisional exception is a
        # one-sample, lower-quality edge digit on an established nonnumeric
        # reading; a second sample confirms the change from its first timestamp.
        if (
            old_numbers != new_numbers
            and not (contains_region(best, line) and not _novel_numbers(line.text, best.text))
            and _confidence(line) >= config.singleton_confidence
            and not _numeric_probe_matches(track, line)
        ):
            contained = False
    if not contained and not same_text(best.text, line.text, ratio=config.same_text_ratio):
        return 0.0
    exact = a == b
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    similarity = matcher.ratio()
    # How many characters of the longer reading are not accounted for. A glyph
    # or two is recogniser noise; a whole word is a different sentence. Ratios
    # cannot tell those apart, because one changed word in a long paragraph
    # scores as high as one changed letter in a short one.
    changed = max(len(a), len(b)) - sum(block.size for block in matcher.get_matching_blocks())
    visual_agreement = False
    if track.box.appearance and line.appearance:
        distance = (int(track.box.appearance, 16) ^ int(line.appearance, 16)).bit_count() / 256
        visual_agreement = distance <= 0.08
    # Two clear but different readings may be successive sentences differing in
    # one word. Fuzzy matching is for uncertainty, not semantic paraphrasing.
    if (
        not exact
        and not (contained and contains_region(best, line))
        and not visual_agreement
        and changed > 2
        and min(_confidence(best), _confidence(line)) >= 0.95
    ):
        return 0.0
    return (2.0 if exact else similarity) + overlap


def _present(lines: Sequence[OCRLine], config: OCRConfig) -> list[OCRLine]:
    kept: list[OCRLine] = []
    for line in sorted(lines, key=lambda x: (-_confidence(x), x.top, x.left, x.text)):
        height = line.bottom - line.top
        width = line.right - line.left
        if not text_key(line.text) or _confidence(line) < config.min_confidence:
            continue
        # The recognizer is horizontal. An extreme vertical crop read with low
        # confidence is not usable vertical text; it is usually several rotated
        # letters hallucinated as a horizontal phrase. Preserve clear vertical
        # readings for CJK material.
        if height > max(width * 3.0, 0.08) and _confidence(line) < config.singleton_confidence:
            continue
        # Only overlapping detections are duplicates; identical labels in two
        # columns of a table are independent observations.
        if any(
            spatial_overlap(line, old) > 0.8 and text_key(line.text) == text_key(old.text)
            for old in kept
        ):
            continue
        kept.append(line)
    # Join detector fragments before applying the text-height floor. A useful
    # row can contain a 2.4%-high fragment beside a 2.6%-high fragment; dropping
    # the former first turns one stable line into a stream of partial tracks.
    rows: list[list[OCRLine]] = []
    # Search the geometric row, not the immediately preceding reading. OCR
    # traversal can interleave a neighbouring column between a marker and its
    # text; list order must not decide whether they join.
    for line in sorted(kept, key=lambda item: (item.left, item.top)):
        candidates = []
        for i, fragments in enumerate(rows):
            old, anchor = fragments[-1], fragments[0]
            h = max(0.001, min(old.bottom - old.top, line.bottom - line.top))
            y = min(old.bottom, line.bottom) - max(old.top, line.top)
            gap = line.left - old.right
            dy = abs(line.top + line.bottom - anchor.top - anchor.bottom) / 2
            if (
                y / h >= 0.6
                and -0.5 * h <= gap <= 0.6 * h
                and dy <= 0.5 * min(anchor.bottom - anchor.top, line.bottom - line.top)
                and not (
                    len(text_key(line.text)) >= 4
                    and any(text_key(line.text) in text_key(part.text) for part in fragments)
                )
            ):
                candidates.append((abs(gap), i))
        if candidates:
            _, i = min(candidates)
            rows[i].append(line)
        else:
            rows.append([line])
    result = []
    for fragments in rows:
        if len(fragments) == 1:
            # Each sample has its own identity even if a caller reuses an
            # immutable OCRLine instance in several observations.
            result.append(replace(fragments[0]))
        else:
            result.append(
                OCRLine(
                    text=" ".join(line.text for line in fragments),
                    confidence=min(_confidence(line) for line in fragments),
                    top=min(line.top for line in fragments),
                    bottom=max(line.bottom for line in fragments),
                    left=min(line.left for line in fragments),
                    right=max(line.right for line in fragments),
                    text_height=median(line.height for line in fragments),
                )
            )
    return reading_order(result)


def _track_observations(
    observations: Sequence[Observation], *, duration_s: float, config: OCRConfig
) -> list[tuple[Track, float]]:
    """One association engine for row identities and their observed regions."""
    active: list[Track] = []
    completed: list[tuple[Track, float]] = []

    def close(track: Track, *, end_at: float | None = None) -> None:
        end = min(
            duration_s,
            track.last + config.max_hold_s,
            duration_s if track.missed is None else track.missed,
        )
        if end_at is not None:
            end = min(end, end_at)
        # Brief detector fragmentation can rejoin. Sustained occlusion does
        # not prove that the complete reading remained visible underneath it.
        if (
            track.cropped_since is not None
            and track.last - track.cropped_since >= config.rejoin_gap_s
        ):
            end = min(end, track.cropped_since)
        if end > track.start:
            completed.append((track, end))

    for observation in sorted(observations, key=lambda o: o.timestamp_s):
        now = observation.timestamp_s
        if now < 0 or now >= duration_s:
            continue
        eligible = []
        for track in active:
            # Expire BEFORE matching: the old implementation joined equal text
            # across arbitrarily long gaps when it happened to reappear.
            limit = config.max_hold_s if track.missed is None else config.rejoin_gap_s
            if now - track.last > limit:
                close(track)
            else:
                eligible.append(track)
        active = eligible
        present = observation.lines
        edges = []
        active_keys = Counter(text_key(t.representative.text) for t in active)
        present_keys = Counter(text_key(line.text) for line in present)
        for i, track in enumerate(active):
            for j, line in enumerate(present):
                score = _match_score(track, line, config)
                key = text_key(line.text)
                # A unique, identical phrase can move across the screen. We
                # track its visibility, not the identity of the object carrying
                # it. Ambiguous repeated table labels still require geometry.
                if (
                    not score
                    and track.missed is None
                    and len(key) >= 4
                    and key == text_key(track.representative.text)
                    and same_text(track.representative.text, line.text, ratio=1.0)
                    and active_keys[key] == present_keys[key] == 1
                    and min(track.box.right - track.box.left, line.right - line.left)
                    >= 0.6 * max(track.box.right - track.box.left, line.right - line.left)
                    and min(track.box.bottom - track.box.top, line.bottom - line.top)
                    >= 0.6 * max(track.box.bottom - track.box.top, line.bottom - line.top)
                ):
                    score = 1.0
                if score:
                    edges.append((-score, i, j))
        used_tracks, used_lines = set(), set()
        for _, i, j in sorted(edges):
            if i in used_tracks or j in used_lines:
                continue
            track, line = active[i], present[j]
            if track.pending_numeric is not None and text_key(
                track.pending_numeric[1].text
            ) == text_key(line.text):
                pending_at, pending_line = track.pending_numeric
                close(track, end_at=pending_at)
                replacement = Track(pending_at, pending_at, pending_line)
                replacement.observe(pending_at, pending_line)
                replacement.observe(now, line)
                active[i] = replacement
            elif _numeric_probe_matches(track, line) and _is_single_digit_extension(
                track.representative.text, line.text
            ):
                track.defer_numeric(now, line)
            else:
                track.pending_numeric = None
                track.observe(now, line)
                if (
                    track.cropped_since is not None
                    and now - track.cropped_since >= config.rejoin_gap_s
                ):
                    # A sustained crop is a real visibility change. End the
                    # complete reading at its first cropped sample, then track
                    # the actually visible remainder independently. Otherwise
                    # an old title can survive for minutes through its logo.
                    cropped_at = track.cropped_since
                    suffix = [(t, sample) for t, sample in track.samples.items() if t >= cropped_at]
                    track.samples = {
                        t: sample for t, sample in track.samples.items() if t < cropped_at
                    }
                    close(track, end_at=cropped_at)
                    first_at, first_line = suffix[0]
                    replacement = Track(first_at, first_at, first_line)
                    for timestamp, sample in suffix:
                        replacement.observe(timestamp, sample)
                    active[i] = replacement
            used_tracks.add(i)
            used_lines.add(j)
        survivors = []
        for i, track in enumerate(active):
            if i not in used_tracks:
                if observation.discovery and not any(
                    left <= track.box.left
                    and top <= track.box.top
                    and right >= track.box.right
                    and bottom >= track.box.bottom
                    for left, top, right, bottom in observation.regions
                ):
                    # A crop is no evidence about text elsewhere in the frame.
                    survivors.append(track)
                    continue
                if track.missed is None:
                    track.missed = now
                # A replacement at the same box is positive evidence of a
                # change. A mere missing detection gets the occlusion grace.
                replaced = any(
                    j not in used_lines
                    and spatial_overlap(track.box, line) > 0.65
                    and (_confidence(line) >= config.singleton_confidence or track.hits < 2)
                    for j, line in enumerate(present)
                )
                if replaced or now - track.last > config.rejoin_gap_s:
                    close(track)
                    continue
            survivors.append(track)
        active = survivors
        for j, line in enumerate(present):
            if j not in used_lines:
                track = Track(now, now, line)
                track.observe(now, line)
                active.append(track)
    for track in active:
        close(track)
    return completed


def _supported_rows(
    tracks: Sequence[tuple[Track, float]],
    *,
    config: OCRConfig,
    verified: set[int] | None = None,
    trusted_times: set[float] | None = None,
) -> set[int]:
    """Require repetition of either the reading or its measured text slot.

    Fast captions may each appear in only one sampled frame. A stable text slot
    in neighbouring samples supplies independent evidence without assuming a
    subtitle location, alphabet, screen type, or expected words. Isolated OCR
    predictions, however confident, do not establish such a slot on their own.
    """
    samples = sorted(
        (t, serial, line)
        for serial, (track, _) in enumerate(tracks)
        for t, line in track.samples.items()
        if trusted_times is None or t in trusted_times
    )
    from bisect import bisect_left, bisect_right

    times = [t for t, _, _ in samples]
    # A legible row can be missed on the second scan of an otherwise stable
    # paragraph. Two independently repeated neighbouring rows establish that
    # layout too; requiring the same slot alone would punch holes in a slide
    # or ingredient list. Isolated glyphs have no such context.
    frames: dict[float, list[OCRLine]] = {}
    owners = {}
    for timestamp, serial, sample in samples:
        if _confidence(sample) >= config.singleton_confidence:
            frames.setdefault(timestamp, []).append(sample)
            owners[id(sample)] = serial
    context_supported = set()
    for lines in frames.values():
        for region in frame_regions(lines):
            if not isinstance(region, RegionLine):
                continue
            peers = {owners[id(row)] for row in region.rows if row.height > config.min_text_height}
            repeated = {
                i
                for i in peers
                if sum(trusted_times is None or t in trusted_times for t in tracks[i][0].samples)
                >= 2
            }
            if len(repeated) >= 2:
                context_supported.update(peers)
    supported = set()
    for serial, (track, _) in enumerate(tracks):
        best = track.representative
        key = text_key(best.text)
        if len(key) <= 2 and _confidence(best) < config.singleton_confidence:
            continue
        count = sum(trusted_times is None or t in trusted_times for t in track.samples)
        if count >= 2 or (verified is not None and id(best) in verified):
            supported.add(id(best))
            continue
        if count == 0:
            continue
        if _confidence(best) < config.singleton_confidence:
            continue
        if serial in context_supported:
            supported.add(id(best))
            continue
        nearby = set()
        radius = config.rejoin_gap_s
        low = bisect_left(times, track.start - radius)
        high = bisect_right(times, track.start + radius)
        for timestamp, _, other in samples[low:high]:
            if timestamp == track.start:
                continue
            h = max(best.height, other.height, 0.001)
            if min(best.height, other.height) < 0.75 * h:
                continue
            if abs(best.top + best.bottom - other.top - other.bottom) / 2 > 0.3 * h:
                continue
            alignment = min(
                abs(best.left - other.left),
                abs(best.right - other.right),
                abs(best.left + best.right - other.left - other.right) / 2,
            )
            if alignment <= 0.6 * h:
                nearby.add(timestamp)
        if len(nearby) >= 2:
            supported.add(id(best))
    return supported


def _prepare_observations(
    observations: Sequence[Observation], config: OCRConfig
) -> list[Observation]:
    return [
        Observation(
            o.timestamp_s,
            tuple(
                _present(
                    [line for line in o.lines if not o.discovery or len(text_key(line.text)) >= 3],
                    config,
                )
            ),
            discovery=o.discovery,
            regions=o.regions,
        )
        for o in sorted(observations, key=lambda o: o.timestamp_s)
        if not o.verification
    ]


def verification_targets(
    observations: Sequence[Observation],
    *,
    duration_s: float,
    config: OCRConfig,
) -> dict[float, list[OCRLine]]:
    """Request neighbours only for readable rows lacking sufficient support.

    One/two-glyph predictions remain under the existing conservative policy.
    More samples must not be an automatic escape hatch for isolated shapes.
    """
    prepared = _prepare_observations(observations, config)
    tracks = _track_observations(prepared, duration_s=duration_s, config=config)
    supported = _supported_rows(
        tracks,
        config=config,
        trusted_times={o.timestamp_s for o in prepared if not o.discovery},
        verified=_verified_rows(tracks, observations, config=config),
    )
    targets: dict[float, list[OCRLine]] = {}
    for track, _ in tracks:
        row = track.representative
        if (
            len(text_key(row.text)) >= 3
            and row.height >= config.min_text_height
            and id(row) not in supported
        ):
            timestamp = max(track.samples, key=lambda t: _confidence(track.samples[t]))
            targets.setdefault(timestamp, []).append(track.samples[timestamp])
    return targets


def _verified_rows(
    tracks: Sequence[tuple[Track, float]],
    observations: Sequence[Observation],
    *,
    config: OCRConfig,
) -> set[int]:
    from bisect import bisect_left, bisect_right

    probes = sorted(observations, key=lambda o: o.timestamp_s)
    times = [o.timestamp_s for o in probes]
    rows = [(*probe.lines, *_present(probe.lines, config)) for probe in probes]
    verified = set()
    radius = max(0.5, 1.0 / config.scan_fps) + 0.001
    for track, _ in tracks:
        best = track.representative
        key = text_key(best.text)
        if len(key) < 3:
            continue
        anchor = max(
            track.samples,
            key=lambda t: (text_key(track.samples[t].text) == key, _confidence(track.samples[t])),
        )
        agreement = {anchor}
        clearest = _confidence(best)
        for i in range(bisect_left(times, anchor - radius), bisect_right(times, anchor + radius)):
            if times[i] == anchor:
                continue
            # Animated captions move inside a shot. Verify against the nearest
            # measured box on the track, not the representative's old position.
            reference = track.samples[min(track.samples, key=lambda t: abs(t - times[i]))]
            # Compare raw detections as well as assembled rows: an unrelated
            # glyph beside the caption must not invalidate its clean reading.
            for line in rows[i]:
                if (
                    text_key(line.text) == key
                    and min(_confidence(best), _confidence(line)) >= config.min_confidence
                    and spatial_overlap(reference, line) >= 0.8
                    and min(reference.height, line.height)
                    >= 0.75 * max(reference.height, line.height)
                ):
                    agreement.add(times[i])
                    clearest = max(clearest, _confidence(line))
                    break
        # Confidence alone is unstable near its threshold. Two measurements
        # need one clear reading; otherwise require three exact, colocated
        # readings. This establishes presence, never a frequency-voted spelling.
        if len(agreement) >= 3 or (len(agreement) >= 2 and clearest >= config.singleton_confidence):
            verified.add(id(best))
    return verified


def track_cues(
    observations: Sequence[Observation], *, duration_s: float, config: OCRConfig
) -> list[TextCue]:
    """Learn co-lived layout, track actual region snapshots, assign one owner.

    Line identities are internal correspondence evidence. Published paragraphs
    always originate in one frame; neighbours with different lifetimes cannot
    drag a persistent label into every successive caption.
    """
    prepared = [
        o for o in _prepare_observations(observations, config) if 0 <= o.timestamp_s < duration_s
    ]
    row_tracks = _track_observations(prepared, duration_s=duration_s, config=config)
    size_evidence: SizeEvidence = {}
    for observation in observations:
        if observation.verification:
            continue
        for line in observation.lines:
            if text_key(line.text) and _confidence(line) >= config.min_confidence:
                size_evidence.setdefault(text_key(line.text), []).append(
                    (line.left, line.right, line.height)
                )
    # Dense acquisition improves evidence; it must not tighten the existing
    # one-second allowance for fragmented/growing text.
    tolerance = min(2.0, max(1.0, 1.0 / config.scan_fps))
    row_runs = [(track.start, end, track.representative) for track, end in row_tracks]
    eligible = {
        id(line)
        for _, _, line in _without_small_tracks(
            row_runs,
            min_text_height=config.min_text_height,
            tolerance=tolerance,
            size_evidence=size_evidence,
        )
    }
    eligible &= _supported_rows(
        row_tracks,
        config=config,
        verified=_verified_rows(row_tracks, observations, config=config),
        trusted_times={o.timestamp_s for o in prepared if not o.discovery},
    )
    identities = {}
    for track, _ in row_tracks:
        if id(track.representative) in eligible:
            for line in track.samples.values():
                identities[id(line)] = track
    compatibility: dict[tuple[int, int], bool] = {}

    def co_lived(a: OCRLine, b: OCRLine) -> bool:
        x, y = identities.get(id(a)), identities.get(id(b))
        if x is None or y is None or x is y:
            return False
        pair = tuple(sorted((id(x), id(y))))
        if pair not in compatibility:
            shared = len(x.samples.keys() & y.samples.keys())
            compatibility[pair] = shared >= 0.6 * max(len(x.samples), len(y.samples))
        return compatibility[pair]

    regions = [
        Observation(
            o.timestamp_s,
            tuple(
                frame_regions(
                    [line for line in o.lines if id(line) in identities],
                    compatible=co_lived,
                )
            ),
            discovery=o.discovery,
            regions=o.regions,
        )
        for o in prepared
    ]
    tracks = _track_observations(regions, duration_s=duration_s, config=config)
    runs = []
    for track, end in tracks:
        best = track.representative
        runs.append((round(track.start, 3), round(end, 3), _recentred(best, track.box)))
    runs = _without_contained_fragments(runs, tolerance=tolerance)
    runs = visible_regions(runs)
    return sorted(
        [TextCue(start, end, line.text, line.position, line.size) for start, end, line in runs],
        key=lambda cue: (cue.start, cue.position, cue.text),
    )


def _without_contained_fragments(
    runs: Sequence[tuple[float, float, OCRLine]],
    *,
    tolerance: float,
) -> list[tuple[float, float, OCRLine]]:
    """Drop partial tracks covered by a longer reading of the same text.

    Streaming text often grows while its container scrolls. The shorter and
    longer readings then occupy the same column but touch in time instead of
    overlapping at the same box. Treat that as one evolving line while keeping
    later recurrences and identical labels in separate columns independent.
    """

    result = []
    for index, candidate in enumerate(runs):
        start, end, line = candidate
        key = text_key(line.text)
        duration = end - start
        covered = False
        for other_index, host in enumerate(runs):
            if index == other_index:
                continue
            host_start, host_end, host_line = host
            host_key = text_key(host_line.text)
            overlap = min(end, host_end) - max(start, host_start)
            candidate_width = max(line.right - line.left, 0.001)
            host_width = max(host_line.right - host_line.left, 0.001)
            x_overlap = max(
                0.0,
                min(line.right, host_line.right) - max(line.left, host_line.left),
            ) / min(candidate_width, host_width)
            candidate_y = (line.top + line.bottom) / 2
            host_y = (host_line.top + host_line.bottom) / 2
            candidate_numbers = _number_tokens(line.text)
            host_numbers = _number_tokens(host_line.text)
            follows_while_growing = (
                0.0 <= host_start - end <= tolerance
                and host_end > end
                and duration <= host_end - host_start
                and candidate_numbers == host_numbers
                and x_overlap >= 0.8
                and abs(line.left - host_line.left) <= 0.04
                and abs(candidate_y - host_y) <= 0.18
            )
            # A message that finishes arriving also scrolls up and re-wraps, so
            # the finished reading sits nowhere near where its partial sat: the
            # geometry above asks for a stillness that streaming text does not
            # have. Waive it when the timing and the text are conclusive on
            # their own -- the partial hands over exactly as the host begins,
            # the host continues it from the first character, no quantity
            # changed, and the partial did not outlive the host.
            streams_into_host = (
                0.0 <= host_start - end <= tolerance
                and host_end > end
                and candidate_numbers == host_numbers
                and host_key.startswith(key)
                and duration <= host_end - host_start
            )
            clipped_after_host = (
                0 <= start - host_end <= tolerance
                and spatial_overlap(line, host_line) >= 0.8
                and _area(line) < 0.8 * _area(host_line)
                and not any(number not in host_numbers for number in candidate_numbers)
            )
            if (
                key != host_key
                and contains_region(host_line, line)
                and len(key) >= 1
                and (
                    len(host_key) >= len(key) * 1.25
                    or ((follows_while_growing or streams_into_host) and len(host_key) > len(key))
                )
                and (
                    (overlap >= 0.8 * duration and spatial_overlap(line, host_line) >= 0.8)
                    or follows_while_growing
                    or streams_into_host
                    or clipped_after_host
                )
            ):
                covered = True
                break
        if not covered:
            result.append(candidate)
    return result


def _without_small_tracks(
    runs: Sequence[tuple[float, float, OCRLine]],
    *,
    min_text_height: float,
    tolerance: float,
    size_evidence: SizeEvidence,
) -> list[tuple[float, float, OCRLine]]:
    """Apply the height floor once per track, after temporal aggregation."""

    height_cache: dict[tuple[str, float, float], float] = {}

    def measured_height(line: OCRLine) -> float:
        identity = (text_key(line.text), line.left, line.right)
        if identity in height_cache:
            return height_cache[identity]
        key = identity[0]
        direct = [
            sample[2] for sample in size_evidence.get(key, ()) if _same_size_span(line, sample)
        ]
        pieces: dict[str, list[float]] = {}
        # OCR sometimes returns a complete row as one tall box and the same row
        # as several smaller boxes on other frames. Include contained raw pieces
        # from the same horizontal span so the occasional combined box cannot
        # lift a tiny UI row over the floor.
        for part, samples in size_evidence.items():
            if part == key:
                continue
            for left, right, height in samples:
                if (
                    len(part) >= 3
                    and part in key
                    and left >= line.left - 0.05
                    and right <= line.right + 0.05
                ):
                    pieces.setdefault(part, []).append(height)
        piece_evidence = [height for heights in pieces.values() for height in heights]
        piece_coverage = sum(len(part) for part in pieces)
        # Prefer direct readings. Override them only when multiple component
        # boxes repeatedly explain the row; a couple of tall partial mistakes
        # must not lift an otherwise tiny complete label.
        components_dominate = (
            len(direct) <= 2
            and len(pieces) >= 2
            and piece_coverage >= 0.6 * len(key)
            and len(piece_evidence) > 2 * len(direct)
        )
        evidence = [*direct, *piece_evidence] if not direct or components_dominate else direct
        value = median(evidence) if evidence else _height(line)
        height_cache[identity] = value
        return value

    def tall(line: OCRLine) -> bool:
        return measured_height(line) > min_text_height + 1e-6

    tall_runs = [run for run in runs if tall(run[2])]
    result = []
    for candidate in runs:
        start, end, line = candidate
        if tall(line):
            result.append(candidate)
            continue
        # A wrapped paragraph line may be slightly shorter than the floor. Keep
        # it only when a normal-sized track directly above shares essentially
        # the same lifetime and left edge. Same-row browser tabs, clocks and
        # garment labels cannot use this exception.
        supported_continuation = False
        for host_start, host_end, host in tall_runs:
            overlap = min(end, host_end) - max(start, host_start)
            shared = overlap > 0 and overlap >= 0.8 * min(end - start, host_end - host_start)
            gap = line.top - host.bottom
            if (
                shared
                and abs(start - host_start) <= tolerance
                and abs(end - host_end) <= tolerance
                and 0.0 <= gap <= 3.0 * max(host.bottom - host.top, line.bottom - line.top)
                and 0.0 <= line.left - host.left <= 0.03
                and min(line.right, host.right) > max(line.left, host.left)
                and host.right - host.left >= 1.5 * (line.right - line.left)
                and len(text_key(line.text)) >= 3
            ):
                supported_continuation = True
                break
        if supported_continuation:
            result.append(candidate)
    return result
