"""Voice activity detection as its own stage, ahead of every audio model.

Until now VAD only existed inside funasr's SenseVoice pipeline, where its
output was invisible: we saw sentences, never the speech/silence timeline that
produced them. The consequence was measurable -- on a BGM-heavy vlog funasr
returned two "segments" of 512s and 464s, so 97% of the video counted as
speech and nothing ever learned where the silence was.

Running VAD first makes the split an artifact in its own right. It partitions
the whole timeline exactly once: every second of the video is either a speech
region or a non-speech region, with no overlap and no hole, and each side is
then handed to the model that suits it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol

from vuc.config import VADConfig
from vuc.timeline import Span, complement, from_milliseconds, merge_spans, split_long


class VADError(RuntimeError):
    pass


class VADProvider(Protocol):
    name: str

    def detect(self, audio_path: Path, *, duration_s: float) -> list[Span]: ...


def normalize_fsmn_result(result: Any, duration_s: float) -> list[Span]:
    """funasr returns [{'key': ..., 'value': [[start_ms, end_ms], ...]}]."""
    spans: list[Span] = []
    items = result if isinstance(result, list) else [result]
    for item in items:
        if not isinstance(item, dict):
            continue
        for pair in item.get("value") or []:
            if not isinstance(pair, list | tuple) or len(pair) < 2:
                continue
            start = from_milliseconds(pair[0])
            end = from_milliseconds(pair[1])
            if end <= start:
                continue
            spans.append(Span(start, end).clamped(duration_s))
    return merge_spans(span for span in spans if span.duration > 0)


class FsmnVAD:
    """FSMN-VAD through funasr, called directly instead of via SenseVoice."""

    name = "fsmn"

    def __init__(self, config: VADConfig) -> None:
        self.config = config
        try:
            from funasr import AutoModel
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise VADError(
                "VAD dependencies are not installed. Run `uv sync --extra sensevoice`."
            ) from exc
        self._model = AutoModel(
            model=config.model,
            hub=config.hub,
            device=config.device,
            ncpu=config.cpu_threads,
            disable_update=True,
        )

    def detect(self, audio_path: Path, *, duration_s: float) -> list[Span]:
        result = self._model.generate(
            input=str(audio_path),
            cache={},
            max_single_segment_time=int(self.config.max_single_segment_s * 1000),
        )
        return normalize_fsmn_result(result, duration_s)


def _bridgeable(gap_s: float, left: Span, right: Span, *, config: VADConfig) -> bool:
    """Whether the silence between two detections is an intake of breath.

    Two limits, and the gap has to clear both. The absolute one says a pause
    longer than merge_gap_s is a pause, not a breath. The proportional one
    reads the detections themselves: bridging 1.4s between two half-second
    blips invents a three-second utterance out of two coughs, while the same
    1.4s between two twenty-second stretches is plainly one person still
    talking. So the bridge may not outlast the shorter neighbour -- measured on
    what the detector actually returned, before any rounding widened it.
    """
    if gap_s > config.merge_gap_s:
        return False
    return gap_s <= max(config.min_bridge_s, min(left.duration, right.duration))


def speech_view(
    spans: Iterable[Span],
    *,
    config: VADConfig,
    duration_s: float,
) -> list[Span]:
    """Turn raw detections into the regions the index is built on.

    Each detection is rounded outwards to whole seconds, which is what stops a
    swarm of sub-second fragments from each becoming an index line -- and, by
    only ever widening, cannot clip a syllable off either end. Neighbours are
    then joined across silence short enough to be a breath, the leftovers too
    short to be regions of their own are absorbed, and anything still too long
    is cut so no region outgrows what an ASR handles well.
    """
    detections = sorted(span for span in spans if span.duration > 0)
    joined: list[Span] = []
    previous: Span | None = None
    for detection in detections:
        span = detection.rounded_out(duration_s)
        if previous is None:
            joined.append(span)
        elif span.start <= joined[-1].end or _bridgeable(
            round(detection.start - previous.end, 3), previous, detection, config=config
        ):
            # Rounding can push two neighbours into each other; regions have to
            # stay disjoint for the timeline to remain a partition.
            joined[-1] = Span(joined[-1].start, max(joined[-1].end, span.end))
        else:
            joined.append(span)
        previous = detection
    # min_region_s deliberately does not apply here. A short stretch of silence
    # is a remainder, but a short stretch of speech may be the whole answer to a
    # question, and there is no way to tell "yes" from a mouth noise by length.
    # The occasional one-second filler is the price of not discarding those.
    absorbed = _absorb_slivers(joined, config=config, duration_s=duration_s)
    return split_long(absorbed, max_s=config.window_max_s)


def _absorb_slivers(
    speech: Sequence[Span],
    *,
    config: VADConfig,
    duration_s: float,
) -> list[Span]:
    """A gap too short to be its own region is not a gap.

    What is left between two speech regions after rounding is often a single
    second. Nothing can be said about one second of room tone -- a tagger given
    it returns noise or nothing -- so emitting it as a line adds an empty row to
    the index for every pause the speaker takes. Dropping it instead would put
    a hole in a timeline whose whole point is that it has none, so the second
    goes to the speech beside it. Speech regions are already widened by
    rounding; this widens them a little further and invents nothing, because
    what gets transcribed there is whatever is actually in the audio.
    """
    absorbed: list[Span] = []
    for span in speech:
        start = 0.0 if not absorbed and span.start < config.min_region_s else span.start
        if absorbed and start - absorbed[-1].end < config.min_region_s:
            absorbed[-1] = Span(absorbed[-1].start, span.end)
        else:
            absorbed.append(Span(start, span.end))
    if absorbed and 0 < duration_s - absorbed[-1].end < config.min_region_s:
        absorbed[-1] = Span(absorbed[-1].start, duration_s)
    return absorbed


def non_speech_view(
    speech: Sequence[Span],
    *,
    config: VADConfig,
    duration_s: float,
) -> list[Span]:
    """Everything the speech view left over, in workable pieces.

    Silence is an observation, not an absence of one. These regions are what
    the event tagger runs on, and they are capped for the same reason speech
    regions are: one tag covering nine minutes says almost nothing.

    The minimum length is enforced here as well as in the speech view. Nothing
    should reach this filter -- the absorption step already took the short
    leftovers -- but a tagger handed 13ms of audio does not return an empty
    answer, it raises, so the invariant is worth one line to hold.
    """
    gaps = complement(speech, duration_s=duration_s)
    return split_long(
        (gap for gap in gaps if gap.duration >= config.min_region_s),
        max_s=config.window_max_s,
    )


def create_vad(config: VADConfig) -> VADProvider:
    if config.provider == "fsmn":
        return FsmnVAD(config)
    raise ValueError(f"unsupported VAD provider: {config.provider}")
