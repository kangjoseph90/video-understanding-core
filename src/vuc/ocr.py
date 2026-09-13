"""Text read off the sampled frames, with no claim about what kind of text it is.

Whether a line is a subtitle, a slide heading, a lower third or a shop sign is
a guess. An earlier version of this made that guess from vertical position and
filed the answer as an evidence type, which put slide body text into the index
as dialogue. This one reports only what it can actually measure -- roughly
where the text sat and roughly how big it was -- and leaves the interpretation
to whoever reads the index.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import unicodedata
from collections.abc import Sequence
from copy import copy
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageFilter

from vuc.config import OCRConfig
from vuc.frames import extract_plain_frames
from vuc.models import TextCue

VERTICAL_BANDS = ((0.33, "top"), (0.66, "middle"), (1.01, "bottom"))
HORIZONTAL_BANDS = ((0.33, "left"), (0.66, "center"), (1.01, "right"))
# Coarse on purpose: a subtitle is a few percent of frame height, a slide
# heading a good deal more, and nothing here needs finer resolution.
SIZE_BANDS = ((0.04, "small"), (0.08, "medium"), (1.01, "large"))


class OCRError(RuntimeError):
    pass


@dataclass(frozen=True)
class OCRLine:
    """One detected line and its box, as 0..1 fractions of the frame."""

    text: str
    confidence: float | None = None
    top: float = 0.0
    bottom: float = 0.0
    left: float = 0.0
    right: float = 1.0
    # Small crop fingerprint supports temporal identity, never recognition.
    appearance: str = ""
    # Glyph thickness from the oriented detection polygon; bbox height includes
    # the tilt of an entire word and is not its font size.
    text_height: float | None = None

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top if self.text_height is None else self.text_height)

    @property
    def position(self) -> str:
        """Coarse ninth of the frame, e.g. ``bottom left``."""
        middle_y = (self.top + self.bottom) / 2
        middle_x = (self.left + self.right) / 2
        vertical = next(name for edge, name in VERTICAL_BANDS if middle_y < edge)
        horizontal = next(name for edge, name in HORIZONTAL_BANDS if middle_x < edge)
        return f"{vertical} {horizontal}"

    @property
    def size(self) -> str:
        return next(name for edge, name in SIZE_BANDS if self.height < edge)


class OCREngine(Protocol):
    name: str

    def read_many(
        self, image_paths: Sequence[Path], *, cropped: bool = False
    ) -> list[list[OCRLine]]: ...


def rapidocr_options(config: OCRConfig) -> dict[str, object]:
    """Flat keyword names only.

    RapidOCR accepts the dotted "Rec.model_path" spelling without complaint and
    then ignores it, so that form leaves the configuration silently dead.
    """
    named = (
        ("det_model_path", config.det_model_path),
        ("rec_model_path", config.rec_model_path),
        ("rec_keys_path", config.rec_keys_path),
    )
    options: dict[str, object] = {key: value for key, value in named if value}
    if config.det_model_config_path:
        # Detector normalisation is part of the model, not a tuning knob.
        # v6 uses ImageNet mean/std; feeding it the wheel's v4 defaults breaks it.
        import yaml

        metadata = yaml.safe_load(Path(config.det_model_config_path).read_text())
        for op in metadata["PreProcess"]["transform_ops"]:
            if "NormalizeImage" in op:
                options["det_mean"] = op["NormalizeImage"]["mean"]
                options["det_std"] = op["NormalizeImage"]["std"]
        post = metadata["PostProcess"]
        for key in ("thresh", "box_thresh", "unclip_ratio", "max_candidates"):
            if key in post:
                options[f"det_{key}"] = post[key]
        options["det_donot_use_dilation"] = True
    return options


class RapidOCREngine:
    """RapidOCR (ONNX Runtime): CPU-only, multilingual, no system packages."""

    name = "rapidocr"

    def __init__(self, config: OCRConfig, *, threads: int = 1) -> None:
        self.config = config
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise OCRError(
                "OCR dependencies are not installed. Run `uv sync --extra ocr`."
            ) from exc
        # The wheel ships Chinese/Latin recognition only. Korean and Japanese
        # need their own PP-OCR recognition model and character dictionary,
        # which are pointed at from config rather than vendored here.
        self._engine = RapidOCR(**rapidocr_options(config), intra_op_num_threads=threads)
        self._fallback = None
        if config.fallback_rec_model_path:
            from rapidocr_onnxruntime.ch_ppocr_rec import TextRecognizer

            self._fallback = TextRecognizer(
                dict(
                    model_path=config.fallback_rec_model_path,
                    rec_keys_path=config.fallback_rec_keys_path,
                    rec_img_shape=[3, 48, 320],
                    rec_batch_num=6,
                    intra_op_num_threads=threads,
                    inter_op_num_threads=1,
                    use_cuda=False,
                    use_dml=False,
                )
            )

    def read(self, image_path: Path, *, cropped: bool = False) -> list[OCRLine]:
        with Image.open(image_path) as image:
            width = float(image.width) or 1.0
            height = float(image.height) or 1.0
            pixels_for_hash = image.convert("L")
        engine = self._engine
        if cropped:
            # Reuse the ONNX sessions, but keep per-call preprocessing private.
            # The wheel otherwise blows a narrow crop up to 736px high.
            engine = copy(self._engine)
            engine.text_det = copy(self._engine.text_det)
            engine.text_det.limit_type = "max"
        if self._fallback is None:
            result, _ = engine(str(image_path), use_cls=self.config.use_angle_cls)
        else:
            # Unknown language: detect once, then compare recognisers on the
            # same crops. Never consult audio or infer a script from a filename.
            import numpy as np

            boxes, _ = engine(str(image_path), use_rec=False, use_cls=False)
            result = []
            if boxes:
                pixels = engine.load_img(str(image_path))
                crops = engine.get_crop_img_list(pixels, np.array(boxes, dtype=np.float32))
                if self.config.use_angle_cls:
                    crops, _, _ = engine.text_cls(crops)
                primary, _ = engine.text_rec(crops)
                # Clear Latin text needs no Hangul second opinion. Keep CJK
                # and short/uncertain readings eligible: a Chinese recogniser
                # can be confidently wrong on an unsupported Korean script.
                uncertain = [
                    i
                    for i, (text, score) in enumerate(primary)
                    if not (text.isascii() and len(text_key(text)) >= 4 and score >= 0.98)
                ]
                alternate = list(primary)
                if uncertain:
                    second_pass, _ = self._fallback([crops[i] for i in uncertain])
                    for i, reading in zip(uncertain, second_pass, strict=True):
                        alternate[i] = reading
                for box, first, second in zip(boxes, primary, alternate, strict=True):
                    # The fallback covers Hangul + Latin. Its Latin is not a
                    # reason to override the multilingual model; its script is.
                    hangul = any("\uac00" <= c <= "\ud7a3" for c in second[0])
                    chosen = second if hangul and second[1] > first[1] else first
                    result.append([box, chosen[0], chosen[1]])
        lines: list[OCRLine] = []
        for item in result or []:
            box, text, score = item[0], str(item[1] or ""), item[2]
            if not text.strip():
                continue
            xs = [float(point[0]) for point in box]
            ys = [float(point[1]) for point in box]
            lines.append(
                OCRLine(
                    text=text.strip(),
                    confidence=None if score is None else float(score),
                    top=min(ys) / height,
                    bottom=max(ys) / height,
                    left=min(xs) / width,
                    right=max(xs) / width,
                    text_height=polygon_text_height(box, height),
                    appearance=crop_fingerprint(
                        pixels_for_hash, (min(xs), min(ys), max(xs), max(ys))
                    ),
                )
            )
        return reading_order(lines)

    def read_many(
        self, image_paths: Sequence[Path], *, cropped: bool = False
    ) -> list[list[OCRLine]]:
        return [self.read(path, cropped=cropped) for path in image_paths]


class IsolatedOCREngine:
    """The OCR stack, kept out of the interpreter that holds torch.

    A worker is started once and reused for every frame, so the model is loaded
    a single time and the cost of the isolation is one process. See
    vuc.ocr_worker for why the isolation is not optional.
    """

    def __init__(self, config: OCRConfig) -> None:
        self.name = config.engine
        self.batch_size = max(1, config.workers * 4)
        self._process = subprocess.Popen(
            [sys.executable, "-m", "vuc.ocr_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        settings = {key: value for key, value in vars(config).items() if key != "rec_by_language"}
        handshake = self._exchange({**settings, "rec_by_language": {}})
        if "error" in handshake:
            self.close()
            raise OCRError(f"OCR worker failed to start: {handshake['error']}")

    def _exchange(self, request: dict[str, object]) -> dict[str, object]:
        assert self._process.stdin is not None and self._process.stdout is not None
        self._process.stdin.write(json.dumps(request) + "\n")
        self._process.stdin.flush()
        reply = self._process.stdout.readline()
        if not reply:
            raise OCRError("OCR worker exited before answering")
        return json.loads(reply)

    def read_many(
        self, image_paths: Sequence[Path], *, cropped: bool = False
    ) -> list[list[OCRLine]]:
        results: list[list[OCRLine]] = []
        for start in range(0, len(image_paths), self.batch_size):
            batch = image_paths[start : start + self.batch_size]
            reply = self._exchange({"paths": [str(path) for path in batch], "cropped": cropped})
            if "error" in reply:
                raise OCRError(str(reply["error"]))
            results.extend([OCRLine(**item) for item in lines] for lines in reply["frames"])
        return results

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:  # pragma: no cover - worker ignored the close
            self._process.kill()


def create_ocr_engine(config: OCRConfig) -> OCREngine:
    if config.engine == "rapidocr":
        return IsolatedOCREngine(config)
    raise ValueError(f"unsupported OCR engine: {config.engine}")


@lru_cache(maxsize=8192)
def normalize_text(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).split()).strip()


@lru_cache(maxsize=8192)
def text_key(text: str) -> str:
    """The comparable form of a line: no case, no spacing, no punctuation."""
    return re.sub(r"[^\w]+", "", normalize_text(text).lower())


def reading_order(lines: Sequence[OCRLine]) -> list[OCRLine]:
    """Read a slanted/jittering row left to right, then the next row.

    Sorting solely by top puts the right half of a slanted line first. Row
    membership is checked against a fixed anchor to prevent vertical drift.
    """
    rows: list[list[OCRLine]] = []
    for line in sorted(lines, key=lambda x: (x.top, x.left)):
        center = (line.top + line.bottom) / 2
        for row in reversed(rows):
            anchor = row[0]
            height = min(line.bottom - line.top, anchor.bottom - anchor.top)
            if abs(center - (anchor.top + anchor.bottom) / 2) <= height / 2:
                row.append(line)
                break
        else:
            rows.append([line])
    return [line for row in rows for line in sorted(row, key=lambda x: x.left)]


def polygon_text_height(box: Sequence[Sequence[float]], frame_height: float) -> float:
    """Short side of the oriented quadrilateral, normalised by frame height.

    Opposite edge means tolerate perspective. The short side also gives glyph
    size for vertical writing, without treating its whole column as one glyph.
    """
    edges = [math.dist(box[i], box[(i + 1) % 4]) for i in range(4)]
    return min((edges[0] + edges[2]) / 2, (edges[1] + edges[3]) / 2) / frame_height


def crop_fingerprint(image: Image.Image, box: tuple[float, float, float, float]) -> str:
    """256-bit horizontal gradient hash of the measured text crop."""
    if box[2] <= box[0] or box[3] <= box[1]:
        return ""
    crop = image.crop(box).resize((33, 8), Image.Resampling.BILINEAR)
    values = list(crop.tobytes())
    bits = 0
    for y in range(8):
        for x in range(32):
            bits = (bits << 1) | (values[y * 33 + x + 1] > values[y * 33 + x])
    return f"{bits:064x}"


@lru_cache(maxsize=8192)
def same_text(left: str, right: str, *, ratio: float) -> bool:
    """Whether two readings are the same line seen twice.

    A logo sitting in the corner for eight minutes is read a little
    differently every time it is sampled -- `open culture`, `open.cuiture`,
    `opencuiture`, `open cultune` -- and on exact comparison each spelling
    started a run of its own, so one unchanging logo became six entries with
    six time ranges. The noise is a glyph here and a space there, which is
    what a similarity ratio is for; nothing shorter than a few characters is
    matched loosely, because at that length everything resembles everything.
    """
    a, b = text_key(left), text_key(right)
    # Quantities, dates and signs must not disappear into fuzzy deduplication.
    if re.findall(r"[+-]?\d+(?:[.,:/-]\d+)*", normalize_text(left)) != re.findall(
        r"[+-]?\d+(?:[.,:/-]\d+)*", normalize_text(right)
    ):
        return False
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    if SequenceMatcher(None, a, b, autojunk=False).ratio() >= ratio:
        return True
    # Hangul OCR often misses one component of a syllable. Compare decomposed
    # glyphs as well, without changing the text we publish.
    return SequenceMatcher(
        None, unicodedata.normalize("NFD", a), unicodedata.normalize("NFD", b), autojunk=False
    ).ratio() >= max(0.85, ratio)


@dataclass(frozen=True)
class Observation:
    """The lines one frame showed."""

    timestamp_s: float
    lines: tuple[OCRLine, ...] = ()
    # Neighbour probes verify an existing candidate. Their other predictions
    # neither create tracks nor change the timeline of unrelated text.
    verification: bool = False
    # Extra change-driven samples require direct textual corroboration; they
    # cannot turn incidental repeated shapes into established text slots.
    discovery: bool = False
    regions: tuple[tuple[float, float, float, float], ...] = ()


def text_signature(image: Image.Image, *, cells: tuple[int, int] = (16, 9)) -> tuple[int, ...]:
    """A cheap description of where the frame has text-like structure.

    Edge magnitude averaged over a coarse grid and quantised hard. Text puts
    far more edge energy into its cell than photographic content does, so a
    caption appearing, changing or leaving moves a handful of cells and nothing
    else does much. It is deliberately blunt: this decides only whether a frame
    is worth paying OCR for, and paying for one too many costs 65ms.
    """
    edges = image.convert("L").filter(ImageFilter.FIND_EDGES)
    coarse = edges.resize(cells, Image.Resampling.BOX)
    return tuple(value // 16 for value in coarse.tobytes())


def signature_changed(left: tuple[int, ...], right: tuple[int, ...], *, threshold: float) -> bool:
    if not left or len(left) != len(right):
        return True
    moved = sum(1 for a, b in zip(left, right, strict=True) if a != b)
    return moved / len(left) >= threshold


def ocr_candidates(
    frames: Sequence[tuple[float, Path]],
    *,
    config: OCRConfig,
    required_s: Sequence[float] = (),
) -> list[int]:
    """Which of the sampled frames are worth reading.

    Sampling OCR where the camera cut sampled the wrong thing -- a caption card
    can come and go inside one shot, and a logo can outlast a hundred cuts --
    but reading every second of a video is mostly re-reading the same subtitle.
    So the scan is dense and the reading is not: a frame is read when its text
    layout has moved since the last one read, and the frames something else
    already asked for are read regardless.
    """
    wanted = sorted(required_s)
    if not frames:
        return []
    chosen: set[int] = set()
    for moment in wanted:
        nearest = min(range(len(frames)), key=lambda index: abs(frames[index][0] - moment))
        chosen.add(nearest)

    def edges(path: Path) -> Image.Image:
        with Image.open(path) as source:
            source.thumbnail((config.scan_width, config.scan_width), Image.Resampling.LANCZOS)
            return source.convert("L").filter(ImageFilter.FIND_EDGES)

    previous: tuple[int, ...] = ()
    current = edges(frames[0][1])
    last_read = -float("inf")
    last_check = -float("inf")
    for index, (timestamp, _path) in enumerate(frames):
        signature = tuple(
            value // 16 for value in current.resize((16, 9), Image.Resampling.BOX).tobytes()
        )
        check_base = timestamp - last_check >= 1.0
        if check_base:
            last_check = timestamp
        base = (
            index in chosen
            or timestamp - last_read >= config.refresh_s
            or (
                check_base
                and signature_changed(previous, signature, threshold=config.change_threshold)
            )
        )
        if base:
            chosen.add(index)
            previous = signature
            last_read = timestamp
        if index + 1 < len(frames):
            current = edges(frames[index + 1][1])
    return sorted(chosen)


def scan_text(
    video_path: Path,
    output_dir: Path,
    engine: OCREngine,
    *,
    config: OCRConfig,
    duration_s: float,
    required_s: Sequence[float] = (),
) -> tuple[list[Observation], int]:
    """Read the video on its own clock, and only where the text moved.

    Returns the observations and how many frames were sampled to get them, so
    the caller can record what the change detection actually saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("text-*.jpg"):
        stale.unlink()
    try:
        frames = extract_plain_frames(
            video_path,
            output_dir,
            prefix="text-scan",
            fps=config.scan_fps,
            width=config.scan_width,
            duration_s=duration_s,
            first_center_s=0.5,
        )
        base_frames = extract_plain_frames(
            video_path,
            output_dir,
            prefix="text-base",
            fps=min(1.0, config.scan_fps),
            width=config.recognition_width,
            duration_s=duration_s,
            first_center_s=0.5,
        )
        base_picked = ocr_candidates(base_frames, config=config, required_s=required_s)
        by_time = {timestamp: i for i, (timestamp, _) in enumerate(frames)}
        picked = [by_time[base_frames[i][0]] for i in base_picked if base_frames[i][0] in by_time]
        paths = {
            by_time[timestamp]: path for timestamp, path in base_frames if timestamp in by_time
        }
        read = engine.read_many([paths[index] for index in picked])
        observations = [
            Observation(frames[i][0], tuple(lines)) for i, lines in zip(picked, read, strict=True)
        ]
        from vuc.ocr_rescan import (
            Region,
            add_region,
            changing_regions,
            in_changing_region,
            padded_box,
            read_regions,
            region_candidates,
        )
        from vuc.ocr_tracking import verification_targets

        windows = changing_regions(observations, config=config)
        plans = region_candidates(frames, observations, config=config)
        available = sorted(
            {
                j
                for i in (*picked, *plans)
                for j in (i - 2, i - 1, i, i + 1, i + 2)
                if 0 <= j < len(frames) and j not in paths
            }
        )
        extra_frames = extract_plain_frames(
            video_path,
            output_dir,
            prefix="text-extra",
            fps=config.scan_fps,
            width=config.recognition_width,
            duration_s=duration_s,
            indices=available,
            first_center_s=0.5,
        )
        paths.update(zip(available, (path for _, path in extra_frames), strict=True))
        observations.extend(read_regions(frames, paths, plans, engine, output_dir))
        needed = verification_targets(observations, duration_s=duration_s, config=config)
        by_time = {timestamp: i for i, (timestamp, _) in enumerate(frames)}
        probes: dict[int, list[Region]] = {}
        base = set(picked)
        for timestamp, rows in needed.items():
            i = by_time[timestamp]
            for line in rows:
                if not in_changing_region(timestamp, line, windows):
                    continue
                for j in (i - 2, i - 1, i + 1, i + 2):
                    if j in paths and j not in base and abs(frames[j][0] - timestamp) <= 0.501:
                        add_region(probes, j, padded_box((line,)), rows=(line,))
        observations.extend(
            read_regions(
                frames,
                paths,
                probes,
                engine,
                output_dir,
                verification=True,
            )
        )
        observations.sort(key=lambda item: item.timestamp_s)
    finally:
        for path in output_dir.glob("text-*.jpg"):
            path.unlink(missing_ok=True)
    # Keep measured boxes/confidences, including rejected readings, for audits
    # and cheap tracker replays without invoking OCR again.
    temporary = output_dir / "observations.jsonl.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        for observation in observations:
            handle.write(json.dumps(asdict(observation), ensure_ascii=False) + "\n")
    temporary.replace(output_dir / "observations.jsonl")
    return observations, len(frames)


def text_cues(
    observations: Sequence[Observation],
    *,
    duration_s: float,
    config: OCRConfig,
) -> list[TextCue]:
    """Track measured lines, select supported readings and group co-lived text."""
    from vuc.ocr_tracking import track_cues

    return track_cues(observations, duration_s=duration_s, config=config)
