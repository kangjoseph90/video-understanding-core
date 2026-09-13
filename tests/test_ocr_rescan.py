from dataclasses import replace

import pytest
from PIL import Image, ImageDraw, ImageFont

from tests.test_ocr_index import line
from tests.test_ocr_sampling import DENSE
from vuc.ocr import Observation, text_cues
from vuc.ocr_rescan import Region, add_region, read_regions, region_candidates


def frames(tmp_path, *, outside=False):
    paths = []
    for i in range(9):
        image = Image.new("RGB", (448, 252), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=17)
        draw.text((160, 200), "First text" if i < 8 else "Last text", font=font, fill="black")
        if i in (2, 3):
            if outside:
                draw.text((10, 10), "Sudden text", font=font, fill="black")
            else:
                draw.rectangle((140, 190, 310, 230), fill="white")
                draw.text((160, 200), "NEXT STEP", font=font, fill="black")
        path = tmp_path / f"{i}.png"
        image.save(path)
        paths.append((i / 4, path))
    return paths


def anchors(end=2):
    a = line("First text", top=0.78, bottom=0.86, left=0.35, right=0.6, confidence=0.99)
    return [Observation(0, (a,)), Observation(end, (replace(a, text="Last text"),))]


def test_extra_reading_stays_in_a_measured_changing_slot(tmp_path):
    images = frames(tmp_path)
    plans = region_candidates(images, anchors(), config=DENSE)
    assert 2 in plans
    assert all(region.box[1] > 0.7 for boxes in plans.values() for region in boxes)
    assert region_candidates(images, [], config=DENSE) == {}


def test_new_text_elsewhere_does_not_trigger_a_rescan(tmp_path):
    assert region_candidates(frames(tmp_path, outside=True), anchors(), config=DENSE) == {}


def test_a_slot_does_not_survive_a_long_gap(tmp_path):
    assert region_candidates(frames(tmp_path), anchors(end=6), config=DENSE) == {}


def test_a_logo_does_not_become_a_changing_slot_from_ocr_jitter(tmp_path):
    obs = anchors()
    obs[1] = Observation(2, (replace(obs[0].lines[0], text="Firsttext"),))
    assert region_candidates(frames(tmp_path), obs, config=DENSE) == {}


def test_cropped_scan_does_not_end_text_outside_its_coverage():
    heading = line("Persistent heading", top=0.1, bottom=0.16, confidence=0.99)
    base = [Observation(0, (heading,)), Observation(2, (heading,)), Observation(4, ())]
    crop = Observation(0.5, (), discovery=True, regions=((0.2, 0.7, 0.8, 0.95),))
    assert text_cues([*base, crop], duration_s=5, config=DENSE) == text_cues(
        base, duration_s=5, config=DENSE
    )


def test_crop_coordinates_and_glyph_height_return_to_full_frame(tmp_path):
    image = Image.new("RGB", (1000, 500), "white")
    path = tmp_path / "full.png"
    image.save(path)

    class Engine:
        def read_many(self, paths, *, cropped=False):
            assert cropped
            assert Image.open(paths[0]).size == (600, 200)
            return [
                [
                    line(
                        "Measured text", left=0.1, right=0.9, top=0.25, bottom=0.75, confidence=0.99
                    ),
                    line("Truncated text", left=0, right=0.95, top=0.1, bottom=0.2),
                ]
            ]

    reference = line("Original text", left=0.26, right=0.74, top=0.4, bottom=0.6)
    result = read_regions(
        [(1, path)],
        {0: path},
        {0: [Region((0.2, 0.3, 0.8, 0.7), (reference,))]},
        Engine(),
        tmp_path,
    )
    measured = result[0].lines[0]
    assert len(result[0].lines) == 1
    assert (measured.left, measured.right, measured.top, measured.bottom) == (0.26, 0.74, 0.4, 0.6)
    assert measured.height == 0.2
    assert result[0].regions == ((0.2, 0.3, 0.8, 0.7),)
    assert not list(tmp_path.glob("text-crop-*.png"))


def test_crop_mode_shares_models_without_changing_full_frame_preprocessing(tmp_path):
    from types import SimpleNamespace

    from vuc.ocr import RapidOCREngine

    states = []

    class Model:
        text_det = SimpleNamespace(limit_type="min")

        def __call__(self, *args, **kwargs):
            states.append(self.text_det.limit_type)
            return [], None

    engine = RapidOCREngine.__new__(RapidOCREngine)
    engine.config = DENSE
    engine._engine = Model()
    engine._fallback = None
    path = tmp_path / "crop.png"
    Image.new("RGB", (400, 100), "white").save(path)
    engine.read_many([path], cropped=True)
    engine.read_many([path])
    assert states == ["max", "min"]
    assert engine._engine.text_det.limit_type == "min"


def test_overlapping_crops_merge_but_separate_rows_and_columns_do_not():
    plans = {}
    add_region(plans, 1, (0.2, 0.7, 0.6, 0.9))
    add_region(plans, 1, (0.21, 0.71, 0.61, 0.91))
    add_region(plans, 1, (0.7, 0.7, 0.9, 0.9))
    add_region(plans, 1, (0.2, 0.3, 0.6, 0.5))
    assert len(plans[1]) == 3
    assert Region((0.2, 0.7, 0.61, 0.91)) in plans[1]


@pytest.mark.parametrize("verification", [False, True])
def test_crop_padding_cannot_publish_a_smaller_background_label(tmp_path, verification):
    path = tmp_path / "frame.png"
    Image.new("RGB", (1000, 500), "white").save(path)
    reference = line("Original caption", left=0.3, right=0.7, top=0.8, bottom=0.86)

    class Engine:
        def read_many(self, paths, *, cropped=False):
            return [
                [
                    line("Next caption", left=0.2, right=0.8, top=0.25, bottom=0.75),
                    line("Any label", left=0.6, right=0.9, top=0.3, bottom=0.5),
                ]
            ]

    result = read_regions(
        [(1, path)],
        {0: path},
        {0: [Region((0.2, 0.77, 0.8, 0.89), (reference,))]},
        Engine(),
        tmp_path,
        verification=verification,
    )
    assert [row.text for row in result[0].lines] == ["Next caption"]


def test_verification_is_limited_to_changing_rows_and_their_lifetime():
    from vuc.ocr_rescan import changing_regions, in_changing_region

    obs = anchors()
    windows = changing_regions(obs, config=DENSE)
    caption = replace(obs[0].lines[0], text="Another caption")
    assert in_changing_region(1, caption, windows)
    assert not in_changing_region(3, caption, windows)
    assert not in_changing_region(1, replace(caption, top=0.1, bottom=0.16), windows)
    assert not in_changing_region(1, replace(caption, bottom=0.81), windows)
