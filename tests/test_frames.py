from pathlib import Path

from PIL import Image

from vuc.frames import create_montages
from vuc.models import FrameArtifact


def _frames(tmp_path: Path, count: int) -> list[FrameArtifact]:
    frames = []
    for index in range(count):
        path = tmp_path / f"frame-{index}.jpg"
        Image.new("RGB", (448, 252), color=(index, 0, 0)).save(path)
        frames.append(FrameArtifact(path=str(path), timestamp_s=float(index)))
    return frames


def test_montage_uses_fixed_reference_canvas_for_full_group(tmp_path: Path) -> None:
    montages = create_montages(
        _frames(tmp_path, 9),
        tmp_path / "montages",
        n=3,
        width=1344,
        height=756,
        jpeg_quality=88,
    )

    with Image.open(montages[0]) as image:
        assert image.size == (1344, 756)


def test_last_montage_shrinks_grid_without_shrinking_cells(tmp_path: Path) -> None:
    montages = create_montages(
        _frames(tmp_path, 13),
        tmp_path / "montages",
        n=3,
        width=1344,
        height=756,
        jpeg_quality=88,
    )

    assert len(montages) == 2
    with Image.open(montages[0]) as full, Image.open(montages[1]) as final:
        assert full.size == (1344, 756)
        assert final.size == (896, 504)


def test_one_by_one_grid_is_supported(tmp_path: Path) -> None:
    montages = create_montages(
        _frames(tmp_path, 1),
        tmp_path / "montages",
        n=1,
        width=1344,
        height=756,
        jpeg_quality=88,
    )

    with Image.open(montages[0]) as image:
        assert image.size == (1344, 756)
