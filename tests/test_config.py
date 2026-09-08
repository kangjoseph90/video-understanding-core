from pathlib import Path

from vuc.config import load_config


def test_load_config_resolves_cache_relative_to_config() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "config.yaml")

    assert config.cache.directory == project_root / ".vuc-cache"
    assert config.indexer.device == "cpu"
    assert config.indexer.hub == "hf"
    assert config.frames.montage_shape == (3, 3)
