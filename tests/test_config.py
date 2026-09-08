import os
from pathlib import Path

from vuc.config import load_config


def test_load_config_resolves_cache_relative_to_config() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "config.yaml")

    assert config.cache.directory == project_root / ".vuc-cache"
    assert config.indexer.device == "cpu"
    assert config.indexer.hub == "hf"
    assert config.frames.montage_shape == (3, 3)


def test_load_config_reads_dotenv_without_overriding_process_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = Path(__file__).parents[1]
    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text(
        "VLM_BASE_URL=https://example.invalid/v1\nVLM_MODEL=test-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("VLM_MODEL", "process-model")
    monkeypatch.delenv("VLM_BASE_URL", raising=False)

    load_config(project_root / "config.yaml", dotenv_path=dotenv_path)

    assert os.environ["VLM_BASE_URL"] == "https://example.invalid/v1"
    assert os.environ["VLM_MODEL"] == "process-model"
