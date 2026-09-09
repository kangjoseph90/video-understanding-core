import os
from pathlib import Path

import pytest

from vuc.config import load_config


def test_load_config_resolves_cache_relative_to_config() -> None:
    project_root = Path(__file__).parents[1]
    config = load_config(project_root / "config.yaml")

    assert config.cache.directory == project_root / ".vuc-cache"
    assert config.run.mode == "agentic"
    assert config.indexer.device == "cpu"
    assert config.indexer.hub == "hf"
    assert config.frames.montage_n == 3
    assert config.frames.index_interval_s == 15
    assert config.frames.baseline_full_interval_s == 1
    assert config.advanced_asr.provider == "local"
    assert config.advanced_asr.local.max_segment_s == 180
    assert config.advanced_asr.cloud.max_segment_s == 600
    assert config.view_frames.max_montages_per_call == 16
    assert config.vision_llm.input_cost_per_million_usd == 0
    assert config.vision_llm.output_cost_per_million_usd == 0


def test_load_config_rejects_automatic_or_unknown_run_mode(tmp_path: Path) -> None:
    project_root = Path(__file__).parents[1]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (project_root / "config.yaml")
        .read_text(encoding="utf-8")
        .replace("mode: agentic", "mode: auto"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run.mode must be agentic, baseline_full"):
        load_config(config_path)


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
