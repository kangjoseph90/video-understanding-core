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
    assert config.frames.index_montage_n == 3
    assert config.frames.baseline_full_montage_n == 3
    assert (config.montage.width, config.montage.height) == (1344, 756)
    assert config.montage.jpeg_quality == 88
    assert config.frames.index_interval_s == 15
    assert config.frames.baseline_full_interval_s == 1
    assert config.advanced_asr.provider == "local"
    assert config.advanced_asr.local.max_segment_s == 180
    assert config.advanced_asr.cloud.max_segment_s == 600
    assert config.view_frames.max_montages_per_call == 16
    assert config.view_frames.fps_options == (0.1, 0.2, 0.5, 1.0, 2.0)
    assert config.view_frames.grid_options == (1, 2, 3, 4)
    assert config.transcribe_segment.max_duration_s == 60
    assert config.vision_llm.input_cost_env == "VLM_INPUT_COST_PER_MTOK"


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


def test_token_rates_come_from_the_environment(tmp_path: Path, monkeypatch) -> None:
    project_root = Path(__file__).parents[1]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (project_root / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setenv("VLM_INPUT_COST_PER_MTOK", "0.30")
    monkeypatch.setenv("VLM_CACHED_INPUT_COST_PER_MTOK", "0.006")
    monkeypatch.setenv("VLM_OUTPUT_COST_PER_MTOK", "1.20")

    vision_llm = load_config(config_path).vision_llm

    assert vision_llm.input_cost_per_million_usd == 0.30
    assert vision_llm.cached_input_cost_per_million_usd == 0.006
    assert vision_llm.output_cost_per_million_usd == 1.20


def test_unset_token_rates_disable_cost_estimation(tmp_path: Path, monkeypatch) -> None:
    from vuc.llm import estimate_vlm_cost_usd

    project_root = Path(__file__).parents[1]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (project_root / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    for name in (
        "VLM_INPUT_COST_PER_MTOK",
        "VLM_CACHED_INPUT_COST_PER_MTOK",
        "VLM_OUTPUT_COST_PER_MTOK",
    ):
        monkeypatch.setenv(name, "")

    vision_llm = load_config(config_path).vision_llm

    # A model whose prices we have not recorded reports no cost rather than zero.
    assert estimate_vlm_cost_usd(vision_llm, 1000, 1000, 0) is None


def test_non_numeric_token_rate_is_rejected(tmp_path: Path, monkeypatch) -> None:
    project_root = Path(__file__).parents[1]
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        (project_root / "config.yaml").read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setenv("VLM_INPUT_COST_PER_MTOK", "cheap")

    with pytest.raises(ValueError, match="VLM_INPUT_COST_PER_MTOK"):
        load_config(config_path)
