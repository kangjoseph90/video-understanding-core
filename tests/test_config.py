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
    assert config.frames.baseline_full_interval_s == 1
    assert config.vad.provider == "fsmn"
    assert config.vad.model == "fsmn-vad"
    assert config.vad.window_max_s == 30
    assert config.audio_events.tagger == "panns"
    assert config.visual_scan.max_interval_s == 30
    assert config.visual_scan.hwaccel == "none"
    assert config.ocr.enabled is True
    assert config.ocr.engine == "rapidocr"
    assert config.ocr.device == "cpu"
    assert config.ocr.rec_model_path.endswith("PP-OCRv6_small_rec.onnx")
    assert config.ocr.for_language("ko").rec_model_path.endswith("korean_PP-OCRv5_rec.onnx")
    assert config.ocr.use_angle_cls is False
    assert config.ocr.for_language("ja").rec_model_path.endswith("PP-OCRv6_small_rec.onnx")
    assert config.ocr.for_language("en").rec_model_path == config.ocr.rec_model_path
    assert config.advanced_asr.provider == "local"
    assert config.advanced_asr.local.max_segment_s == 180
    assert config.advanced_asr.cloud.max_segment_s == 600
    assert config.view_frames.max_montages_per_call == 16
    assert config.view_frames.fps_options == (0.1, 0.2, 0.5, 1.0, 2.0)
    assert config.view_frames.grid_options == (1, 2, 3, 4)
    assert config.transcribe_segment.max_duration_s == 60
    assert config.vision_llm.input_cost_env == "VLM_INPUT_COST_PER_MTOK"
    assert config.vision_llm.temperature is None


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


def _with(tmp_path: Path, old: str, new: str) -> Path:
    project_root = Path(__file__).parents[1]
    path = tmp_path / "config.yaml"
    path.write_text(
        (project_root / "config.yaml").read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )
    return path


def test_load_config_rejects_an_unknown_event_tagger(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="audio_events.tagger"):
        load_config(_with(tmp_path, "tagger: panns", "tagger: magic"))


def test_load_config_rejects_sampling_intervals_that_cannot_hold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_interval_s must not exceed"):
        load_config(_with(tmp_path, "min_interval_s: 2", "min_interval_s: 45"))


def test_load_config_rejects_a_zero_length_vad_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="vad window lengths"):
        load_config(_with(tmp_path, "window_max_s: 30", "window_max_s: 0"))


def test_load_config_custom_and_zero_temperature(tmp_path: Path) -> None:
    cfg_07 = load_config(_with(tmp_path, "temperature: null", "temperature: 0.7"))
    assert cfg_07.vision_llm.temperature == 0.7

    cfg_00 = load_config(_with(tmp_path, "temperature: null", "temperature: 0.0"))
    assert cfg_00.vision_llm.temperature == 0.0


def test_load_config_null_and_omitted_temperature(tmp_path: Path) -> None:
    cfg_null = load_config(_with(tmp_path, "temperature: null", "temperature: null"))
    assert cfg_null.vision_llm.temperature is None

    cfg_omitted = load_config(_with(tmp_path, "  temperature: null\n", ""))
    assert cfg_omitted.vision_llm.temperature is None


def test_load_config_rejects_out_of_range_temperature(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="vision_llm temperature must be between 0.0 and 2.0"):
        load_config(_with(tmp_path, "temperature: null", "temperature: -0.1"))

    with pytest.raises(ValueError, match="vision_llm temperature must be between 0.0 and 2.0"):
        load_config(_with(tmp_path, "temperature: null", "temperature: 2.1"))


def test_load_config_custom_hwaccel_and_ocr_device(tmp_path: Path) -> None:
    path = _with(tmp_path, "hwaccel: none", "hwaccel: d3d11va")
    text = path.read_text(encoding="utf-8").replace(
        "device: cpu\n  scan_fps", "device: dml\n  scan_fps"
    )
    path.write_text(text, encoding="utf-8")
    cfg = load_config(path)
    assert cfg.visual_scan.hwaccel == "d3d11va"
    assert cfg.ocr.device == "dml"


def test_load_config_rejects_unsupported_hwaccel(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported visual_scan.hwaccel"):
        load_config(_with(tmp_path, "hwaccel: none", "hwaccel: invalid_accel"))


def test_load_config_rejects_unsupported_ocr_device(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    project_root = Path(__file__).parents[1]
    path.write_text(
        (project_root / "config.yaml")
        .read_text(encoding="utf-8")
        .replace("device: cpu\n  scan_fps", "device: rocm\n  scan_fps"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported ocr.device"):
        load_config(path)
