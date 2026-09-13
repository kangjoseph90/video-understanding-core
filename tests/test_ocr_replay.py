import json
from dataclasses import asdict

import pytest

from scripts.replay_ocr import main
from tests.test_ocr_index import line
from vuc.ocr import Observation


def test_replay_is_read_only_and_refuses_to_overwrite_results(tmp_path):
    source = tmp_path / "observations.jsonl"
    raw = "\n".join(
        json.dumps(asdict(Observation(t, (line("Readable text", confidence=0.99),))))
        for t in (0, 1)
    )
    source.write_text(raw)
    out = tmp_path / "replay"
    args = [str(source), "--duration", "2", "--output-dir", str(out)]
    assert main(args) == 0
    assert source.read_text() == raw
    assert (out / "text_index.txt").read_text() == "[0-2, bottom center, medium] Readable text"
    with pytest.raises(SystemExit):
        main(args)
    assert source.read_text() == raw
