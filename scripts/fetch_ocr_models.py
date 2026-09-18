"""Fetch reproducible, matched OCR weights and character dictionaries.

Both recognisers are official PaddlePaddle releases. Models are pinned to a
Hub revision; model/keys hashes and provenance are saved beside the assets.
No model download occurs implicitly while indexing a user's video.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

MODELS = {
    "PP-OCRv6_tiny_det": (
        "PaddlePaddle/PP-OCRv6_tiny_det_onnx",
        "2ba1506c0380b8f0b03dd142459aac66d4421f6c",
    ),
    "korean_PP-OCRv5_rec": (
        "PaddlePaddle/korean_PP-OCRv5_mobile_rec_onnx",
        "5c6f574b8e2230adf4287b33e736d71b9fabd28e",
    ),
    "PP-OCRv6_small_rec": (
        "PaddlePaddle/PP-OCRv6_small_rec_onnx",
        "b8f84f0b80c529de40b4fbb3544b84fa7233a513",
    ),
}


def fetch(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, (repo, revision) in MODELS.items():
        model_path = output_dir / f"{name}.onnx"
        keys_path = output_dir / f"{name.removesuffix('_rec')}_dict.txt"
        source = Path(hf_hub_download(repo, "inference.onnx", revision=revision))
        metadata_path = Path(hf_hub_download(repo, "inference.yml", revision=revision))
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        characters = metadata.get("PostProcess", {}).get("character_dict")
        if name.endswith("_rec") and not characters:
            raise SystemExit(f"{repo}@{revision} has no character_dict in inference.yml")
        temporary = model_path.with_suffix(".onnx.tmp")
        shutil.copyfile(source, temporary)
        temporary.replace(model_path)
        if characters:
            keys_path.write_text("\n".join(characters) + "\n", encoding="utf-8")
        shutil.copyfile(metadata_path, output_dir / f"{name}.yml")
        manifest[name] = dict(
            repo=repo,
            revision=revision,
            bytes=model_path.stat().st_size,
            model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
            keys_sha256=hashlib.sha256(keys_path.read_bytes()).hexdigest() if characters else None,
            characters=len(characters or []),
        )
        print(f"{name}: {model_path.stat().st_size // 1024}KB, {len(characters or [])} characters")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[1] / "models/ocr"
    fetch(target)
