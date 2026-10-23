"""Download only public, pinned model assets into the project-local model folder."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="configs/local_qwen3_1_7b.json")
    args = parser.parse_args()
    config = json.loads((ROOT / args.profile).read_text(encoding="utf-8"))["model"]
    destination = (ROOT / config["path"]).resolve()
    if not destination.is_relative_to(ROOT / "models"):
        raise ValueError("local model path must stay under this checkout's models directory")
    from huggingface_hub import snapshot_download
    snapshot_download(config["id"], revision=config["revision"], local_dir=str(destination),
                      token=False, max_workers=2,
                      allow_patterns=["*.json", "*.txt", "*.safetensors", "LICENSE", "README.md"])
    (destination / "memturn_source.json").write_text(json.dumps(
        {"model_id": config["id"], "revision": config["revision"]}, indent=2), encoding="utf-8")
    print(json.dumps({"model_directory": str(destination), "revision": config["revision"]}))


if __name__ == "__main__":
    main()
