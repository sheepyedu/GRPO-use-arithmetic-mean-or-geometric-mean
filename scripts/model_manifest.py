"""Hash local model files without copying weights or loading a GPU model."""

import argparse
import hashlib
from pathlib import Path

from src.data import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("output")
    args = parser.parse_args()
    root = Path(args.model)
    files = sorted(root.glob("*.safetensors"))
    files += [
        root / name
        for name in (
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "model.safetensors.index.json",
        )
    ]
    manifest = {}
    for path in files:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        manifest[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    write_json(args.output, manifest)
    print(f"Hashed {len(manifest)} model files")


if __name__ == "__main__":
    main()
