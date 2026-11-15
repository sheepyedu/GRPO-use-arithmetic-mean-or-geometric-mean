"""Read-only verification of prepared data and paired training inputs."""

import argparse
import json
from collections import Counter
from pathlib import Path

from src.data import canonical, content_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    root = Path(parser.parse_args().directory)
    manifest = json.loads((root / "manifest.json").read_text())
    splits = {
        name: json.loads((root / f"{name}.json").read_text())
        for name in manifest["split_hashes"]
    }
    for name, rows in splits.items():
        assert content_hash(rows) == manifest["split_hashes"][name], name
    n = manifest["train_total"]
    assert len(splits["math"]) == len(splits["mixed"]) == n
    assert Counter(row["task"] for row in splits["mixed"]) == {
        "gsm8k": n // 2,
        "arc": n // 4,
        "boolq": n // 4,
    }
    train = {canonical(row["question"]) for row in splits["math"] + splits["mixed"]}
    validation = {canonical(row["question"]) for row in splits["validation"]}
    test = {canonical(row["question"]) for row in splits["test"]}
    assert not train.intersection(validation)
    assert not train.intersection(test)
    assert not validation.intersection(test)
    math_ids = {row["id"] for row in splits["math"]}
    assert {row["id"] for row in splits["mixed"] if row["task"] == "gsm8k"}.issubset(
        math_ids
    )
    print(
        json.dumps(
            {
                "verified": True,
                "train_per_environment": n,
                "validation": dict(
                    Counter(row["task"] for row in splits["validation"])
                ),
                "test": dict(Counter(row["task"] for row in splits["test"])),
                "manifest_content_sha256": content_hash(manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
