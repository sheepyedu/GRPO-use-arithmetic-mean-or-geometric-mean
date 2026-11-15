"""Fixed splits, leakage exclusion, and deterministic paired prompt scheduling."""

import argparse
import hashlib
import json
import random
import re
import unicodedata
from pathlib import Path


def canonical(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[+-]?\d+(?:[.,/]\d+)*|[^\W\d_]+", normalized))


def content_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def normalize_row(task, split, index, row):
    question = row["question"]
    result = {"id": f"{task}/{split}/{index}", "task": task, "question": question}
    if task == "gsm8k":
        result["answer"] = row["answer"].rsplit("####", 1)[1].strip()
    elif task == "arc":
        labels = row["choices"]["label"]
        result.update(labels=labels, answer=row["answerKey"])
        result["question"] += "\n" + "\n".join(
            f"{label}. {text}" for label, text in zip(labels, row["choices"]["text"])
        )
    else:
        result.update(
            answer="yes" if row["answer"] else "no",
            question=row["passage"] + "\n\n" + question,
        )
    return result


def near_key(text):
    # Retain every digit. Token bigram similarity catches reworded punctuation
    # and very close copies without conflating different numerical problems.
    tokens = canonical(text).split()
    return set(zip(tokens, tokens[1:])) or set(tokens)


def exclude_duplicates(candidates, protected):
    seen = {canonical(row["question"]) for row in protected}
    signatures = [near_key(row["question"]) for row in protected]
    numbers = [
        re.findall(r"[+-]?\d+(?:[.,/]\d+)*", row["question"]) for row in protected
    ]
    index = {}
    for position, signature in enumerate(signatures):
        for token in signature:
            index.setdefault(token, set()).add(position)
    kept, counts = [], {"exact": 0, "near": 0}
    for row in candidates:
        normalized = canonical(row["question"])
        signature = near_key(row["question"])
        if normalized in seen:
            counts["exact"] += 1
            continue
        overlaps = {}
        for token in signature:
            for position in index.get(token, ()):
                overlaps[position] = overlaps.get(position, 0) + 1
        row_numbers = re.findall(r"[+-]?\d+(?:[.,/]\d+)*", row["question"])
        duplicate = any(
            numbers[position] == row_numbers
            and overlap / (len(signature) + len(signatures[position]) - overlap) >= 0.9
            for position, overlap in overlaps.items()
        )
        if duplicate:
            counts["near"] += 1
            continue
        position = len(signatures)
        signatures.append(signature)
        numbers.append(row_numbers)
        for token in signature:
            index.setdefault(token, set()).add(position)
        seen.add(normalized)
        kept.append(row)
    return kept, counts


def prepare(destination):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    destination = Path(destination)
    if (destination / "manifest.json").exists():
        raise FileExistsError(
            "Prepared data already exist; verify and reuse their manifest"
        )
    sources = {
        "gsm8k": ("openai/gsm8k", "main"),
        "arc": ("allenai/ai2_arc", "ARC-Easy"),
        "boolq": ("google/boolq", "default"),
    }
    data, revisions = {}, {}
    for task, (repo, config) in sources.items():
        info = HfApi().dataset_info(repo)
        revisions[task] = {
            "repo": repo,
            "revision": info.sha,
            "config": config,
            "card": info.card_data.to_dict() if info.card_data else {},
        }
        raw = load_dataset(repo, config, revision=info.sha)
        data[task] = {
            split: [normalize_row(task, split, i, row) for i, row in enumerate(rows)]
            for split, rows in raw.items()
        }
        print("loaded", task, {k: len(v) for k, v in data[task].items()}, flush=True)
    test = data["gsm8k"]["test"] + data["arc"]["test"] + data["boolq"]["validation"]
    protected = list(test)
    audit = {}
    splits = {"test": test, "validation": [], "train_pools": {}}
    for task in sources:
        rows, counts = exclude_duplicates(data[task]["train"], protected)
        random.Random(2026).shuffle(rows)
        if task in ("gsm8k", "boolq"):
            validation, train = rows[:1000], rows[1000:]
        else:
            validation, extra = exclude_duplicates(
                data[task]["validation"], protected + rows
            )
            counts["validation_excluded"] = extra
            train = rows
        splits["validation"].extend(validation)
        splits["train_pools"][task] = train
        protected.extend(rows + validation)
        audit[task] = counts
    pools = splits["train_pools"]
    total = min(
        6000, len(pools["gsm8k"]), 4 * len(pools["arc"]), 4 * len(pools["boolq"])
    )
    total -= total % 4
    splits["math"] = pools["gsm8k"][:total]
    splits["mixed"] = (
        pools["gsm8k"][: total // 2]
        + pools["arc"][: total // 4]
        + pools["boolq"][: total // 4]
    )
    for name, rows in splits.items():
        write_json(destination / f"{name}.json", rows)
    manifest = {
        "data_seed": 2026,
        "normalization": "NFKC/casefold/word tokens preserving digits",
        "near_duplicate": "word bigram Jaccard >=0.9",
        "sources": revisions,
        "exclusions": audit,
        "train_total": total,
        "split_hashes": {name: content_hash(rows) for name, rows in splits.items()},
        "test_labels_used_for_selection": False,
    }
    write_json(destination / "manifest.json", manifest)
    print(
        json.dumps({"train_total": total, "hash": content_hash(manifest)}), flush=True
    )


class PromptSchedule:
    def __init__(self, rows, environment, seed):
        self.random = random.Random(seed)
        self.pools = {
            task: [row for row in rows if row["task"] == task]
            for task in ("gsm8k", "arc", "boolq")
        }
        self.orders = {task: [] for task in self.pools}
        self.environment = environment

    def batch(self):
        result = []
        counts = (
            {"gsm8k": 64}
            if self.environment == "math"
            else {"gsm8k": 32, "arc": 16, "boolq": 16}
        )
        for task, count in counts.items():
            for _ in range(count):
                if not self.orders[task]:
                    self.orders[task] = self.pools[task].copy()
                    self.random.shuffle(self.orders[task])
                result.append(self.orders[task].pop())
        self.random.shuffle(result)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    prepare(parser.parse_args().output)
