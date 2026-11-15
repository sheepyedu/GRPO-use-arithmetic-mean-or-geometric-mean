"""Accept four terminal runs and write the paired single-seed result."""

import argparse
import json
from pathlib import Path

from src.data import content_hash, file_sha256, write_json


def load(path):
    return json.loads(path.read_text())


def summarize(root):
    runs = []
    common_config = None
    common_source = None
    paired_terminals = {}
    for environment in ("math", "mixed"):
        for operator in ("am", "gm"):
            path = root / f"formal-{environment}-{operator}"
            complete = load(path / "COMPLETE.json")
            config = load(path / "config.json")
            assert complete["formal"] is True
            actual_rounds = complete["rounds"]
            assert config["rounds"] == 200
            assert 1 <= actual_rounds <= config["rounds"]
            stop_decision = complete.get("paired_stop_decision")
            if actual_rounds < config["rounds"] or config.get("early_stopping"):
                assert stop_decision and stop_decision["stop"] is True
                assert stop_decision["round"] == actual_rounds
                assert stop_decision["reason"] in (
                    "paired_plateau",
                    "wall_time",
                    "max_rounds",
                )
                assert load(path / "STOP_DECISION.json") == stop_decision
                assert (
                    load(root / f"formal-{environment}-control" / "STOP_DECISION.json")
                    == stop_decision
                )
            terminal = (actual_rounds, stop_decision)
            if environment in paired_terminals:
                assert terminal == paired_terminals[environment], (
                    "Paired stopping rounds or decisions differ"
                )
            paired_terminals[environment] = terminal
            assert not (path / "FAILURE.json").exists()
            checkpoint = path / f"checkpoint-{actual_rounds:03d}"
            assert (checkpoint / "COMMITTED.json").exists()
            assert (checkpoint / "training_state.pt").stat().st_size > 0
            committed = load(checkpoint / "COMMITTED.json")
            assert committed["round"] == actual_rounds
            assert committed["updates"] == complete["updates"]
            for name, digest in committed["files_sha256"].items():
                assert file_sha256(checkpoint / name) == digest, name
            assert config["environment"] == environment
            assert config["operator"] == operator
            shared = {
                key: value
                for key, value in config.items()
                if key not in ("environment", "operator")
            }
            source = load(path / "source_manifest.json")
            if common_config is None:
                common_config, common_source = shared, source
            assert shared == common_config, "Paired run configurations differ"
            assert source == common_source, "Paired run sources differ"
            metrics = complete["metrics"]
            learning = []
            for file in sorted(path.glob("validation-*-metrics.json")):
                step = int(file.name.split("-")[1])
                value = load(file)
                primary = (
                    value["gsm8k"]["accuracy"]
                    if environment == "math"
                    else value["mixed_weighted"]
                )
                learning.append((step, primary))
            expected_validation = list(
                range(0, actual_rounds + 1, config.get("validation_every", 20))
            )
            if expected_validation[-1] != actual_rounds:
                expected_validation.append(actual_rounds)
            assert [step for step, _ in learning] == expected_validation
            auc = (
                sum(
                    (b[0] - a[0]) * (a[1] + b[1]) / 2
                    for a, b in zip(learning, learning[1:])
                )
                / actual_rounds
            )
            rounds = [
                json.loads(line)
                for line in (path / "metrics.jsonl").read_text().splitlines()
            ]
            assert [row["round"] for row in rounds] == list(range(1, actual_rounds + 1))
            runs.append(
                {
                    "environment": environment,
                    "operator": operator,
                    "seed": config["seed"],
                    "rounds": actual_rounds,
                    "planned_rounds": config["rounds"],
                    "stop_reason": complete.get("stop_reason", "max_rounds"),
                    "paired_stop_decision": stop_decision,
                    "primary": metrics["gsm8k"]["accuracy"]
                    if environment == "math"
                    else metrics["mixed_weighted"],
                    "test_metrics": metrics,
                    "validation_auc_normalized": auc,
                    "training_generated_tokens": sum(row["tokens"] for row in rounds),
                    "allocated_gpu_hours": complete["elapsed_seconds"] / 3600,
                    "directory": str(path),
                }
            )
    paired = {
        environment: next(
            r["primary"]
            for r in runs
            if r["environment"] == environment and r["operator"] == "gm"
        )
        - next(
            r["primary"]
            for r in runs
            if r["environment"] == environment and r["operator"] == "am"
        )
        for environment in ("math", "mixed")
    }
    return {
        "seed": common_config["seed"],
        "complete_runs": 4,
        "planned_runs": 20,
        "common_config": common_config,
        "source_manifest_hash": content_hash(common_source),
        "runs": runs,
        "gm_minus_am": paired,
        "statistical_scope": "One paired seed; five-seed confidence intervals remain pending",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(Path(args.directory).resolve())
    write_json(args.output, result)
    print(
        json.dumps(
            {
                "seed": result["seed"],
                "complete_runs": 4,
                "gm_minus_am": result["gm_minus_am"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
