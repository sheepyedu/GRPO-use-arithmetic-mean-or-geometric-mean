import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_seed import summarize
from src.data import file_sha256, write_json


class ResultTests(unittest.TestCase):
    def fixture(self, root, actual_rounds=200, early=False):
        for environment in ("math", "mixed"):
            decision = {
                "round": actual_rounds,
                "stop": True,
                "reason": "paired_plateau",
            }
            if early:
                write_json(
                    root / f"formal-{environment}-control" / "STOP_DECISION.json",
                    decision,
                )
            for operator in ("am", "gm"):
                path = root / f"formal-{environment}-{operator}"
                path.mkdir()
                offset = 0.02 if operator == "gm" else 0
                metrics = {
                    "gsm8k": {"accuracy": 0.8 + offset},
                    "arc": {"accuracy": 0.6 + offset},
                    "boolq": {"accuracy": 0.7 + offset},
                    "mixed_weighted": 0.725 + offset,
                }
                write_json(
                    path / "config.json",
                    {
                        "environment": environment,
                        "operator": operator,
                        "seed": 17,
                        "rounds": 200,
                        **({"early_stopping": {"min_rounds": 20}} if early else {}),
                    },
                )
                write_json(path / "source_manifest.json", {"src/train.py": "fixture"})
                write_json(
                    path / "COMPLETE.json",
                    {
                        "formal": True,
                        "rounds": actual_rounds,
                        "updates": actual_rounds * 16,
                        **(
                            {
                                "paired_stop_decision": decision,
                                "stop_reason": decision["reason"],
                            }
                            if early
                            else {}
                        ),
                        "elapsed_seconds": 3600,
                        "metrics": metrics,
                    },
                )
                if early:
                    write_json(path / "STOP_DECISION.json", decision)
                checkpoint = path / f"checkpoint-{actual_rounds:03d}"
                checkpoint.mkdir()
                for name in (
                    "adapter_config.json",
                    "adapter_model.safetensors",
                    "training_state.pt",
                ):
                    (checkpoint / name).write_bytes(b"test fixture")
                write_json(
                    checkpoint / "COMMITTED.json",
                    {
                        "round": actual_rounds,
                        "updates": actual_rounds * 16,
                        "files_sha256": {
                            name: file_sha256(checkpoint / name)
                            for name in (
                                "adapter_config.json",
                                "adapter_model.safetensors",
                                "training_state.pt",
                            )
                        },
                    },
                )
                for step in range(0, actual_rounds + 1, 20):
                    write_json(path / f"validation-{step:03d}-metrics.json", metrics)
                (path / "metrics.jsonl").write_text(
                    "\n".join(
                        json.dumps({"round": step, "tokens": 100})
                        for step in range(1, actual_rounds + 1)
                    )
                )

    def test_paired_summary_and_integrity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            result = summarize(root)
            self.assertEqual(result["complete_runs"], 4)
            self.assertAlmostEqual(result["gm_minus_am"]["math"], 0.02)
            self.assertAlmostEqual(result["gm_minus_am"]["mixed"], 0.02)
            self.assertEqual(result["runs"][0]["training_generated_tokens"], 20000)
            self.assertAlmostEqual(result["runs"][0]["validation_auc_normalized"], 0.8)
            checkpoint = root / "formal-math-am/checkpoint-200"
            (checkpoint / "training_state.pt").write_bytes(b"corrupt")
            with self.assertRaises(AssertionError):
                summarize(root)

    def test_incomplete_seed_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                summarize(Path(temporary))

    def test_early_stopped_pair_and_asymmetric_rejection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root, actual_rounds=20, early=True)
            result = summarize(root)
            self.assertEqual(result["runs"][0]["rounds"], 20)
            self.assertEqual(result["runs"][0]["planned_rounds"], 200)
            self.assertEqual(result["runs"][0]["stop_reason"], "paired_plateau")
            path = root / "formal-math-gm/COMPLETE.json"
            complete = json.loads(path.read_text())
            complete["rounds"] = 21
            write_json(path, complete)
            with self.assertRaises(AssertionError):
                summarize(root)


if __name__ == "__main__":
    unittest.main()
