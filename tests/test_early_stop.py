import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_seed import supervise_rounds
from src.data import write_json
from src.early_stop import PairedEarlyStopping, exchange_round


class EarlyStopTests(unittest.TestCase):
    def policy(self):
        return PairedEarlyStopping(
            {"min_rounds": 20, "patience": 2, "min_delta": 0.005, "max_pair_hours": 12},
            200,
        )

    def test_both_must_plateau(self):
        policy = self.policy()
        self.assertFalse(policy.decide(0, {"am": 0.5, "gm": 0.5}, 0)["stop"])
        self.assertFalse(policy.decide(10, {"am": 0.501, "gm": 0.52}, 100)["stop"])
        self.assertFalse(policy.decide(20, {"am": 0.502, "gm": 0.53}, 200)["stop"])
        self.assertFalse(policy.decide(30, {"am": 0.502, "gm": 0.531}, 300)["stop"])
        result = policy.decide(40, {"am": 0.502, "gm": 0.531}, 400)
        self.assertTrue(result["stop"])
        self.assertEqual(result["reason"], "paired_plateau")
        self.assertEqual(result["round"], 40)

    def test_minimum_and_delta(self):
        policy = self.policy()
        policy.decide(0, {"am": 0.5, "gm": 0.5}, 0)
        policy.decide(10, {"am": 0.504, "gm": 0.504}, 10)
        self.assertTrue(policy.decide(20, {"am": 0.504, "gm": 0.504}, 20)["stop"])
        policy = self.policy()
        policy.decide(0, {"am": 0.5, "gm": 0.5}, 0)
        policy.decide(10, {"am": 0.506, "gm": 0.506}, 10)
        self.assertFalse(policy.decide(20, {"am": 0.506, "gm": 0.506}, 20)["stop"])

    def test_time_and_maximum(self):
        policy = self.policy()
        self.assertEqual(policy.decide(25, None, 12 * 3600)["reason"], "wall_time")
        self.assertEqual(policy.decide(200, None, 10)["reason"], "max_rounds")

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            self.policy().decide(10, {"am": float("nan"), "gm": 0.5}, 0)

    def test_round_exchange_and_abort(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(
                root / "decision-020.json",
                {"round": 20, "stop": True, "reason": "paired_plateau"},
            )
            decision = exchange_round(root, "am", 20, 0.8)
            self.assertTrue(decision["stop"])
            self.assertTrue((root / "ready-am-020.json").exists())
            write_json(root / "ABORT.json", {"reason": "partner failed"})
            with self.assertRaises(RuntimeError):
                exchange_round(root, "gm", 21, None)

    def test_two_processes_stop_on_the_same_checkpoint(self):
        worker = """import sys
from pathlib import Path
from src.data import write_json,file_sha256
from src.early_stop import exchange_round
control,operator,output=Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3])
for number in range(6):
 digest=None
 if number:
  committed=output/f"checkpoint-{number:03d}"/"COMMITTED.json"
  write_json(committed,{"round":number,"fixture":True})
  digest=file_sha256(committed)
 decision=exchange_round(control,operator,number,.8,digest,timeout=15)
 if decision["stop"]:
  write_json(output/"COMPLETE.json",{"rounds":number,"paired_stop_decision":decision})
  break
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            control = root / "control"
            control.mkdir()
            children = []
            repository = Path(__file__).resolve().parent.parent
            for operator in ("am", "gm"):
                output = root / operator
                process = subprocess.Popen(
                    [sys.executable, "-c", worker, str(control), operator, str(output)],
                    env={**os.environ, "PYTHONPATH": str(repository)},
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
                children.append((process, output, None))
            started = time.monotonic()
            original_sleep = time.sleep

            def bounded_sleep(seconds):
                if time.monotonic() - started > 15:
                    raise TimeoutError("Controller fixture timed out")
                original_sleep(seconds)

            try:
                with patch("scripts.run_seed.time.sleep", bounded_sleep):
                    supervise_rounds(
                        children,
                        control,
                        {
                            "rounds": 5,
                            "early_stopping": {
                                "min_rounds": 2,
                                "patience": 2,
                                "min_delta": 0.005,
                                "max_pair_hours": 12,
                            },
                        },
                        started,
                    )
                for process, output, _ in children:
                    self.assertEqual(
                        process.wait(timeout=5), 0, process.stderr.read().decode()
                    )
                    complete = json.loads((output / "COMPLETE.json").read_text())
                    self.assertEqual(complete["rounds"], 2)
                    self.assertEqual(
                        complete["paired_stop_decision"]["reason"], "paired_plateau"
                    )
            finally:
                for process, _, _ in children:
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=5)
                    process.stderr.close()


if __name__ == "__main__":
    unittest.main()
