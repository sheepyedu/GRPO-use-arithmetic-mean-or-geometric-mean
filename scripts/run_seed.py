"""Sequential paired stages, one process per card, no implicit run retries."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.data import write_json


def pair(root, config, environment, phase):
    children = []
    for gpu, operator in enumerate(("am", "gm")):
        output = root / f"{phase}-{environment}-{operator}"
        if output.exists():
            raise FileExistsError(f"Refusing to reuse run directory {output}")
        log = (root / f"{phase}-{environment}-{operator}.log").open("xb")
        command = [
            sys.executable,
            "-u",
            "-m",
            "src.train",
            "--config",
            str(config),
            "--environment",
            environment,
            "--operator",
            operator,
            "--output",
            str(output),
        ]
        if phase == "pilot":
            command.append("--calibration")
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "false",
            "USE_TF": "0",
            "USE_FLAX": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
        }
        process = subprocess.Popen(
            command,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        children.append((process, output, log))
    write_json(
        root / "pair.json",
        {
            "phase": phase,
            "environment": environment,
            "children": [{"pid": p.pid, "output": str(out)} for p, out, _ in children],
            "start": time.time(),
        },
    )
    # Both sides finish before advancing the environment. A failure blocks the
    # next pair and retains the still-running partner's valid result.
    codes = []
    for process, output, log in children:
        codes.append(process.wait())
        log.close()
    write_json(
        root / f"{phase}-{environment}-exit.json", {"codes": codes, "time": time.time()}
    )
    if codes != [0, 0] or any(
        not (out / "COMPLETE.json").exists() for _, out, _ in children
    ):
        raise RuntimeError(f"{phase}/{environment} pair failed, codes={codes}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["pilot", "formal"], required=True)
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "manager.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = Path(args.config).resolve()
        expected_rounds = 10 if args.phase == "pilot" else 200
        if json.loads(config.read_text())["rounds"] != expected_rounds:
            raise ValueError(
                f"The {args.phase} phase requires {expected_rounds} rounds"
            )
        for environment in ("math", "mixed"):
            pair(root, config, environment, args.phase)
        write_json(
            root / f"{args.phase}-four-complete.json", {"time": time.time(), "runs": 4}
        )


if __name__ == "__main__":
    main()
