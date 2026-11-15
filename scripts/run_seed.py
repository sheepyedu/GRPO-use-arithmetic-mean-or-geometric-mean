"""Sequential paired stages, one process per card, no implicit run retries."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.data import file_sha256, write_json
from src.early_stop import PairedEarlyStopping, process_start_ticks


def supervise_rounds(children, control, config, started):
    policy = PairedEarlyStopping(config.get("early_stopping"), config["rounds"])
    number = 0
    stopping = False
    while any(process.poll() is None for process, _, _ in children):
        failed = [
            process.pid for process, _, _ in children if process.poll() not in (None, 0)
        ]
        premature = not stopping and any(
            process.poll() == 0 for process, _, _ in children
        )
        if failed or premature:
            if not (control / "ABORT.json").exists():
                write_json(
                    control / "ABORT.json",
                    {"failed_pids": failed, "premature_exit": premature},
                )
            stopping = True
        if not stopping:
            paths = {
                operator: control / f"ready-{operator}-{number:03d}.json"
                for operator in ("am", "gm")
            }
            if all(path.exists() for path in paths.values()):
                ready = {
                    operator: json.loads(path.read_text())
                    for operator, path in paths.items()
                }
                for operator, receipt in ready.items():
                    if receipt["round"] != number or receipt["operator"] != operator:
                        raise ValueError("Invalid paired round receipt")
                if number:
                    for (_, output, _), operator in zip(children, ("am", "gm")):
                        committed = (
                            output / f"checkpoint-{number:03d}" / "COMMITTED.json"
                        )
                        if (
                            file_sha256(committed)
                            != ready[operator]["checkpoint_sha256"]
                        ):
                            raise ValueError(
                                "Checkpoint changed before the paired decision"
                            )
                primary = {
                    operator: receipt["primary"] for operator, receipt in ready.items()
                }
                if any(value is None for value in primary.values()) and any(
                    value is not None for value in primary.values()
                ):
                    raise ValueError("Paired validation schedules differ")
                scores = (
                    primary
                    if all(value is not None for value in primary.values())
                    else None
                )
                decision = policy.decide(number, scores, time.monotonic() - started)
                write_json(control / f"decision-{number:03d}.json", decision)
                if decision["stop"]:
                    write_json(control / "STOP_DECISION.json", decision)
                    stopping = True
                number += 1
        time.sleep(1)


def pair(root, config, environment, phase, gpus=(0, 1)):
    settings = json.loads(config.read_text())
    started = time.monotonic()
    control = None
    if phase == "formal":
        control = root / f"{phase}-{environment}-control"
        control.mkdir(exist_ok=False)
        write_json(
            control / "controller.json",
            {
                "pid": os.getpid(),
                "start_ticks": process_start_ticks(os.getpid()),
                "config_sha256": file_sha256(config),
                "environment": environment,
                "seed": settings["seed"],
            },
        )
    children = []
    for gpu, operator in zip(gpus, ("am", "gm")):
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
        else:
            command.extend(["--pair-control", str(control)])
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "OMP_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "false",
            "USE_TF": "0",
            "USE_FLAX": "0",
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "TMPDIR": "/tmp",
            "VLLM_RPC_BASE_PATH": tempfile.mkdtemp(
                prefix=f"grpo-{environment}-{operator}-", dir="/tmp"
            ),
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
        root / f"{phase}-{environment}-pair.json",
        {
            "phase": phase,
            "environment": environment,
            "children": [
                {"pid": p.pid, "output": str(out), "gpu": gpu}
                for (p, out, _), gpu in zip(children, gpus)
            ],
            "start": time.time(),
        },
    )
    controller_error = None
    if control:
        try:
            supervise_rounds(children, control, settings, started)
        except BaseException as exc:
            write_json(control / "ABORT.json", {"reason": str(exc)})
            controller_error = exc
    # Both sides finish before advancing the environment.
    codes = []
    for process, output, log in children:
        codes.append(process.wait())
        log.close()
    write_json(
        root / f"{phase}-{environment}-exit.json", {"codes": codes, "time": time.time()}
    )
    if controller_error:
        raise RuntimeError("Paired controller failed") from controller_error
    if codes != [0, 0] or any(
        not (out / "COMPLETE.json").exists() for _, out, _ in children
    ):
        raise RuntimeError(f"{phase}/{environment} pair failed, codes={codes}")
    if control:
        decision = json.loads((control / "STOP_DECISION.json").read_text())
        for _, output, _ in children:
            complete = json.loads((output / "COMPLETE.json").read_text())
            if (
                complete["rounds"] != decision["round"]
                or complete["paired_stop_decision"] != decision
            ):
                raise ValueError(
                    "Paired terminal results differ from the shared stop decision"
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=["pilot", "formal"], required=True)
    parser.add_argument("--parallel-environments", action="store_true")
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
        if args.parallel_environments:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(pair, root, config, environment, args.phase, gpus)
                    for environment, gpus in (("math", (0, 1)), ("mixed", (2, 3)))
                ]
                for future in futures:
                    future.result()
        else:
            for environment in ("math", "mixed"):
                pair(root, config, environment, args.phase)
        write_json(
            root / f"{args.phase}-four-complete.json", {"time": time.time(), "runs": 4}
        )


if __name__ == "__main__":
    main()
