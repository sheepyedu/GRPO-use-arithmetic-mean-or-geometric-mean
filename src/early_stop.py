"""Paired validation-based stopping at committed round boundaries."""

import json
import math
import time
from pathlib import Path

from src.data import write_json


class PairedEarlyStopping:
    def __init__(self, settings, max_rounds):
        self.settings = settings
        self.max_rounds = max_rounds
        self.best = {"am": None, "gm": None}
        self.bad_checks = {"am": 0, "gm": 0}
        if settings:
            if not 1 <= settings["min_rounds"] <= max_rounds:
                raise ValueError("Invalid minimum round count")
            if (
                settings["patience"] < 1
                or not math.isfinite(settings["min_delta"])
                or settings["min_delta"] < 0
            ):
                raise ValueError("Invalid early-stopping patience or delta")
            if (
                not math.isfinite(settings["max_pair_hours"])
                or settings["max_pair_hours"] <= 0
            ):
                raise ValueError("The paired time budget must be positive")

    def decide(self, number, scores, elapsed_seconds):
        if scores is not None:
            if set(scores) != {"am", "gm"}:
                raise ValueError("Both validation scores are required")
            delta = self.settings["min_delta"] if self.settings else 0
            for operator, score in scores.items():
                if not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("Invalid validation accuracy")
                previous = self.best[operator]
                if previous is None or score >= previous + delta:
                    self.best[operator] = score
                    self.bad_checks[operator] = 0
                else:
                    self.bad_checks[operator] += 1
        reason = None
        if number >= self.max_rounds:
            reason = "max_rounds"
        elif self.settings and number > 0:
            if elapsed_seconds >= self.settings["max_pair_hours"] * 3600:
                reason = "wall_time"
            elif (
                scores is not None
                and number >= self.settings["min_rounds"]
                and all(
                    value >= self.settings["patience"]
                    for value in self.bad_checks.values()
                )
            ):
                reason = "paired_plateau"
        return {
            "round": number,
            "stop": reason is not None,
            "reason": reason,
            "best": self.best.copy(),
            "bad_checks": self.bad_checks.copy(),
            "scores": scores,
            "elapsed_seconds": elapsed_seconds,
        }


def process_start_ticks(pid):
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    if stat[0] == "Z":
        raise ProcessLookupError(pid)
    return int(stat[19])


def exchange_round(
    control, operator, number, primary, checkpoint_sha=None, timeout=10800
):
    control = Path(control)
    ready = {
        "operator": operator,
        "round": number,
        "primary": primary,
        "checkpoint_sha256": checkpoint_sha,
    }
    path = control / f"ready-{operator}-{number:03d}.json"
    if path.exists():
        if json.loads(path.read_text()) != ready:
            raise ValueError("Conflicting round receipt")
    else:
        write_json(path, ready)
    deadline = time.monotonic() + timeout
    decision = control / f"decision-{number:03d}.json"
    while True:
        abort = control / "ABORT.json"
        if abort.exists():
            raise RuntimeError(f"Paired run aborted: {abort.read_text()}")
        if decision.exists():
            result = json.loads(decision.read_text())
            if result["round"] != number or type(result["stop"]) is not bool:
                raise ValueError("Invalid paired decision")
            return result
        controller_file = control / "controller.json"
        if controller_file.exists():
            controller = json.loads(controller_file.read_text())
            try:
                if process_start_ticks(controller["pid"]) != controller["start_ticks"]:
                    raise ProcessLookupError(controller["pid"])
            except (FileNotFoundError, ProcessLookupError) as exc:
                raise RuntimeError(
                    "Paired controller exited before its decision"
                ) from exc
        if time.monotonic() >= deadline:
            raise TimeoutError("Paired round barrier timed out")
        time.sleep(1)
