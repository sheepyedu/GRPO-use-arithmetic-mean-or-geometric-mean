#!/usr/bin/env python3
"""Summarize timestamp intervals without inferring overlapping device activity."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path


def profile(path, time_key="time", phase_key="phase", interval_owner="previous"):
    events = []
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            timestamp = event[time_key]
            phase = event[phase_key]
            if (isinstance(timestamp, bool)
                    or not isinstance(timestamp, (int, float))
                    or not math.isfinite(timestamp)):
                raise ValueError("timestamp must be a finite number")
            if not isinstance(phase, str) or not phase:
                raise ValueError("phase must be a nonempty string")
            events.append((timestamp, phase))
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"{path}, line {number}: {error}") from error
    seconds = defaultdict(float)
    intervals = defaultdict(int)
    for previous, current in zip(events, events[1:]):
        elapsed = current[0] - previous[0]
        if not math.isfinite(elapsed):
            raise ValueError(f"{path}: timestamp interval exceeds finite range")
        if elapsed < 0:
            raise ValueError(f"{path}: timestamps move backward")
        phase = previous[1] if interval_owner == "previous" else current[1]
        seconds[phase] += elapsed
        intervals[phase] += 1
    total = sum(seconds.values())
    return {
        "path": str(path),
        "event_count": len(events),
        "first_timestamp": events[0][0] if events else None,
        "last_timestamp": events[-1][0] if events else None,
        "covered_interval_seconds": total,
        "interval_owner": interval_owner,
        "phases": {
            phase: {
                "seconds": value,
                "fraction": value / total if total else None,
                "interval_count": intervals[phase],
            }
            for phase, value in sorted(seconds.items(), key=lambda item: -item[1])
        },
        "tail_after_last_event": "unobserved",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path)
    parser.add_argument("--time-key", default="time")
    parser.add_argument("--phase-key", default="phase")
    parser.add_argument("--interval-owner", choices=("previous", "next"), default="previous")
    args = parser.parse_args()
    try:
        results = [profile(path, args.time_key, args.phase_key, args.interval_owner)
                   for path in args.files]
    except (OSError, ValueError) as error:
        parser.exit(2, f"{error}\n")
    print(json.dumps({
        "files": results,
        "interpretation": "Intervals per serial event stream. Concurrent streams may overlap. "
                          "Covered process seconds require device ownership evidence before conversion to GPU-hours.",
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
