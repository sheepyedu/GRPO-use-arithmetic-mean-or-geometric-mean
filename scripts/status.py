"""Bounded read-only status projection for an existing paired stage."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def snapshot(root):
    pair_files = sorted(root.glob("*-pair.json"))
    if not pair_files and (root / "pair.json").exists():
        pair_files = [root / "pair.json"]
    pairs = [json.loads(path.read_text()) for path in pair_files]
    runs = []
    for pair in pairs:
        for child in pair["children"]:
            path = Path(child["output"])
            status_file = path / "status.json"
            status = (
                json.loads(status_file.read_text()) if status_file.exists() else None
            )
            if status:
                status.pop("prompt_ids", None)
            failure_file = path / "FAILURE.json"
            failure = (
                json.loads(failure_file.read_text()) if failure_file.exists() else None
            )
            pid = child["pid"]
            proc = Path(f"/proc/{pid}")
            try:
                raw = (proc / "cmdline").read_bytes()
                stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
                identity = {
                    "pid": pid,
                    "uid": proc.stat().st_uid,
                    "start_ticks": int(stat[19]),
                    "cmdline_sha256": hashlib.sha256(raw).hexdigest(),
                    "matches_output": os.fsencode(str(path)) in raw,
                    "state": stat[0],
                }
            except FileNotFoundError:
                identity = {"pid": pid, "gone": True}
            runs.append(
                {
                    "directory": str(path),
                    "status": status,
                    "identity": identity,
                    "failure": failure,
                    "complete": (path / "COMPLETE.json").exists(),
                }
            )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        text=True,
        capture_output=True,
        timeout=10,
    )
    return {
        "observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "root": str(root),
        "pair": pairs[0] if len(pairs) == 1 else None,
        "pairs": pairs,
        "runs": runs,
        "gpu_csv": gpu.stdout.strip(),
        "disk_free_bytes": shutil.disk_usage(root).free,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    print(json.dumps(snapshot(Path(parser.parse_args().directory).resolve())))
