from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def process_exists(pid: int) -> bool:
    process_query_limited_information = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(
        process_query_limited_information, False, pid
    )
    if not handle:
        return False
    exit_code = ctypes.c_ulong()
    try:
        if not ctypes.windll.kernel32.GetExitCodeProcess(
            handle, ctypes.byref(exit_code)
        ):
            return False
        return exit_code.value == 259  # STILL_ACTIVE
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def run(command: list[str], repo: Path, environment: dict[str, str]) -> dict:
    started = time.time()
    print("START " + " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=repo,
        env=environment,
        check=False,
    )
    record = {
        "command": command,
        "returncode": completed.returncode,
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)
    if completed.returncode:
        raise SystemExit(completed.returncode)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = repo / "outputs" / "full" / "eight_class_20260725"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo / "src")
    print(f"WAIT primary_pid={args.wait_pid}", flush=True)
    while process_exists(args.wait_pid):
        time.sleep(10)
    records = []
    phases = [
        [sys.executable, "scripts/run_eight_class_experiments.py", "--phase", "primary"],
        [sys.executable, "scripts/run_moe_eight_class_experiments.py", "--phase", "repeats"],
        [sys.executable, "scripts/run_eight_class_experiments.py", "--phase", "repeats"],
        [sys.executable, "scripts/run_moe_eight_class_experiments.py", "--phase", "ablations"],
        [sys.executable, "scripts/run_eight_class_experiments.py", "--phase", "ablations"],
        [sys.executable, "scripts/finalize_eight_class_experiment.py"],
    ]
    for command in phases:
        records.append(run(command, repo, environment))
        (root / "remaining_chain_manifest.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
