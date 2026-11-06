from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run(repo: Path, command: list[str], records: list[dict]) -> None:
    started = datetime.now().isoformat(timespec="seconds")
    print("PHASE_START", started, " ".join(command), flush=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo / "src")
    completed = subprocess.run(
        command, cwd=repo, env=environment, check=False
    )
    record = {
        "command": command,
        "started": started,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "returncode": completed.returncode,
    }
    records.append(record)
    state = (
        repo
        / "outputs"
        / "full"
        / "eight_class_20260725"
        / "complete_chain_state.json"
    )
    state.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if completed.returncode:
        raise RuntimeError(f"Phase failed: {' '.join(command)}")
    print("PHASE_DONE", record["finished"], flush=True)


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    records: list[dict] = []
    phases = [
        [
            sys.executable,
            "scripts/run_eight_class_experiments.py",
            "--phase",
            "ablations",
        ],
        [
            sys.executable,
            "scripts/run_hfcnn_eight_class_experiments.py",
            "--phase",
            "ablations",
        ],
        [sys.executable, "scripts/finalize_eight_class_experiment.py"],
        [sys.executable, "scripts/audit_eight_class_outputs.py"],
    ]
    for command in phases:
        run(repo, command, records)


if __name__ == "__main__":
    main()
