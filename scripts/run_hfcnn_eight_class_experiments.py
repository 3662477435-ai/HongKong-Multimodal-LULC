from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from run_eight_class_experiments import VARIANTS


def run(command: list[str], repo: Path, log: Path, error: Path) -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(repo / "src")
    print("RUN", " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as stdout, error.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=repo,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"Command failed; see {error}")


def run_one(
    repo: Path,
    root: Path,
    variant: str,
    seed: int,
    force: bool,
) -> dict:
    final_dir = root / f"ours_hfcnn_{variant}_seed{seed}"
    final_metric = final_dir / "metrics.json"
    if final_metric.exists() and not force:
        return {
            "variant": variant,
            "seed": seed,
            "status": "skipped_existing",
            "metrics": str(final_metric),
        }
    baseline = root / f"fcnn_{variant}_seed{seed}" / "best_model.pt"
    if not baseline.exists():
        raise FileNotFoundError(
            f"FCNN initialization is required before the hierarchy: {baseline}"
        )
    train_name = f"eight_class_20260725/ours_hfcnn_{variant}_train_seed{seed}"
    train_dir = root / f"ours_hfcnn_{variant}_train_seed{seed}"
    checkpoint = train_dir / "best_final.pt"
    modalities = ",".join(VARIANTS[variant])
    full = variant == "full"
    stage_epochs = "8,8,12" if full else "5,5,8"
    patience = "4" if full else "3"
    batch_size = "4"
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    train_command = [
        sys.executable,
        "-m",
        "hk_landuse.train_hierarchical_fcnn_8class",
        "--config",
        "config_hierarchical.yaml",
        "--baseline-checkpoint",
        str(baseline),
        "--run-name",
        train_name,
        "--stage-epochs",
        stage_epochs,
        "--patience",
        patience,
        "--seed",
        str(seed),
        "--batch-size",
        batch_size,
        "--num-workers",
        "0",
        "--modalities",
        modalities,
        "--skip-test",
    ]
    if force or not checkpoint.exists():
        run(
            train_command,
            repo,
            logs / f"ours_hfcnn_{variant}_train_seed{seed}.log",
            logs / f"ours_hfcnn_{variant}_train_seed{seed}.err.log",
        )
    evaluate_command = [
        sys.executable,
        "-m",
        "hk_landuse.train_hierarchical_fcnn_8class",
        "--config",
        "config_hierarchical.yaml",
        "--baseline-checkpoint",
        str(baseline),
        "--run-name",
        f"eight_class_20260725/ours_hfcnn_{variant}_seed{seed}",
        "--seed",
        str(seed),
        "--batch-size",
        batch_size,
        "--num-workers",
        "0",
        "--modalities",
        modalities,
        "--evaluate-checkpoint",
        str(checkpoint),
    ]
    run(
        evaluate_command,
        repo,
        logs / f"ours_hfcnn_{variant}_seed{seed}.log",
        logs / f"ours_hfcnn_{variant}_seed{seed}.err.log",
    )
    return {
        "variant": variant,
        "seed": seed,
        "status": "completed",
        "checkpoint": str(checkpoint),
        "metrics": str(final_metric),
    }


def schedule(phase: str) -> list[tuple[str, int]]:
    repeats = [("full", seed) for seed in (43, 44)]
    ablations = [(variant, 42) for variant in VARIANTS if variant != "full"]
    return {
        "repeats": repeats,
        "ablations": ablations,
        "all": repeats + ablations,
    }[phase]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("repeats", "ablations", "all"), default="all"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = repo / "outputs" / "full" / "eight_class_20260725"
    records = []
    for variant, seed in schedule(args.phase):
        records.append(run_one(repo, root, variant, seed, args.force))
        (root / f"run_manifest_ours_hfcnn_{args.phase}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
