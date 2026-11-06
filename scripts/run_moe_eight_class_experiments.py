from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


ALL = ("s2", "s1", "highres", "nighttime", "building", "roads", "poi", "aoi", "text")
VARIANTS = {
    "full": ALL,
    "s2": ("s2",),
    "no_osm_roads": ("s2", "s1", "highres", "nighttime", "building"),
    "no_building": ("s2", "s1", "highres", "nighttime", "roads", "poi", "aoi", "text"),
    "no_s1": ("s2", "highres", "nighttime", "building", "roads", "poi", "aoi", "text"),
    "no_nighttime": ("s2", "s1", "highres", "building", "roads", "poi", "aoi", "text"),
    "no_highres": ("s2", "s1", "nighttime", "building", "roads", "poi", "aoi", "text"),
}


def execute(
    command: list[str],
    repo: Path,
    log: Path,
    error: Path,
) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")
    print("COMMAND " + " ".join(command), flush=True)
    with log.open("w", encoding="utf-8") as stdout, error.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            command,
            cwd=repo,
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode:
        raise RuntimeError(f"Command failed; see {error}")


def ensure_fcnn(
    repo: Path,
    root: Path,
    log_dir: Path,
    variant: str,
    seed: int,
    force: bool,
) -> Path:
    run_name = f"fcnn_{variant}_seed{seed}"
    run = root / run_name
    checkpoint = run / "best_model.pt"
    metrics = run / "metrics.json"
    if force or not (checkpoint.exists() and metrics.exists()):
        command = [
            sys.executable,
            "-m",
            "hk_landuse.train_comparison_8class",
            "--config",
            "config_hierarchical.yaml",
            "--model",
            "fcnn",
            "--run-name",
            f"eight_class_20260725/{run_name}",
            "--epochs",
            "20" if variant == "full" else "12",
            "--patience",
            "5" if variant == "full" else "3",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--seed",
            str(seed),
            "--modalities",
            ",".join(VARIANTS[variant]),
        ]
        execute(
            command,
            repo,
            log_dir / f"{run_name}.log",
            log_dir / f"{run_name}.err.log",
        )
    return checkpoint


def ensure_hierarchy(
    repo: Path,
    root: Path,
    log_dir: Path,
    variant: str,
    seed: int,
    force: bool,
) -> Path:
    # Reuse the already completed, hierarchy-only seed-42 full run.
    if variant == "full" and seed == 42 and not force:
        existing = root / "ours_full_seed42" / "best_joint.pt"
        if existing.exists():
            return existing
    run_name = f"ours_hierarchy_{variant}_seed{seed}"
    run = root / run_name
    checkpoint = run / "best_joint.pt"
    if force or not checkpoint.exists():
        command = [
            sys.executable,
            "-m",
            "hk_landuse.hierarchical_train_8class",
            "--config",
            "config_hierarchical.yaml",
            "--run-name",
            f"eight_class_20260725/{run_name}",
            "--stage-epochs",
            "1,1,2,10",
            "--patience",
            "3",
            "--seed",
            str(seed),
            "--num-workers",
            "0",
            "--modalities",
            ",".join(VARIANTS[variant]),
            "--hierarchy-only",
            "--skip-test",
        ]
        execute(
            command,
            repo,
            log_dir / f"{run_name}.log",
            log_dir / f"{run_name}.err.log",
        )
    return checkpoint


def ensure_hybrid_head(
    repo: Path,
    root: Path,
    log_dir: Path,
    variant: str,
    seed: int,
    hierarchy_checkpoint: Path,
    force: bool,
) -> Path:
    if variant == "full" and seed == 42 and not force:
        existing = root / "ours_hybrid_warmup_seed42" / "best_head.pt"
        if existing.exists():
            return existing
    run_name = f"ours_hybrid_{variant}_seed{seed}"
    run = root / run_name
    checkpoint = run / "best_head.pt"
    if force or not checkpoint.exists():
        command = [
            sys.executable,
            "-m",
            "hk_landuse.hierarchical_train_8class",
            "--config",
            "config_hierarchical.yaml",
            "--run-name",
            f"eight_class_20260725/{run_name}",
            "--stage-epochs",
            "0,0,0,1",
            "--patience",
            "5",
            "--seed",
            str(seed),
            "--num-workers",
            "0",
            "--modalities",
            ",".join(VARIANTS[variant]),
            "--resume-checkpoint",
            str(hierarchy_checkpoint),
            "--head-only",
            "--head-warmup-epochs",
            "5",
            "--head-learning-rate",
            "0.0003",
            "--skip-test",
            "--leaf-aux-weight",
            "0.15",
            "--consistency-weight",
            "0.01",
        ]
        execute(
            command,
            repo,
            log_dir / f"{run_name}.log",
            log_dir / f"{run_name}.err.log",
        )
    return checkpoint


def calibrate(
    repo: Path,
    root: Path,
    log_dir: Path,
    variant: str,
    seed: int,
    ours_checkpoint: Path,
    expert_checkpoint: Path,
    force: bool,
) -> Path:
    run_name = f"ours_moe_{variant}_seed{seed}"
    destination = root / run_name / "metrics.json"
    if force or not destination.exists():
        command = [
            sys.executable,
            "-m",
            "hk_landuse.calibrate_hybrid_ensemble_8class",
            "--config",
            "config_hierarchical.yaml",
            "--ours-checkpoint",
            str(ours_checkpoint),
            "--expert-checkpoint",
            str(expert_checkpoint),
            "--output",
            str(destination),
            "--seed",
            str(seed),
            "--num-workers",
            "0",
        ]
        execute(
            command,
            repo,
            log_dir / f"{run_name}.log",
            log_dir / f"{run_name}.err.log",
        )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("primary", "repeats", "ablations", "all"), default="all"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    root = repo / "outputs" / "full" / "eight_class_20260725"
    log_dir = root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    schedule = {
        "primary": [("full", 42)],
        "repeats": [("full", 43), ("full", 44)],
        "ablations": [
            (variant, 42) for variant in VARIANTS if variant != "full"
        ],
    }
    jobs = (
        schedule["primary"] + schedule["repeats"] + schedule["ablations"]
        if args.phase == "all"
        else schedule[args.phase]
    )
    records = []
    for variant, seed in jobs:
        hierarchy = ensure_hierarchy(
            repo, root, log_dir, variant, seed, args.force
        )
        head = ensure_hybrid_head(
            repo, root, log_dir, variant, seed, hierarchy, args.force
        )
        expert = ensure_fcnn(
            repo, root, log_dir, variant, seed, args.force
        )
        metrics = calibrate(
            repo, root, log_dir, variant, seed, head, expert, args.force
        )
        records.append(
            {
                "variant": variant,
                "seed": seed,
                "hierarchy_checkpoint": str(hierarchy),
                "hybrid_checkpoint": str(head),
                "expert_checkpoint": str(expert),
                "metrics": str(metrics),
            }
        )
        (root / f"run_manifest_moe_{args.phase}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
