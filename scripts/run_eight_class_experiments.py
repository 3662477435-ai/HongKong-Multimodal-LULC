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
COMPARATORS = ("cnn", "fcnn", "transformer", "fcn_resnet50", "vgg_unet_transformer")


def metrics_path(repo: Path, model: str, variant: str, seed: int) -> Path:
    run = repo / "outputs" / "full" / "eight_class_20260725" / f"{model}_{variant}_seed{seed}"
    return run / ("stage_metrics.json" if model == "ours" else "metrics.json")


def command(repo: Path, model: str, variant: str, seed: int) -> list[str]:
    run_name = f"eight_class_20260725/{model}_{variant}_seed{seed}"
    modalities = ",".join(VARIANTS[variant])
    if model == "ours":
        return [
            sys.executable, "-m", "hk_landuse.hierarchical_train_8class",
            "--config", "config_hierarchical.yaml",
            "--run-name", run_name,
            "--stage-epochs", "1,1,2,10",
            "--patience", "3",
            "--seed", str(seed),
            "--num-workers", "0",
            "--modalities", modalities,
        ]
    batch_size = "4" if model in ("cnn", "fcnn", "transformer") else "2"
    epochs = "20" if variant == "full" else "12"
    patience = "5" if variant == "full" else "3"
    return [
        sys.executable, "-m", "hk_landuse.train_comparison_8class",
        "--config", "config_hierarchical.yaml",
        "--model", model,
        "--run-name", run_name,
        "--epochs", epochs,
        "--patience", patience,
        "--batch-size", batch_size,
        "--num-workers", "0",
        "--seed", str(seed),
        "--modalities", modalities,
    ]


def run_one(repo: Path, log_dir: Path, model: str, variant: str, seed: int, force: bool) -> dict:
    destination = metrics_path(repo, model, variant, seed)
    if destination.exists() and not force:
        return {
            "model": model, "variant": variant, "seed": seed,
            "status": "skipped_existing", "metrics": str(destination),
        }
    cmd = command(repo, model, variant, seed)
    log = log_dir / f"{model}_{variant}_seed{seed}.log"
    error = log_dir / f"{model}_{variant}_seed{seed}.err.log"
    print(
        f"START model={model} variant={variant} seed={seed}\n"
        f"COMMAND {' '.join(cmd)}",
        flush=True,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo / "src")
    with log.open("w", encoding="utf-8") as stdout, error.open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            cmd, cwd=repo, env=env, stdout=stdout, stderr=stderr, check=False
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Training failed for {model}/{variant}/seed{seed}; see {error}"
        )
    print(f"DONE model={model} variant={variant} seed={seed}", flush=True)
    return {
        "model": model, "variant": variant, "seed": seed,
        "status": "completed", "metrics": str(destination),
        "log": str(log), "stderr": str(error),
    }


def schedule(phase: str) -> list[tuple[str, str, int]]:
    primary = [
        (model, "full", 42) for model in COMPARATORS
    ]
    repeat_full = [
        (model, "full", seed)
        for seed in (43, 44)
        for model in COMPARATORS
    ]
    ablations = [
        (model, variant, 42)
        for variant in VARIANTS
        if variant != "full"
        for model in COMPARATORS
    ]
    return {
        "primary": primary,
        "repeats": repeat_full,
        "ablations": ablations,
        "all": primary + repeat_full + ablations,
    }[phase]


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
    records = []
    for model, variant, seed in schedule(args.phase):
        records.append(run_one(repo, log_dir, model, variant, seed, args.force))
        (root / f"run_manifest_{args.phase}.json").write_text(
            json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
