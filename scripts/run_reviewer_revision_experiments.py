from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ALL = ("s2", "s1", "highres", "nighttime", "building", "roads", "poi", "aoi", "text")
COMPARATORS = ("cnn", "fcnn", "transformer", "fcn_resnet50", "vgg_unet_transformer")
ROOT_NAME = "eight_class_review_revision_20260801"


def metric_values(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data["test"]["official8"]
    return {
        "OA": float(values["overall_accuracy"]),
        "mIoU": float(values["macro_iou_present"]),
        "F1": float(values["macro_f1_present"]),
    }


class Runner:
    def __init__(self, repo: Path, force: bool = False) -> None:
        self.repo = repo
        self.force = force
        self.output = repo / "outputs" / "full" / ROOT_NAME
        self.logs = self.output / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.records_path = self.output / "RUN_MANIFEST.json"
        self.records: list[dict] = []
        if self.records_path.exists():
            self.records = json.loads(self.records_path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.records_path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def run(self, name: str, command: list[str], expected: Path, phase: str) -> Path:
        if expected.exists() and not self.force:
            values = metric_values(expected) if expected.name == "metrics.json" else {}
            print(f"SKIP phase={phase} name={name} metrics={values}", flush=True)
            return expected
        started = time.strftime("%Y-%m-%dT%H:%M:%S")
        print(f"START phase={phase} name={name}\nCOMMAND {' '.join(command)}", flush=True)
        log = self.logs / f"{name}.log"
        error = self.logs / f"{name}.err.log"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo / "src")
        with log.open("w", encoding="utf-8") as stdout, error.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command, cwd=self.repo, env=env, stdout=stdout, stderr=stderr, check=False
            )
        record = {
            "phase": phase,
            "name": name,
            "started": started,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "returncode": completed.returncode,
            "expected": str(expected),
            "log": str(log),
            "error_log": str(error),
        }
        if expected.exists() and expected.name == "metrics.json":
            record["metrics"] = metric_values(expected)
        self.records.append(record)
        self.save()
        if completed.returncode != 0 or not expected.exists():
            print(f"ERROR phase={phase} name={name} returncode={completed.returncode}", flush=True)
            raise RuntimeError(f"{name} failed; see {error}")
        print(f"DONE phase={phase} name={name} metrics={record.get('metrics', {})}", flush=True)
        return expected

    def baseline(
        self,
        name: str,
        model: str,
        seed: int,
        modalities: tuple[str, ...] = ALL,
        manifest: str = "manifest_hier_v2.csv",
        vocab: str = "vocab_hier_v2.json",
        normalization: str = "normalization_hier_v2.json",
        phase: str = "baseline",
    ) -> Path:
        run_name = f"{ROOT_NAME}/{name}"
        out = self.repo / "outputs" / "full" / run_name
        batch = "4" if model in ("cnn", "fcnn", "transformer") else "2"
        command = [
            sys.executable, "-m", "hk_landuse.train_comparison_8class",
            "--config", "config_hierarchical.yaml",
            "--model", model,
            "--run-name", run_name,
            "--epochs", "20",
            "--patience", "5",
            "--batch-size", batch,
            "--num-workers", "0",
            "--seed", str(seed),
            "--modalities", ",".join(modalities),
            "--manifest-name", manifest,
            "--vocab-name", vocab,
            "--normalization-name", normalization,
        ]
        self.run(name, command, out / "metrics.json", phase)
        return out / "best_model.pt"

    def hierarchy(
        self,
        name: str,
        baseline: Path,
        seed: int,
        modalities: tuple[str, ...] = ALL,
        component_mode: str = "full",
        manifest: str = "manifest_hier_v2.csv",
        vocab: str = "vocab_hier_v2.json",
        normalization: str = "normalization_hier_v2.json",
        phase: str = "hierarchy",
    ) -> Path:
        train_name = f"{name}_train"
        train_run = f"{ROOT_NAME}/{train_name}"
        train_out = self.repo / "outputs" / "full" / train_run
        common = [
            "--config", "config_hierarchical.yaml",
            "--baseline-checkpoint", str(baseline),
            "--seed", str(seed),
            "--batch-size", "4",
            "--num-workers", "0",
            "--component-mode", component_mode,
            "--modalities", ",".join(modalities),
            "--manifest-name", manifest,
            "--vocab-name", vocab,
            "--normalization-name", normalization,
        ]
        train_command = [
            sys.executable, "-m", "hk_landuse.train_hierarchical_fcnn_8class",
            "--run-name", train_run,
            "--stage-epochs", "8,8,12",
            "--patience", "4",
            *common,
        ]
        self.run(train_name, train_command, train_out / "best_final.pt", phase)
        eval_run = f"{ROOT_NAME}/{name}"
        eval_out = self.repo / "outputs" / "full" / eval_run
        eval_command = [
            sys.executable, "-m", "hk_landuse.train_hierarchical_fcnn_8class",
            "--run-name", eval_run,
            "--evaluate-checkpoint", str(train_out / "best_final.pt"),
            *common,
        ]
        self.run(name, eval_command, eval_out / "metrics.json", phase)
        return eval_out / "metrics.json"


def structural(runner: Runner) -> None:
    baseline = runner.repo / "outputs/full/eight_class_20260725/fcnn_full_seed42/best_model.pt"
    for mode in ("no_hierarchy", "no_residual", "shared_gate"):
        runner.hierarchy(
            f"structural_{mode}_seed42", baseline, 42,
            component_mode=mode, phase="structural",
        )


def seeds(runner: Runner) -> None:
    for seed in (43, 44):
        checkpoints: dict[str, Path] = {}
        for model in COMPARATORS:
            checkpoints[model] = runner.baseline(
                f"{model}_full_seed{seed}", model, seed, phase="multiseed"
            )
        runner.hierarchy(
            f"ours_full_seed{seed}", checkpoints["fcnn"], seed, phase="multiseed"
        )


def osm(runner: Runner) -> None:
    variants = {
        "no_poi": tuple(x for x in ALL if x != "poi"),
        "no_aoi": tuple(x for x in ALL if x != "aoi"),
        "no_roads": tuple(x for x in ALL if x != "roads"),
        "no_text": tuple(x for x in ALL if x != "text"),
    }
    for label, modalities in variants.items():
        baseline = runner.baseline(
            f"fcnn_{label}_seed42", "fcnn", 42, modalities, phase="osm_ablation"
        )
        runner.hierarchy(
            f"ours_{label}_seed42", baseline, 42, modalities,
            component_mode="full", phase="osm_ablation",
        )


def spatial(runner: Runner) -> None:
    protocols = {
        "block256": ("manifest_block256.csv", "vocab_block256.json", "normalization_block256.json"),
        "block1024": ("manifest_block1024.csv", "vocab_block1024.json", "normalization_block1024.json"),
        "block512_buffer128": ("manifest_block512_buffer128.csv", "vocab_block512_buffer128.json", "normalization_block512_buffer128.json"),
        "loro_west": ("manifest_loro_west.csv", "vocab_loro_west.json", "normalization_loro_west.json"),
    }
    for label, (manifest, vocab, normalization) in protocols.items():
        baseline = runner.baseline(
            f"fcnn_{label}_seed42", "fcnn", 42, ALL, manifest, vocab,
            normalization, phase="spatial_validation",
        )
        runner.hierarchy(
            f"ours_{label}_seed42", baseline, 42, ALL, "full", manifest,
            vocab, normalization, phase="spatial_validation",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("structural", "seeds", "osm", "spatial", "all"), default="all"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    runner = Runner(repo, args.force)
    phases = ("structural", "seeds", "osm", "spatial") if args.phase == "all" else (args.phase,)
    for phase in phases:
        print(f"PHASE_START {phase}", flush=True)
        globals()[phase](runner)
        print(f"PHASE_DONE {phase}", flush=True)


if __name__ == "__main__":
    main()
