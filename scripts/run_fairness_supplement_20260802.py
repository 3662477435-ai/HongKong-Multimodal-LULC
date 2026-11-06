from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


ROOT_NAME = "eight_class_fairness_supplement_20260802"
ALL = "s2,s1,highres,nighttime,building,roads,poi,aoi,text"


class Runner:
    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.root = repo / "outputs" / "full" / ROOT_NAME
        self.logs = self.root / "logs"
        self.logs.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "RUN_MANIFEST.json"
        self.records = (
            json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if self.manifest_path.exists() else []
        )

    def save(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def run(self, name: str, command: list[str], expected: Path) -> None:
        if expected.exists():
            print(f"SKIP {name}: {expected}", flush=True)
            return
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo / "src")
        log = self.logs / f"{name}.log"
        error = self.logs / f"{name}.err.log"
        started = time.time()
        print(f"START {name}\nCOMMAND {' '.join(command)}", flush=True)
        with log.open("w", encoding="utf-8") as stdout, error.open("w", encoding="utf-8") as stderr:
            completed = subprocess.run(
                command, cwd=self.repo, env=env, stdout=stdout, stderr=stderr, check=False
            )
        record = {
            "name": name,
            "command": command,
            "started_unix": started,
            "finished_unix": time.time(),
            "wall_seconds": time.time() - started,
            "returncode": completed.returncode,
            "expected": str(expected),
            "log": str(log),
            "error_log": str(error),
        }
        self.records.append(record)
        self.save()
        if completed.returncode != 0 or not expected.exists():
            raise RuntimeError(f"{name} failed; inspect {error}")
        print(f"DONE {name} wall_hours={record['wall_seconds']/3600:.3f}", flush=True)


def train_commands(repo: Path):
    python = sys.executable
    base = [python, "-m", "hk_landuse.train_comparison_8class",
            "--config", "config_hierarchical.yaml", "--modalities", ALL,
            "--num-workers", "0", "--seed", "42"]
    return (
        (
            "fcnn_equal_budget_47ep_seed42",
            base + ["--model", "fcnn", "--run-name", f"{ROOT_NAME}/fcnn_equal_budget_47ep_seed42",
                    "--epochs", "47", "--patience", "0", "--batch-size", "4"],
        ),
        (
            "fcn_resnet50_direct48_seed42",
            base + ["--model", "fcn_resnet50_direct48",
                    "--run-name", f"{ROOT_NAME}/fcn_resnet50_direct48_seed42",
                    "--epochs", "20", "--patience", "5", "--batch-size", "2"],
        ),
        (
            "vgg_unet_transformer_direct48_seed42",
            base + ["--model", "vgg_unet_transformer_direct48",
                    "--run-name", f"{ROOT_NAME}/vgg_unet_transformer_direct48_seed42",
                    "--epochs", "20", "--patience", "5", "--batch-size", "2"],
        ),
    )


def metric_row(name: str, path: Path, design: str) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = payload["test"]["official8"]
    return {
        "model": name,
        "design": design,
        "OA": float(values["overall_accuracy"]),
        "mIoU": float(values["macro_iou_present"]),
        "macro_F1": float(values["macro_f1_present"]),
        "best_epoch": payload.get("best_epoch"),
        "actual_epochs_run": payload.get("actual_epochs_run"),
        "training_wall_seconds": payload.get("training_wall_seconds"),
        "parameter_count": payload.get("parameter_count"),
        "metrics_path": str(path),
    }


def summarize(repo: Path, root: Path) -> None:
    old = repo / "outputs" / "full" / "eight_class_20260725"
    rows = [
        metric_row("本文模型", old / "ours_fcnn_hierarchy_semantic_final_seed42" / "metrics.json",
                   "FCNN+层级+语义残差+类别门控；20+8+12+7 epoch"),
        metric_row("FCNN原始预算", old / "fcnn_full_seed42" / "metrics.json", "最多20 epoch"),
        metric_row("FCNN同等预算", root / "fcnn_equal_budget_47ep_seed42" / "metrics.json", "固定47 epoch"),
        metric_row("FCN-ResNet50 48→3", old / "fcn_resnet50_full_seed42" / "metrics.json", "原3通道适配瓶颈"),
        metric_row("FCN-ResNet50直接48通道", root / "fcn_resnet50_direct48_seed42" / "metrics.json", "第一层直接接收48通道"),
        metric_row("VGG-UNet-Transformer 48→3", old / "vgg_unet_transformer_full_seed42" / "metrics.json", "原3通道适配瓶颈"),
        metric_row("VGG-UNet-Transformer直接48通道", root / "vgg_unet_transformer_direct48_seed42" / "metrics.json", "第一层直接接收48通道"),
    ]
    frame = pd.DataFrame(rows)
    ours = frame.loc[frame.model == "本文模型"].iloc[0]
    frame["delta_OA_vs_ours"] = frame.OA - ours.OA
    frame["delta_mIoU_vs_ours"] = frame.mIoU - ours.mIoU
    frame["delta_F1_vs_ours"] = frame.macro_F1 - ours.macro_F1
    frame.to_csv(root / "fairness_accuracy_comparison.csv", index=False, encoding="utf-8-sig")
    (root / "fairness_accuracy_comparison.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    runner = Runner(repo)
    for name, command in train_commands(repo):
        runner.run(name, command, runner.root / name / "metrics.json")

    expected_dir = runner.root / "expected_gradients_full128"
    runner.run(
        "expected_gradients_full128",
        [sys.executable, "-m", "hk_landuse.model_expected_gradients_fullpatch_8class",
         "--config", "config_hierarchical.yaml",
         "--checkpoint", str(repo / "outputs/full/eight_class_20260725/ours_fcnn_hierarchy_semantic_seed42/best_final.pt"),
         "--output-dir", str(expected_dir), "--samples-per-class", "10",
         "--background-size", "8", "--patch-size", "128", "--aggregation-window", "49",
         "--nsamples", "64", "--explainer-batch-size", "1", "--seed", "42"],
        expected_dir / "expected_gradients_metadata.json",
    )

    specs = [
        {"name": "本文模型", "kind": "ours",
         "checkpoint": str(repo / "outputs/full/eight_class_20260725/ours_fcnn_hierarchy_semantic_seed42/best_final.pt")},
        {"name": "FCNN原始预算", "kind": "baseline", "model": "fcnn",
         "checkpoint": str(repo / "outputs/full/eight_class_20260725/fcnn_full_seed42/best_model.pt")},
        {"name": "FCNN同等预算", "kind": "baseline", "model": "fcnn",
         "checkpoint": str(runner.root / "fcnn_equal_budget_47ep_seed42/best_model.pt")},
        {"name": "CNN", "kind": "baseline", "model": "cnn",
         "checkpoint": str(repo / "outputs/full/eight_class_20260725/cnn_full_seed42/best_model.pt")},
        {"name": "Transformer", "kind": "baseline", "model": "transformer",
         "checkpoint": str(repo / "outputs/full/eight_class_20260725/transformer_full_seed42/best_model.pt")},
        {"name": "FCN-ResNet50 48→3", "kind": "baseline", "model": "fcn_resnet50",
         "checkpoint": str(repo / "outputs/full/eight_class_20260725/fcn_resnet50_full_seed42/best_model.pt")},
        {"name": "FCN-ResNet50直接48通道", "kind": "baseline", "model": "fcn_resnet50_direct48",
         "checkpoint": str(runner.root / "fcn_resnet50_direct48_seed42/best_model.pt")},
        {"name": "VGG-UNet-Transformer 48→3", "kind": "baseline", "model": "vgg_unet_transformer",
         "checkpoint": str(repo / "outputs/full/eight_class_20260725/vgg_unet_transformer_full_seed42/best_model.pt")},
        {"name": "VGG-UNet-Transformer直接48通道", "kind": "baseline", "model": "vgg_unet_transformer_direct48",
         "checkpoint": str(runner.root / "vgg_unet_transformer_direct48_seed42/best_model.pt")},
    ]
    specs_path = runner.root / "profiling_specs.json"
    specs_path.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")
    profile_dir = runner.root / "profiling_all"
    runner.run(
        "profile_models",
        [sys.executable, "-m", "hk_landuse.profile_fairness_models_8class",
         "--config", "config_hierarchical.yaml", "--specs", str(specs_path),
         "--output-dir", str(profile_dir), "--warmup", "30", "--repeats", "100"],
        profile_dir / "model_complexity_and_inference_speed.csv",
    )
    summarize(repo, runner.root)
    required = [
        runner.root / "fcnn_equal_budget_47ep_seed42/metrics.json",
        runner.root / "fcn_resnet50_direct48_seed42/metrics.json",
        runner.root / "vgg_unet_transformer_direct48_seed42/metrics.json",
        expected_dir / "expected_gradients_raster_modalities.csv",
        expected_dir / "expected_gradients_text_separate.csv",
        profile_dir / "model_complexity_and_inference_speed.csv",
        runner.root / "fairness_accuracy_comparison.csv",
        runner.root / "fairness_supplement_summary.png",
    ]
    audit = {"complete": all(path.exists() for path in required),
             "files": {str(path): path.exists() for path in required}}
    (runner.root / "FINAL_FAIRNESS_SUPPLEMENT_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not audit["complete"]:
        raise RuntimeError("Final supplement audit failed")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
