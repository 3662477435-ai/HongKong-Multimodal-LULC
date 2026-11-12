"""Controlled building-function lineage ablations for the final eight-class model.

Each variant retains the same 14 building slots and therefore the identical
95->96->48 multimodal fusion geometry.  The only intervention is zeroing the
functional (first nine) and/or morphology-height (last five) building bands.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[1]
OUT_PREFIX = "eight_class_lineage_strong_baselines_20260813"
MODALITIES = "s2,s1,highres,nighttime,building,roads,poi,aoi,text"
VARIANTS = {
    "no_building_function": "morphology_only",
    "no_building_morphology_height": "function_only",
    "no_building_all": "none",
}


def run(command: list[str], log) -> None:
    log.write("COMMAND " + " ".join(command) + "\n")
    log.flush()
    # Large tiled GeoTIFF reads can otherwise let GDAL claim most of the
    # Windows file-backed cache during long sequential retraining runs.
    # A bounded cache is sufficient for 128x128 patches and prevents
    # transient TIFFFillTile allocation failures on the system drive.
    environment = {
        **os.environ,
        "PYTHONPATH": str(REPO / "src"),
        "GDAL_CACHEMAX": "128",
    }
    process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True)
    if process.wait() != 0:
        raise RuntimeError(f"failed with code {process.returncode}: {' '.join(command)}")


def metric(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["test"]["official8"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    mode = VARIANTS[args.variant]
    root = REPO / "outputs/full" / OUT_PREFIX
    logs = root / "logs"; logs.mkdir(parents=True, exist_ok=True)
    base_name = f"{args.variant}_fcnn_seed42"
    train_name = f"{args.variant}_hierarchy_train_seed42"
    eval_name = f"{args.variant}_hierarchy_seed42"
    base_dir = root / base_name
    train_dir = root / train_name
    eval_dir = root / eval_name
    log_path = logs / f"{args.variant}.log"
    with log_path.open("a", encoding="utf-8") as log:
        if not (base_dir / "metrics.json").exists():
            run([args.python, "-u", "-m", "hk_landuse.train_comparison_8class", "--config", "config_release.yaml", "--model", "fcnn", "--run-name", f"{OUT_PREFIX}/{base_name}", "--epochs", "20", "--patience", "5", "--batch-size", "4", "--num-workers", "0", "--seed", "42", "--modalities", MODALITIES, "--building-feature-mode", mode, "--manifest-name", "manifest_hier_v2.csv", "--vocab-name", "vocab_hier_v2.json", "--normalization-name", "normalization_hier_v2.json"], log)
        if not (train_dir / "best_final.pt").exists():
            run([args.python, "-u", "-m", "hk_landuse.train_hierarchical_fcnn_8class", "--config", "config_release.yaml", "--baseline-checkpoint", str(base_dir / "best_model.pt"), "--run-name", f"{OUT_PREFIX}/{train_name}", "--stage-epochs", "8,8,12", "--patience", "4", "--batch-size", "4", "--num-workers", "0", "--seed", "42", "--component-mode", "full", "--modalities", MODALITIES, "--building-feature-mode", mode, "--manifest-name", "manifest_hier_v2.csv", "--vocab-name", "vocab_hier_v2.json", "--normalization-name", "normalization_hier_v2.json"], log)
        if not (eval_dir / "metrics.json").exists():
            run([args.python, "-u", "-m", "hk_landuse.train_hierarchical_fcnn_8class", "--config", "config_release.yaml", "--baseline-checkpoint", str(base_dir / "best_model.pt"), "--evaluate-checkpoint", str(train_dir / "best_final.pt"), "--run-name", f"{OUT_PREFIX}/{eval_name}", "--batch-size", "4", "--num-workers", "0", "--seed", "42", "--component-mode", "full", "--modalities", MODALITIES, "--building-feature-mode", mode, "--manifest-name", "manifest_hier_v2.csv", "--vocab-name", "vocab_hier_v2.json", "--normalization-name", "normalization_hier_v2.json"], log)
        result = metric(eval_dir / "metrics.json")
        if result:
            log.write(json.dumps({"variant": args.variant, "OA": result["overall_accuracy"], "mIoU": result["macro_iou_present"], "F1": result["macro_f1_present"]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
