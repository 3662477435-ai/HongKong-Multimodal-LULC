from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import rasterio


MODELS = (
    "ours",
    "cnn",
    "fcnn",
    "transformer",
    "fcn_resnet50",
    "vgg_unet_transformer",
)
BASELINES = MODELS[1:]
VARIANTS = (
    "full",
    "s2",
    "no_osm_roads",
    "no_building",
    "no_s1",
    "no_nighttime",
    "no_highres",
)


def metric_path(root: Path, model: str, variant: str) -> Path:
    if model == "ours":
        if variant == "full":
            return root / "ours_fcnn_hierarchy_semantic_final_seed42" / "metrics.json"
        return root / f"ours_hfcnn_{variant}_seed42" / "metrics.json"
    return root / f"{model}_{variant}_seed42" / "metrics.json"


def metric_payload(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["test"]["official8"]


def row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    root = repo / "outputs" / "full" / "eight_class_20260725"
    results = root / "final_results"
    maps = root / "prediction_maps"
    figures = root / "figures"
    checks: dict[str, object] = {}

    metric_checks = []
    for model in MODELS:
        for variant in VARIANTS:
            path = metric_path(root, model, variant)
            if not path.exists():
                raise FileNotFoundError(path)
            metric = metric_payload(path)
            matrix = np.asarray(metric["confusion_matrix"], dtype=np.int64)
            if matrix.shape != (8, 8):
                raise AssertionError(f"{path}: confusion shape={matrix.shape}")
            if len(metric["per_class"]) != 8:
                raise AssertionError(f"{path}: per_class count is not 8")
            values = np.asarray([
                metric["overall_accuracy"],
                metric["macro_iou_present"],
                metric["macro_f1_present"],
            ])
            if not np.isfinite(values).all():
                raise AssertionError(f"{path}: non-finite primary metric")
            if not ((values >= 0) & (values <= 1)).all():
                raise AssertionError(f"{path}: primary metric outside [0,1]")
            if variant == "full" and len(metric["spatial_group_confusions"]) != 14:
                raise AssertionError(f"{path}: expected 14 spatial groups")
            metric_checks.append({
                "model": model,
                "variant": variant,
                "OA": float(values[0]),
                "mIoU": float(values[1]),
                "macro_F1": float(values[2]),
            })
    checks["metrics"] = metric_checks

    label_path = repo / "artifacts" / "full" / "prepared" / "labels_leaf14.tif"
    map_checks = []
    with rasterio.open(label_path) as labels:
        for model in MODELS:
            path = maps / f"{model}_official8_classification.tif"
            if not path.exists():
                raise FileNotFoundError(path)
            with rasterio.open(path) as predicted:
                if (
                    predicted.width != labels.width
                    or predicted.height != labels.height
                    or predicted.transform != labels.transform
                    or predicted.crs != labels.crs
                ):
                    raise AssertionError(f"{path}: grid mismatch")
                observed: set[int] = set()
                invalid_nonzero = 0
                valid_zero = 0
                for _, window in predicted.block_windows(1):
                    pred = predicted.read(1, window=window)
                    truth = labels.read(1, window=window)
                    valid = (truth != 255) & (truth != 1) & (truth != 12)
                    invalid_nonzero += int(np.count_nonzero((~valid) & (pred != 0)))
                    valid_zero += int(np.count_nonzero(valid & (pred == 0)))
                    observed.update(int(value) for value in np.unique(pred))
                if invalid_nonzero:
                    raise AssertionError(
                        f"{path}: {invalid_nonzero} excluded/BLU0 pixels predicted"
                    )
                if valid_zero:
                    raise AssertionError(
                        f"{path}: {valid_zero} active pixels unexpectedly zero"
                    )
                if not observed <= set(range(9)):
                    raise AssertionError(f"{path}: values outside 0..8: {observed}")
                map_checks.append({
                    "model": model,
                    "classes_observed": sorted(observed),
                    "invalid_nonzero": invalid_nonzero,
                    "valid_zero": valid_zero,
                })
    checks["maps"] = map_checks

    expected_rows = {
        "table_overall_metrics_with_ci.csv": 6,
        "table_per_class_metrics_with_ci.csv": 48,
        "table_confusion_long.csv": 384,
        "table_paired_significance.csv": 15,
        "table_ablation_all_models.csv": 42,
        "table_dataset_support_8class.csv": 24,
    }
    table_checks = {}
    for name, expected in expected_rows.items():
        path = results / name
        actual = row_count(path)
        if actual != expected:
            raise AssertionError(f"{path}: rows={actual}, expected={expected}")
        table_checks[name] = actual
    checks["tables"] = table_checks

    stems = (
        "figure_8class_six_model_maps_2x3",
        "figure_8class_confusion_matrices_2x3",
        "table_8class_extended_modality_ablation",
        "table_8class_overall_results_with_ci",
        "table_8class_per_class_iou_all_models",
        "table_8class_ours_per_class_metrics_with_ci",
        "table_8class_paired_significance",
    )
    figure_checks = {}
    for stem in stems:
        paths = [figures / f"{stem}{suffix}" for suffix in (".png", ".pdf", ".svg", ".tif")]
        missing = [str(path) for path in paths if not path.exists() or path.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(", ".join(missing))
        figure_checks[stem] = {
            path.suffix: path.stat().st_size for path in paths
        }
    checks["figures"] = figure_checks

    report = root / "FINAL_OUTPUT_AUDIT.json"
    report.write_text(
        json.dumps(
            {"status": "passed", "checks": checks},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
