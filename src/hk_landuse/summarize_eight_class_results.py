from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# Keep the statistics-only summarizer independent of PyTorch. Importing task8
# pulls in the CUDA runtime even though this module only needs fixed metadata;
# on memory-constrained Windows runs that can fail while initializing c10.dll.
ACTIVE_OFFICIAL_IDS = (0, 2, 3, 4, 5, 6, 7, 9)
CLASS_NAMES_8 = (
    "\u4f4f\u5b85",
    "\u5de5\u4e1a\u4ed3\u50a8",
    "\u516c\u5171\u673a\u6784\u4e0e\u4f11\u95f2",
    "\u4ea4\u901a\u8bbe\u65bd",
    "\u5176\u4ed6\u5efa\u8bbe\u7528\u5730",
    "\u519c\u4e1a",
    "\u6797\u5730/\u704c\u4e1b/\u8349\u5730/\u6e7f\u5730",
    "\u6c34\u4f53",
)


MODEL_ORDER = (
    "ours", "cnn", "fcnn", "transformer", "fcn_resnet50", "vgg_unet_transformer"
)
MODEL_LABELS = {
    "ours": "本文模型",
    "cnn": "CNN",
    "fcnn": "FCNN",
    "transformer": "Transformer",
    "fcn_resnet50": "FCN-ResNet50",
    "vgg_unet_transformer": "VGG-UNet-Transformer",
}
VARIANT_ORDER = (
    "full", "s2", "no_osm_roads", "no_building",
    "no_s1", "no_nighttime", "no_highres",
)
VARIANT_LABELS = {
    "full": "全部模态",
    "s2": "仅S2",
    "no_osm_roads": "去除POI/AOI/道路与OSM文本",
    "no_building": "去除建筑功能与高度",
    "no_s1": "去除S1",
    "no_nighttime": "去除夜间灯光",
    "no_highres": "去除高分辨率影像",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_metric(root: Path, model: str, variant: str, seed: int) -> dict:
    if model == "ours":
        if variant == "full" and seed == 42:
            run = root / "ours_fcnn_hierarchy_semantic_final_seed42"
        else:
            run = root / f"ours_hfcnn_{variant}_seed{seed}"
        data = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        return data["test"]["official8"]
    run = root / f"{model}_{variant}_seed{seed}"
    data = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    return data["test"]["official8"]


def metric_file(root: Path, model: str, variant: str, seed: int) -> Path:
    if model == "ours":
        if variant == "full" and seed == 42:
            return (
                root
                / "ours_fcnn_hierarchy_semantic_final_seed42"
                / "metrics.json"
            )
        return root / f"ours_hfcnn_{variant}_seed{seed}" / "metrics.json"
    return root / f"{model}_{variant}_seed{seed}" / "metrics.json"


def metric_from_matrix(matrix: np.ndarray) -> dict[str, np.ndarray | float]:
    matrix = matrix.astype(float)
    true_positive = np.diag(matrix)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    union = support + predicted - true_positive
    precision = np.divide(
        true_positive, predicted, out=np.full(8, np.nan), where=predicted > 0
    )
    recall = np.divide(
        true_positive, support, out=np.full(8, np.nan), where=support > 0
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.full(8, np.nan),
        where=(precision + recall) > 0,
    )
    iou = np.divide(
        true_positive, union, out=np.full(8, np.nan), where=union > 0
    )
    return {
        "OA": float(true_positive.sum() / max(matrix.sum(), 1)),
        "macro_mIoU": float(np.nanmean(iou)),
        "macro_F1": float(np.nanmean(f1)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "support": support,
    }


def bootstrap_groups(
    group_matrices: dict[str, list[list[int]]],
    repetitions: int,
    seed: int,
) -> dict[str, np.ndarray]:
    groups = sorted(group_matrices)
    matrices = np.asarray([group_matrices[group] for group in groups], dtype=np.int64)
    rng = np.random.default_rng(seed)
    values = {
        "OA": np.empty(repetitions),
        "macro_mIoU": np.empty(repetitions),
        "macro_F1": np.empty(repetitions),
        "precision": np.empty((repetitions, 8)),
        "recall": np.empty((repetitions, 8)),
        "f1": np.empty((repetitions, 8)),
        "iou": np.empty((repetitions, 8)),
    }
    for index in range(repetitions):
        sampled = rng.integers(0, len(groups), size=len(groups))
        result = metric_from_matrix(matrices[sampled].sum(axis=0))
        for key in ("OA", "macro_mIoU", "macro_F1", "precision", "recall", "f1", "iou"):
            values[key][index] = result[key]
    return values


def interval(values: np.ndarray) -> tuple[float, float]:
    return tuple(np.nanpercentile(values, [2.5, 97.5]).tolist())


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * p_values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def exact_paired_permutation(
    matrices_a: dict[str, list[list[int]]],
    matrices_b: dict[str, list[list[int]]],
    metric_key: str,
) -> tuple[float, float]:
    groups = sorted(set(matrices_a) & set(matrices_b))
    a = np.asarray([matrices_a[group] for group in groups], dtype=np.int64)
    b = np.asarray([matrices_b[group] for group in groups], dtype=np.int64)
    observed = (
        metric_from_matrix(a.sum(axis=0))[metric_key]
        - metric_from_matrix(b.sum(axis=0))[metric_key]
    )
    differences = np.empty(2 ** len(groups), dtype=float)
    for index, bits in enumerate(itertools.product((False, True), repeat=len(groups))):
        swap = np.asarray(bits)
        perm_a = np.where(swap[:, None, None], b, a).sum(axis=0)
        perm_b = np.where(swap[:, None, None], a, b).sum(axis=0)
        differences[index] = (
            metric_from_matrix(perm_a)[metric_key]
            - metric_from_matrix(perm_b)[metric_key]
        )
    p_value = float(
        np.count_nonzero(np.abs(differences) >= abs(observed) - 1e-15)
        / len(differences)
    )
    return float(observed), p_value


def paired_bootstrap_difference(
    matrices_a: dict[str, list[list[int]]],
    matrices_b: dict[str, list[list[int]]],
    metric_key: str,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    groups = sorted(set(matrices_a) & set(matrices_b))
    a = np.asarray([matrices_a[group] for group in groups], dtype=np.int64)
    b = np.asarray([matrices_b[group] for group in groups], dtype=np.int64)
    rng = np.random.default_rng(seed)
    differences = np.empty(repetitions)
    for index in range(repetitions):
        sampled = rng.integers(0, len(groups), size=len(groups))
        differences[index] = (
            metric_from_matrix(a[sampled].sum(axis=0))[metric_key]
            - metric_from_matrix(b[sampled].sum(axis=0))[metric_key]
        )
    return interval(differences)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260725)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    support_rows = []
    for split in ("train", "val", "test"):
        subset = manifest.loc[manifest["split"] == split]
        for old_id, class_name in zip(ACTIVE_OFFICIAL_IDS, CLASS_NAMES_8):
            support_rows.append({
                "数据划分": split,
                "类别": class_name,
                "空间patch数": int((subset[f"pixels_class_{old_id}"] > 0).sum()),
                "有效像元数": int(subset[f"pixels_class_{old_id}"].sum()),
                "独立空间组数": int(
                    subset.loc[subset[f"pixels_class_{old_id}"] > 0, "group"].nunique()
                ),
            })
    write_csv(args.output_dir / "table_dataset_support_8class.csv", support_rows)

    primary: dict[str, dict] = {}
    bootstrap: dict[str, dict[str, np.ndarray]] = {}
    overall_rows = []
    class_rows = []
    seed_rows = []
    confusion_rows = []
    for model in MODEL_ORDER:
        metric = load_metric(args.experiment_root, model, "full", 42)
        primary[model] = metric
        group_matrices = metric["spatial_group_confusions"]
        boot = bootstrap_groups(
            group_matrices, args.bootstrap, args.bootstrap_seed
        )
        bootstrap[model] = boot
        oa_ci = interval(boot["OA"])
        miou_ci = interval(boot["macro_mIoU"])
        f1_ci = interval(boot["macro_F1"])
        overall_rows.append({
            "模型": MODEL_LABELS[model],
            "OA": metric["overall_accuracy"],
            "OA_95CI下限": oa_ci[0],
            "OA_95CI上限": oa_ci[1],
            "macro_mIoU": metric["macro_iou_present"],
            "mIoU_95CI下限": miou_ci[0],
            "mIoU_95CI上限": miou_ci[1],
            "macro_F1": metric["macro_f1_present"],
            "F1_95CI下限": f1_ci[0],
            "F1_95CI上限": f1_ci[1],
            "空间测试块数": len(group_matrices),
            "bootstrap次数": args.bootstrap,
        })
        for class_index, class_name in enumerate(CLASS_NAMES_8):
            values = metric["per_class"][class_name]
            class_rows.append({
                "模型": MODEL_LABELS[model],
                "类别": class_name,
                "测试像元数": values["support"],
                "Precision": values["precision"],
                "Precision_95CI下限": interval(boot["precision"][:, class_index])[0],
                "Precision_95CI上限": interval(boot["precision"][:, class_index])[1],
                "Recall": values["recall"],
                "Recall_95CI下限": interval(boot["recall"][:, class_index])[0],
                "Recall_95CI上限": interval(boot["recall"][:, class_index])[1],
                "F1": values["f1"],
                "F1_95CI下限": interval(boot["f1"][:, class_index])[0],
                "F1_95CI上限": interval(boot["f1"][:, class_index])[1],
                "IoU": values["iou"],
                "IoU_95CI下限": interval(boot["iou"][:, class_index])[0],
                "IoU_95CI上限": interval(boot["iou"][:, class_index])[1],
            })
        normalized = np.asarray(metric["confusion_matrix_row_normalized"])
        for true_index, true_name in enumerate(CLASS_NAMES_8):
            for pred_index, pred_name in enumerate(CLASS_NAMES_8):
                confusion_rows.append({
                    "模型": MODEL_LABELS[model],
                    "真实类别": true_name,
                    "预测类别": pred_name,
                    "行归一化比例": normalized[true_index, pred_index],
                })
        for seed in (42, 43, 44):
            if not metric_file(args.experiment_root, model, "full", seed).exists():
                continue
            seed_metric = load_metric(args.experiment_root, model, "full", seed)
            seed_rows.append({
                "模型": MODEL_LABELS[model],
                "seed": seed,
                "OA": seed_metric["overall_accuracy"],
                "macro_mIoU": seed_metric["macro_iou_present"],
                "macro_F1": seed_metric["macro_f1_present"],
            })

    # Three independent training seeds: descriptive mean, SD and t interval.
    seed_summary_rows = []
    for model in MODEL_ORDER:
        rows = [row for row in seed_rows if row["模型"] == MODEL_LABELS[model]]
        summary = {"模型": MODEL_LABELS[model], "训练随机种子数": len(rows)}
        for key in ("OA", "macro_mIoU", "macro_F1"):
            values = np.asarray([row[key] for row in rows], dtype=float)
            mean = float(values.mean())
            if len(values) >= 2:
                sd = float(values.std(ddof=1))
                critical = 12.7062047361747 if len(values) == 2 else 4.302652729911275
                half = critical * sd / math.sqrt(len(values))
            else:
                sd = float("nan")
                half = float("nan")
            summary[f"{key}_均值"] = mean
            summary[f"{key}_SD"] = sd
            summary[f"{key}_t95CI下限"] = mean - half
            summary[f"{key}_t95CI上限"] = mean + half
        seed_summary_rows.append(summary)

    significance_rows = []
    for metric_key, label in (
        ("OA", "OA"), ("macro_mIoU", "macro_mIoU"), ("macro_F1", "macro_F1")
    ):
        raw_p = []
        temporary = []
        ours_groups = primary["ours"]["spatial_group_confusions"]
        for index, model in enumerate(MODEL_ORDER[1:]):
            comparison_groups = primary[model]["spatial_group_confusions"]
            difference, p_value = exact_paired_permutation(
                ours_groups, comparison_groups, metric_key
            )
            ci = paired_bootstrap_difference(
                ours_groups, comparison_groups, metric_key,
                args.bootstrap, args.bootstrap_seed + index + 1,
            )
            raw_p.append(p_value)
            temporary.append({
                "指标": label,
                "比较": f"本文模型 vs {MODEL_LABELS[model]}",
                "差值_本文减基线": difference,
                "差值95CI下限": ci[0],
                "差值95CI上限": ci[1],
                "精确配对置换p值": p_value,
                "空间测试块数": len(ours_groups),
                "置换数": 2 ** len(ours_groups),
            })
        adjusted = holm_adjust(raw_p)
        for row, value in zip(temporary, adjusted):
            row["Holm校正p值"] = value
            row["Holm_0.05显著"] = value < 0.05
            significance_rows.append(row)

    ablation_rows = []
    for model in MODEL_ORDER:
        full = load_metric(args.experiment_root, model, "full", 42)
        for variant in VARIANT_ORDER:
            metric = load_metric(args.experiment_root, model, variant, 42)
            ablation_rows.append({
                "模型": MODEL_LABELS[model],
                "变体": VARIANT_LABELS[variant],
                "变体代码": variant,
                "OA": metric["overall_accuracy"],
                "macro_mIoU": metric["macro_iou_present"],
                "macro_F1": metric["macro_f1_present"],
                "相对全模态_OA变化": metric["overall_accuracy"] - full["overall_accuracy"],
                "相对全模态_mIoU变化": metric["macro_iou_present"] - full["macro_iou_present"],
                "相对全模态_F1变化": metric["macro_f1_present"] - full["macro_f1_present"],
            })

    write_csv(args.output_dir / "table_overall_metrics_with_ci.csv", overall_rows)
    write_csv(args.output_dir / "table_per_class_metrics_with_ci.csv", class_rows)
    write_csv(args.output_dir / "table_confusion_long.csv", confusion_rows)
    write_csv(args.output_dir / "table_training_seed_runs.csv", seed_rows)
    write_csv(args.output_dir / "table_training_seed_summary.csv", seed_summary_rows)
    write_csv(args.output_dir / "table_paired_significance.csv", significance_rows)
    write_csv(args.output_dir / "table_ablation_all_models.csv", ablation_rows)
    protocol = {
        "task": "Eight active classes; commercial/office and barren excluded before training and evaluation",
        "primary_point_estimate": "seed 42 checkpoint selected only by validation macro mIoU",
        "test_design": "fixed spatially independent test partition with 14 non-overlapping 5.12-km groups",
        "confidence_interval": (
            f"{args.bootstrap}-replicate paired spatial-group bootstrap; "
            "percentile 95% interval; groups sampled with replacement"
        ),
        "significance": (
            "exact paired group-label permutation over all 2^14 assignments; "
            "two-sided p-values; Holm correction across five baselines per metric"
        ),
        "training_stability": (
            "Available independently trained seeds are reported descriptively; "
            "sample SD and t95 interval are emitted only when at least two complete runs exist"
        ),
        "class_support_table": "table_dataset_support_8class.csv",
        "warning": "Intervals quantify the sampled spatial groups; only the three-seed table reflects optimizer/initialization variability.",
    }
    (args.output_dir / "validation_protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "overall_rows": len(overall_rows),
        "per_class_rows": len(class_rows),
        "significance_rows": len(significance_rows),
        "ablation_rows": len(ablation_rows),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
