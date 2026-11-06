from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .hierarchical_train import make_dataset
from .labels import CLASS_NAMES, IGNORE_INDEX, LEAF_TO_OFFICIAL
from .predict_experiment_models import (
    make_comparator,
    make_main_model,
    make_multimodal_comparator,
)


def metrics_from_confusion(cm: np.ndarray) -> dict:
    cm = cm.astype(np.float64)
    support = cm.sum(1)
    predicted = cm.sum(0)
    tp = np.diag(cm)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    union = support + predicted - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    present = support > 0
    total = cm.sum()
    return {
        "overall_accuracy": float(tp.sum() / total) if total else 0.0,
        "macro_iou_present": float(iou[present].mean()),
        "macro_f1_present": float(f1[present].mean()),
        "macro_precision_present": float(precision[present].mean()),
        "macro_recall_present": float(recall[present].mean()),
        "per_class": [
            {
                "class_id": i + 1,
                "class_name": CLASS_NAMES[i],
                "support": int(support[i]),
                "predicted": int(predicted[i]),
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "iou": float(iou[i]),
            }
            for i in range(10)
        ],
        "confusion_matrix": cm.astype(np.int64).tolist(),
        "confusion_matrix_row_normalized": np.divide(
            cm, support[:, None], out=np.zeros_like(cm), where=support[:, None] > 0
        ).tolist(),
    }


def bootstrap_intervals(group_confusions: dict[str, np.ndarray], seed: int, draws: int) -> dict:
    groups = sorted(group_confusions)
    matrices = np.stack([group_confusions[group] for group in groups])
    rng = np.random.default_rng(seed)
    oa = np.empty(draws, dtype=np.float64)
    miou = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = rng.integers(0, len(groups), size=len(groups))
        metric = metrics_from_confusion(matrices[sampled].sum(0))
        oa[draw] = metric["overall_accuracy"]
        miou[draw] = metric["macro_iou_present"]
    return {
        "method": "spatial-block cluster bootstrap",
        "unit": "512x512-pixel split group",
        "groups": len(groups),
        "draws": draws,
        "seed": seed,
        "OA_95CI": [float(x) for x in np.quantile(oa, (0.025, 0.975))],
        "macro_mIoU_95CI": [float(x) for x in np.quantile(miou, (0.025, 0.975))],
    }


def evaluate_model(
    name, model, loader, dataset, device, hierarchical, bootstrap_draws, seed,
    multimodal: bool = False,
):
    model = model.to(device).eval()
    patch_to_group = dict(
        zip(dataset.frame["patch_id"].astype(str), dataset.frame["group"].astype(str))
    )
    group_confusions: dict[str, np.ndarray] = {}
    pixels = 0
    start = time.perf_counter()
    official_lut = torch.as_tensor(LEAF_TO_OFFICIAL, device=device)
    with torch.inference_mode():
        for batch in loader:
            s2 = batch["s2"].to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16
            ):
                if hierarchical:
                    output = model(
                        batch["s1"].to(device, non_blocking=True), s2,
                        batch["input_ids"].to(device, non_blocking=True),
                        batch["text_mask"].to(device, non_blocking=True),
                        osm=batch["osm"].to(device, non_blocking=True),
                        urban=batch["urban"].to(device, non_blocking=True),
                        roads=batch["roads"].to(device, non_blocking=True),
                    )
                    leaf_probs = output["leaf_log_probs"].exp()
                elif multimodal:
                    leaf_probs = model(
                        s2,
                        batch["s1"].to(device, non_blocking=True),
                        batch["urban"].to(device, non_blocking=True),
                        batch["roads"].to(device, non_blocking=True),
                        batch["osm"].to(device, non_blocking=True),
                        batch["input_ids"].to(device, non_blocking=True),
                        batch["text_mask"].to(device, non_blocking=True),
                    ).softmax(1)
                else:
                    leaf_probs = model(s2).softmax(1)
            pred = official_lut[leaf_probs.argmax(1)]
            target = batch["label"].to(device, non_blocking=True)
            valid = target != IGNORE_INDEX
            safe = target.masked_fill(~valid, 0)
            target = official_lut[safe]
            for i, patch_id in enumerate(batch["patch_id"]):
                mask = valid[i]
                pair = (target[i][mask] * 10 + pred[i][mask]).cpu().numpy()
                cm = np.bincount(pair, minlength=100).reshape(10, 10)
                group = patch_to_group[str(patch_id)]
                group_confusions[group] = group_confusions.get(group, np.zeros((10, 10), dtype=np.int64)) + cm
                pixels += int(mask.sum())
    seconds = time.perf_counter() - start
    cm = sum(group_confusions.values(), np.zeros((10, 10), dtype=np.int64))
    result = metrics_from_confusion(cm)
    result["bootstrap"] = bootstrap_intervals(group_confusions, seed, bootstrap_draws)
    result["inference"] = {
        "evaluated_pixels": pixels,
        "seconds": seconds,
        "megapixels_per_second": pixels / max(seconds, 1e-9) / 1e6,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"model={name} test_OA={result['overall_accuracy']:.4f} "
        f"test_mIoU={result['macro_iou_present']:.4f}", flush=True
    )
    return result


def validation_metrics(path: Path, hierarchical: bool) -> tuple[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if hierarchical:
        return int(data["best_epoch"]), data["validation"]["official10_projection"]
    return int(data["best_epoch"]), data["validation"]["official10"]


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate three models and make paper-ready tables.")
    parser.add_argument("--config", default="config_hierarchical.yaml")
    parser.add_argument("--main-checkpoint", required=True, type=Path)
    parser.add_argument("--main-metrics", required=True, type=Path)
    parser.add_argument("--fcn-checkpoint", required=True, type=Path)
    parser.add_argument("--fcn-metrics", required=True, type=Path)
    parser.add_argument("--vgg-checkpoint", required=True, type=Path)
    parser.add_argument("--vgg-metrics", required=True, type=Path)
    parser.add_argument("--fcn-all-checkpoint", type=Path)
    parser.add_argument("--fcn-all-metrics", type=Path)
    parser.add_argument("--vgg-all-checkpoint", type=Path)
    parser.add_argument("--vgg-all-metrics", type=Path)
    parser.add_argument("--old-main-metrics", type=Path)
    parser.add_argument("--building-report", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["data"]["manifest_name"] = "manifest_hier_v2.csv"
    cfg["data"]["vocab_name"] = "vocab_hier_v2.json"
    cfg["data"]["normalization_name"] = "normalization_hier_v2.json"
    dataset = make_dataset(cfg, "test", False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=int(cfg["train"].get("num_workers", 0)),
                        pin_memory=device.type == "cuda")
    specifications = [
        ("本文模型（建筑高度）", "main", args.main_checkpoint, args.main_metrics, True, False),
        ("FCN-ResNet50（S2）", "fcn_resnet50", args.fcn_checkpoint, args.fcn_metrics, False, False),
        ("VGG-UNet-Transformer（S2）", "vgg_unet_transformer", args.vgg_checkpoint, args.vgg_metrics, False, False),
    ]
    if args.fcn_all_checkpoint and args.fcn_all_metrics:
        specifications.append((
            "FCN-ResNet50（全模态）", "fcn_resnet50", args.fcn_all_checkpoint,
            args.fcn_all_metrics, False, True,
        ))
    if args.vgg_all_checkpoint and args.vgg_all_metrics:
        specifications.append((
            "VGG-UNet-Transformer（全模态）", "vgg_unet_transformer",
            args.vgg_all_checkpoint, args.vgg_all_metrics, False, True,
        ))
    results = {}
    validation = {}
    epochs = {}
    for name, kind, checkpoint, metrics_path, hierarchical, multimodal in specifications:
        model = (
            make_main_model(cfg, dataset, checkpoint)
            if hierarchical
            else (
                make_multimodal_comparator(cfg, dataset, kind, checkpoint)
                if multimodal
                else make_comparator(kind, dataset.s2_output_channels, checkpoint)
            )
        )
        results[name] = evaluate_model(
            name, model, loader, dataset, device, hierarchical,
            args.bootstrap_draws, int(cfg["data"]["seed"]), multimodal,
        )
        epochs[name], validation[name] = validation_metrics(metrics_path, hierarchical)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "canonical_test_metrics_with_spatial_bootstrap.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_rows = []
    for name in results:
        test = results[name]
        ci = test["bootstrap"]
        summary_rows.append({
            "模型": name,
            "最佳epoch": epochs[name],
            "验证OA": f"{validation[name]['overall_accuracy']:.4f}",
            "验证mIoU": f"{validation[name]['macro_iou_present']:.4f}",
            "测试OA": f"{test['overall_accuracy']:.4f}",
            "测试OA_95%CI": f"[{ci['OA_95CI'][0]:.4f}, {ci['OA_95CI'][1]:.4f}]",
            "测试macro_mIoU": f"{test['macro_iou_present']:.4f}",
            "测试mIoU_95%CI": f"[{ci['macro_mIoU_95CI'][0]:.4f}, {ci['macro_mIoU_95CI'][1]:.4f}]",
            "测试macro_F1": f"{test['macro_f1_present']:.4f}",
            "参数量": test["inference"]["parameters"],
            "推理MPix/s": f"{test['inference']['megapixels_per_second']:.3f}",
        })
    write_csv(args.output_dir / "table_model_comparison.csv", list(summary_rows[0]), summary_rows)

    class_rows = []
    for name, metric in results.items():
        for row in metric["per_class"]:
            class_rows.append({"模型": name, **row})
        matrix = np.asarray(metric["confusion_matrix_row_normalized"])
        write_csv(
            args.output_dir / f"confusion_row_normalized_{len(class_rows)}_{name.split('（')[0]}.csv",
            ["真实类别"] + CLASS_NAMES,
            [{"真实类别": CLASS_NAMES[i], **{CLASS_NAMES[j]: f"{matrix[i, j]:.6f}" for j in range(10)}} for i in range(10)],
        )
    write_csv(args.output_dir / "table_per_class_metrics.csv", list(class_rows[0]), class_rows)

    ablation_rows = []
    if args.old_main_metrics:
        old = json.loads(args.old_main_metrics.read_text(encoding="utf-8"))["test"]["official10_projection"]
        new = results["本文模型（建筑高度）"]
        ablation_rows = [
            {"建筑输入": "旧建筑功能轮廓（9通道）", "测试OA": old["overall_accuracy"],
             "测试macro_mIoU": old["macro_iou_present"]},
            {"建筑输入": "建筑功能+高度结构（14通道）", "测试OA": new["overall_accuracy"],
             "测试macro_mIoU": new["macro_iou_present"]},
            {"建筑输入": "高度版本－旧版本", "测试OA": new["overall_accuracy"] - old["overall_accuracy"],
             "测试macro_mIoU": new["macro_iou_present"] - old["macro_iou_present"]},
        ]
        write_csv(args.output_dir / "table_building_height_ablation.csv", list(ablation_rows[0]), ablation_rows)

    # Paper-style staged and conditional tables for the selected model.
    main_metric_document = json.loads(args.main_metrics.read_text(encoding="utf-8"))
    stage_rows = []
    for output_name, metric_key in (
        ("一级物理地表4类", "coarse4"),
        ("建成区条件6类", "built6_conditional"),
        ("自然地表条件8类", "natural8_conditional"),
        ("叶类14类软路由", "leaf14_soft_route"),
        ("官方一级组10类", "official10_projection"),
    ):
        metric = main_metric_document["test"][metric_key]
        stage_rows.append({
            "输出层级": output_name,
            "OA": metric["overall_accuracy"],
            "macro_mIoU": metric["macro_iou_present"],
            "macro_F1": metric["macro_f1_present"],
            "macro_Precision": metric["macro_precision_present"],
            "macro_Recall": metric["macro_recall_present"],
        })
    write_csv(args.output_dir / "table_hierarchical_outputs.csv", list(stage_rows[0]), stage_rows)

    subset_rows = []
    for subset, metric in main_metric_document["test"].get(
        "condition_subsets_official10", {}
    ).items():
        subset_rows.append({
            "测试子集": subset,
            "OA": metric["overall_accuracy"],
            "macro_mIoU": metric["macro_iou_present"],
            "macro_F1": metric["macro_f1_present"],
        })
    if subset_rows:
        write_csv(args.output_dir / "table_condition_subsets.csv", list(subset_rows[0]), subset_rows)

    progressive_rows = []
    experiment_root = Path(cfg["paths"]["output_dir"])
    for label, relative in (
        ("全量多模态基础模型", "hierarchical_s1_dualpol_v2_full/metrics_joint.json"),
        ("+S2指数+道路（直接融合）", "hierarchical_indices_roads_full/metrics_joint.json"),
        ("+S2指数+道路（独立门控）", "hierarchical_indices_roads_gated_full/metrics_joint.json"),
        ("+建筑高度结构（本文最终）", "hierarchical_height_buildings_full/metrics_joint.json"),
    ):
        path = experiment_root / relative
        if not path.exists():
            continue
        document = json.loads(path.read_text(encoding="utf-8"))
        val = document["validation"]["official10_projection"]
        test = document["test"]["official10_projection"]
        progressive_rows.append({
            "实验": label,
            "验证OA": val["overall_accuracy"],
            "验证macro_mIoU": val["macro_iou_present"],
            "测试OA": test["overall_accuracy"],
            "测试macro_mIoU": test["macro_iou_present"],
        })
    if progressive_rows:
        write_csv(args.output_dir / "table_progressive_feature_ablation.csv", list(progressive_rows[0]), progressive_rows)

    latex = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{不同模型在统一空间独立测试集上的精度比较。}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"模型 & 验证OA & 验证mIoU & 测试OA & 测试mIoU & 测试macro-F1 \\",
        r"\midrule",
    ]
    for row in summary_rows:
        latex.append(
            f"{row['模型']} & {row['验证OA']} & {row['验证mIoU']} & "
            f"{row['测试OA']} & {row['测试macro_mIoU']} & {row['测试macro_F1']} \\\\"
        )
    latex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (args.output_dir / "table_model_comparison.tex").write_text("\n".join(latex), encoding="utf-8")

    building = None
    if args.building_report:
        building = json.loads(args.building_report.read_text(encoding="utf-8"))
    report_lines = [
        "# 全香港建筑高度多模态土地利用／覆盖实验结果",
        "",
        "## 实验口径",
        "",
        "三种模型使用完全相同的 512×512 像元空间分组训练／验证／测试划分；模型和 epoch 只依据验证集 macro mIoU 选择。测试集未参与调参。置信区间采用测试空间块聚类 bootstrap，不能替代多随机种子重复训练。",
        "",
        "## 模型比较",
        "",
        "| 模型 | 验证 OA | 验证 mIoU | 测试 OA | 测试 macro mIoU | 测试 macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        report_lines.append(
            f"| {row['模型']} | {row['验证OA']} | {row['验证mIoU']} | {row['测试OA']} | "
            f"{row['测试macro_mIoU']} | {row['测试macro_F1']} |"
        )
    if building:
        report_lines += [
            "", "## 建筑高度数据处理", "",
            f"两份高度 shp 经接缝去重、几何修复和研究区裁剪后保留 {building['merged_clipped_building_count']:,} 个建筑；"
            f"功能标签匹配 {building['function_matched_count']:,} 个，Unknown {building['function_unknown_count']:,} 个。"
            f"建筑高度中位数 {building['height_statistics_m']['median']:.2f} m，均值 {building['height_statistics_m']['mean']:.2f} m。",
            "",
            "高度/3 仅用于构造楼层与容积率代理变量，不是真实楼层数或规划容积率。",
        ]
    if ablation_rows:
        report_lines += ["", "## 建筑高度消融", "", "| 建筑输入 | 测试 OA | 测试 macro mIoU |", "|---|---:|---:|"]
        report_lines += [f"| {r['建筑输入']} | {r['测试OA']:.4f} | {r['测试macro_mIoU']:.4f} |" for r in ablation_rows]
    report_lines += [
        "", "## 论文使用限制", "",
        "当前比较是单随机种子实验；空间 bootstrap 描述测试区域抽样不确定性，但不能证明训练随机性的稳定性。若要写“显著优于”，应补做至少 3 个随机种子并报告均值±标准差或配对空间块检验。",
    ]
    (args.output_dir / "PAPER_RESULTS.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
