from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODEL_ORDER = (
    "本文分层多模态模型",
    "CNN",
    "FCNN",
    "Transformer",
    "FCN-ResNet50",
    "VGG-UNet-Transformer",
)
VARIANT_ORDER = (
    "full",
    "s2",
    "no_osm_roads",
    "no_building",
    "no_s1",
    "no_nighttime",
    "no_highres",
)
VARIANT_LABELS = {
    "full": "全部模态",
    "s2": "仅S2",
    "no_osm_roads": "去除POI/AOI/道路",
    "no_building": "去除建筑物特征",
    "no_s1": "去除S1",
    "no_nighttime": "去除夜间灯光",
    "no_highres": "去除高分辨率影像",
}
CLASS_NAMES = (
    "住宅",
    "商业办公",
    "工业仓储",
    "公共机构与休闲",
    "交通设施",
    "其他建设用地",
    "农业",
    "林地/灌丛/草地/湿地",
    "荒地",
    "水体",
)
CLASS_SHORT = ("住宅", "商业", "工业", "公共", "交通", "其他", "农业", "林地等", "荒地", "水体")


@dataclass(frozen=True)
class ResultSpec:
    model: str
    variant: str
    path: Path
    hierarchical: bool = False


def _read_metric(spec: ResultSpec) -> dict:
    document = json.loads(spec.path.read_text(encoding="utf-8"))
    key = "official10_projection" if spec.hierarchical else "official10"
    return {
        "best_epoch": document.get("best_epoch"),
        "modalities": document.get("modalities"),
        "validation": document["validation"][key],
        "test": document["test"][key],
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _normalise(matrix: np.ndarray) -> np.ndarray:
    denominator = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, denominator, out=np.zeros_like(matrix), where=denominator > 0)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "font.size": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _safe_name(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_").replace("/", "_")


def _draw_matrix(ax, matrix: np.ndarray, title: str, annotate: bool = True):
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(10), CLASS_SHORT, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(10), CLASS_SHORT, fontsize=7)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title, fontsize=10, fontweight="bold")
    if annotate:
        for row in range(10):
            for col in range(10):
                value = matrix[row, col]
                if value >= 0.005:
                    ax.text(
                        col,
                        row,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5.2,
                        color="white" if value > 0.52 else "black",
                    )
    return image


def _write_confusions(output: Path, results: dict[tuple[str, str], dict]) -> None:
    confusion_dir = output / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    for model in MODEL_ORDER:
        for variant in VARIANT_ORDER:
            metric = results[(model, variant)]["test"]
            raw = np.asarray(metric["confusion_matrix"], dtype=np.float64)
            normalised = _normalise(raw)
            stem = f"{_safe_name(model)}_{variant}"
            np.savetxt(confusion_dir / f"{stem}_raw.csv", raw, delimiter=",", fmt="%.0f")
            np.savetxt(
                confusion_dir / f"{stem}_row_normalized.csv",
                normalised,
                delimiter=",",
                fmt="%.8f",
            )
            fig, ax = plt.subplots(figsize=(7.2, 6.3), constrained_layout=True)
            image = _draw_matrix(ax, normalised, f"{model}（{VARIANT_LABELS[variant]}）")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="行归一化比例")
            fig.savefig(confusion_dir / f"{stem}.png", dpi=300, facecolor="white")
            fig.savefig(confusion_dir / f"{stem}.pdf", facecolor="white")
            plt.close(fig)

    for variant in VARIANT_ORDER:
        fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.2), constrained_layout=True)
        image = None
        for ax, model in zip(axes.flat, MODEL_ORDER):
            raw = np.asarray(results[(model, variant)]["test"]["confusion_matrix"], dtype=np.float64)
            image = _draw_matrix(ax, _normalise(raw), model)
        assert image is not None
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="行归一化比例")
        fig.suptitle(
            f"六种模型测试集混淆矩阵（{VARIANT_LABELS[variant]}）",
            fontsize=15,
            fontweight="bold",
        )
        fig.savefig(output / f"confusion_overview_{variant}.png", dpi=300, facecolor="white")
        fig.savefig(output / f"confusion_overview_{variant}.pdf", facecolor="white")
        plt.close(fig)


def _write_ablation_heatmap(output: Path, results: dict[tuple[str, str], dict]) -> None:
    variants = VARIANT_ORDER[2:]
    drops = np.asarray(
        [
            [
                results[(model, "full")]["test"]["macro_iou_present"]
                - results[(model, variant)]["test"]["macro_iou_present"]
                for variant in variants
            ]
            for model in MODEL_ORDER
        ],
        dtype=np.float64,
    )
    limit = max(0.02, float(np.max(np.abs(drops))))
    fig, ax = plt.subplots(figsize=(10.8, 5.7), constrained_layout=True)
    image = ax.imshow(drops, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(variants)), [VARIANT_LABELS[item] for item in variants], rotation=20, ha="right")
    ax.set_yticks(range(len(MODEL_ORDER)), MODEL_ORDER)
    ax.set_title("去除单一模态组后的测试集 mIoU 下降量", fontweight="bold")
    for row in range(drops.shape[0]):
        for col in range(drops.shape[1]):
            value = drops[row, col]
            ax.text(col, row, f"{value:+.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, label="全模态 mIoU − 消融 mIoU")
    fig.savefig(output / "extended_modality_ablation_heatmap.png", dpi=300, facecolor="white")
    fig.savefig(output / "extended_modality_ablation_heatmap.pdf", facecolor="white")
    plt.close(fig)


def _specs(root: Path) -> list[ResultSpec]:
    ours = {
        "full": "hierarchical_height_buildings_full",
        "s2": "comparison_ours_s2_ablation",
        "no_osm_roads": "ablation_ours_no_osm_roads",
        "no_building": "ablation_ours_no_building",
        "no_s1": "ablation_ours_no_s1",
        "no_nighttime": "ablation_ours_no_nighttime",
        "no_highres": "ablation_ours_no_highres",
    }
    specs = [
        ResultSpec(
            "本文分层多模态模型",
            variant,
            root / "outputs" / "full" / run_name / "metrics_joint.json",
            True,
        )
        for variant, run_name in ours.items()
    ]
    baseline_names = {
        "CNN": "cnn",
        "FCNN": "fcnn",
        "Transformer": "transformer",
        "FCN-ResNet50": "fcn_resnet50",
        "VGG-UNet-Transformer": "vgg_unet_transformer",
    }
    for model, slug in baseline_names.items():
        for variant in VARIANT_ORDER:
            run_variant = "all" if variant == "full" else variant
            run_name = f"comparison_{slug}_{run_variant}_uniform"
            specs.append(ResultSpec(model, variant, root / "outputs" / "full" / run_name / "metrics.json"))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总六模型扩展模态消融实验。")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/full/final_height_experiment/ablation_extended_modalities"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    specs = _specs(root)
    missing = [str(spec.path) for spec in specs if not spec.path.exists()]
    if missing:
        raise FileNotFoundError("缺少以下训练结果：\n" + "\n".join(missing))
    results = {(spec.model, spec.variant): _read_metric(spec) for spec in specs}

    long_rows: list[dict] = []
    wide_rows: list[dict] = []
    class_rows: list[dict] = []
    for model in MODEL_ORDER:
        full_miou = results[(model, "full")]["test"]["macro_iou_present"]
        wide_row: dict = {"模型": model}
        for variant in VARIANT_ORDER:
            result = results[(model, variant)]
            test = result["test"]
            long_rows.append(
                {
                    "模型": model,
                    "实验设置": VARIANT_LABELS[variant],
                    "变体代码": variant,
                    "最佳epoch": result["best_epoch"],
                    "验证OA": result["validation"]["overall_accuracy"],
                    "验证mIoU": result["validation"]["macro_iou_present"],
                    "测试OA": test["overall_accuracy"],
                    "测试mIoU": test["macro_iou_present"],
                    "测试macro_F1": test["macro_f1_present"],
                    "相对全模态mIoU下降": full_miou - test["macro_iou_present"],
                }
            )
            wide_row[f"{VARIANT_LABELS[variant]}_mIoU"] = test["macro_iou_present"]
            wide_row[f"{VARIANT_LABELS[variant]}_相对全模态下降"] = full_miou - test["macro_iou_present"]
            per_class_values = list(test["per_class"].values())
            if len(per_class_values) != len(CLASS_NAMES):
                raise ValueError(f"{model}/{variant} 类别数不是10。")
            for class_name, item in zip(CLASS_NAMES, per_class_values):
                class_rows.append(
                    {
                        "模型": model,
                        "实验设置": VARIANT_LABELS[variant],
                        "类别": class_name,
                        "测试像元数": item["support"],
                        "Precision": item["precision"],
                        "Recall": item["recall"],
                        "F1": item["f1"],
                        "IoU": item["iou"],
                    }
                )
        wide_rows.append(wide_row)

    _write_csv(output / "table_extended_modality_ablation_long.csv", long_rows)
    _write_csv(output / "table_extended_modality_ablation_wide.csv", wide_rows)
    _write_csv(output / "table_extended_modality_ablation_per_class.csv", class_rows)
    _style()
    _write_confusions(output, results)
    _write_ablation_heatmap(output, results)

    lines = [
        "# 六模型扩展模态消融实验",
        "",
        "所有消融均从头重新训练相应网络，而不是在完整模型推理时将输入置零。训练、验证和测试空间划分、标签映射、优化器与评价代码保持不变。",
        "其中“去除POI/AOI/道路”同时移除POI栅格、AOI栅格、道路栅格与OSM文本编码，防止通过文本支路泄漏同源语义。",
        "",
        "| 模型 | 全模态 | 仅S2 | 去POI/AOI/道路 | 去建筑 | 去S1 | 去夜光 | 去高分影像 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in wide_rows:
        lines.append(
            f"| {row['模型']} | {row['全部模态_mIoU']:.4f} | {row['仅S2_mIoU']:.4f} | "
            f"{row['去除POI/AOI/道路_mIoU']:.4f} | {row['去除建筑物特征_mIoU']:.4f} | "
            f"{row['去除S1_mIoU']:.4f} | {row['去除夜间灯光_mIoU']:.4f} | "
            f"{row['去除高分辨率影像_mIoU']:.4f} |"
        )
    ours_row = next(row for row in wide_rows if row["模型"] == "本文分层多模态模型")
    ours_ranked = sorted(
        (
            (VARIANT_LABELS[variant], ours_row[f"{VARIANT_LABELS[variant]}_相对全模态下降"])
            for variant in VARIANT_ORDER[2:]
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    lines += [
        "",
        "## 本文模型的模态贡献",
        "",
        "按移除后测试 mIoU 的下降量排序："
        + "，".join(f"{label} {drop:+.4f}" for label, drop in ours_ranked)
        + "。因此，本次单随机种子实验中，POI/AOI/道路与高分辨率影像贡献最大；S1和建筑物特征提供中等增益；夜间灯光的边际增益较小。",
        "",
        "仅使用S2时，本文模型 mIoU 为 "
        f"{ours_row['仅S2_mIoU']:.4f}，全部模态为 {ours_row['全部模态_mIoU']:.4f}，"
        f"总增益为 {ours_row['仅S2_相对全模态下降']:+.4f}。",
        "",
        "## 公平输入说明",
        "",
        "FCN-ResNet50 与 VGG-UNet-Transformer 的原始版本只输入 Sentinel-2。为公平比较，本实验为二者以及 CNN、FCNN、Transformer 增加与本文模型一致的多模态输入头；每次消融均从六种模型的输入头中删除同一组数据，并重新训练。因而差异主要反映网络结构与融合机制，而不是可用数据源不同。",
        "",
        "## 统计限制",
        "",
        "当前表格为固定随机种子的一次完整训练。可用于报告本次测试集结果和结构性趋势；小幅负下降值表示消融模型在单次随机训练中略高于全模态模型，不能解释为该模态有害。若论文使用“显著提升”等措辞，应再运行至少3个随机种子并报告均值±标准差或配对空间块检验。",
    ]
    (output / "EXTENDED_ABLATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "rows": len(long_rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
