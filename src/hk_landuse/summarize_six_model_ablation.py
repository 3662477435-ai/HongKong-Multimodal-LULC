from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .labels import CLASS_NAMES


SHORT_NAMES = (
    "住宅", "商业", "工业", "公共", "交通",
    "其他", "农业", "林地等", "荒地", "水体",
)


@dataclass(frozen=True)
class ResultSpec:
    model: str
    variant: str
    path: Path
    hierarchical: bool = False


def _read_metric(spec: ResultSpec) -> dict:
    document = json.loads(spec.path.read_text(encoding="utf-8"))
    split_key = "official10_projection" if spec.hierarchical else "official10"
    return {
        "best_epoch": document.get("best_epoch"),
        "validation": document["validation"][split_key],
        "test": document["test"][split_key],
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
    plt.rcParams.update({
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _draw_matrix(ax, matrix: np.ndarray, title: str, annotate: bool = True):
    image = ax.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(10), SHORT_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(10), SHORT_NAMES, fontsize=7)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(title, fontsize=10, fontweight="bold")
    if annotate:
        for row in range(10):
            for col in range(10):
                value = matrix[row, col]
                if value >= 0.005:
                    ax.text(
                        col, row, f"{value:.2f}", ha="center", va="center",
                        fontsize=5.3, color="white" if value > 0.52 else "black",
                    )
    return image


def _write_confusions(output: Path, results: dict[tuple[str, str], dict]) -> None:
    confusion_dir = output / "confusion_matrices"
    confusion_dir.mkdir(parents=True, exist_ok=True)
    _style()
    for (model, variant), result in results.items():
        raw = np.asarray(result["test"]["confusion_matrix"], dtype=np.float64)
        normalised = _normalise(raw)
        safe = model.lower().replace("-", "_").replace(" ", "_")
        np.savetxt(confusion_dir / f"{safe}_{variant}_raw.csv", raw, delimiter=",", fmt="%.0f")
        np.savetxt(
            confusion_dir / f"{safe}_{variant}_row_normalized.csv",
            normalised, delimiter=",", fmt="%.8f",
        )
        fig, ax = plt.subplots(figsize=(7.2, 6.3), constrained_layout=True)
        image = _draw_matrix(ax, normalised, f"{model}（{'全模态' if variant == 'all' else 'S2-only'}）")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="行归一化比例")
        fig.savefig(confusion_dir / f"{safe}_{variant}.png", dpi=300, facecolor="white")
        plt.close(fig)

    model_order = ["本文模型", "CNN", "FCNN", "Transformer", "FCN-ResNet50", "VGG-UNet-Transformer"]
    for variant in ("s2", "all"):
        fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.2), constrained_layout=True)
        image = None
        for ax, model in zip(axes.flat, model_order):
            raw = np.asarray(results[(model, variant)]["test"]["confusion_matrix"], dtype=np.float64)
            image = _draw_matrix(ax, _normalise(raw), model, annotate=True)
        assert image is not None
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="行归一化比例")
        label = "all_modalities" if variant == "all" else "s2_only"
        fig.suptitle(
            f"六种模型测试集混淆矩阵（{'全模态' if variant == 'all' else 'S2-only'}）",
            fontsize=15, fontweight="bold",
        )
        fig.savefig(output / f"confusion_overview_{label}.png", dpi=300, facecolor="white")
        fig.savefig(output / f"confusion_overview_{label}.pdf", facecolor="white")
        plt.close(fig)


def _write_gain_plot(output: Path, ablation_rows: list[dict]) -> None:
    _style()
    models = [row["模型"] for row in ablation_rows]
    s2 = np.array([float(row["S2-only测试mIoU"]) for row in ablation_rows])
    all_modal = np.array([float(row["全模态测试mIoU"]) for row in ablation_rows])
    x = np.arange(len(models))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 5.3), constrained_layout=True)
    ax.bar(x - width / 2, s2, width, label="S2-only", color="#9ecae1")
    ax.bar(x + width / 2, all_modal, width, label="全模态", color="#2171b5")
    for index, (left, right) in enumerate(zip(s2, all_modal)):
        ax.text(index + width / 2, right + 0.008, f"+{right-left:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x, models, rotation=15, ha="right")
    ax.set_ylabel("测试集 macro mIoU")
    ax.set_ylim(0, max(all_modal.max(), s2.max()) + 0.08)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("S2-only与全模态输入消融", fontweight="bold")
    fig.savefig(output / "ablation_s2_vs_all_modalities.png", dpi=300, facecolor="white")
    fig.savefig(output / "ablation_s2_vs_all_modalities.pdf", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise six-model S2/full-modal ablations.")
    parser.add_argument(
        "--project-root", type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/full/final_height_experiment/ablation_six_models"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    specs = [
        ResultSpec("本文模型", "s2", root / "outputs/full/comparison_ours_s2_ablation/metrics_joint.json", True),
        ResultSpec("本文模型", "all", root / "outputs/full/hierarchical_height_buildings_full/metrics_joint.json", True),
        ResultSpec("CNN", "s2", root / "outputs/full/comparison_cnn_s2_uniform/metrics.json"),
        ResultSpec("CNN", "all", root / "outputs/full/comparison_cnn_all_uniform/metrics.json"),
        ResultSpec("FCNN", "s2", root / "outputs/full/comparison_fcnn_s2_uniform/metrics.json"),
        ResultSpec("FCNN", "all", root / "outputs/full/comparison_fcnn_all_uniform/metrics.json"),
        ResultSpec("Transformer", "s2", root / "outputs/full/comparison_transformer_s2_uniform/metrics.json"),
        ResultSpec("Transformer", "all", root / "outputs/full/comparison_transformer_all_uniform/metrics.json"),
        ResultSpec("FCN-ResNet50", "s2", root / "outputs/full/comparison_fcn_resnet50_s2_uniform/metrics.json"),
        ResultSpec("FCN-ResNet50", "all", root / "outputs/full/comparison_fcn_resnet50_all_uniform/metrics.json"),
        ResultSpec("VGG-UNet-Transformer", "s2", root / "outputs/full/comparison_vgg_unet_transformer_s2_uniform/metrics.json"),
        ResultSpec("VGG-UNet-Transformer", "all", root / "outputs/full/comparison_vgg_unet_transformer_all_uniform/metrics.json"),
    ]
    missing = [str(spec.path) for spec in specs if not spec.path.exists()]
    if missing:
        raise FileNotFoundError("Missing metrics:\n" + "\n".join(missing))
    results = {(spec.model, spec.variant): _read_metric(spec) for spec in specs}
    model_order = ["本文模型", "CNN", "FCNN", "Transformer", "FCN-ResNet50", "VGG-UNet-Transformer"]
    ablation_rows = []
    full_rows = []
    class_rows = []
    for model in model_order:
        s2 = results[(model, "s2")]
        all_modal = results[(model, "all")]
        ablation_rows.append({
            "模型": model,
            "S2-only最佳epoch": s2["best_epoch"],
            "S2-only测试OA": s2["test"]["overall_accuracy"],
            "S2-only测试mIoU": s2["test"]["macro_iou_present"],
            "全模态最佳epoch": all_modal["best_epoch"],
            "全模态测试OA": all_modal["test"]["overall_accuracy"],
            "全模态测试mIoU": all_modal["test"]["macro_iou_present"],
            "OA增益": all_modal["test"]["overall_accuracy"] - s2["test"]["overall_accuracy"],
            "mIoU增益": all_modal["test"]["macro_iou_present"] - s2["test"]["macro_iou_present"],
        })
        full_rows.append({
            "模型": model,
            "验证OA": all_modal["validation"]["overall_accuracy"],
            "验证mIoU": all_modal["validation"]["macro_iou_present"],
            "测试OA": all_modal["test"]["overall_accuracy"],
            "测试mIoU": all_modal["test"]["macro_iou_present"],
            "测试macro_F1": all_modal["test"]["macro_f1_present"],
        })
        for variant in ("s2", "all"):
            metric = results[(model, variant)]["test"]
            for class_name in CLASS_NAMES:
                item = metric["per_class"][class_name]
                class_rows.append({
                    "模型": model,
                    "输入": "全模态" if variant == "all" else "S2-only",
                    "类别": class_name,
                    "support": item["support"],
                    "precision": item["precision"],
                    "recall": item["recall"],
                    "f1": item["f1"],
                    "iou": item["iou"],
                })
    _write_csv(output / "table_six_model_ablation.csv", ablation_rows)
    _write_csv(output / "table_six_model_full_modal_comparison.csv", full_rows)
    _write_csv(output / "table_per_class_metrics_s2_and_all.csv", class_rows)
    _write_confusions(output, results)
    _write_gain_plot(output, ablation_rows)

    lines = [
        "# 六模型输入消融与公平比较说明", "",
        "## S2-only与全模态消融", "",
        "| 模型 | S2-only OA | S2-only mIoU | 全模态 OA | 全模态 mIoU | mIoU增益 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in ablation_rows:
        lines.append(
            f"| {row['模型']} | {row['S2-only测试OA']:.4f} | {row['S2-only测试mIoU']:.4f} | "
            f"{row['全模态测试OA']:.4f} | {row['全模态测试mIoU']:.4f} | {row['mIoU增益']:+.4f} |"
        )
    lines += [
        "", "## FCN-ResNet50与VGG-UNet-Transformer的公平输入设计", "",
        "原始对比设置中的 FCN-ResNet50 和 VGG-UNet-Transformer 只接收 Sentinel-2 影像。"
        "为了避免本文模型因使用更多数据源而获得不公平的信息优势，本研究为两种网络增加了完全相同的多模态输入头。",
        "",
        "该输入头首先在10 m统一网格上拼接S2 19通道、S1 7通道、高分辨率RGB 3通道、"
        "VIIRS 3通道、建筑功能与高度14通道、道路13通道以及POI/AOI 20通道。"
        "OSM语义文本使用2层Transformer编码为128维向量，再投影为16个空间通道并广播至整幅patch。"
        "由此得到95通道联合特征，随后通过共享的早期融合模块压缩为48通道，再分别送入FCN-ResNet50和"
        "VGG-UNet-Transformer。两种网络使用相同输入头、相同训练/验证/测试空间划分和相同评价指标，"
        "因此全模态实验中的差异主要反映分割主干结构的差异，而不是输入数据可用性的差异。",
        "",
        "CNN、FCNN和Transformer基线的全模态版本同样使用该共享输入头；S2-only版本则只使用19通道S2特征。",
        "",
        "## 统计解释限制", "",
        "以上结果均为随机种子42的一次训练。可以报告测试集上的数值差异，但若要使用“显著优于”等统计措辞，"
        "仍需进行至少3个随机种子重复训练或配对空间块检验。",
    ]
    (output / "FAIR_COMPARISON_AND_ABLATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "ablation": ablation_rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
