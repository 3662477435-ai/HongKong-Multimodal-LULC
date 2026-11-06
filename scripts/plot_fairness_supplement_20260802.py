from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    root = repo / "outputs/full/eight_class_fairness_supplement_20260802"
    accuracy = pd.read_csv(root / "fairness_accuracy_comparison.csv")
    profile = pd.read_csv(root / "profiling_all/model_complexity_and_inference_speed.csv")
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    labels = {
        "本文模型": "本文模型",
        "FCNN原始预算": "FCNN-20",
        "FCNN同等预算": "FCNN-47",
        "FCN-ResNet50 48→3": "R50-3",
        "FCN-ResNet50直接48通道": "R50-48",
        "VGG-UNet-Transformer 48→3": "VGG-3",
        "VGG-UNet-Transformer直接48通道": "VGG-48",
    }
    colors = ["#B2182B", "#4C78A8", "#72A0C1", "#999999", "#666666", "#59A14F", "#1B7837"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.35), constrained_layout=True)
    x = np.arange(len(accuracy))
    bars = ax1.bar(x, 100 * accuracy.mIoU, color=colors, width=0.72)
    ax1.set_xticks(x, [labels[value] for value in accuracy.model], rotation=30, ha="right")
    ax1.set_ylabel("macro mIoU（%）")
    ax1.set_title("a  等预算与输入适配公平性实验", loc="left", fontweight="bold")
    ax1.set_ylim(35, 63)
    ax1.grid(axis="y", color="#dddddd", lw=0.5)
    ax1.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, accuracy.mIoU):
        ax1.text(bar.get_x() + bar.get_width()/2, 100*value + 0.5, f"{value:.3f}",
                 ha="center", va="bottom", fontsize=6.2, rotation=90)

    merged = profile.merge(accuracy[["model", "mIoU"]], left_on="name", right_on="model", how="left")
    merged = merged[np.isfinite(merged.mIoU)]
    sizes = 25 + 145 * merged.parameter_count / merged.parameter_count.max()
    ax2.scatter(merged.forward_gflops, 100 * merged.mIoU, s=sizes,
                c=["#B2182B" if x == "本文模型" else "#4C78A8" for x in merged.name],
                alpha=0.78, edgecolor="white", linewidth=0.7)
    short = {
        "本文模型": "本文模型", "FCNN原始预算": "FCNN-20", "FCNN同等预算": "FCNN-47",
        "FCN-ResNet50 48→3": "R50-3", "FCN-ResNet50直接48通道": "R50-48",
        "VGG-UNet-Transformer 48→3": "VGG-3",
        "VGG-UNet-Transformer直接48通道": "VGG-48",
    }
    for row in merged.itertuples():
        ax2.annotate(short.get(row.name, row.name), (row.forward_gflops, 100*row.mIoU),
                     xytext=(3, 3), textcoords="offset points", fontsize=6.2)
    ax2.set_xlabel("单个128×128 patch前向计算量（GFLOPs）")
    ax2.set_ylabel("macro mIoU（%）")
    ax2.set_title("b  精度—计算量关系", loc="left", fontweight="bold")
    ax2.grid(color="#dddddd", lw=0.5)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.text(0.03, 0.03, "气泡面积表示参数量", transform=ax2.transAxes,
             ha="left", va="bottom", fontsize=6.5, color="#555555")
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(root / f"fairness_supplement_summary.{suffix}",
                    dpi=400 if suffix == "png" else None,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
