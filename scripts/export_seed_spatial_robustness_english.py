from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


MODEL_LABELS = {
    "本文模型": "Proposed",
    "CNN": "CNN",
    "FCNN": "FCNN",
    "Transformer": "Transformer",
    "FCN-ResNet50": "FCN-ResNet50",
    "VGG-UNet-Transformer": "VGG-UNet-Transformer",
}

PROTOCOL_LABELS = {
    "5.12 km（原协议）": "5.12 km\n(main)",
    "2.56 km空间块": "2.56 km\nblocks",
    "10.24 km空间块": "10.24 km\nblocks",
    "5.12 km + 1.28 km缓冲": "5.12 km +\n1.28 km buffer",
    "西部留区验证": "Western\nleave-region-out",
}


def configure_plot() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def draw_figure(results_dir: Path, output_dir: Path, dpi: int) -> list[Path]:
    seed_summary = pd.read_csv(
        results_dir / "table_multiseed_summary.csv", encoding="utf-8-sig"
    )
    spatial = pd.read_csv(
        results_dir / "table_spatial_robustness.csv", encoding="utf-8-sig"
    )
    correlation = pd.read_csv(
        results_dir / "table_spatial_label_composition_correlogram.csv",
        encoding="utf-8-sig",
    )

    configure_plot()
    # Extra height is reserved for the legends below panels a and b.  Keeping
    # them outside the plotting areas avoids covering data or category labels.
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.3), constrained_layout=True)

    x = np.arange(len(seed_summary))
    for key, colour, marker in (
        ("OA", "#4C78A8", "o"),
        ("mIoU", "#F58518", "s"),
        ("F1", "#54A24B", "^"),
    ):
        axes[0].errorbar(
            x,
            seed_summary[f"{key}_均值"].to_numpy(),
            yerr=seed_summary[f"{key}_SD"].to_numpy(),
            fmt=marker,
            color=colour,
            capsize=3,
            label=key,
        )
    axes[0].set_xticks(
        x,
        [MODEL_LABELS[value] for value in seed_summary["模型"]],
        rotation=20,
        ha="right",
        fontsize=7.4,
    )
    axes[0].set_ylabel("Mean ± SD across 3 seeds")
    axes[0].set_title("a  Training stability across seeds", loc="left", fontweight="bold")
    axes[0].legend(
        frameon=False,
        ncol=3,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        borderaxespad=0,
        handletextpad=0.45,
        columnspacing=1.35,
    )

    width = 0.24
    positions = np.arange(len(spatial))
    for offset, key, colour in (
        (-width, "OA", "#4C78A8"),
        (0, "mIoU", "#F58518"),
        (width, "F1", "#54A24B"),
    ):
        axes[1].bar(positions + offset, spatial[key], width, label=key, color=colour)
    axes[1].set_xticks(
        positions,
        [PROTOCOL_LABELS[value] for value in spatial["空间验证协议"]],
        rotation=15,
        ha="right",
        fontsize=6.7,
    )
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("b  Spatial-protocol sensitivity", loc="left", fontweight="bold")
    axes[1].legend(
        frameon=False,
        ncol=3,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.30),
        borderaxespad=0,
        handletextpad=0.45,
        columnspacing=1.35,
    )

    axes[2].plot(
        correlation["距离中点_km"],
        correlation["八类组成平均标准化相关"],
        marker="o",
        ms=3,
        color="#4C78A8",
    )
    axes[2].axhline(0.1, color="#E45756", ls="--", lw=1, label="Reference 0.1")
    axes[2].axvline(
        5.12,
        color="#777777",
        ls=":",
        lw=1,
        label="Original 5.12 km block",
    )
    axes[2].set_xlabel("Patch-center distance (km)")
    axes[2].set_ylabel("Mean standardized correlation")
    axes[2].set_title(
        "c  Spatial correlation of label composition", loc="left", fontweight="bold"
    )
    axes[2].legend(frameon=False, fontsize=7)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "figure_seed_spatial_robustness_english_600dpi"
    outputs = [stem.with_suffix(suffix) for suffix in (".png", ".tif", ".pdf", ".svg")]
    fig.savefig(outputs[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(
        outputs[1],
        dpi=dpi,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(outputs[2], bbox_inches="tight")
    fig.savefig(outputs[3], bbox_inches="tight")
    plt.close(fig)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    outputs = draw_figure(args.results_dir, args.output_dir, args.dpi)
    checks = {}
    for path in outputs[:2]:
        with Image.open(path) as image:
            checks[path.name] = {
                "pixels": list(image.size),
                "dpi": [float(value) for value in image.info.get("dpi", ())],
                "mode": image.mode,
                "bytes": path.stat().st_size,
            }
    report = {
        "source_plot_script": str(
            Path(__file__).with_name("summarize_reviewer_revision_experiments.py")
        ),
        "source_tables": [
            str(args.results_dir / "table_multiseed_summary.csv"),
            str(args.results_dir / "table_spatial_robustness.csv"),
            str(args.results_dir / "table_spatial_label_composition_correlogram.csv"),
        ],
        "outputs": [str(path) for path in outputs],
        "checks": checks,
    }
    report_path = args.output_dir / "figure_seed_spatial_robustness_english_qc.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
