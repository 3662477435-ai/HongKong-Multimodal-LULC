from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image

from .labels import CLASS_NAMES, IGNORE_INDEX, LEAF_TO_OFFICIAL


# Tableau categorical palette: consistent, high-contrast and substantially
# more color-vision-deficiency friendly than a rainbow ramp.
CLASS_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]

MODELS = (
    (
        "ours_height_hierarchical",
        "本文分层多模态模型",
        "OA=0.8554, mIoU=0.4753",
    ),
    ("fcn_resnet50", "FCN-ResNet50", "OA=0.7549, mIoU=0.2639"),
    (
        "vgg_unet_transformer",
        "VGG-UNet-Transformer",
        "OA=0.7596, mIoU=0.2098",
    ),
)

ALL_MODALITY_MODELS = (
    (
        "fcn_resnet50_all_modalities",
        "FCN-ResNet50（全模态）",
        "OA=0.7610, mIoU=0.2631",
    ),
    (
        "vgg_unet_transformer_all_modalities",
        "VGG-UNet-Transformer（全模态）",
        "OA=0.8171, mIoU=0.3272",
    ),
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 10,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def load_map(path: Path, stride: int) -> tuple[np.ma.MaskedArray, tuple[float, ...]]:
    with rasterio.open(path) as src:
        array = src.read(1)[::stride, ::stride]
        bounds = src.bounds
    # Express map coordinates in kilometres for the optional scale bar.
    extent = (bounds.left / 1000, bounds.right / 1000, bounds.bottom / 1000, bounds.top / 1000)
    return np.ma.masked_equal(array, 0), extent


def load_blu_map(path: Path, stride: int) -> tuple[np.ma.MaskedArray, tuple[float, ...]]:
    """Load the aligned original BLU leaves and project them to official10."""
    with rasterio.open(path) as src:
        leaf = src.read(1)[::stride, ::stride]
        bounds = src.bounds
    official = np.zeros(leaf.shape, dtype=np.uint8)
    valid = leaf != IGNORE_INDEX
    official[valid] = LEAF_TO_OFFICIAL[leaf[valid]] + 1
    extent = (bounds.left / 1000, bounds.right / 1000, bounds.bottom / 1000, bounds.top / 1000)
    return np.ma.masked_equal(official, 0), extent


def decorate_map(ax, array, extent, title: str, subtitle: str, panel: str | None = None) -> None:
    cmap = ListedColormap(CLASS_COLORS)
    cmap.set_bad("white", alpha=0.0)
    ax.imshow(array, cmap=cmap, vmin=1, vmax=10, interpolation="nearest", extent=extent)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{title}\n{subtitle}", pad=5, fontweight="semibold")
    if panel:
        ax.text(0.01, 0.99, panel, transform=ax.transAxes, va="top", ha="left",
                fontsize=11, fontweight="bold", color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.5})
    # 10 km scale bar and north arrow, in projected kilometre coordinates.
    x0 = extent[0] + 0.04 * (extent[1] - extent[0])
    y0 = extent[2] + 0.045 * (extent[3] - extent[2])
    ax.plot([x0, x0 + 10], [y0, y0], color="black", lw=2.0, solid_capstyle="butt")
    ax.text(x0 + 5, y0 + 0.8, "10 km", ha="center", va="bottom", fontsize=7)
    ax.annotate("N", xy=(extent[1] - 3.0, extent[3] - 2.0),
                xytext=(extent[1] - 3.0, extent[3] - 8.0),
                ha="center", va="center", fontsize=8, fontweight="bold",
                arrowprops={"arrowstyle": "-|>", "lw": 1.2, "color": "black"})


def legend_handles() -> list[Patch]:
    return [Patch(facecolor=color, edgecolor="none", label=f"{i + 1} {name}")
            for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS))]


def export_individual(input_dir: Path, output_dir: Path) -> list[Path]:
    outputs = []
    for stem, title, metric in MODELS + ALL_MODALITY_MODELS:
        path = input_dir / f"{stem}_official10_classification.tif"
        if not path.exists():
            continue
        array, extent = load_map(path, stride=2)
        fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=False)
        decorate_map(ax, array, extent, title, metric)
        fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False,
                   fontsize=7, bbox_to_anchor=(0.5, 0.005), handlelength=1.2,
                   columnspacing=1.1, handletextpad=0.4)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.13)
        output = output_dir / f"{stem}_masked_classification.png"
        fig.savefig(output, dpi=300, pil_kwargs={"compress_level": 6})
        plt.close(fig)
        outputs.append(output)
    return outputs


def export_combined(input_dir: Path, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), constrained_layout=False)
    for panel, ax, (stem, title, metric) in zip("abc", axes, MODELS):
        array, extent = load_map(
            input_dir / f"{stem}_official10_classification.tif", stride=3
        )
        decorate_map(ax, array, extent, title, metric, panel)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.01), handlelength=1.3,
               columnspacing=1.4, handletextpad=0.45)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.15, wspace=0.025)
    output = output_dir / "comparison_all_three_models_masked.png"
    fig.savefig(output, dpi=300, pil_kwargs={"compress_level": 6})
    plt.close(fig)
    return output


def export_blu_individual(blu_path: Path, output_dir: Path) -> Path:
    array, extent = load_blu_map(blu_path, stride=2)
    fig, ax = plt.subplots(figsize=(7.2, 5.5), constrained_layout=False)
    decorate_map(ax, array, extent, "原始 BLU 参考标签", "官方一级组（10类）")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False,
               fontsize=7, bbox_to_anchor=(0.5, 0.005), handlelength=1.2,
               columnspacing=1.1, handletextpad=0.4)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.94, bottom=0.13)
    output = output_dir / "original_BLU_official10.png"
    fig.savefig(output, dpi=300, pil_kwargs={"compress_level": 6})
    plt.close(fig)
    return output


def export_blu_and_models(input_dir: Path, blu_path: Path, output_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.2), constrained_layout=False)
    blu, extent = load_blu_map(blu_path, stride=3)
    decorate_map(axes[0], blu, extent, "原始 BLU 参考标签", "官方一级组（10类）", "a")
    for panel, ax, (stem, title, metric) in zip("bcd", axes[1:], MODELS):
        array, extent = load_map(
            input_dir / f"{stem}_official10_classification.tif", stride=3
        )
        decorate_map(ax, array, extent, title, metric, panel)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.01), handlelength=1.3,
               columnspacing=1.4, handletextpad=0.45)
    fig.subplots_adjust(left=0.008, right=0.992, top=0.91, bottom=0.15, wspace=0.02)
    output = output_dir / "comparison_original_BLU_and_three_models.png"
    fig.savefig(output, dpi=300, pil_kwargs={"compress_level": 6})
    plt.close(fig)
    return output


def export_blu_and_five_models(input_dir: Path, blu_path: Path, output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2), constrained_layout=False)
    flat = axes.ravel()
    blu, extent = load_blu_map(blu_path, stride=3)
    decorate_map(flat[0], blu, extent, "原始 BLU 参考标签", "官方一级组（10类）", "a")
    ordered = (
        MODELS[0],
        MODELS[1],
        ALL_MODALITY_MODELS[0],
        MODELS[2],
        ALL_MODALITY_MODELS[1],
    )
    for panel, ax, (stem, title, metric) in zip("bcdef", flat[1:], ordered):
        array, extent = load_map(
            input_dir / f"{stem}_official10_classification.tif", stride=3
        )
        decorate_map(ax, array, extent, title, metric, panel)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.008), handlelength=1.3,
               columnspacing=1.4, handletextpad=0.45)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.95, bottom=0.105,
                        wspace=0.025, hspace=0.08)
    output = output_dir / "comparison_original_BLU_and_five_models.png"
    fig.savefig(output, dpi=300, pil_kwargs={"compress_level": 6})
    plt.close(fig)
    return output


def export_blu_ours_and_two_all_models(
    input_dir: Path, blu_path: Path, output_dir: Path
) -> Path:
    """Main-paper comparison: truth, proposed model and two same-data baselines."""
    fig, axes = plt.subplots(1, 4, figsize=(18.0, 5.2), constrained_layout=False)
    blu, extent = load_blu_map(blu_path, stride=3)
    decorate_map(axes[0], blu, extent, "原始 BLU 参考标签", "官方一级组（10类）", "a")
    selected = (MODELS[0], ALL_MODALITY_MODELS[0], ALL_MODALITY_MODELS[1])
    for panel, ax, (stem, title, metric) in zip("bcd", axes[1:], selected):
        array, extent = load_map(
            input_dir / f"{stem}_official10_classification.tif", stride=3
        )
        decorate_map(ax, array, extent, title, metric, panel)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.01), handlelength=1.3,
               columnspacing=1.4, handletextpad=0.45)
    fig.subplots_adjust(left=0.008, right=0.992, top=0.91, bottom=0.15, wspace=0.02)
    output = output_dir / "comparison_BLU_ours_FCNall_VGGall.png"
    fig.savefig(output, dpi=300, pil_kwargs={"compress_level": 6})
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--blu-label", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    outputs = export_individual(args.input_dir, args.output_dir)
    outputs.append(export_combined(args.input_dir, args.output_dir))
    if args.blu_label:
        outputs.append(export_blu_individual(args.blu_label, args.output_dir))
        outputs.append(
            export_blu_and_models(args.input_dir, args.blu_label, args.output_dir)
        )
        if all(
            (args.input_dir / f"{stem}_official10_classification.tif").exists()
            for stem, _, _ in ALL_MODALITY_MODELS
        ):
            outputs.append(
                export_blu_and_five_models(
                    args.input_dir, args.blu_label, args.output_dir
                )
            )
            outputs.append(
                export_blu_ours_and_two_all_models(
                    args.input_dir, args.blu_label, args.output_dir
                )
            )
    checks = {}
    for path in outputs:
        with Image.open(path) as image:
            checks[path.name] = {
                "size_pixels": list(image.size),
                "mode": image.mode,
                "dpi": list(image.info.get("dpi", ())),
                "bytes": path.stat().st_size,
            }
    report = {"status": "passed", "figures": checks, "palette": CLASS_COLORS}
    (args.output_dir / "figure_export_qc.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
