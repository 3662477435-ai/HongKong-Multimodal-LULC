from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image


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
CLASS_COLORS = (
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AC",
)
MODEL_SPECS = (
    ("ours_height_hierarchical", "本文分层多模态模型", "本文分层多模态模型"),
    ("cnn_all_modalities", "CNN（全模态）", "CNN"),
    ("fcnn_all_modalities", "FCNN（全模态）", "FCNN"),
    ("transformer_all_modalities", "Transformer（全模态）", "Transformer"),
    ("fcn_resnet50_all_modalities", "FCN-ResNet50（全模态）", "FCN-ResNet50"),
    (
        "vgg_unet_transformer_all_modalities",
        "VGG-UNet-Transformer（全模态）",
        "VGG-UNet-Transformer",
    ),
)


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 10,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_full_metrics(path: Path) -> dict[str, tuple[float, float]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["模型"]: (float(row["测试OA"]), float(row["测试mIoU"]))
        for row in rows
        if row["变体代码"] == "full"
    }


def load_map(path: Path, stride: int) -> tuple[np.ma.MaskedArray, tuple[float, ...]]:
    with rasterio.open(path) as source:
        values = source.read(1)[::stride, ::stride]
        bounds = source.bounds
    extent = (
        bounds.left / 1000.0,
        bounds.right / 1000.0,
        bounds.bottom / 1000.0,
        bounds.top / 1000.0,
    )
    return np.ma.masked_equal(values, 0), extent


def decorate_map(
    ax,
    values: np.ma.MaskedArray,
    extent: tuple[float, ...],
    title: str,
    oa: float,
    miou: float,
    panel: str,
) -> None:
    color_map = ListedColormap(CLASS_COLORS)
    color_map.set_bad("white", alpha=0.0)
    ax.imshow(
        values,
        cmap=color_map,
        vmin=1,
        vmax=10,
        interpolation="nearest",
        extent=extent,
        rasterized=True,
    )
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{title}\nOA={oa:.4f}, mIoU={miou:.4f}", pad=4, fontweight="semibold")
    ax.text(
        0.005,
        0.985,
        panel,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
    )
    width = extent[1] - extent[0]
    height = extent[3] - extent[2]
    scale_x = extent[0] + 0.04 * width
    scale_y = extent[2] + 0.045 * height
    ax.plot(
        [scale_x, scale_x + 10],
        [scale_y, scale_y],
        color="black",
        linewidth=1.8,
        solid_capstyle="butt",
    )
    ax.text(scale_x + 5, scale_y + 0.7, "10 km", ha="center", va="bottom", fontsize=6.5)
    north_x = extent[1] - 3.0
    ax.annotate(
        "N",
        xy=(north_x, extent[3] - 2.0),
        xytext=(north_x, extent[3] - 8.0),
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "linewidth": 1.1, "color": "black"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制六种全模态模型的2×3香港分类结果图。")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--metrics-table", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--stride", type=int, default=3)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    metrics = read_full_metrics(args.metrics_table)
    missing_metrics = [key for _, _, key in MODEL_SPECS if key not in metrics]
    missing_maps = [
        str(args.input_dir / f"{stem}_official10_classification.tif")
        for stem, _, _ in MODEL_SPECS
        if not (args.input_dir / f"{stem}_official10_classification.tif").exists()
    ]
    if missing_metrics or missing_maps:
        raise FileNotFoundError(
            f"Missing metrics={missing_metrics}; missing maps={missing_maps}"
        )

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.3), constrained_layout=False)
    source_records = []
    for panel, ax, (stem, title, metric_key) in zip("abcdef", axes.ravel(), MODEL_SPECS):
        source = args.input_dir / f"{stem}_official10_classification.tif"
        values, extent = load_map(source, args.stride)
        oa, miou = metrics[metric_key]
        decorate_map(ax, values, extent, title, oa, miou, panel)
        source_records.append(
            {
                "panel": panel,
                "model": metric_key,
                "raster": str(source),
                "OA": oa,
                "mIoU": miou,
            }
        )
    handles = [
        Patch(facecolor=color, edgecolor="none", label=f"{index + 1} {name}")
        for index, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS))
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=7.5,
        bbox_to_anchor=(0.5, 0.008),
        handlelength=1.2,
        columnspacing=1.25,
        handletextpad=0.4,
    )
    fig.subplots_adjust(
        left=0.012,
        right=0.988,
        top=0.965,
        bottom=0.105,
        wspace=0.025,
        hspace=0.075,
    )
    stem = args.output_dir / "comparison_six_full_modal_models_2x3"
    fig.savefig(stem.with_suffix(".png"), dpi=300, pil_kwargs={"compress_level": 6})
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    fig.savefig(
        stem.with_suffix(".tif"),
        dpi=300,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)

    with Image.open(stem.with_suffix(".png")) as preview:
        preview_qc = {
            "size_pixels": list(preview.size),
            "dpi": list(preview.info.get("dpi", ())),
            "mode": preview.mode,
        }
    report = {
        "status": "passed",
        "layout": "2x3",
        "backend": "Python/Matplotlib",
        "BLU_zero": "masked as white/transparent map background",
        "context_margin_pixels": 16,
        "preview": preview_qc,
        "panels": source_records,
        "outputs": [str(stem.with_suffix(suffix)) for suffix in (".png", ".pdf", ".svg", ".tif")],
    }
    (args.output_dir / "comparison_six_full_modal_models_2x3_qc.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
