from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle
from PIL import Image

# Figure export only needs immutable class metadata. Keeping it local prevents
# Matplotlib-only jobs from loading the PyTorch/CUDA runtime on Windows.
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
CLASS_NAMES_8_EN = (
    "Residential", "Industrial/warehouse", "Public/institutional and recreation",
    "Transportation", "Other built-up", "Agriculture",
    "Woodland/shrubland/grassland/wetland", "Water",
)


MODEL_ORDER = (
    "本文模型", "CNN", "FCNN", "Transformer", "FCN-ResNet50", "VGG-UNet-Transformer"
)
MODEL_ORDER_EN = (
    "Proposed model", "CNN", "FCNN", "Transformer", "FCN-ResNet50", "VGG-UNet-Transformer"
)
CLASS_NAMES_8_SHORT = (
    "住宅", "工业", "公共", "交通", "其他", "农业", "林地等", "水体"
)
CLASS_NAMES_8_SHORT_EN = (
    "Residential", "Industrial", "Public", "Transportation",
    "Other built-up", "Agriculture", "Woodland etc.", "Water",
)
MODEL_STEMS = (
    "ours", "cnn", "fcnn", "transformer", "fcn_resnet50", "vgg_unet_transformer"
)
VARIANTS = (
    ("full", "全部模态", (1, 1, 1, 1, 1, 1)),
    ("s2", "仅S2", (1, 0, 0, 0, 0, 0)),
    ("no_osm_roads", "去除POI/AOI/道路", (1, 1, 1, 1, 1, 0)),
    ("no_building", "去除建筑特征", (1, 1, 1, 1, 0, 1)),
    ("no_s1", "去除S1", (1, 0, 1, 1, 1, 1)),
    ("no_nighttime", "去除夜间灯光", (1, 1, 1, 0, 1, 1)),
    ("no_highres", "去除高分影像", (1, 1, 0, 1, 1, 1)),
)
CLASS_COLORS = (
    "#4E79A7", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#2980B9",
)


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        # Keep a CJK-capable face first; modality presence uses a broadly
        # supported solid-circle glyph instead of a font-sensitive check mark.
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial"],
        "font.size": 7,
        "axes.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save_pub(fig, stem: Path, dpi: int = 600) -> list[str]:
    outputs = []
    for suffix in (".png", ".pdf", ".svg", ".tif"):
        path = stem.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight"}
        if suffix in (".png", ".tif"):
            kwargs["dpi"] = dpi
        if suffix == ".tif":
            kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
        fig.savefig(path, **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def load_map(path: Path, stride: int = 3):
    with rasterio.open(path) as source:
        values = source.read(1)[::stride, ::stride]
        bounds = source.bounds
    extent = (
        bounds.left / 1000, bounds.right / 1000,
        bounds.bottom / 1000, bounds.top / 1000,
    )
    return np.ma.masked_equal(values, 0), extent


def decorate_map(ax, values, extent, title: str, panel: str) -> None:
    colour_map = ListedColormap(CLASS_COLORS)
    colour_map.set_bad("white", alpha=0.0)
    ax.imshow(
        values, cmap=colour_map, vmin=1, vmax=8, interpolation="nearest",
        extent=extent, rasterized=True,
    )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontweight="semibold", pad=3)
    ax.text(
        0.006, 0.985, panel, transform=ax.transAxes,
        va="top", ha="left", fontsize=10, fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
    )
    width = extent[1] - extent[0]
    height = extent[3] - extent[2]
    x = extent[0] + 0.04 * width
    # Keep the scale bar in the clear lower margin of every panel.  The
    # previous 4.5% offset could sit on the southern map polygons; 2.0% moves
    # it slightly downward without letting it intrude into the adjacent row.
    y = extent[2] + 0.020 * height
    ax.plot([x, x + 10], [y, y], color="black", linewidth=1.6)
    ax.text(x + 5, y + 0.7, "10 km", ha="center", va="bottom", fontsize=6)
    north_x = extent[1] - 3
    ax.annotate(
        "N", xy=(north_x, extent[3] - 2), xytext=(north_x, extent[3] - 8),
        ha="center", fontsize=7, fontweight="bold",
        arrowprops={"arrowstyle": "-|>", "linewidth": 1.0, "color": "black"},
    )


def map_figure(map_dir: Path, overall: pd.DataFrame, output: Path, english: bool = False) -> list[str]:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.3))
    model_labels = MODEL_ORDER_EN if english else MODEL_ORDER
    class_labels = CLASS_NAMES_8_EN if english else CLASS_NAMES_8
    for panel, ax, label, stem in zip("abcdef", axes.ravel(), model_labels, MODEL_STEMS):
        values, extent = load_map(map_dir / f"{stem}_official8_classification.tif")
        decorate_map(ax, values, extent, label, panel)
    handles = [
        Patch(facecolor=colour, edgecolor="none", label=f"{index + 1} {name}")
        for index, (name, colour) in enumerate(zip(class_labels, CLASS_COLORS))
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, 0.006), fontsize=7.3,
        handlelength=1.2, handletextpad=0.4, columnspacing=1.2,
    )
    fig.subplots_adjust(
        left=0.012, right=0.988, top=0.96, bottom=0.115,
        wspace=0.025, hspace=0.11,
    )
    suffix = "_english" if english else ""
    return save_pub(fig, output / f"figure_8class_six_model_maps_2x3{suffix}")


def confusion_figure(
    confusion: pd.DataFrame,
    output: Path,
    english: bool = False,
) -> list[str]:
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 8.0))
    image = None
    short = CLASS_NAMES_8_SHORT_EN if english else CLASS_NAMES_8_SHORT
    display_models = MODEL_ORDER_EN if english else MODEL_ORDER
    for panel, ax, source_label, display_label in zip(
        "abcdef", axes.ravel(), MODEL_ORDER, display_models
    ):
        subset = confusion.loc[confusion["模型"] == source_label]
        matrix = subset.pivot(
            index="真实类别", columns="预测类别", values="行归一化比例"
        ).reindex(index=CLASS_NAMES_8, columns=CLASS_NAMES_8).to_numpy()
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", aspect="equal")
        for row in range(8):
            for column in range(8):
                value = matrix[row, column]
                if value >= 0.005:
                    ax.text(
                        column, row, f"{value:.2f}", ha="center", va="center",
                        fontsize=5.1, color="white" if value > 0.55 else "#202020",
                    )
        ax.set_xticks(range(8), short, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(8), short, fontsize=6)
        ax.set_xlabel("Predicted class" if english else "预测类别", fontsize=7)
        ax.set_ylabel("True class" if english else "真实类别", fontsize=7)
        ax.set_title(display_label, fontweight="semibold")
        ax.text(
            -0.17, 1.06, panel, transform=ax.transAxes,
            fontsize=10, fontweight="bold",
        )
    figure_title = (
        "Eight-class spatial-test confusion matrices for six models\n"
        "(commercial/office and barren land excluded)"
        if english
        else "六种模型的8类空间测试混淆矩阵（商业办公与荒地已排除）"
    )
    fig.suptitle(figure_title, y=0.995, fontsize=11, fontweight="semibold")
    fig.subplots_adjust(
        left=0.07, right=0.885, top=0.92 if english else 0.94, bottom=0.08,
        wspace=0.30, hspace=0.32,
    )
    if image is not None:
        # Give the shared colour bar its own axis.  Otherwise a later
        # subplots_adjust call can move the third column underneath it.
        colour_bar_axis = fig.add_axes([0.915, 0.22, 0.016, 0.56])
        colour_bar = fig.colorbar(image, cax=colour_bar_axis)
        colour_bar.set_label("Row-normalized proportion" if english else "行归一化比例")
    suffix = "_english" if english else ""
    return save_pub(fig, output / f"figure_8class_confusion_matrices_2x3{suffix}")


def draw_ablation_table(ablation: pd.DataFrame, output: Path) -> list[str]:
    columns = (
        ("模型", 2.30), ("组合", 0.62), ("S2", 0.52), ("S1", 0.52),
        ("高分影像", 0.78), ("夜间灯光", 0.78), ("建筑特征", 0.78),
        ("POI/AOI/道路", 1.12), ("OA (%)", 0.82), ("mIoU", 0.78),
        ("ΔmIoU", 0.84),
    )
    total_width = sum(width for _, width in columns)
    rows = len(MODEL_ORDER) * len(VARIANTS)
    row_height = 0.31
    header_height = 0.78
    fig, ax = plt.subplots(figsize=(10.0, 14.0))
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, header_height + rows * row_height)
    ax.axis("off")
    header_colour = "#626977"
    full_colour = "#E8F0F7"
    grid_colour = "#AAB2BD"
    x_positions = np.cumsum([0] + [width for _, width in columns])
    for index, (label, width) in enumerate(columns):
        x = x_positions[index]
        ax.add_patch(Rectangle(
            (x, rows * row_height), width, header_height,
            facecolor=header_colour, edgecolor="white", linewidth=0.6,
        ))
        ax.text(
            x + width / 2, rows * row_height + header_height / 2,
            label, ha="center", va="center", color="white",
            fontsize=7.2, fontweight="semibold",
        )
    row_index = 0
    for model in MODEL_ORDER:
        model_rows = ablation.loc[ablation["模型"] == model].set_index("变体代码")
        block_top = rows * row_height - row_index * row_height
        block_height = len(VARIANTS) * row_height
        ax.add_patch(Rectangle(
            (0, block_top - block_height), columns[0][1], block_height,
            facecolor="#EDF2F7", edgecolor=grid_colour, linewidth=0.7,
        ))
        ax.text(
            columns[0][1] / 2, block_top - block_height / 2,
            model.replace("VGG-UNet-", "VGG-UNet-\n"),
            ha="center", va="center", fontsize=7.0, fontweight="semibold",
        )
        for combination, (variant, _, flags) in enumerate(VARIANTS, start=1):
            y = rows * row_height - (row_index + 1) * row_height
            row = model_rows.loc[variant]
            face = full_colour if variant == "full" else "white"
            for column_index in range(1, len(columns)):
                x = x_positions[column_index]
                width = columns[column_index][1]
                ax.add_patch(Rectangle(
                    (x, y), width, row_height,
                    facecolor=face, edgecolor=grid_colour, linewidth=0.45,
                ))
            values = [
                str(combination),
                *["●" if flag else "" for flag in flags],
                f"{100 * row['OA']:.2f}",
                f"{row['macro_mIoU']:.4f}",
                "—" if variant == "full" else f"{row['相对全模态_mIoU变化']:+.4f}",
            ]
            for column_index, value in enumerate(values, start=1):
                x = x_positions[column_index]
                width = columns[column_index][1]
                colour = "#C0392B" if (
                    column_index == len(columns) - 1
                    and variant != "full"
                    and row["相对全模态_mIoU变化"] < 0
                ) else "#1F2933"
                ax.text(
                    x + width / 2, y + row_height / 2, value,
                    ha="center", va="center", fontsize=6.4, color=colour,
                    fontweight="bold" if variant == "full" else "normal",
                )
            row_index += 1
    ax.text(
        0, -0.28,
        "注：ΔmIoU=当前变体mIoU−同一模型全模态mIoU；红色负值表示移除该模态后下降。",
        ha="left", va="top", fontsize=6.5, color="#4A5568",
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.995, bottom=0.035)
    return save_pub(fig, output / "table_8class_extended_modality_ablation", dpi=400)


def draw_overall_table(overall: pd.DataFrame, output: Path) -> list[str]:
    rows = []
    for label in MODEL_ORDER:
        row = overall.loc[overall["模型"] == label].iloc[0]
        rows.append([
            label,
            f"{row['OA']:.4f}\n[{row['OA_95CI下限']:.4f}, {row['OA_95CI上限']:.4f}]",
            f"{row['macro_mIoU']:.4f}\n[{row['mIoU_95CI下限']:.4f}, {row['mIoU_95CI上限']:.4f}]",
            f"{row['macro_F1']:.4f}\n[{row['F1_95CI下限']:.4f}, {row['F1_95CI上限']:.4f}]",
        ])
    fig, ax = plt.subplots(figsize=(8.3, 3.5))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=("模型", "OA（95% CI）", "macro mIoU（95% CI）", "macro-F1（95% CI）"),
        cellLoc="center", colLoc="center", loc="center",
        colWidths=(0.30, 0.23, 0.25, 0.24),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1, 1.75)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#AAB2BD")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor("#626977")
            cell.set_text_props(color="white", fontweight="semibold")
        elif row == 1:
            cell.set_facecolor("#E8F0F7")
            cell.set_text_props(fontweight="semibold")
        else:
            cell.set_facecolor("white")
    ax.set_title(
        "八类任务中六种统一全模态模型的空间独立测试结果",
        fontsize=10, fontweight="semibold", pad=8,
    )
    return save_pub(fig, output / "table_8class_overall_results_with_ci", dpi=400)


def draw_per_class_iou(per_class: pd.DataFrame, output: Path) -> list[str]:
    matrix = per_class.pivot(index="类别", columns="模型", values="IoU").reindex(
        index=CLASS_NAMES_8, columns=MODEL_ORDER
    )
    values = matrix.to_numpy()
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=max(0.8, np.nanmax(values)))
    for row in range(values.shape[0]):
        best = int(np.nanargmax(values[row]))
        for column in range(values.shape[1]):
            value = values[row, column]
            ax.text(
                column, row, f"{value:.3f}",
                ha="center", va="center", fontsize=6.7,
                color="white" if value > 0.58 else "#202020",
                fontweight="bold" if column == best else "normal",
            )
    ax.set_xticks(range(len(MODEL_ORDER)), MODEL_ORDER, rotation=25, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES_8)), CLASS_NAMES_8)
    ax.set_title("六种全模态模型的逐类IoU（每行最优值加粗）",
                 fontsize=10, fontweight="semibold")
    colour_bar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colour_bar.set_label("IoU")
    fig.subplots_adjust(left=0.21, right=0.93, top=0.92, bottom=0.22)
    return save_pub(fig, output / "table_8class_per_class_iou_all_models", dpi=400)


def draw_ours_per_class_table(
    per_class: pd.DataFrame, output: Path
) -> list[str]:
    subset = (
        per_class.loc[per_class["模型"] == "本文模型"]
        .set_index("类别")
        .reindex(CLASS_NAMES_8)
    )
    rows = []
    for class_name, row in subset.iterrows():
        rows.append([
            class_name,
            f"{int(row['测试像元数']):,}",
            (
                f"{row['Precision']:.3f}\n"
                f"[{row['Precision_95CI下限']:.3f}, {row['Precision_95CI上限']:.3f}]"
            ),
            (
                f"{row['Recall']:.3f}\n"
                f"[{row['Recall_95CI下限']:.3f}, {row['Recall_95CI上限']:.3f}]"
            ),
            f"{row['F1']:.3f}\n[{row['F1_95CI下限']:.3f}, {row['F1_95CI上限']:.3f}]",
            f"{row['IoU']:.3f}\n[{row['IoU_95CI下限']:.3f}, {row['IoU_95CI上限']:.3f}]",
        ])
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=(
            "类别", "测试像元数", "Precision（95% CI）",
            "Recall（95% CI）", "F1（95% CI）", "IoU（95% CI）",
        ),
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=(0.22, 0.13, 0.18, 0.18, 0.16, 0.16),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.0)
    table.scale(1, 1.72)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#AAB2BD")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#626977")
            cell.set_text_props(color="white", fontweight="semibold")
        elif row % 2 == 1:
            cell.set_facecolor("#F4F7FA")
    ax.set_title(
        "本文模型逐类精度与空间分组 Bootstrap 95% 置信区间",
        fontsize=10,
        fontweight="semibold",
        pad=8,
    )
    return save_pub(fig, output / "table_8class_ours_per_class_metrics_with_ci", dpi=400)


def draw_significance_table(
    significance: pd.DataFrame, output: Path
) -> list[str]:
    rows = []
    for _, row in significance.iterrows():
        rows.append([
            row["指标"],
            row["比较"].replace("本文模型 vs ", ""),
            (
                f"{row['差值_本文减基线']:+.4f}\n"
                f"[{row['差值95CI下限']:+.4f}, {row['差值95CI上限']:+.4f}]"
            ),
            f"{row['精确配对置换p值']:.4f}",
            f"{row['Holm校正p值']:.4f}",
            "是" if bool(row["Holm_0.05显著"]) else "否",
        ])
    fig, ax = plt.subplots(figsize=(10.5, 8.4))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=(
            "指标", "基线模型", "本文模型−基线（95% CI）",
            "配对置换 p", "Holm 校正 p", "显著",
        ),
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=(0.12, 0.23, 0.26, 0.13, 0.14, 0.10),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(6.8)
    table.scale(1, 1.48)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#AAB2BD")
        cell.set_linewidth(0.5)
        if row == 0:
            cell.set_facecolor("#626977")
            cell.set_text_props(color="white", fontweight="semibold")
        elif row % 2 == 1:
            cell.set_facecolor("#F4F7FA")
        if row > 0 and column == 5 and cell.get_text().get_text() == "是":
            cell.set_facecolor("#DDEEDF")
            cell.set_text_props(color="#26643B", fontweight="semibold")
    ax.set_title(
        "本文模型与五种基线的空间配对显著性检验",
        fontsize=10,
        fontweight="semibold",
        pad=8,
    )
    ax.text(
        0.0,
        0.01,
        "注：以14个独立空间组为配对单位；双侧精确置换检验；每项指标内对5次比较作Holm校正。",
        transform=ax.transAxes,
        fontsize=6.7,
        color="#4A5568",
    )
    return save_pub(fig, output / "table_8class_paired_significance", dpi=400)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--english",
        action="store_true",
        help="Use English text in both the map and confusion-matrix plates.",
    )
    parser.add_argument("--english-maps", action="store_true", help="Use English titles and legend labels in the map plate.")
    parser.add_argument("--maps-only", action="store_true", help="Export only the six-model map plate.")
    parser.add_argument(
        "--core-only",
        action="store_true",
        help="Export only the map and confusion-matrix plates.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    overall = pd.read_csv(
        args.results_dir / "table_overall_metrics_with_ci.csv", encoding="utf-8-sig"
    )
    outputs = []
    outputs += map_figure(
        args.map_dir,
        overall,
        args.output_dir,
        english=args.english or args.english_maps,
    )
    if not args.maps_only:
        confusion = pd.read_csv(
            args.results_dir / "table_confusion_long.csv", encoding="utf-8-sig"
        )
        outputs += confusion_figure(confusion, args.output_dir, english=args.english)
        if not args.core_only:
            per_class = pd.read_csv(
                args.results_dir / "table_per_class_metrics_with_ci.csv", encoding="utf-8-sig"
            )
            ablation = pd.read_csv(
                args.results_dir / "table_ablation_all_models.csv", encoding="utf-8-sig"
            )
            significance = pd.read_csv(
                args.results_dir / "table_paired_significance.csv", encoding="utf-8-sig"
            )
            outputs += draw_ablation_table(ablation, args.output_dir)
            outputs += draw_overall_table(overall, args.output_dir)
            outputs += draw_per_class_iou(per_class, args.output_dir)
            outputs += draw_ours_per_class_table(per_class, args.output_dir)
            outputs += draw_significance_table(significance, args.output_dir)
    checks = {}
    for path_string in outputs:
        path = Path(path_string)
        if path.suffix.lower() in (".png", ".tif"):
            with Image.open(path) as image:
                checks[path.name] = {
                    "pixels": list(image.size),
                    "dpi": [float(value) for value in image.info.get("dpi", ())],
                    "mode": image.mode,
                    "bytes": path.stat().st_size,
                }
    contract = {
        "core_conclusion": (
            "After removing commercial/office and barren land, compare six models "
            "on the same eight-class spatial hold-out while displaying uncertainty."
        ),
        "backend": "Python/Matplotlib only",
        "archetypes": {
            "maps": "image plate + quant",
            "confusion": "quantitative grid",
            "ablation": "quantitative grid",
            "tables": "quantitative grid",
        },
        "statistics": "95% spatial-group bootstrap intervals; exact paired permutation tests in source tables",
        "image_integrity": "categorical rasters use nearest-neighbour display; excluded/BLU0 pixels are masked; no smoothing",
        "outputs": outputs,
        "preview_checks": checks,
    }
    (args.output_dir / "figure_contract_and_qc.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
