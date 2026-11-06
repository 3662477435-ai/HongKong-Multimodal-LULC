"""Create a publication-ready pixel-level validation distribution figure.

The spatial blocks determine the train/validation/test partition.  Every valid
pixel inside the retained blocks is still used to compute the pixel-level
confusion matrix and all reported land-use metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


# Eight active official classes after commercial/office and barren/rock masking.
CLASS_IDS = (0, 2, 3, 4, 5, 6, 7, 9)
CLASS_NAMES = (
    "住宅",
    "工业仓储",
    "公共机构\n与休闲",
    "交通设施",
    "其他建设\n用地",
    "农业",
    "林地等",
    "水体",
)
SPLITS = ("train", "val", "test")
SPLIT_NAMES = {"train": "训练集", "val": "验证集", "test": "测试集"}
SPLIT_COLORS = {"train": "#9BB7D4", "val": "#E7B77B", "test": "#C85A5A"}
CLASS_COLORS = (
    "#4C78A8",
    "#E45756",
    "#72B7B2",
    "#54A24B",
    "#B279A2",
    "#ECA82C",
    "#59A14F",
    "#9D9D9D",
)


def apply_style() -> None:
    # SVG text stays editable; the first available CJK font is used for labels.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "Arial",
        "DejaVu Sans",
        "Liberation Sans",
    ]
    plt.rcParams["svg.fonttype"] = "none"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["legend.frameon"] = False


def summarize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    rows: list[dict[str, object]] = []
    patch_counts: dict[str, int] = {}
    for split in SPLITS:
        selected = frame.loc[frame["split"] == split].copy()
        patch_counts[split] = int(len(selected))
        for class_id, class_name in zip(CLASS_IDS, CLASS_NAMES):
            pixels = int(selected[f"pixels_class_{class_id}"].sum())
            rows.append(
                {
                    "split": split,
                    "split_name": SPLIT_NAMES[split],
                    "class_id_original10": class_id,
                    "class_name": class_name.replace("\n", ""),
                    "pixels": pixels,
                    "area_km2": pixels * 0.0001,
                    "patches_total": patch_counts[split],
                }
            )
    table = pd.DataFrame(rows)
    table["split_total_valid_pixels"] = table.groupby("split")["pixels"].transform("sum")
    table["share_within_split_pct"] = 100 * table["pixels"] / table["split_total_valid_pixels"]
    return table, patch_counts


def draw_figure(frame: pd.DataFrame, table: pd.DataFrame, patch_counts: dict[str, int], out_base: Path) -> dict[str, object]:
    apply_style()
    fig = plt.figure(figsize=(7.2, 3.9))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.15, 1.06, 1.14),
        left=0.055,
        right=0.985,
        bottom=0.205,
        top=0.89,
        wspace=0.47,
    )
    ax_map = fig.add_subplot(grid[0, 0])
    ax_stack = fig.add_subplot(grid[0, 1])
    ax_heat = fig.add_subplot(grid[0, 2])

    # a. Spatial allocation: patches are the leakage-control unit, not the metric unit.
    for split in SPLITS:
        selected = frame.loc[frame["split"] == split]
        for row in selected.itertuples(index=False):
            ax_map.add_patch(
                Rectangle(
                    (row.col_off, row.row_off),
                    row.width,
                    row.height,
                    facecolor=SPLIT_COLORS[split],
                    edgecolor="white",
                    linewidth=0.12,
                    alpha=0.92,
                )
            )
    ax_map.set_aspect("equal")
    ax_map.invert_yaxis()
    ax_map.set_xlim(frame["col_off"].min(), (frame["col_off"] + frame["width"]).max())
    ax_map.set_ylim((frame["row_off"] + frame["height"]).max(), frame["row_off"].min())
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=SPLIT_COLORS[s], markeredgecolor="none", markersize=5, label=f"{SPLIT_NAMES[s]} (n={patch_counts[s]})")
        for s in SPLITS
    ]
    ax_map.legend(handles=handles, loc="lower left", fontsize=5.6, handletextpad=0.35, borderpad=0.1)
    ax_map.set_title("空间独立划分", fontsize=8.2, pad=5, fontweight="bold")
    ax_map.text(0.0, -0.055, "每个方块：128 × 128像元（1.28 × 1.28 km）", transform=ax_map.transAxes, fontsize=5.4, va="top")

    # b. Composition: percentages make split imbalance immediately visible.
    bottoms = np.zeros(len(SPLITS))
    positions = np.arange(len(SPLITS))
    for class_name, color in zip(CLASS_NAMES, CLASS_COLORS):
        values = np.array(
            [
                table.loc[(table["split"] == split) & (table["class_name"] == class_name.replace("\n", "")), "share_within_split_pct"].iloc[0]
                for split in SPLITS
            ]
        )
        ax_stack.bar(positions, values, bottom=bottoms, width=0.66, color=color, edgecolor="white", linewidth=0.25)
        bottoms += values
    ax_stack.set_ylim(0, 100)
    ax_stack.set_yticks([0, 25, 50, 75, 100])
    ax_stack.set_ylabel("有效标签像元占比（%）")
    ax_stack.set_xticks(positions)
    ax_stack.set_xticklabels([SPLIT_NAMES[x] for x in SPLITS])
    ax_stack.grid(axis="y", color="#D9D9D9", linewidth=0.45, zorder=0)
    ax_stack.set_axisbelow(True)
    ax_stack.set_title("八类像元组成", fontsize=8.2, pad=5, fontweight="bold")
    class_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="none", label=name.replace("\n", ""))
        for name, color in zip(CLASS_NAMES, CLASS_COLORS)
    ]
    fig.legend(
        handles=class_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.61, 0.025),
        fontsize=5.25,
        handlelength=0.9,
        handletextpad=0.25,
        columnspacing=0.9,
    )

    # c. Support matrix: raw pixel counts, shown on log scale to reveal rare classes.
    matrix = np.array(
        [
            [
                table.loc[(table["split"] == split) & (table["class_name"] == cls.replace("\n", "")), "pixels"].iloc[0]
                for split in SPLITS
            ]
            for cls in CLASS_NAMES
        ]
    )
    log_matrix = np.log10(np.maximum(matrix, 1))
    im = ax_heat.imshow(log_matrix, cmap=ListedColormap(["#F7FBFF", "#DCEAF7", "#A7CBE7", "#679DC6", "#1E5B8F"]), aspect="auto", vmin=3.7, vmax=6.8)
    ax_heat.set_xticks(range(len(SPLITS)))
    ax_heat.set_xticklabels([SPLIT_NAMES[x] for x in SPLITS])
    ax_heat.set_yticks(range(len(CLASS_NAMES)))
    ax_heat.set_yticklabels(CLASS_NAMES, fontsize=5.8)
    ax_heat.tick_params(length=0)
    ax_heat.set_title("类别支持量（像元）", fontsize=8.2, pad=5, fontweight="bold")
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            value = matrix[r, c]
            text_color = "white" if log_matrix[r, c] > 5.75 else "#272727"
            label = f"{value / 1e6:.2f}M" if value >= 100000 else f"{value / 1000:.0f}k"
            ax_heat.text(c, r, label, ha="center", va="center", fontsize=5.1, color=text_color)
    colorbar = fig.colorbar(im, ax=ax_heat, fraction=0.045, pad=0.025)
    colorbar.set_label(r"log$_{10}$(像元数)", fontsize=5.8)
    colorbar.ax.tick_params(labelsize=5.4, length=2)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    for label, axis in zip(("a", "b", "c"), (ax_map, ax_stack, ax_heat)):
        axis.text(-0.13, 1.045, label, transform=axis.transAxes, fontsize=9.2, fontweight="bold", va="bottom")

    for extension, kwargs in {
        "png": {"dpi": 600},
        "tiff": {"dpi": 600},
        "pdf": {},
        "svg": {},
    }.items():
        fig.savefig(out_base.with_suffix(f".{extension}"), bbox_inches="tight", **kwargs)
    plt.close(fig)

    split_totals = {split: int(table.loc[table["split"] == split, "pixels"].sum()) for split in SPLITS}
    return {
        "pixels_per_split": split_totals,
        "patches_per_split": patch_counts,
        "metric_unit": "all valid pixels in held-out spatial test patches",
        "test_pixels": split_totals["test"],
        "test_patches": patch_counts["test"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    required = {"split", "row_off", "col_off", "height", "width"} | {f"pixels_class_{x}" for x in CLASS_IDS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    frame = frame.loc[frame["split"].isin(SPLITS)].copy()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table, patch_counts = summarize(frame)
    table.to_csv(args.out_dir / "pixel_level_split_class_counts.csv", index=False, encoding="utf-8-sig")
    report = draw_figure(frame, table, patch_counts, args.out_dir / "pixel_level_validation_distribution")
    report["source_manifest"] = str(args.manifest)
    report["active_original_blu_groups"] = list(CLASS_IDS)
    with (args.out_dir / "pixel_level_validation_distribution_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
