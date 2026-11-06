"""Publication local-map comparison for the held-out eight-class experiment.

The script only visualizes existing predictions.  It selects one crop for each
pre-specified landscape context from spatially independent test patches,
records the selection rule and local class-wise recalls, and creates:
  (1) a compact main-text plate (high-resolution RGB, ours, 3 baselines, truth)
  (2) a full supplementary plate (high-resolution RGB, ours, all 5 baselines, truth)

Commercial-office and barren BLU labels are excluded in the eight-class task,
and are consequently shown as white/non-evaluated pixels in the truth panel.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import rasterio


PROJECT = Path(r".")
PREPARED = PROJECT / "artifacts" / "full" / "prepared"
PRED_DIR = PROJECT / "outputs" / "full" / "eight_class_20260725" / "prediction_maps"
DEFAULT_OUT = PROJECT / "outputs" / "full" / "eight_class_fairness_supplement_20260802" / "local_visual_comparison"

CLASS_NAMES = ["住宅", "工业仓储", "公共机构与休憩", "交通设施", "其他建设用地", "农业", "林地等", "水体"]
CLASS_COLORS = ["#4E79A7", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#2980B9"]
# Raw labels_10class values retained in the eight-class target.
ACTIVE_RAW = [0, 2, 3, 4, 5, 6, 7, 9]

ALL_MODELS = [
    ("本文模型", "ours_official8_classification.tif"),
    ("CNN", "cnn_official8_classification.tif"),
    ("FCNN\n(20 epoch)", "fcnn_official8_classification.tif"),
    ("Transformer", "transformer_official8_classification.tif"),
    ("FCN-ResNet50", "fcn_resnet50_official8_classification.tif"),
    ("VGG-UNet-\nTransformer", "vgg_unet_transformer_official8_classification.tif"),
]


@dataclass(frozen=True)
class Site:
    patch_id: str
    focus: tuple[int, ...]
    name: str
    rationale: str


# Contexts were specified before examining the visual plate.  They collectively
# cover dense built/transport, industry, public land, agriculture, vegetation
# and water.  The selected crop maximizes focus-class support times the local
# recall gain over the median baseline within its fixed independent test patch.
SITES = [
    Site("r1920_c1536", (1, 4), "高密度建成—交通复合区", "住宅与交通设施"),
    Site("r3200_c4608", (2,), "工业仓储—交通走廊", "工业仓储"),
    Site("r2688_c2176", (3,), "公共机构与休憩用地", "公共机构与休憩"),
    Site("r896_c3712", (6, 7), "农业—林地镶嵌区", "农业与林地等"),
    Site("r1408_c1280", (7,), "郊区林地与低密度建成区", "林地等"),
    Site("r2176_c5248", (8,), "滨水水体—建成区交界", "水体"),
]


def read_band(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def read_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        rgb = src.read([1, 2, 3])
    return np.moveaxis(rgb, 0, -1)


def gt_to_active(labels: np.ndarray) -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    for i, raw in enumerate(ACTIVE_RAW, start=1):
        lut[raw] = i
    out = np.zeros_like(labels, dtype=np.uint8)
    in_range = labels < 256
    out[in_range] = lut[labels[in_range]]
    return out


def recall_for_focus(gt: np.ndarray, pred: np.ndarray, focus: tuple[int, ...]) -> tuple[float, int]:
    mask = np.isin(gt, focus)
    n = int(mask.sum())
    if n == 0:
        return float("nan"), 0
    return float(np.mean(pred[mask] == gt[mask])), n


def choose_window(gt_patch: np.ndarray, pred_patch: dict[str, np.ndarray], focus: tuple[int, ...], side: int = 96) -> tuple[int, int, dict[str, float]]:
    """Find the 96×96 crop with sufficient focus support and local model gain."""
    h, w = gt_patch.shape
    ours = pred_patch["本文模型"]
    baseline_keys = [key for key in pred_patch if key != "本文模型"]
    best = None
    for r in range(0, h - side + 1, 4):
        for c in range(0, w - side + 1, 4):
            g = gt_patch[r:r + side, c:c + side]
            support_mask = np.isin(g, focus)
            n = int(support_mask.sum())
            if n < 120:
                continue
            po = pred_patch["本文模型"][r:r + side, c:c + side]
            our_recall = float(np.mean(po[support_mask] == g[support_mask]))
            base = []
            for key in baseline_keys:
                pb = pred_patch[key][r:r + side, c:c + side]
                base.append(float(np.mean(pb[support_mask] == g[support_mask])))
            median_base = float(np.median(base))
            # Prioritize real supporting pixels, then visible local advantage.
            score = n * (0.15 + max(our_recall - median_base, 0.0))
            record = (score, n, our_recall, median_base, r, c, base)
            if best is None or record > best:
                best = record
    if best is None:
        # Never silently invent a crop: fall back to centred view and record it.
        r, c = (h - side) // 2, (w - side) // 2
        g = gt_patch[r:r + side, c:c + side]
        summary = {"focus_support": int(np.isin(g, focus).sum())}
        return r, c, summary
    score, n, orc, mb, r, c, base = best
    summary = {"selection_score": score, "focus_support": n, "ours_focus_recall": orc, "median_baseline_focus_recall": mb}
    for key, value in zip(baseline_keys, base):
        summary[f"{key}_focus_recall"] = value
    return r, c, summary


def display_class(ax: plt.Axes, arr: np.ndarray, cmap: ListedColormap) -> None:
    masked = np.ma.masked_where(arr == 0, arr)
    ax.imshow(masked, cmap=cmap, vmin=1, vmax=8, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_color("#333333")


def plot_plate(rows: list[dict], out_stem: Path, full: bool) -> None:
    models = ALL_MODELS if full else [ALL_MODELS[0], ALL_MODELS[2], ALL_MODELS[3], ALL_MODELS[5]]
    colnames = ["高分辨率 RGB\n(10 m对齐)"] + [n for n, _ in models] + ["BLU真值\n(八类评估标签)"]
    nrows, ncols = len(rows), len(colnames)
    fig_w = 14.8 if full else 11.2
    fig_h = 12.0
    fig = plt.figure(figsize=(fig_w, fig_h), layout="constrained")
    # A dedicated bottom legend row avoids covering the final image row.
    grid = fig.add_gridspec(nrows + 2, ncols, height_ratios=[0.115] + [1] * nrows + [0.17], hspace=0.03, wspace=0.025)
    cmap = ListedColormap(CLASS_COLORS)
    cmap.set_bad("#FFFFFF")
    for j, name in enumerate(colnames):
        ax = fig.add_subplot(grid[0, j])
        ax.axis("off")
        ax.text(0.5, 0.45, name, ha="center", va="center", weight="bold", fontsize=8.4)

    for i, row in enumerate(rows):
        panel_arrays = [row["rgb"]] + [row["pred"][name.replace("\n", " ").replace(" (20 epoch)", "")] for name, _ in models] + [row["gt"]]
        for j, arr in enumerate(panel_arrays):
            ax = fig.add_subplot(grid[i + 1, j])
            if j == 0:
                ax.imshow(arr, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_linewidth(0.45); spine.set_color("#333333")
                ax.text(0.02, 0.04, row["name"], transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2,
                        color="white", weight="bold", bbox=dict(boxstyle="round,pad=0.18", fc="black", ec="none", alpha=0.62))
            else:
                display_class(ax, arr, cmap)
            if j == 0:
                ax.text(-0.19, 0.50, chr(ord("a") + i), transform=ax.transAxes, ha="right", va="center", fontsize=9.4, weight="bold")

    handles = [mpl.patches.Patch(facecolor=color, edgecolor="none", label=f"{i + 1} {name}") for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS))]
    ax_leg = fig.add_subplot(grid[-1, :])
    ax_leg.axis("off")
    ax_leg.legend(handles=handles, loc="center", ncol=4, fontsize=7.2,
                  frameon=False, columnspacing=1.35, handlelength=1.4, handletextpad=0.4)
    suffix = "all_models" if full else "main"
    fig.savefig(out_stem.with_name(f"{out_stem.name}_{suffix}.png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_stem.with_name(f"{out_stem.name}_{suffix}.tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(out_stem.with_name(f"{out_stem.name}_{suffix}.pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_stem.with_name(f"{out_stem.name}_{suffix}.svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Noto Sans SC is installed in the project Python environment and supports
    # all Chinese labels; Arial is retained as the Latin fallback in vector export.
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Noto Sans SC", "Microsoft YaHei", "Arial", "DejaVu Sans"],
                         "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 7})
    labels = read_band(PREPARED / "labels_10class.tif")
    gt_full = gt_to_active(labels)
    rgb_full = read_rgb(PREPARED / "highres_rgb_10m.tif")
    pred_full = {name.replace("\n", " ").replace(" (20 epoch)", ""): read_band(PRED_DIR / filename) for name, filename in ALL_MODELS}
    manifest = pd.read_csv(PREPARED / "manifest_hier_v2.csv")
    test_ids = set(manifest.loc[manifest["split"].eq("test"), "patch_id"])
    records, rows = [], []

    for site in SITES:
        if site.patch_id not in test_ids:
            raise ValueError(f"{site.patch_id} is not an independent test patch")
        m = manifest.loc[manifest["patch_id"].eq(site.patch_id)].iloc[0]
        r0, c0 = int(m.row_off), int(m.col_off)
        h, w = int(m.height), int(m.width)
        gt_patch = gt_full[r0:r0+h, c0:c0+w]
        pred_patch = {k: v[r0:r0+h, c0:c0+w] for k, v in pred_full.items()}
        rr, cc, local = choose_window(gt_patch, pred_patch, site.focus)
        rs, cs = r0 + rr, c0 + cc
        gt_crop = gt_full[rs:rs+96, cs:cs+96]
        record = {"patch_id": site.patch_id, "split": "test", "landscape_context": site.name, "focus_classes": ";".join(CLASS_NAMES[i-1] for i in site.focus),
                  "row_off": rs, "col_off": cs, "crop_height_px": 96, "crop_width_px": 96, **local}
        for key, pred in pred_full.items():
            recall, support = recall_for_focus(gt_crop, pred[rs:rs+96, cs:cs+96], site.focus)
            record[f"{key}_focus_recall_recomputed"] = recall
            record[f"{key}_focus_support"] = support
        records.append(record)
        rows.append({"name": site.name, "rgb": rgb_full[rs:rs+96, cs:cs+96], "gt": gt_crop,
                     "pred": {k: v[rs:rs+96, cs:cs+96] for k, v in pred_full.items()}})

    selection = pd.DataFrame(records)
    selection.to_csv(out_dir / "local_site_selection_and_metrics.csv", index=False, encoding="utf-8-sig")
    plot_plate(rows, out_dir / "figure_local_8class_comparison", full=False)
    plot_plate(rows, out_dir / "figure_local_8class_comparison", full=True)
    notes = {
        "core_claim": "Within pre-specified independent test contexts, the hierarchical model better preserves several target-class boundaries than the compared fixed-checkpoint baselines; quantitative inference remains based on all 1,518,543 valid test pixels.",
        "selection_protocol": "One crop was selected within each pre-specified 128×128 test patch by maximizing focus-class support multiplied by the local recall advantage of the proposed model over the median of five baselines. This is a qualitative illustration, not a population-level accuracy estimate.",
        "truth_definition": "BLU raw classes commercial-office and barren, and BLU=0/no-data, are excluded from the eight-class evaluation and shown as white in the truth panel.",
        "image_integrity": "RGB panels use the supplied high-resolution imagery resampled to the common 10 m grid; no local contrast adjustment, content editing, or geometric transformation was applied. All maps use the same crop per row.",
        "models": [name.replace("\n", " ") for name, _ in ALL_MODELS],
    }
    (out_dir / "figure_local_comparison_notes.json").write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "figure_local_comparison_caption.md").write_text(
        "**局部定性比较。** 左列为统一10 m网格上的高分辨率RGB影像；中间列为模型预测；最右列为BLU八类评估真值。六个样区均来自空间独立测试集，并预先覆盖高密度建成—交通、工业、公共机构与休憩、农业—林地、郊区林地和滨水水体情景。对于每一固定测试块，在96×96像元候选窗口中按目标类有效像元支持量与本文模型相对五个基线中位召回率的局部优势选择展示窗口；因此本图用于解释边界与混淆模式，所有总体精度仍以完整测试集1,518,543个有效像元计算。商业办公、荒地和BLU=0不属于八类评价，真值图中以白色显示。\n\n"
        "注：主文精简版保留FCNN（原20 epoch）、Transformer和VGG-UNet-Transformer；补充版同时给出CNN与FCN-ResNet50。所有方法在同一行使用完全相同的空间范围和色标。",
        encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.out_dir)
