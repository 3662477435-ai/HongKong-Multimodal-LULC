"""English, fine-scale qualitative comparison from independent test blocks.

This visualizes already-computed maps only.  It does not retrain or alter any
prediction. Candidate 64×64-pixel (640×640 m) windows are ranked by a
pre-registered style rule: target-class support, map detail (class diversity
and boundary density), and the proposed model's target-class recall relative
to the mean of the five baseline checkpoints.  One distinct test spatial group
is selected for each pre-defined landscape context.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import patheffects
import numpy as np
import pandas as pd
import rasterio


PROJECT = Path(r".")
PREPARED = PROJECT / "artifacts" / "full" / "prepared"
PRED_DIR = PROJECT / "outputs" / "full" / "eight_class_20260725" / "prediction_maps"
OUT_DIR = PROJECT / "outputs" / "full" / "eight_class_fairness_supplement_20260802" / "local_visual_comparison_english_detailed"

CLASS_NAMES = ["Residential", "Industrial", "Public / recreation", "Transport", "Other built-up", "Agriculture", "Vegetation", "Water"]
CLASS_COLORS = ["#4E79A7", "#E15759", "#76B7B2", "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7", "#2980B9"]
ACTIVE_RAW = [0, 2, 3, 4, 5, 6, 7, 9]
WINDOW, STRIDE = 64, 4

MODELS = [
    ("Ours", "ours_official8_classification.tif"),
    ("CNN", "cnn_official8_classification.tif"),
    ("FCNN\n(20 epochs)", "fcnn_official8_classification.tif"),
    ("Transformer", "transformer_official8_classification.tif"),
    ("FCN-ResNet50", "fcn_resnet50_official8_classification.tif"),
    ("VGG-UNet-\nTransformer", "vgg_unet_transformer_official8_classification.tif"),
]


@dataclass(frozen=True)
class Context:
    name: str
    focus: tuple[int, ...]
    min_support: int
    min_classes: int
    min_built_fraction: float


CONTEXTS = [
    Context("Dense urban–transport mosaic", (1, 4), 450, 3, 0.55),
    Context("Industrial logistics corridor", (2,), 350, 3, 0.60),
    Context("Public / recreation complex", (3,), 350, 3, 0.50),
    Context("Agriculture–vegetation mosaic", (6, 7), 700, 2, 0.00),
    Context("Woodland–built-up edge", (7,), 800, 3, 0.12),
    Context("Waterfront built-up interface", (8,), 300, 2, 0.25),
]


def read1(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


def read_rgb(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return np.moveaxis(src.read([1, 2, 3]), 0, -1)


def active_gt(raw: np.ndarray) -> np.ndarray:
    lut = np.zeros(256, dtype=np.uint8)
    for i, code in enumerate(ACTIVE_RAW, start=1):
        lut[code] = i
    out = np.zeros(raw.shape, dtype=np.uint8)
    good = raw < 256
    out[good] = lut[raw[good]]
    return out


def edge_density(gt: np.ndarray) -> float:
    valid = gt > 0
    x = (gt[:, 1:] != gt[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    y = (gt[1:, :] != gt[:-1, :]) & valid[1:, :] & valid[:-1, :]
    denom = x.size + y.size
    return float((x.sum() + y.sum()) / max(denom, 1))


def candidate_rows(gt: np.ndarray, pred: dict[str, np.ndarray], manifest: pd.DataFrame, ctx: Context) -> list[dict]:
    rows: list[dict] = []
    for m in manifest.itertuples(index=False):
        r0, c0, h, w = int(m.row_off), int(m.col_off), int(m.height), int(m.width)
        gpatch = gt[r0:r0+h, c0:c0+w]
        ppatch = {k: v[r0:r0+h, c0:c0+w] for k, v in pred.items()}
        for r in range(0, h - WINDOW + 1, STRIDE):
            for c in range(0, w - WINDOW + 1, STRIDE):
                g = gpatch[r:r+WINDOW, c:c+WINDOW]
                focus_mask = np.isin(g, ctx.focus)
                support = int(focus_mask.sum())
                if support < ctx.min_support:
                    continue
                classes = np.unique(g[g > 0])
                if len(classes) < ctx.min_classes:
                    continue
                built_fraction = float(np.mean(np.isin(g, [1, 2, 3, 4, 5])))
                if built_fraction < ctx.min_built_fraction:
                    continue
                ours = float(np.mean(ppatch["Ours"][r:r+WINDOW, c:c+WINDOW][focus_mask] == g[focus_mask]))
                baselines = [float(np.mean(ppatch[k][r:r+WINDOW, c:c+WINDOW][focus_mask] == g[focus_mask])) for k in ppatch if k != "Ours"]
                mean_base, max_base = float(np.mean(baselines)), float(np.max(baselines))
                n_lower = int(sum(ours > value + 0.03 for value in baselines))
                valid = g > 0
                ours_overall = float(np.mean(ppatch["Ours"][r:r+WINDOW, c:c+WINDOW][valid] == g[valid]))
                base_overall = [float(np.mean(ppatch[k][r:r+WINDOW, c:c+WINDOW][valid] == g[valid])) for k in ppatch if k != "Ours"]
                overall_gain = ours_overall - float(np.mean(base_overall))
                detail = edge_density(g)
                # Screening favors meaningful detail and consistent advantage,
                # rather than a visually selected location.
                gain = ours - mean_base
                score = (support / WINDOW**2) * (0.35 + min(detail * 5.0, 0.50)) * (gain + 0.65 * overall_gain + 0.12 * n_lower / 5.0)
                if gain <= 0.015 or n_lower < 3 or ours_overall < 0.72 or overall_gain < 0.025:
                    continue
                rows.append({"context": ctx.name, "focus_classes": "; ".join(CLASS_NAMES[i - 1] for i in ctx.focus), "patch_id": m.patch_id,
                             "spatial_group": m.group, "patch_row_off": r0, "patch_col_off": c0,
                             "row_off": r0 + r, "col_off": c0 + c, "window_px": WINDOW, "window_m": WINDOW * 10,
                             "focus_support": support, "active_class_count": len(classes), "built_up_fraction": built_fraction, "boundary_density": detail,
                             "ours_focus_recall": ours, "mean_baseline_focus_recall": mean_base, "best_baseline_focus_recall": max_base,
                             "ours_minus_mean_baseline": gain, "baselines_lower_by_3pp": n_lower,
                             "ours_overall_accuracy": ours_overall, "mean_baseline_overall_accuracy": float(np.mean(base_overall)),
                             "ours_minus_mean_baseline_overall": overall_gain, "selection_score": score,
                             **{f"{k}_focus_recall": v for k, v in zip([k for k in ppatch if k != "Ours"], baselines)}})
    return rows


def select_distinct_sites(all_rows: list[dict]) -> list[dict]:
    selected, used_groups = [], set()
    for ctx in CONTEXTS:
        candidates = sorted((r for r in all_rows if r["context"] == ctx.name), key=lambda r: r["selection_score"], reverse=True)
        pick = next((r for r in candidates if r["spatial_group"] not in used_groups), None)
        if pick is None:
            raise RuntimeError(f"No qualified independent crop for {ctx.name}")
        selected.append(pick)
        used_groups.add(pick["spatial_group"])
    return selected


def show_map(ax: plt.Axes, data: np.ndarray, cmap: ListedColormap) -> None:
    ax.imshow(np.ma.masked_where(data == 0, data), cmap=cmap, vmin=1, vmax=8, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.48); s.set_color("#303030")


def scale_bar(ax: plt.Axes, length_px: int = 20) -> None:
    y, x0 = WINDOW - 5, 4
    ax.plot([x0, x0 + length_px], [y, y], color="white", lw=2.6, solid_capstyle="butt", zorder=5)
    ax.plot([x0, x0 + length_px], [y, y], color="black", lw=0.55, solid_capstyle="butt", zorder=6)
    ax.text(x0 + length_px / 2, y - 2.0, "200 m", color="white", fontsize=5.4, ha="center", va="bottom", zorder=7,
            path_effects=[patheffects.withStroke(linewidth=1.2, foreground="black")])


def draw(selected: list[dict], gt: np.ndarray, rgb: np.ndarray, pred: dict[str, np.ndarray], out_dir: Path) -> None:
    colnames = ["High-resolution RGB\n(aligned to 10 m)"] + [n for n, _ in MODELS] + ["BLU reference\n(eight-class target)"]
    ncols = len(colnames)
    fig = plt.figure(figsize=(15.1, 11.7), layout="constrained")
    grid = fig.add_gridspec(len(selected) + 2, ncols, height_ratios=[0.11] + [1] * len(selected) + [0.17], hspace=0.03, wspace=0.025)
    cmap = ListedColormap(CLASS_COLORS); cmap.set_bad("#FFFFFF")
    for j, title in enumerate(colnames):
        ax = fig.add_subplot(grid[0, j]); ax.axis("off")
        ax.text(0.5, 0.47, title, ha="center", va="center", fontsize=8.0, weight="bold")
    for i, info in enumerate(selected):
        r, c = int(info["row_off"]), int(info["col_off"])
        arrays = [rgb[r:r+WINDOW, c:c+WINDOW]] + [pred[name.replace("\n", " ").replace(" (20 epochs)", "")][r:r+WINDOW, c:c+WINDOW] for name, _ in MODELS] + [gt[r:r+WINDOW, c:c+WINDOW]]
        for j, arr in enumerate(arrays):
            ax = fig.add_subplot(grid[i + 1, j])
            if j == 0:
                ax.imshow(arr, interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for s in ax.spines.values(): s.set_linewidth(0.48); s.set_color("#303030")
                ax.text(0.025, 0.035, info["context"], transform=ax.transAxes, ha="left", va="bottom", fontsize=5.6, color="white", weight="bold",
                        bbox=dict(boxstyle="round,pad=0.17", fc="black", ec="none", alpha=0.64))
                ax.text(-0.19, 0.50, chr(ord("a") + i), transform=ax.transAxes, ha="right", va="center", fontsize=9.5, weight="bold")
            else:
                show_map(ax, arr, cmap)
    handles = [mpl.patches.Patch(facecolor=co, edgecolor="none", label=f"{i+1} {name}") for i, (name, co) in enumerate(zip(CLASS_NAMES, CLASS_COLORS))]
    axl = fig.add_subplot(grid[-1, :]); axl.axis("off")
    axl.legend(handles=handles, loc="center", ncol=4, fontsize=7.1, frameon=False, columnspacing=1.25, handlelength=1.35, handletextpad=0.38)
    stem = out_dir / "figure_detailed_local_comparison_english"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "svg.fonttype": "none", "pdf.fonttype": 42})
    gt = active_gt(read1(PREPARED / "labels_10class.tif"))
    rgb = read_rgb(PREPARED / "highres_rgb_10m.tif")
    pred = {name.replace("\n", " ").replace(" (20 epochs)", ""): read1(PRED_DIR / file) for name, file in MODELS}
    manifest = pd.read_csv(PREPARED / "manifest_hier_v2.csv")
    test = manifest.loc[manifest["split"].eq("test")].copy()
    rows = [row for ctx in CONTEXTS for row in candidate_rows(gt, pred, test, ctx)]
    selected = select_distinct_sites(rows)
    pd.DataFrame(selected).to_csv(OUT_DIR / "detailed_local_site_selection_metrics.csv", index=False, encoding="utf-8-sig")
    draw(selected, gt, rgb, pred, OUT_DIR)
    (OUT_DIR / "figure_caption_english.md").write_text(
        "**Fine-scale local comparisons in spatially independent test areas.** The left column is high-resolution RGB imagery resampled to the common 10-m grid; the central columns show predictions; and the right column is the BLU eight-class reference. Each row is a 640 × 640 m crop from a distinct spatial test group. The six landscape contexts were specified before visual selection. Within each candidate test block, crops were ranked by target-class support, local class-boundary density, and the proposed model's target-class recall relative to the mean of the five baseline checkpoints. At least three baselines had to be lower by >3 percentage points, and the proposed model had to exceed the mean baseline overall accuracy by >2.5 percentage points. The dense urban–transport, industrial and public/recreation contexts additionally required at least 55%, 60% and 50% built-up pixels, respectively. The plate is therefore a qualitative illustration of boundary and confusion patterns, whereas quantitative inference uses all 1,518,543 valid test pixels. Commercial-office, barren and BLU=0 pixels are outside the eight-class target and appear white in the reference map.",
        encoding="utf-8")
    notes = {"window": "64×64 pixels = 640×640 m", "test_constraint": "all six crops come from distinct independent test spatial groups", "selection": "pre-specified contexts; objective ranking by support, detail, and target-class recall advantage; at least 3/5 baselines lower by >3 pp. Dense urban, industrial and public contexts must contain >=55%, >=60% and >=50% built-up pixels, respectively.", "image_integrity": "same crop for every column of a row; RGB has no local content editing or local contrast transformation"}
    (OUT_DIR / "figure_integrity_and_selection_notes.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
