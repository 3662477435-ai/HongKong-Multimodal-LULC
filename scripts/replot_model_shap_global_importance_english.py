"""Render the two-panel model-SHAP summary in English from saved CSV outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUP_ORDER = ("s2", "s1", "highres", "nighttime", "building", "roads", "poi", "aoi", "text")
GROUP_LABEL = {
    "s2": "Sentinel-2",
    "s1": "Sentinel-1",
    "highres": "High-resolution imagery",
    "nighttime": "Night-time lights",
    "building": "Building function\nand height",
    "roads": "OSM roads",
    "poi": "OSM POI",
    "aoi": "OSM AOI",
    "text": "OSM text",
}
GROUP_SHORT_LABEL = {
    "s2": "S2", "s1": "S1", "highres": "High-res.", "nighttime": "NTL",
    "building": "Buildings", "roads": "Roads", "poi": "POI", "aoi": "AOI", "text": "Text",
}
GROUP_COLOR = {
    "s2": "#4477AA", "s1": "#66CCEE", "highres": "#228833", "nighttime": "#CCBB44",
    "building": "#EE6677", "roads": "#AA3377", "poi": "#999999", "aoi": "#EE8866", "text": "#555555",
}
CLASS_ORDER = tuple(range(8))
CLASS_LABEL = {
    0: "Residential",
    1: "Industrial/warehouse",
    2: "Public/institutional\nand recreation",
    3: "Transportation",
    4: "Other built-up",
    5: "Agriculture",
    6: "Woodland/shrubland/\ngrassland/wetland",
    7: "Water",
}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "axes.titlesize": 8.5,
        "axes.labelsize": 7.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })


def save(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(
        base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality-csv", type=Path, required=True)
    parser.add_argument("--class-modality-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    modality = pd.read_csv(args.modality_csv)
    class_modality = pd.read_csv(args.class_modality_csv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    style()

    # Figure conclusion: optical observations dominate model sensitivity, while
    # contextual OSM inputs vary systematically across individual classes.
    fig = plt.figure(figsize=(7.15, 3.5), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.00, 1.08), wspace=0.35)
    ax_bar = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[0, 1])

    ordered = modality.set_index("group").loc[list(GROUP_ORDER)].sort_values("importance_percent")
    y = np.arange(len(ordered))
    values = ordered["importance_percent"].to_numpy()
    lower = values - ordered["ci_low_percent"].to_numpy()
    upper = ordered["ci_high_percent"].to_numpy() - values
    ax_bar.barh(y, values, height=0.68, color=[GROUP_COLOR[x] for x in ordered.index], edgecolor="white", linewidth=0.4)
    ax_bar.errorbar(values, y, xerr=np.vstack((lower, upper)), fmt="none", ecolor="#202020", capsize=2.0, lw=0.75)
    # Place labels beyond the upper CI limit, not beyond the bar end, so the
    # reported value never overlaps its horizontal confidence interval.
    for yy, value, ci_high in zip(y, values, ordered["ci_high_percent"].to_numpy()):
        ax_bar.text(ci_high + 0.85, yy, f"{value:.1f}", va="center", fontsize=6.5)
    ax_bar.set_yticks(y, [GROUP_LABEL[x] for x in ordered.index])
    ax_bar.set_xlabel("Normalized mean |SHAP| (%)")
    ax_bar.set_xlim(0, max(57.5, float(ordered["ci_high_percent"].max()) + 5.0))
    ax_bar.set_title("Overall modality contribution (95% bootstrap CI)", loc="left", fontweight="bold")
    ax_bar.grid(axis="x", color="#E1E1E1", lw=0.55)
    ax_bar.set_axisbelow(True)

    matrix = class_modality.pivot(index="class_id", columns="group", values="row_percent").loc[
        list(CLASS_ORDER), list(GROUP_ORDER)
    ]
    im = ax_heat.imshow(matrix.to_numpy(), cmap="YlGnBu", vmin=0, vmax=max(35, float(matrix.to_numpy().max())), aspect="auto")
    ax_heat.set_xticks(range(len(GROUP_ORDER)), [GROUP_SHORT_LABEL[x] for x in GROUP_ORDER], rotation=35, ha="right")
    ax_heat.set_yticks(range(len(CLASS_ORDER)), [CLASS_LABEL[x] for x in CLASS_ORDER])
    ax_heat.tick_params(axis="both", labelsize=6.2, length=2.3)
    ax_heat.set_title("Within-class modality contribution", loc="left", fontweight="bold")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix.iat[row, col]
            ax_heat.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=5.35,
                         color="white" if value > 0.60 * im.norm.vmax else "#202020")
    colorbar = fig.colorbar(im, ax=ax_heat, fraction=0.045, pad=0.025)
    colorbar.set_label("Contribution (%)", fontsize=7)
    colorbar.ax.tick_params(labelsize=6, length=2)

    for label, axis in (("a", ax_bar), ("c", ax_heat)):
        axis.text(-0.15, 1.035, label, transform=axis.transAxes, fontsize=9, fontweight="bold", va="bottom")

    output = args.out_dir / "model_shap_modality_importance_english"
    save(fig, output)
    plt.close(fig)

    (args.out_dir / "model_shap_modality_importance_english_figure_contract.json").write_text(
        json.dumps({
            "core_conclusion": "Sentinel-2 dominates normalized mean absolute SHAP attribution, while class-specific reliance on semantic and spatial context varies by land-use class.",
            "archetype": "quantitative grid",
            "panels": {"a": "overall modality attribution with bootstrap confidence intervals", "c": "class-conditional modality attribution composition"},
            "sources": [str(args.modality_csv), str(args.class_modality_csv)],
            "export_formats": ["png", "pdf", "svg", "tiff"],
        }, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
