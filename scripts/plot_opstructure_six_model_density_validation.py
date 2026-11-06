"""Plot external Hong Kong OP-building-use agreement for six 8-class maps.

One labelled building contributes one observation.  The reference class comes
from the Building Master's approved-use text and the predicted class is sampled
at the building representative point.  A categorical hexbin (with a fixed,
seeded sub-category jitter) visualises the count at each reference--prediction
combination without pretending that class codes are continuous measurements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LogNorm
from rasterio.transform import rowcol


MODELS = [
    ("Proposed model", "ours_official8_classification.tif"),
    ("CNN", "cnn_official8_classification.tif"),
    ("FCNN", "fcnn_official8_classification.tif"),
    ("Transformer", "transformer_official8_classification.tif"),
    ("FCN-ResNet50", "fcn_resnet50_official8_classification.tif"),
    ("VGG-UNet-Transformer", "vgg_unet_transformer_official8_classification.tif"),
]

CLASS_NAMES = {
    1: "Residential",
    2: "Industrial",
    3: "Public/institutional",
    4: "Transportation",
    5: "Other built-up",
    6: "Agriculture",
    7: "Vegetation/wetland",
    8: "Water",
}
REFERENCE_CODES = (1, 2, 4, 5)


def sample_predictions(points: gpd.GeoSeries, raster_path: Path) -> np.ndarray:
    """Sample a classification raster at each building representative point."""
    with rasterio.open(raster_path) as src:
        projected = points.to_crs(src.crs)
        rows, cols = rowcol(src.transform, projected.x.to_numpy(), projected.y.to_numpy())
        rows = np.asarray(rows)
        cols = np.asarray(cols)
        values = np.zeros(len(points), dtype=np.uint8)
        inside = (rows >= 0) & (rows < src.height) & (cols >= 0) & (cols < src.width)
        image = src.read(1)
        values[inside] = image[rows[inside], cols[inside]]
    return values


def load_pairs(validation_gpkg: Path, prediction_dir: Path, split: str) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    gdf = gpd.read_file(validation_gpkg, layer="validation8_strict")
    if split == "test":
        gdf = gdf[gdf["split_code"] == 3].copy()
    elif split != "all":
        raise ValueError("split must be 'all' or 'test'")
    gdf = gdf[gdf["lulc8_code"].notna()].copy()
    gdf["lulc8_code"] = gdf["lulc8_code"].astype(int)

    table = pd.DataFrame({
        "BuildingID": gdf["BuildingID"].to_numpy(),
        "reference_code": gdf["lulc8_code"].to_numpy(),
        "reference_class": gdf["lulc8_class"].to_numpy(),
    })
    representative_points = gpd.GeoSeries(gdf.geometry.representative_point(), crs=gdf.crs)
    for name, file_name in MODELS:
        table[name] = sample_predictions(representative_points, prediction_dir / file_name)
    return gdf, table


def make_figure(table: pd.DataFrame, out_file: Path, title_prefix: str) -> pd.DataFrame:
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 9,
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    rng = np.random.default_rng(20260830)
    x = table["reference_code"].to_numpy(float)
    xj = x + rng.normal(0, 0.115, size=len(x))
    panels = []
    max_count = 1
    # First determine a comparable log-scale upper bound for all panels.
    for name, _ in MODELS:
        y = table[name].to_numpy(float)
        valid = y > 0
        hist, _, _ = np.histogram2d(xj[valid], y[valid], bins=[36, 48], range=[[0.5, 5.5], [0.5, 8.5]])
        max_count = max(max_count, int(hist.max()))
    norm = LogNorm(vmin=1, vmax=max(2, max_count))

    # Three columns × two rows keeps the six-model comparison in the compact,
    # landscape format conventionally used for map/model panels.
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 8.0), sharex=True, sharey=True, constrained_layout=False)
    mappable = None
    for idx, ((name, _), ax) in enumerate(zip(MODELS, axes.ravel())):
        y = table[name].to_numpy(float)
        valid = y > 0
        # Independent but deterministic y jitter preserves each model's class
        # density while avoiding opaque overplotting of categorical points.
        local_rng = np.random.default_rng(1000 + idx)
        yj = y[valid] + local_rng.normal(0, 0.115, size=int(valid.sum()))
        hb = ax.hexbin(
            xj[valid], yj, gridsize=(28, 40), extent=(0.5, 5.5, 0.5, 8.5),
            mincnt=1, cmap="viridis", norm=norm, linewidths=0,
        )
        mappable = hb
        ax.plot([0.5, 5.5], [0.5, 5.5], ls="--", lw=0.8, color="white", alpha=0.65)
        agreement = float((table.loc[valid, "reference_code"].to_numpy() == y[valid]).mean())
        panel = chr(ord("a") + idx)
        ax.set_title(f"{panel}  {name}", loc="left", fontsize=10, pad=7)
        ax.text(
            0.97, 0.95, f"Agreement = {agreement:.3f}\nn = {int(valid.sum()):,}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.8},
        )
        ax.grid(color="0.85", lw=0.5, alpha=0.55)
        ax.set_axisbelow(True)
        panels.append({"model": name, "n_buildings": int(valid.sum()), "exact_class_agreement": agreement})

    xlabels = [CLASS_NAMES[c] for c in REFERENCE_CODES]
    for ax in axes[-1, :]:
        ax.set_xlabel("Reference building-use class", fontsize=9)
        ax.set_xticks(REFERENCE_CODES, xlabels, rotation=18, ha="right")
    for ax in axes[:, 0]:
        ax.set_ylabel("Predicted 8-class label", fontsize=9)
        ax.set_yticks(range(1, 9), [CLASS_NAMES[c] for c in range(1, 9)])
    for ax in axes[:, 1:].ravel():
        ax.set_yticks(range(1, 9))
    fig.suptitle(title_prefix, fontsize=12, fontweight="bold", y=0.985)
    # Allocate a dedicated colour-bar axis so it cannot cover the annotations
    # in the third-column panels.
    cax = fig.add_axes([0.905, 0.205, 0.014, 0.58])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("Building count per hexagon (log scale)", fontsize=9)
    fig.subplots_adjust(left=0.13, right=0.88, bottom=0.12, top=0.90, hspace=0.26, wspace=0.11)
    fig.savefig(out_file.with_suffix(".png"), dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(out_file.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(out_file.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return pd.DataFrame(panels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-gpkg", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["all", "test"], default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _, pairs = load_pairs(args.validation_gpkg, args.prediction_dir, args.split)
    pairs.to_csv(args.output_dir / f"opstructure_building_predictions_{args.split}.csv", index=False, encoding="utf-8-sig")
    heading = (
        "External OP-building-use agreement across six 8-class maps"
        if args.split == "all" else
        "Spatially blocked external OP-building-use agreement across six 8-class maps"
    )
    summary = make_figure(pairs, args.output_dir / f"opstructure_six_model_density_{args.split}", heading)
    summary.to_csv(args.output_dir / f"opstructure_six_model_agreement_{args.split}.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    print("Buildings by reference class:")
    print(pairs.groupby(["reference_code", "reference_class"]).size().to_string())


if __name__ == "__main__":
    main()
