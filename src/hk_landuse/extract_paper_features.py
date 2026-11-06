from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box
from skimage.feature import graycomatrix, graycoprops

from .config import load_config
from .labels import IGNORE_INDEX, LEAF_TO_COARSE, LEAF_TO_OFFICIAL


BUILDING_FUNCTIONS = (
    "Residential",
    "Commercial",
    "Industrial",
    "Educational",
    "Healthcare",
    "Administrative",
    "Public service",
    "Sport and art",
    "Unknown",
)


def finite_stats(values: np.ndarray, prefix: str, result: dict[str, float]) -> None:
    values = values[np.isfinite(values)]
    if not values.size:
        for suffix in ("mean", "std", "p10", "p50", "p90"):
            result[f"{prefix}_{suffix}"] = np.nan
        return
    for suffix, value in zip(
        ("mean", "std", "p10", "p50", "p90"),
        (values.mean(), values.std(), *np.percentile(values, (10, 50, 90))),
    ):
        result[f"{prefix}_{suffix}"] = float(value)


def glcm_features(rgb: np.ndarray, result: dict[str, float]) -> None:
    gray = 0.2989 * rgb[0] + 0.5870 * rgb[1] + 0.1141 * rgb[2]
    valid = np.isfinite(gray)
    if not valid.any():
        for name in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "entropy"):
            result[f"highres_glcm_{name}"] = np.nan
        return
    lo, hi = np.percentile(gray[valid], (1, 99))
    quantized = np.clip((gray - lo) / max(hi - lo, 1e-6) * 15, 0, 15).astype(np.uint8)
    matrix = graycomatrix(
        quantized,
        distances=(1, 2),
        angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
        levels=16,
        symmetric=True,
        normed=True,
    )
    for name in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        result[f"highres_glcm_{name}"] = float(np.nanmean(graycoprops(matrix, name)))
    probabilities = matrix[matrix > 0]
    result["highres_glcm_entropy"] = float(-(probabilities * np.log2(probabilities)).sum())


def building_features(
    buildings: gpd.GeoDataFrame,
    geometry,
    result: dict[str, float],
) -> None:
    indices = list(buildings.sindex.query(geometry, predicate="intersects"))
    patch_area = max(geometry.area, 1.0)
    result["building_count"] = float(len(indices))
    result["building_count_km2"] = float(len(indices) / patch_area * 1e6)
    if not indices:
        result["building_coverage"] = 0.0
        for function in BUILDING_FUNCTIONS:
            key = function.lower().replace(" ", "_")
            result[f"building_{key}_count"] = 0.0
            result[f"building_{key}_area_fraction"] = 0.0
        return
    subset = buildings.iloc[indices][["Function", "geometry"]].copy()
    subset["intersection_area"] = subset.geometry.intersection(geometry).area
    result["building_coverage"] = float(min(subset.intersection_area.sum() / patch_area, 1.0))
    for function in BUILDING_FUNCTIONS:
        selected = subset[subset.Function == function]
        key = function.lower().replace(" ", "_")
        result[f"building_{key}_count"] = float(len(selected))
        result[f"building_{key}_area_fraction"] = float(selected.intersection_area.sum() / patch_area)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract block features following Wang et al. (2026).")
    parser.add_argument("--config", default="config_paper_replication.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    prepared = Path(cfg["paths"]["prepared_dir"])
    output = Path(cfg["paths"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(prepared / cfg["data"]["manifest_name"])

    buildings = gpd.read_file(cfg["paths"]["buildings"], columns=["Function", "geometry"])
    rows: list[dict] = []
    with (
        rasterio.open(cfg["paths"]["image"]) as image,
        rasterio.open(cfg["paths"]["high_resolution"]) as high_source,
        rasterio.open(cfg["paths"]["nighttime_light"]) as ntl_source,
        rasterio.open(cfg["paths"]["osm_poi_raster"]) as poi,
        rasterio.open(cfg["paths"]["osm_aoi_raster"]) as aoi,
        rasterio.open(prepared / cfg["data"]["label_name"]) as labels,
    ):
        buildings = buildings.to_crs(image.crs)
        high = WarpedVRT(
            high_source,
            crs=image.crs,
            transform=image.transform,
            width=image.width,
            height=image.height,
            resampling=Resampling.average,
        )
        ntl = WarpedVRT(
            ntl_source,
            crs=image.crs,
            transform=image.transform,
            width=image.width,
            height=image.height,
            src_nodata=ntl_source.nodata,
            nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        try:
            for number, row in enumerate(manifest.itertuples(), start=1):
                window = Window(int(row.col_off), int(row.row_off), int(row.width), int(row.height))
                cube = image.read(window=window, out_dtype="float32")
                high_rgb = high.read(window=window, out_dtype="float32")
                ntl_cube = ntl.read(window=window, out_dtype="float32")
                poi_cube = poi.read(window=window, out_dtype="float32")
                aoi_values = aoi.read(1, window=window, out_dtype="uint8")
                target = labels.read(1, window=window, out_dtype="uint8")
                valid = target != IGNORE_INDEX
                leaf_hist = np.bincount(target[valid], minlength=14) if valid.any() else np.zeros(14, dtype=int)
                leaf_majority = int(leaf_hist.argmax()) if leaf_hist.sum() else -1
                left, bottom, right, top = window_bounds(window, image.transform)
                geometry = box(left, bottom, right, top)
                result: dict[str, object] = {
                    "patch_id": row.patch_id,
                    "split": row.split,
                    "group": row.group,
                    "row_off": int(row.row_off),
                    "col_off": int(row.col_off),
                    "valid_pixels": int(leaf_hist.sum()),
                    "leaf_target": leaf_majority,
                    "official_target": int(LEAF_TO_OFFICIAL[leaf_majority]) if leaf_majority >= 0 else -1,
                    "coarse_target": int(LEAF_TO_COARSE[leaf_majority]) if leaf_majority >= 0 else -1,
                    "leaf_purity": float(leaf_hist.max() / max(leaf_hist.sum(), 1)),
                }
                # S2: 12 physical bands + NDVI; S1: VV, VH, VV-VH.
                for band in range(13):
                    finite_stats(cube[band], f"s2_b{band + 1:02d}", result)
                for band, name in zip(range(13, 16), ("vv", "vh", "vv_minus_vh")):
                    finite_stats(cube[band], f"s1_{name}", result)
                for band, name in enumerate(("r", "g", "b")):
                    finite_stats(high_rgb[band], f"highres_{name}", result)
                glcm_features(high_rgb, result)
                for band, name in enumerate(("avg_rad", "log1p", "cf_cvg")):
                    finite_stats(ntl_cube[band], f"ntl_{name}", result)
                for category in range(14):
                    result[f"poi_count_{category}"] = float(np.nansum(poi_cube[category]))
                area_pixels = max(aoi_values.size, 1)
                for category in range(1, 7):
                    mask = aoi_values == category
                    # AD is area fraction. AFD uses component frequency times area fraction.
                    result[f"aoi_ad_{category}"] = float(mask.sum() / area_pixels)
                    if mask.any():
                        from scipy.ndimage import label as component_label
                        components = int(component_label(mask)[1])
                    else:
                        components = 0
                    result[f"aoi_afd_{category}"] = float(components * mask.sum() / area_pixels)
                building_features(buildings, geometry, result)
                relative_light = float(np.nanmean(ntl_cube[1])) if np.isfinite(ntl_cube[1]).any() else 0.0
                result["building_nightlight_proxy"] = float(
                    np.log1p(max(relative_light, 0.0))
                    * np.log(np.e + float(result["building_coverage"]))
                )
                rows.append(result)
                if number % 50 == 0:
                    print(f"features {number}/{len(manifest)}", flush=True)
        finally:
            high.close()
            ntl.close()

    frame = pd.DataFrame(rows)
    train = frame.split == "train"
    poi_columns = [f"poi_count_{i}" for i in range(14)]
    counts = frame[poi_columns].to_numpy(dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    tf = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    document_frequency = (counts[train] > 0).sum(axis=0)
    idf = np.log((train.sum() + 1) / (document_frequency + 1)) + 1
    for category in range(14):
        frame[f"poi_tfidf_{category}"] = tf[:, category] * idf[category]
    frame.to_csv(output / "block_features.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "paper": "Wang et al. (2026), doi:10.3389/frsc.2026.1736773",
        "sample_count": len(frame),
        "split_counts": frame.split.value_counts().to_dict(),
        "building_has_height": False,
        "nightlight_native_resolution": "approximately 500 m",
        "high_resolution_native_resolution": "approximately 1.1 m RGB",
        "proxy_warning": "building_nightlight_proxy is not PRENI because building height/floor data are absent",
        "columns": frame.columns.tolist(),
    }
    (output / "feature_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
