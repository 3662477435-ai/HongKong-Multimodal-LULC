from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import box

from .config import load_config


FUNCTIONS = (
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


def _load_height_tiles(root: Path, bbox_wgs84: tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    parts = []
    for path in sorted(root.glob("*.shp")):
        frame = gpd.read_file(path, bbox=bbox_wgs84, columns=["Height", "id", "geometry"])
        frame = frame[~frame.geometry.is_empty & frame.geometry.notna()].copy()
        centroid_y = frame.geometry.centroid.y
        if path.name.startswith("2269_"):
            frame = frame[centroid_y < 22.5].copy()
        else:
            frame = frame[centroid_y >= 22.5].copy()
        frame["source_tile"] = path.stem.split("_")[0]
        frame["height_m"] = pd.to_numeric(frame["Height"], errors="coerce")
        parts.append(frame[["height_m", "source_tile", "geometry"]])
    merged = pd.concat(parts, ignore_index=True)
    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    invalid_before = int((~merged.geometry.is_valid).sum())
    if invalid_before:
        merged.geometry = merged.geometry.make_valid()
    merged = merged[merged.geom_type.isin(("Polygon", "MultiPolygon"))].copy()
    merged = merged.clip(box(*bbox_wgs84))
    merged = merged[~merged.geometry.is_empty].copy()
    merged["height_valid"] = np.isfinite(merged.height_m) & (merged.height_m > 0)
    median_height = float(merged.loc[merged.height_valid, "height_m"].median())
    merged["height_m"] = merged.height_m.where(merged.height_valid, median_height).clip(1.0, 300.0)
    merged.attrs["invalid_geometry_before_repair"] = invalid_before
    merged.attrs["median_height_imputation_m"] = median_height
    return merged


def _transfer_function(height: gpd.GeoDataFrame, function_path: Path) -> tuple[gpd.GeoDataFrame, dict]:
    function = gpd.read_file(function_path, columns=["Function", "geometry"])
    function = function[function.geometry.notna() & ~function.geometry.is_empty].copy()
    if (~function.geometry.is_valid).any():
        function.geometry = function.geometry.make_valid()
    function = function.to_crs(height.crs)
    centroids = gpd.GeoDataFrame(
        {"height_index": height.index}, geometry=height.geometry.representative_point(), crs=height.crs
    )
    joined = gpd.sjoin(centroids, function[["Function", "geometry"]], predicate="within", how="left")
    joined = joined.drop_duplicates("height_index", keep="first").set_index("height_index")
    height["Function"] = joined["Function"].reindex(height.index).fillna("Unknown")
    height.loc[~height.Function.isin(FUNCTIONS), "Function"] = "Unknown"
    report = {
        "function_reference_count": int(len(function)),
        "function_matched_count": int((height.Function != "Unknown").sum()),
        "function_unknown_count": int((height.Function == "Unknown").sum()),
        "function_counts": height.Function.value_counts().to_dict(),
    }
    return height, report


def _write_features(height: gpd.GeoDataFrame, reference_path: str, output: Path) -> None:
    with rasterio.open(reference_path) as reference:
        profile = reference.profile.copy()
        transform = reference.transform
        height_px, width_px = reference.height, reference.width
    function_bands = []
    for name in FUNCTIONS:
        subset = height[height.Function == name]
        function_bands.append(
            rasterize(
                ((geometry, 1) for geometry in subset.geometry),
                out_shape=(height_px, width_px),
                transform=transform,
                fill=0,
                all_touched=True,
                dtype="float32",
            )
        )

    scale = 2
    fine_shape = (height_px * scale, width_px * scale)
    fine_transform = transform * Affine.scale(1.0 / scale, 1.0 / scale)
    fine_mask = rasterize(
        ((geometry, 1) for geometry in height.geometry),
        out_shape=fine_shape,
        transform=fine_transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )
    # Sort so the tallest building wins where polygons overlap.
    ordered = height.sort_values("height_m")
    fine_height = rasterize(
        ((geometry, value) for geometry, value in zip(ordered.geometry, ordered.height_m)),
        out_shape=fine_shape,
        transform=fine_transform,
        fill=0,
        all_touched=True,
        dtype="float32",
    )
    mask_blocks = fine_mask.reshape(height_px, scale, width_px, scale)
    height_blocks = fine_height.reshape(height_px, scale, width_px, scale)
    coverage = mask_blocks.mean(axis=(1, 3)).astype(np.float32)
    count = mask_blocks.sum(axis=(1, 3)).astype(np.float32)
    mean_height = np.divide(
        (height_blocks * mask_blocks).sum(axis=(1, 3)),
        count,
        out=np.zeros((height_px, width_px), dtype=np.float32),
        where=count > 0,
    )
    max_height = height_blocks.max(axis=(1, 3)).astype(np.float32)
    floor_area_ratio_proxy = coverage * mean_height / 3.0
    height_range = np.maximum(max_height - mean_height, 0.0)
    bands = [
        *function_bands,
        coverage,
        mean_height,
        max_height,
        floor_area_ratio_proxy,
        height_range,
    ]
    descriptions = [
        *(f"building_{name.lower().replace(' ', '_')}" for name in FUNCTIONS),
        "building_coverage_fraction_5m_subpixels",
        "building_mean_height_m",
        "building_max_height_m",
        "building_floor_area_ratio_proxy_3m_floor",
        "building_height_range_m",
    ]
    profile.update(
        count=len(bands),
        dtype="float32",
        nodata=0,
        compress="deflate",
        predictor=3,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **profile) as dst:
        for index, (band, description) in enumerate(zip(bands, descriptions), 1):
            dst.write(band.astype(np.float32), index)
            dst.set_band_description(index, description)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge height tiles and build height-aware 10 m features.")
    parser.add_argument("--config", default="config_release.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    vector_output = Path(cfg["paths"]["building_height_vector"])
    raster_output = Path(cfg["paths"]["building_aligned"])
    with rasterio.open(cfg["paths"]["image"]) as reference:
        bbox_wgs84 = transform_bounds(reference.crs, "EPSG:4326", *reference.bounds, densify_pts=21)
        study_box_utm = box(*reference.bounds)
        target_crs = reference.crs
    height = _load_height_tiles(Path(cfg["paths"]["building_height_dir"]), bbox_wgs84)
    load_attributes = dict(height.attrs)
    height = height.to_crs(target_crs)
    height = height.clip(study_box_utm)
    height = height[~height.geometry.is_empty].reset_index(drop=True)
    height, function_report = _transfer_function(
        height, Path(cfg["paths"]["building_function_source"])
    )
    vector_output.parent.mkdir(parents=True, exist_ok=True)
    height.to_file(vector_output, layer="buildings", driver="GPKG")
    _write_features(height, cfg["paths"]["image"], raster_output)
    report = {
        "study_bbox_wgs84": list(bbox_wgs84),
        "target_crs": str(target_crs),
        "merged_clipped_building_count": int(len(height)),
        "invalid_geometry_before_repair": load_attributes.get("invalid_geometry_before_repair", 0),
        "median_height_imputation_m": load_attributes.get("median_height_imputation_m"),
        "height_statistics_m": {
            "min": float(height.height_m.min()),
            "p01": float(height.height_m.quantile(0.01)),
            "median": float(height.height_m.median()),
            "mean": float(height.height_m.mean()),
            "p99": float(height.height_m.quantile(0.99)),
            "max": float(height.height_m.max()),
        },
        **function_report,
        "vector_output": str(vector_output),
        "raster_output": str(raster_output),
    }
    report_path = raster_output.with_name("building_height_preparation_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
