from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import rasterio

from .config import load_config


DESCRIPTIONS = (
    "road_class_1_motorway_trunk",
    "road_class_2_primary",
    "road_class_3_secondary",
    "road_class_4_tertiary",
    "road_class_5_residential_local",
    "road_class_6_service_unclassified",
    "road_class_7_foot_cycle",
    "road_class_8_rural_track",
    "road_class_9_other",
    "road_surface_mask",
    "road_centerline_mask",
    "road_width_0_30m_quantized",
    "road_proximity_exp_250m_quantized",
)


def _assert_grid(source, reference) -> None:
    if (
        source.width != reference.width
        or source.height != reference.height
        or source.crs != reference.crs
        or source.transform != reference.transform
    ):
        raise ValueError(f"Road raster grid does not match S1/S2 reference: {source.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack aligned OSM roads into compact 13-band features.")
    parser.add_argument("--config", default="config_hierarchical.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["paths"]["roads_dir"])
    output = Path(cfg["paths"]["roads_aligned"])
    output.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(cfg["paths"]["image"]) as reference:
        profile = reference.profile.copy()
        with rasterio.open(root / "osm_road_surface_class_10m.tif") as src:
            _assert_grid(src, reference)
            road_class = src.read(1)
        with rasterio.open(root / "osm_road_surface_mask_10m.tif") as src:
            _assert_grid(src, reference)
            surface = src.read(1).astype(np.uint8)
        with rasterio.open(root / "osm_road_centerline_mask_10m.tif") as src:
            _assert_grid(src, reference)
            centerline = src.read(1).astype(np.uint8)
        with rasterio.open(root / "osm_road_width_10m.tif") as src:
            _assert_grid(src, reference)
            width = src.read(1).astype(np.float32)

    # cv2 returns float32 pixel distances and avoids a full float64 31M-pixel array.
    distance_pixels = cv2.distanceTransform((centerline == 0).astype(np.uint8), cv2.DIST_L2, 5)
    proximity = np.exp(-(distance_pixels * 10.0) / 250.0)
    bands = [(road_class == class_id).astype(np.uint8) for class_id in range(1, 10)]
    bands.extend(
        [
            surface,
            centerline,
            np.rint(np.clip(width, 0.0, 30.0) / 30.0 * 255.0).astype(np.uint8),
            np.rint(np.clip(proximity, 0.0, 1.0) * 255.0).astype(np.uint8),
        ]
    )
    profile.update(
        count=len(bands),
        dtype="uint8",
        nodata=0,
        compress="deflate",
        predictor=2,
        tiled=True,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(output, "w", **profile) as dst:
        for index, (band, description) in enumerate(zip(bands, DESCRIPTIONS), 1):
            dst.write(band, index)
            dst.set_band_description(index, description)
    print(output)


if __name__ == "__main__":
    main()
