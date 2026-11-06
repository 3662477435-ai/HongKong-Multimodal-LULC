from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.warp import reproject

from .config import load_config


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Align high-resolution, VIIRS and buildings to the 10 m grid.")
    parser.add_argument("--config", default="config_paper_replication.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    prepared = Path(cfg["paths"]["prepared_dir"])
    prepared.mkdir(parents=True, exist_ok=True)
    high_path = prepared / "highres_rgb_10m.tif"
    ntl_path = prepared / "viirs_10m.tif"
    building_path = prepared / "building_function_9band_10m.tif"

    with rasterio.open(cfg["paths"]["image"]) as reference:
        common = reference.profile.copy()
        common.update(tiled=True, blockxsize=256, blockysize=256, compress="deflate")
        with rasterio.open(cfg["paths"]["high_resolution"]) as source:
            profile = common.copy()
            profile.update(count=3, dtype="uint8", nodata=0)
            with rasterio.open(high_path, "w", **profile) as destination:
                for band, name in enumerate(("highres_red", "highres_green", "highres_blue"), start=1):
                    reproject(
                        source=rasterio.band(source, band),
                        destination=rasterio.band(destination, band),
                        src_transform=source.transform,
                        src_crs=source.crs,
                        src_nodata=source.nodata,
                        dst_transform=reference.transform,
                        dst_crs=reference.crs,
                        dst_nodata=0,
                        resampling=Resampling.average,
                        num_threads=4,
                    )
                    destination.set_band_description(band, name)
                    print(f"high-resolution band {band}/3", flush=True)
        with rasterio.open(cfg["paths"]["nighttime_light"]) as source:
            profile = common.copy()
            profile.update(count=3, dtype="float32", nodata=np.nan)
            with rasterio.open(ntl_path, "w", **profile) as destination:
                for band, name in enumerate(("VIIRS_avg_rad", "VIIRS_log1p", "VIIRS_cf_cvg"), start=1):
                    reproject(
                        source=rasterio.band(source, band),
                        destination=rasterio.band(destination, band),
                        src_transform=source.transform,
                        src_crs=source.crs,
                        src_nodata=source.nodata,
                        dst_transform=reference.transform,
                        dst_crs=reference.crs,
                        dst_nodata=np.nan,
                        resampling=Resampling.bilinear,
                        num_threads=4,
                    )
                    destination.set_band_description(band, name)
                    print(f"nightlight band {band}/3", flush=True)
        buildings = gpd.read_file(cfg["paths"]["buildings"], columns=["Function", "geometry"]).to_crs(reference.crs)
        profile = common.copy()
        profile.update(count=9, dtype="uint8", nodata=0)
        with rasterio.open(building_path, "w", **profile) as destination:
            for band, function in enumerate(BUILDING_FUNCTIONS, start=1):
                geometries = buildings.loc[buildings.Function == function, "geometry"]
                values = rasterize(
                    ((geometry, 1) for geometry in geometries),
                    out_shape=(reference.height, reference.width),
                    transform=reference.transform,
                    fill=0,
                    all_touched=True,
                    dtype="uint8",
                )
                destination.write(values, band)
                destination.set_band_description(band, f"building_{function.lower().replace(' ', '_')}")
                print(f"building function {band}/9: {function}", flush=True)
    print(high_path)
    print(ntl_path)
    print(building_path)


if __name__ == "__main__":
    main()
