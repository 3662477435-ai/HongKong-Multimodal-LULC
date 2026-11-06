"""Clip a native-grid VIIRS product to the valid BLU 1--9 map footprint."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, shapes
from rasterio.windows import Window, from_bounds, transform as window_transform
from shapely.geometry import box, shape
from shapely.ops import unary_union


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blu", type=Path, required=True)
    parser.add_argument("--viirs", type=Path, required=True)
    parser.add_argument("--output-tif", type=Path, required=True)
    parser.add_argument("--output-shp", type=Path, required=True)
    args = parser.parse_args()

    # BLU values 1--9 are the eight target classes plus display-only Other.
    with rasterio.open(args.blu) as blu:
        data = blu.read(1)
        valid = (data >= 1) & (data <= 9)
        polygons = [shape(geometry) for geometry, value in shapes(valid.astype(np.uint8), mask=valid, transform=blu.transform) if value == 1]
        if not polygons:
            raise ValueError("No BLU values in the requested 1--9 range.")
        valid_geometry = unary_union(polygons)
        valid_gdf = gpd.GeoDataFrame({"maskclass": ["BLU_1_to_9"]}, geometry=[valid_geometry], crs=blu.crs)

    with rasterio.open(args.viirs) as source:
        geometry_viirs = valid_gdf.to_crs(source.crs).geometry.iloc[0]
        left, bottom, right, top = geometry_viirs.bounds
        raw_window = from_bounds(left, bottom, right, top, source.transform)
        col_off = max(0, int(np.floor(raw_window.col_off)))
        row_off = max(0, int(np.floor(raw_window.row_off)))
        col_end = min(source.width, int(np.ceil(raw_window.col_off + raw_window.width)))
        row_end = min(source.height, int(np.ceil(raw_window.row_off + raw_window.height)))
        window = Window(col_off, row_off, col_end - col_off, row_end - row_off)
        data = source.read(window=window)
        transform = window_transform(window, source.transform)
        inside = geometry_mask(
            [geometry_viirs.__geo_interface__],
            out_shape=(int(window.height), int(window.width)),
            transform=transform,
            invert=True,
            all_touched=True,
        )
        nodata = source.nodata if source.nodata is not None else -9999.0
        data[:, ~inside] = nodata
        profile = source.profile.copy()
        profile.update(
            width=int(window.width), height=int(window.height), transform=transform,
            nodata=nodata, compress="deflate", tiled=True,
        )
        args.output_tif.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(args.output_tif, "w", **profile) as destination:
            destination.write(data)
            for band, description in enumerate(source.descriptions, 1):
                if description:
                    destination.set_band_description(band, description)
            destination.update_tags(
                clip_mask="BLU values 1--9",
                mask_method="BLU polygon; native VIIRS cells intersecting polygon retained (all_touched=True)",
                mask_crs=str(source.crs),
            )

    # Export the same valid region in the VIIRS CRS.  This represents the
    # retained geographic footprint, rather than merely the output bbox.
    output_gdf = valid_gdf.to_crs(source.crs)
    output_gdf["viirsgrid"] = "native"
    args.output_shp.parent.mkdir(parents=True, exist_ok=True)
    output_gdf.to_file(args.output_shp, driver="ESRI Shapefile", encoding="UTF-8")
    print({
        "tif": str(args.output_tif),
        "shp": str(args.output_shp),
        "native_grid_shape": [int(window.height), int(window.width)],
        "retained_cells": int(inside.sum()),
    })


if __name__ == "__main__":
    main()
