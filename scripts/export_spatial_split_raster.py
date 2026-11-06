"""Export the panel-a spatial-block allocation as a georeferenced raster.

Values: 0 = outside retained patches, 1 = training, 2 = validation,
3 = test.  The raster exactly follows the reference 10-m grid.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import rasterio


SPLIT_VALUE = {"train": 1, "val": 2, "test": 3}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    required = {"split", "row_off", "col_off", "height", "width"}
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        rows = [row for row in reader if row.get("split") in SPLIT_VALUE]
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    with rasterio.open(args.reference) as reference:
        allocation = np.zeros((reference.height, reference.width), dtype=np.uint8)
        for row in rows:
            value = SPLIT_VALUE[row["split"]]
            row0, col0 = int(row["row_off"]), int(row["col_off"])
            row1 = min(row0 + int(row["height"]), reference.height)
            col1 = min(col0 + int(row["width"]), reference.width)
            allocation[row0:row1, col0:col1] = value

        profile = reference.profile.copy()
        profile.update(
            count=1,
            dtype="uint8",
            nodata=0,
            compress="deflate",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(args.output, "w", **profile) as destination:
            destination.write(allocation, 1)
            destination.set_band_description(1, "spatial_split")
            destination.update_tags(
                value_0="outside_retained_patch",
                value_1="training",
                value_2="validation",
                value_3="test",
                source_manifest=str(args.manifest),
                patch_size_pixels="128x128",
                pixel_size_m="10",
            )

    counts = {name: int((allocation == value).sum()) for name, value in SPLIT_VALUE.items()}
    print({"output": str(args.output), "pixel_counts": counts})


if __name__ == "__main__":
    main()
