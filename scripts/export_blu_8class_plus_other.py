"""Export a 10-m BLU reference raster with eight target classes plus Other.

Output values: 0 = BLU no-data/background; 1--8 = the active eight-class
target; 9 = Other (commercial/office or barren/rock), retained only for map
display and excluded from the eight-class training/evaluation protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio


# Prepared labels_10class values: 0=residential, 1=commercial, 2=industrial,
# 3=public/recreation, 4=transport, 5=other built-up, 6=agriculture,
# 7=woodland etc., 8=barren, 9=water; 255=no-data.
OUTPUT_VALUES = {
    0: 1,  # residential
    2: 2,  # industrial/warehouse
    3: 3,  # public/institutional and recreation
    4: 4,  # transportation
    5: 5,  # other built-up
    6: 6,  # agriculture
    7: 7,  # woodland/shrubland/grassland/wetland
    9: 8,  # water
    1: 9,  # commercial/office -> Other
    8: 9,  # barren/rock -> Other
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with rasterio.open(args.input) as source:
        label10 = source.read(1)
        result = np.zeros(label10.shape, dtype=np.uint8)
        for old_value, new_value in OUTPUT_VALUES.items():
            result[label10 == old_value] = new_value
        profile = source.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="deflate", tiled=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(args.output, "w", **profile) as destination:
            destination.write(result, 1)
            destination.set_band_description(1, "BLU_8class_plus_other")
            destination.update_tags(
                value_0="no_data_or_background",
                value_1="residential",
                value_2="industrial_warehouse",
                value_3="public_institutional_recreation",
                value_4="transportation",
                value_5="other_built_up",
                value_6="agriculture",
                value_7="woodland_shrubland_grassland_wetland",
                value_8="water",
                value_9="other_commercial_office_or_barren_rock",
                source_label="BLU_10_coarse_classes",
                resampling="nearest_to_10m_reference_grid",
            )

    print({"output": str(args.output), "values": {i: int((result == i).sum()) for i in range(10)}})


if __name__ == "__main__":
    main()
