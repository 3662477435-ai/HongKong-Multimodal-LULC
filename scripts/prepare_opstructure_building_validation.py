"""Prepare transparent 8-class external building-use validation labels.

The Hong Kong Building Master contains a structural/occupation-permit (OP)
record for many buildings, but an explicit approved-use description only for a
subset.  This script never invents a functional label for records without
``OPBuildingUses``.  It maps only semantically unambiguous use descriptions to
the project's eight-class code space:

    1 Residential; 2 Industrial/warehouse; 4 Transportation;
    5 Other built-up.

The ``Other built-up`` validation label is intentionally a heterogeneous
catch-all for declared non-residential, non-industrial and non-transport
building uses.  It should therefore be used as an external building-use check,
not claimed to be identical to the BLU ``other built-up`` class.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol


CLASS_NAMES = {
    1: "Residential",
    2: "Industrial/warehouse",
    4: "Transportation",
    5: "Other built-up",
}

# These patterns deliberately target *primary functional uses*.  We avoid a
# generic "station" pattern because electricity, pumping and communications
# stations are not transport facilities; and avoid a generic "storage" pattern
# because many residential developments mention refuse storage.
PATTERNS = {
    "residential": re.compile(
        r"\b(residence|residential|apartment|flat|domestic|single family house|"
        r"transitional housing|dwelling|housing)\b",
        flags=re.IGNORECASE,
    ),
    "industrial": re.compile(
        r"\b(industrial|factory|warehouse|godown|logistics?|manufactur\w*|"
        r"workshop)\b",
        flags=re.IGNORECASE,
    ),
    "transport": re.compile(
        r"\b(public transport|transport interchange|transport terminus|"
        r"bus (?:depot|terminus|terminal)|ferry (?:terminal|pier)|passenger ferry|"
        r"railway|mtr\b|kcrc|airport|air cargo|airfreight|cargo terminal|"
        r"passenger terminal|cable car|funicular|petrol filling|aircraft|pier\b)",
        flags=re.IGNORECASE,
    ),
}

# Workshop can denote a school or assisted-workshop facility rather than an
# industrial building.  In those cases it is retained as heterogeneous other
# built-up rather than labelled industrial.
INDUSTRIAL_WORKSHOP_EXCLUSION = re.compile(
    r"(sheltered workshop|school.{0,80}workshop|college.{0,80}workshop|"
    r"drama studio|variety studio|broadcasting)", flags=re.IGNORECASE
)


def derive_mapping(use: object) -> tuple[float, str, str, bool]:
    """Return code, class, deterministic rule and strict eligibility."""
    if not isinstance(use, str) or not use.strip():
        return np.nan, "Unlabelled", "No approved-use text", False

    text = " ".join(use.split())
    flags = {name: bool(pattern.search(text)) for name, pattern in PATTERNS.items()}
    if flags["industrial"] and INDUSTRIAL_WORKSHOP_EXCLUSION.search(text):
        # Retain the original value in the audit; only remove a workshop-only
        # signal, so explicit 'industrial' or 'factory' remains industrial.
        flags["industrial"] = bool(
            re.search(r"\b(industrial|factory|warehouse|godown|logistics?|manufactur\w*)\b", text, re.I)
        )

    active = [name for name, matched in flags.items() if matched]
    if len(active) > 1:
        return np.nan, "Mixed use / excluded", "Conflicting target-use terms: " + "+".join(active), False
    if flags["residential"]:
        return 1.0, CLASS_NAMES[1], "Approved-use text contains residential term", True
    if flags["industrial"]:
        return 2.0, CLASS_NAMES[2], "Approved-use text contains industrial term", True
    if flags["transport"]:
        return 4.0, CLASS_NAMES[4], "Approved-use text contains transport-facility term", True
    return 5.0, CLASS_NAMES[5], "Declared non-target building use mapped to other built-up", True


def sample_split(geoms: gpd.GeoSeries, split_raster: Path) -> np.ndarray:
    """Assign a spatial-block split using polygon representative points."""
    with rasterio.open(split_raster) as src:
        points = gpd.GeoSeries(geoms.representative_point(), crs=geoms.crs).to_crs(src.crs)
        out = np.zeros(len(points), dtype=np.uint8)
        for i, point in enumerate(points):
            try:
                r, c = rowcol(src.transform, point.x, point.y)
                if 0 <= r < src.height and 0 <= c < src.width:
                    value = src.read(1, window=((r, r + 1), (c, c + 1)))[0, 0]
                    out[i] = int(value)
            except Exception:
                pass
    return out


def export_raster(labels: gpd.GeoDataFrame, reference_raster: Path, output_raster: Path) -> dict[int, int]:
    """Rasterize reference polygons to the fixed 10 m project grid."""
    with rasterio.open(reference_raster) as ref:
        project = labels.to_crs(ref.crs)
        valid = project[project["lulc8_code"].notna()].copy()
        shapes = ((geom, int(code)) for geom, code in zip(valid.geometry, valid["lulc8_code"]))
        arr = rasterize(
            shapes=shapes,
            out_shape=(ref.height, ref.width),
            transform=ref.transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )
        profile = ref.profile.copy()
        profile.update(count=1, dtype="uint8", nodata=0, compress="lzw")
        with rasterio.open(output_raster, "w", **profile) as dst:
            dst.write(arr, 1)
    values, counts = np.unique(arr, return_counts=True)
    return {int(v): int(n) for v, n in zip(values, counts)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference-raster", type=Path, required=True)
    parser.add_argument("--split-raster", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_columns = [
        "BuildingCSUID", "BuildingID", "GeoRefNo", "Status", "BuildingBlockType",
        "PreferredName_EN", "PreferredName_ZH", "BaseHeight_m", "TopHeight_m", "Storeys",
        "ShapeArea_m2", "DateCreated", "DateStamp", "HasOPStructure", "OPStructureCount",
        "OPBuildingUses", "OPStructureDetails", "OPBlockTypes", "OPPermitDetails",
    ]
    gdf = pyogrio.read_dataframe(
        args.source, layer="building_master", columns=source_columns, use_arrow=False
    )
    op = gdf[gdf["HasOPStructure"] == True].copy()  # noqa: E712
    results = op["OPBuildingUses"].apply(derive_mapping)
    op[["lulc8_code", "lulc8_class", "mapping_rule", "validation_eligible"]] = pd.DataFrame(
        results.tolist(), index=op.index
    )
    op["lulc8_code"] = op["lulc8_code"].astype("Float64")
    op["split_code"] = sample_split(op.geometry, args.split_raster)
    op["split_name"] = op["split_code"].map({0: "Outside reference grid", 1: "Train", 2: "Validation", 3: "Test"})
    op["source_dataset"] = args.source.name
    op["label_source"] = "OPBuildingUses"

    # Audit every distinct source term, including source terms that are mixed
    # and therefore excluded from the validation subset.
    term_audit = (
        op.groupby(["OPBuildingUses", "lulc8_code", "lulc8_class", "mapping_rule", "validation_eligible"], dropna=False)
        .size().rename("building_count").reset_index()
        .sort_values(["validation_eligible", "building_count", "OPBuildingUses"], ascending=[False, False, True])
    )
    term_audit.to_csv(args.output_dir / "OPBuildingUses_all_terms_and_mapping.csv", index=False, encoding="utf-8-sig")

    # All OP structures are included here with no invented labels for records
    # without function descriptions; this is the audit source product.
    keep_all = [
        "BuildingCSUID", "BuildingID", "GeoRefNo", "Status", "BuildingBlockType", "PreferredName_EN",
        "PreferredName_ZH", "BaseHeight_m", "TopHeight_m", "Storeys", "ShapeArea_m2", "DateCreated",
        "DateStamp", "HasOPStructure", "OPStructureCount", "OPBuildingUses", "OPStructureDetails",
        "OPBlockTypes", "OPPermitDetails", "lulc8_code", "lulc8_class", "mapping_rule",
        "validation_eligible", "split_code", "split_name", "source_dataset", "label_source", "geometry",
    ]
    op[keep_all].to_file(args.output_dir / "HK_Building_OPStructure_all_52886.gpkg", layer="opstructure_all", driver="GPKG")

    validation = op[op["validation_eligible"]].copy()
    validation[keep_all].to_file(
        args.output_dir / "HK_Building_OPStructure_validation8_strict.gpkg",
        layer="validation8_strict", driver="GPKG"
    )
    test_validation = validation[validation["split_code"] == 3].copy()
    test_validation[keep_all].to_file(
        args.output_dir / "HK_Building_OPStructure_validation8_test_blocks.gpkg",
        layer="validation8_test_blocks", driver="GPKG"
    )

    full_counts = export_raster(
        validation, args.reference_raster,
        args.output_dir / "HK_Building_OPStructure_validation8_strict_10m.tif"
    )
    test_counts = export_raster(
        test_validation, args.reference_raster,
        args.output_dir / "HK_Building_OPStructure_validation8_test_blocks_10m.tif"
    )

    summary = pd.DataFrame([
        {"scope": "All Building Master records", "records": len(gdf)},
        {"scope": "HasOPStructure=True", "records": len(op)},
        {"scope": "Has OP structure but no OPBuildingUses", "records": int(op["OPBuildingUses"].isna().sum())},
        {"scope": "OPBuildingUses present", "records": int(op["OPBuildingUses"].notna().sum())},
        {"scope": "Strict validation eligible", "records": len(validation)},
        {"scope": "Strict validation eligible in test blocks", "records": len(test_validation)},
    ])
    class_summary = (
        validation.groupby(["lulc8_code", "lulc8_class"]).size().rename("building_count").reset_index()
    )
    class_summary["lulc8_code"] = class_summary["lulc8_code"].astype(int)
    class_summary["scope"] = "Strict validation eligible"
    pixel_summary = pd.DataFrame([
        {"scope": "All strict validation labels rasterized", "lulc8_code": code, "raster_pixels": n}
        for code, n in full_counts.items() if code != 0
    ] + [
        {"scope": "Test-block strict validation labels rasterized", "lulc8_code": code, "raster_pixels": n}
        for code, n in test_counts.items() if code != 0
    ])
    summary.to_csv(args.output_dir / "validation8_preparation_summary.csv", index=False, encoding="utf-8-sig")
    class_summary.to_csv(args.output_dir / "validation8_class_counts.csv", index=False, encoding="utf-8-sig")
    pixel_summary.to_csv(args.output_dir / "validation8_raster_pixel_counts.csv", index=False, encoding="utf-8-sig")

    print("Output directory:", args.output_dir)
    print(summary.to_string(index=False))
    print("\nClass counts:\n", class_summary.to_string(index=False))
    print("\nRaster pixels (full / test):", full_counts, test_counts)


if __name__ == "__main__":
    main()
