from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from fiona import listlayers


DATA_ROOT = Path(r"<third_party_sources>\数据")
LABEL_PATH = DATA_ROOT.parent / "BLU.tif"
OSM_ROOT = DATA_ROOT / "HK_Kowloon_OSM_training_data"


def raster_summary(path: Path) -> dict:
    with rasterio.open(path) as src:
        result = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "shape": [src.height, src.width],
            "count": src.count,
            "dtypes": list(src.dtypes),
            "crs": str(src.crs),
            "transform": list(src.transform)[:6],
            "bounds": list(src.bounds),
            "nodata": src.nodata,
            "descriptions": list(src.descriptions),
            "colorinterp": [str(x) for x in src.colorinterp],
            "bands": [],
        }
        for idx in range(1, src.count + 1):
            band = src.read(idx, masked=True)
            valid = band.compressed()
            stats = {
                "index": idx,
                "valid_pixels": int(valid.size),
                "masked_pixels": int(band.size - valid.size),
            }
            if valid.size:
                q = np.percentile(valid, [0, 1, 50, 99, 100])
                stats.update(
                    min=float(q[0]),
                    p01=float(q[1]),
                    median=float(q[2]),
                    p99=float(q[3]),
                    max=float(q[4]),
                    mean=float(valid.mean()),
                    std=float(valid.std()),
                )
                if np.issubdtype(valid.dtype, np.integer) and np.unique(valid).size <= 100:
                    values, counts = np.unique(valid, return_counts=True)
                    stats["value_counts"] = {
                        str(int(v)): int(c) for v, c in zip(values, counts)
                    }
            result["bands"].append(stats)
    return result


def csv_summary(path: Path) -> dict:
    df = pd.read_csv(path, low_memory=False)
    return {
        "path": str(path),
        "rows": len(df),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "missing": {c: int(n) for c, n in df.isna().sum().items() if n},
        "duplicates": int(df.duplicated().sum()),
        "head": df.head(3).fillna("").to_dict(orient="records"),
        "top_values": {
            c: df[c].fillna("<NA>").astype(str).value_counts().head(12).to_dict()
            for c in df.columns
            if df[c].nunique(dropna=False) <= 500
        },
    }


def vector_summary(path: Path) -> dict:
    output = {"path": str(path), "layers": {}}
    for layer in listlayers(path):
        gdf = gpd.read_file(path, layer=layer)
        invalid = int((~gdf.geometry.is_valid & gdf.geometry.notna()).sum())
        output["layers"][layer] = {
            "rows": len(gdf),
            "crs": str(gdf.crs),
            "bounds": list(gdf.total_bounds),
            "columns": list(gdf.columns),
            "geometry_types": gdf.geometry.geom_type.value_counts(dropna=False).to_dict(),
            "null_geometry": int(gdf.geometry.isna().sum()),
            "empty_geometry": int(gdf.geometry.is_empty.sum()),
            "invalid_geometry": invalid,
            "duplicate_geometry": int(gdf.geometry.to_wkb().duplicated().sum()),
            "missing": {c: int(n) for c, n in gdf.isna().sum().items() if n},
            "head": gdf.drop(columns="geometry").head(3).fillna("").to_dict(orient="records"),
        }
    return output


def main() -> None:
    rasters = sorted(DATA_ROOT.rglob("*.tif")) + [LABEL_PATH]
    csvs = sorted(OSM_ROOT.glob("*.csv"))
    vectors = sorted(OSM_ROOT.glob("*.gpkg"))
    report = {
        "rasters": [raster_summary(p) for p in rasters],
        "csvs": [csv_summary(p) for p in csvs],
        "vectors": [vector_summary(p) for p in vectors],
    }
    out = Path(__file__).resolve().parents[1] / "data_inspection.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
