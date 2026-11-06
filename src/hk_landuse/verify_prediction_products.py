from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio


def summarize(path: Path, reference) -> dict:
    histogram: dict[int, int] = {}
    minimum = np.inf
    maximum = -np.inf
    total = 0.0
    count = 0
    with rasterio.open(path) as src:
        aligned = (
            src.width == reference.width
            and src.height == reference.height
            and src.crs == reference.crs
            and src.transform == reference.transform
        )
        for _, window in src.block_windows(1):
            array = src.read(1, window=window)
            finite = np.isfinite(array)
            if not finite.any():
                continue
            values = array[finite]
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
            total += float(values.sum())
            count += int(values.size)
            if src.dtypes[0] == "uint8":
                ids, counts = np.unique(values, return_counts=True)
                for key, value in zip(ids, counts):
                    histogram[int(key)] = histogram.get(int(key), 0) + int(value)
        return {
            "path": str(path),
            "size": [src.width, src.height],
            "crs": str(src.crs),
            "transform": list(src.transform),
            "dtype": src.dtypes[0],
            "nodata": src.nodata,
            "aligned_to_reference": aligned,
            "min": minimum,
            "max": maximum,
            "mean": total / max(count, 1),
            "histogram": histogram,
            "overviews": src.overviews(1),
            "tags": src.tags(),
        }


def check_mask(path: Path, mask_path: Path) -> dict:
    invalid_nonzero = 0
    valid_zero = 0
    with rasterio.open(path) as src, rasterio.open(mask_path) as mask_src:
        for _, window in src.block_windows(1):
            data = src.read(1, window=window)
            valid = mask_src.read(1, window=window) != 255
            invalid_nonzero += int(((data != 0) & ~valid).sum())
            if "classification" in path.name:
                valid_zero += int(((data == 0) & valid).sum())
    return {"invalid_area_nonzero": invalid_nonzero, "valid_area_zero": valid_zero}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--prediction-dir", required=True, type=Path)
    parser.add_argument("--valid-mask", type=Path)
    args = parser.parse_args()
    with rasterio.open(args.reference) as reference:
        summaries = {
            path.name: summarize(path, reference)
            for path in sorted(args.prediction_dir.glob("*.tif"))
        }
    failures = []
    for name, summary in summaries.items():
        if not summary["aligned_to_reference"]:
            failures.append(f"{name}: grid mismatch")
        if "classification" in name and not (summary["min"] >= 0 and summary["max"] <= 10):
            failures.append(f"{name}: class values outside nodata=0 or classes 1..10")
        if "confidence" in name and not (summary["min"] >= 0 and summary["max"] <= 1):
            failures.append(f"{name}: confidence outside 0..1")
        if args.valid_mask:
            mask_check = check_mask(args.prediction_dir / name, args.valid_mask)
            summary["BLU_mask_check"] = mask_check
            if mask_check["invalid_area_nonzero"]:
                failures.append(f"{name}: nonzero pixels exist where BLU is zero/unmapped")
            if "classification" in name and mask_check["valid_area_zero"]:
                failures.append(f"{name}: zero class exists inside valid BLU area")
    report = {"status": "passed" if not failures else "failed", "failures": failures, "products": summaries}
    (args.prediction_dir / "prediction_qc.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "files": len(summaries), "failures": failures}, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
