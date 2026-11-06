from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from skimage.feature import graycomatrix, graycoprops

from .config import load_config
from .sar_features import dual_pol_numpy


def stats(values: np.ndarray, prefix: str, row: dict[str, float]) -> None:
    finite = values[np.isfinite(values)]
    for suffix, value in zip(
        ("mean", "std", "p10", "p50", "p90"),
        (finite.mean(), finite.std(), *np.percentile(finite, (10, 50, 90))),
    ):
        row[f"s1_index_{prefix}_{suffix}"] = float(value)


def texture(values: np.ndarray, prefix: str, row: dict[str, float]) -> None:
    finite = np.isfinite(values)
    lo, hi = np.percentile(values[finite], (1, 99))
    quantized = np.clip((values - lo) / max(hi - lo, 1e-6) * 15, 0, 15).astype(np.uint8)
    matrix = graycomatrix(
        quantized,
        distances=(1, 2),
        angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
        levels=16,
        symmetric=True,
        normed=True,
    )
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        row[f"s1_texture_{prefix}_{prop}"] = float(np.nanmean(graycoprops(matrix, prop)))
    probabilities = matrix[matrix > 0]
    row[f"s1_texture_{prefix}_entropy"] = float(-(probabilities * np.log2(probabilities)).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description="Add dual-pol indices and textures to block features.")
    parser.add_argument("--config", default="config_paper_replication.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(cfg["paths"]["output_dir"])
    frame = pd.read_csv(output / "block_features.csv")
    additions: list[dict[str, float | str]] = []
    with rasterio.open(cfg["paths"]["image"]) as image:
        for number, record in enumerate(frame.itertuples(), start=1):
            window = Window(int(record.col_off), int(record.row_off), 32, 32)
            vv_db, vh_db = image.read((14, 15), window=window, out_dtype="float32")
            derived = dual_pol_numpy(vv_db, vh_db)
            row: dict[str, float | str] = {"patch_id": record.patch_id}
            for name, values in derived.items():
                stats(values, name, row)
            texture(vv_db, "vv", row)
            texture(vh_db, "vh", row)
            texture(vh_db - vv_db, "vh_minus_vv_db", row)
            additions.append(row)
            if number % 500 == 0:
                print(f"SAR features {number}/{len(frame)}", flush=True)
    added = pd.DataFrame(additions)
    old_sar = [column for column in frame if column.startswith("s1_index_") or column.startswith("s1_texture_")]
    frame = frame.drop(columns=old_sar).merge(added, on="patch_id", how="left", validate="one_to_one")
    frame.to_csv(output / "block_features.csv", index=False, encoding="utf-8-sig")
    print(f"added {len(added.columns) - 1} SAR combination/texture features")


if __name__ == "__main__":
    main()
