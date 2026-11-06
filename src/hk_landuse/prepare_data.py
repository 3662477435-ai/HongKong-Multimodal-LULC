from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import Resampling, reproject
from rasterio.windows import Window, bounds as window_bounds
from shapely.geometry import box

from .config import load_config
from .labels import (
    CLASS_NAMES,
    IGNORE_INDEX,
    LEAF_CLASS_NAMES,
    RAW_TO_LEAF,
    RAW_TO_TRAIN,
    remap_blu,
    remap_blu_leaf,
)
from .text import build_vocab


def _count_bin(count: int) -> str:
    if count == 1:
        return "one"
    if count <= 4:
        return "few"
    return "many"


def _area_bin(area_m2: float) -> str:
    if area_m2 < 1_000:
        return "small"
    if area_m2 < 10_000:
        return "medium"
    return "large"


def patch_text(poi: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame, patch_geom) -> str:
    fragments: list[str] = []
    poi_idx = list(poi.sindex.query(patch_geom, predicate="intersects"))
    if poi_idx:
        counts = poi.iloc[poi_idx]["semantic_text"].value_counts().head(24)
        fragments.extend(
            f"{semantic} count_bin={_count_bin(int(count))}"
            for semantic, count in counts.items()
        )
    aoi_idx = list(aoi.sindex.query(patch_geom, predicate="intersects"))
    if aoi_idx:
        candidates = aoi.iloc[aoi_idx][["semantic_text", "geometry"]].copy()
        candidates["intersection_area"] = candidates.geometry.intersection(patch_geom).area
        areas = candidates.groupby("semantic_text")["intersection_area"].sum().sort_values(ascending=False).head(16)
        fragments.extend(
            f"{semantic} area_bin={_area_bin(float(area))}"
            for semantic, area in areas.items()
            if area > 0
        )
    return " ; ".join(fragments) if fragments else "no_osm_context"


def choose_group_splits(
    group_hist: dict[str, np.ndarray],
    train_fraction: float,
    val_fraction: float,
    seed: int,
    trials: int = 1000,
) -> dict[str, str]:
    """Random-search spatial group split with class-distribution matching."""
    groups = sorted(group_hist)
    n = len(groups)
    n_train = max(1, round(n * train_fraction))
    n_val = max(1, round(n * val_fraction))
    if n_train + n_val >= n:
        n_train = n - 2
        n_val = 1
    num_classes = len(next(iter(group_hist.values())))
    global_hist = sum(group_hist.values(), start=np.zeros(num_classes, dtype=np.int64))
    support = np.array([sum(hist[c] > 0 for hist in group_hist.values()) for c in range(num_classes)])
    rng = np.random.default_rng(seed)
    best_score = math.inf
    best: dict[str, str] | None = None
    for _ in range(trials):
        perm = list(rng.permutation(groups))
        sets = {
            "train": perm[:n_train],
            "val": perm[n_train : n_train + n_val],
            "test": perm[n_train + n_val :],
        }
        hists = {
            split: sum((group_hist[g] for g in members), start=np.zeros(num_classes, dtype=np.int64))
            for split, members in sets.items()
        }
        score = 0.0
        for split, target_fraction in (
            ("train", train_fraction),
            ("val", val_fraction),
            ("test", 1.0 - train_fraction - val_fraction),
        ):
            total_fraction = hists[split].sum() / max(global_hist.sum(), 1)
            score += 5.0 * abs(total_fraction - target_fraction)
            present = global_hist > 0
            global_prop = global_hist[present] / global_hist[present].sum()
            split_prop = hists[split][present] / max(hists[split][present].sum(), 1)
            score += float(np.mean(np.abs(split_prop - global_prop)))
        for class_id in np.flatnonzero(global_hist > 0):
            if hists["train"][class_id] == 0:
                score += 100.0
            if support[class_id] >= 3:
                score += 10.0 * sum(hists[s][class_id] == 0 for s in ("val", "test"))
        if score < best_score:
            best_score = score
            best = {group: split for split, members in sets.items() for group in members}
    assert best is not None
    return best


def _write_raster(path: Path, data: np.ndarray, reference, nodata, description: str) -> None:
    profile = reference.profile.copy()
    profile.update(
        count=1,
        dtype=str(data.dtype),
        nodata=nodata,
        compress="deflate",
        tiled=True,
    )
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, description)


def _stream_finite_mask(image_src) -> np.ndarray:
    """Build an exact all-band finite mask without loading the full image cube."""
    mask = np.zeros((image_src.height, image_src.width), dtype=bool)
    for _, window in image_src.block_windows(1):
        block = image_src.read(window=window)
        r0, c0 = int(window.row_off), int(window.col_off)
        h, w = int(window.height), int(window.width)
        mask[r0 : r0 + h, c0 : c0 + w] = np.isfinite(block).all(axis=0)
    return mask


def _balanced_subset(
    manifest: pd.DataFrame,
    limits: dict[str, int],
    min_per_class: int,
    seed: int,
    class_prefix: str = "pixels_class_",
    num_classes: int = 10,
) -> pd.DataFrame:
    """Select a reproducible, class-aware lightweight subset within fixed spatial splits."""
    rng = np.random.default_rng(seed)
    selected_parts: list[pd.DataFrame] = []
    class_cols = [f"{class_prefix}{i}" for i in range(num_classes)]
    for split in ("train", "val", "test"):
        part = manifest[manifest.split == split].copy()
        limit = min(int(limits[split]), len(part))
        selected: set[int] = set()
        class_support = {col: int((part[col] > 0).sum()) for col in class_cols}
        for col in sorted(class_cols, key=lambda name: class_support[name]):
            candidates = part.index[part[col] > 0].to_numpy()
            rng.shuffle(candidates)
            selected.update(int(i) for i in candidates[:min(min_per_class, len(candidates))])
        if len(selected) > limit:
            # Preserve patches carrying the rarest classes first.
            rarity = np.zeros(len(part), dtype=float)
            for col in class_cols:
                support = max(class_support[col], 1)
                rarity += (part[col].to_numpy() > 0) / support
            ranked = part.index[np.argsort(-rarity)]
            selected = set(int(i) for i in ranked[:limit])
        remaining = part.index[~part.index.isin(selected)].to_numpy()
        slots = limit - len(selected)
        if slots > 0 and len(remaining):
            weight = np.full(len(remaining), 0.05, dtype=float)
            remaining_frame = part.loc[remaining]
            for col in class_cols:
                support = max(class_support[col], 1)
                weight += (remaining_frame[col].to_numpy() > 0) / np.sqrt(support)
            weight /= weight.sum()
            chosen = rng.choice(remaining, size=min(slots, len(remaining)), replace=False, p=weight)
            selected.update(int(i) for i in chosen)
        selected_parts.append(part.loc[sorted(selected)])
    return pd.concat(selected_parts).sort_values(["split", "row_off", "col_off"]).reset_index(drop=True)


def _compute_normalization(
    image_path: str,
    manifest: pd.DataFrame,
    sample_stride: int,
) -> dict:
    samples: list[list[np.ndarray]] = []
    with rasterio.open(image_path) as src:
        samples = [[] for _ in range(src.count)]
        for row in manifest[manifest.split == "train"].itertuples():
            window = Window(row.col_off, row.row_off, row.width, row.height)
            block = src.read(window=window, out_dtype="float32")
            block = block[:, ::sample_stride, ::sample_stride]
            finite = np.isfinite(block).all(axis=0)
            for band_idx in range(src.count):
                values = block[band_idx][finite]
                if values.size:
                    samples[band_idx].append(values)
    normalization = {"bands": []}
    for idx, chunks in enumerate(samples):
        if not chunks:
            raise ValueError(f"No finite training samples for band {idx + 1}")
        values = np.concatenate(chunks)
        p01, p99 = np.percentile(values, [1, 99])
        clipped = np.clip(values, p01, p99)
        normalization["bands"].append(
            {
                "index": idx + 1,
                "sample_count": int(values.size),
                "clip_min": float(p01),
                "clip_max": float(p99),
                "mean": float(clipped.mean()),
                "std": float(max(clipped.std(), 1e-6)),
            }
        )
    return normalization


def main() -> None:
    parser = argparse.ArgumentParser(description="Align BLU, create spatial splits and OSM text context.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cleaned_dir = Path(cfg["paths"]["cleaned_dir"])
    out_dir = Path(cfg["paths"]["prepared_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    poi = gpd.read_file(cleaned_dir / "poi_clean.gpkg", layer="poi")
    aoi = gpd.read_file(cleaned_dir / "aoi_clean.gpkg", layer="aoi")
    patch_size = int(cfg["data"]["patch_size"])
    block_size = int(cfg["data"]["spatial_block_size"])
    if block_size % patch_size != 0:
        raise ValueError("spatial_block_size must be a multiple of patch_size")

    with rasterio.open(cfg["paths"]["image"]) as image_src, rasterio.open(cfg["paths"]["blu"]) as blu_src:
        raw_aligned = np.zeros((image_src.height, image_src.width), dtype=np.uint8)
        reproject(
            source=rasterio.band(blu_src, 1),
            destination=raw_aligned,
            src_transform=blu_src.transform,
            src_crs=blu_src.crs,
            src_nodata=blu_src.nodata,
            dst_transform=image_src.transform,
            dst_crs=image_src.crs,
            dst_nodata=0,
            resampling=Resampling.nearest,
        )
        labels = remap_blu(raw_aligned)
        leaf_labels = remap_blu_leaf(raw_aligned)
        finite_mask = _stream_finite_mask(image_src)
        labels[~finite_mask] = IGNORE_INDEX
        leaf_labels[~finite_mask] = IGNORE_INDEX
        _write_raster(out_dir / "blu_aligned_raw.tif", raw_aligned, image_src, 0, "BLU_aligned_nearest")
        _write_raster(out_dir / "labels_10class.tif", labels, image_src, IGNORE_INDEX, "BLU_10_coarse_classes")
        _write_raster(
            out_dir / "labels_leaf14.tif",
            leaf_labels,
            image_src,
            IGNORE_INDEX,
            "BLU_14_hierarchy_leaves",
        )
        transform = image_src.transform
        height, width = image_src.height, image_src.width

    rows: list[dict] = []
    inference_rows: list[dict] = []
    group_hist: dict[str, np.ndarray] = {}
    for row_off in range(0, height, patch_size):
        for col_off in range(0, width, patch_size):
            win_h = min(patch_size, height - row_off)
            win_w = min(patch_size, width - col_off)
            label_patch = labels[row_off : row_off + win_h, col_off : col_off + win_w]
            leaf_patch = leaf_labels[row_off : row_off + win_h, col_off : col_off + win_w]
            valid = label_patch != IGNORE_INDEX
            label_fraction = float(valid.sum() / (patch_size * patch_size))
            hist = np.bincount(label_patch[valid], minlength=10).astype(np.int64)
            leaf_valid = leaf_patch != IGNORE_INDEX
            leaf_hist = np.bincount(leaf_patch[leaf_valid], minlength=14).astype(np.int64)
            group = f"r{row_off // block_size}_c{col_off // block_size}"
            win = Window(col_off, row_off, win_w, win_h)
            left, bottom, right, top = window_bounds(win, transform)
            text = patch_text(poi, aoi, box(left, bottom, right, top))
            record = {
                "patch_id": f"r{row_off}_c{col_off}",
                "row_off": row_off,
                "col_off": col_off,
                "height": win_h,
                "width": win_w,
                "group": group,
                "label_fraction": label_fraction,
                "text": text,
            }
            record.update({f"pixels_class_{i}": int(hist[i]) for i in range(10)})
            record.update({f"leaf_pixels_class_{i}": int(leaf_hist[i]) for i in range(14)})
            inference_rows.append(record.copy())
            if label_fraction < float(cfg["data"]["min_label_fraction"]):
                continue
            group_hist.setdefault(group, np.zeros(14, dtype=np.int64))
            group_hist[group] += leaf_hist
            rows.append(record)

    split_map = choose_group_splits(
        group_hist,
        float(cfg["data"]["train_fraction"]),
        float(cfg["data"]["val_fraction"]),
        int(cfg["data"]["seed"]),
        int(cfg["data"].get("split_search_trials", 1000)),
    )
    manifest = pd.DataFrame(rows)
    manifest["split"] = manifest["group"].map(split_map)
    manifest = manifest.sort_values(["split", "row_off", "col_off"]).reset_index(drop=True)
    manifest.to_csv(out_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(inference_rows).sort_values(["row_off", "col_off"]).to_csv(
        out_dir / "inference_manifest.csv", index=False, encoding="utf-8-sig"
    )

    vocab = build_vocab(manifest.loc[manifest.split == "train", "text"].tolist())
    (out_dir / "vocab.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    normalization = _compute_normalization(
        cfg["paths"]["image"],
        manifest,
        int(cfg["data"].get("normalization_sample_stride", 4)),
    )
    (out_dir / "normalization.json").write_text(
        json.dumps(normalization, indent=2), encoding="utf-8"
    )

    light_manifest = None
    light_cfg = cfg.get("light_experiment", {})
    if bool(light_cfg.get("enabled", False)):
        light_manifest = _balanced_subset(
            manifest,
            limits={
                "train": int(light_cfg["max_train_patches"]),
                "val": int(light_cfg["max_val_patches"]),
                "test": int(light_cfg["max_test_patches"]),
            },
            min_per_class=int(light_cfg.get("min_patches_per_class", 8)),
            seed=int(cfg["data"]["seed"]),
            class_prefix="leaf_pixels_class_",
            num_classes=14,
        )
        light_manifest.to_csv(out_dir / "manifest_light.csv", index=False, encoding="utf-8-sig")
        light_vocab = build_vocab(light_manifest.loc[light_manifest.split == "train", "text"].tolist())
        (out_dir / "vocab_light.json").write_text(
            json.dumps(light_vocab, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        light_normalization = _compute_normalization(
            cfg["paths"]["image"],
            light_manifest,
            int(cfg["data"].get("normalization_sample_stride", 4)),
        )
        (out_dir / "normalization_light.json").write_text(
            json.dumps(light_normalization, indent=2), encoding="utf-8"
        )

    class_counts = {
        split: [int(manifest.loc[manifest.split == split, f"pixels_class_{i}"].sum()) for i in range(10)]
        for split in ("train", "val", "test")
    }
    leaf_class_counts = {
        split: [
            int(manifest.loc[manifest.split == split, f"leaf_pixels_class_{i}"].sum())
            for i in range(14)
        ]
        for split in ("train", "val", "test")
    }
    warnings: list[str] = []
    total_counts = np.array(class_counts["train"]) + np.array(class_counts["val"]) + np.array(class_counts["test"])
    for idx, count in enumerate(total_counts):
        if count == 0:
            warnings.append(f"类别 {idx}（{CLASS_NAMES[idx]}）在 S1/S2 范围内无 BLU 样本，无法学习该类。")
        elif count < 100:
            warnings.append(f"类别 {idx}（{CLASS_NAMES[idx]}）仅有 {count} 个像元，统计上极不稳定。")
    report = {
        "reference_grid": {
            "width": width,
            "height": height,
            "crs": str(poi.crs),
            "bounds": list(array_bounds(height, width, transform)),
            "transform": list(transform)[:6],
        },
        "raw_to_train": {str(k): v for k, v in RAW_TO_TRAIN.items()},
        "raw_to_leaf": {str(k): v for k, v in RAW_TO_LEAF.items()},
        "class_names": CLASS_NAMES,
        "leaf_class_names": LEAF_CLASS_NAMES,
        "patch_counts": manifest.split.value_counts().to_dict(),
        "spatial_group_counts": manifest.groupby("split").group.nunique().to_dict(),
        "class_pixel_counts": class_counts,
        "leaf_class_pixel_counts": leaf_class_counts,
        "finite_image_pixels": int(finite_mask.sum()),
        "usable_label_pixels": int((labels != IGNORE_INDEX).sum()),
        "vocabulary_size": len(vocab),
        "light_patch_counts": (
            light_manifest.split.value_counts().to_dict() if light_manifest is not None else None
        ),
        "light_class_pixel_counts": (
            {
                split: [
                    int(light_manifest.loc[light_manifest.split == split, f"pixels_class_{i}"].sum())
                    for i in range(10)
                ]
                for split in ("train", "val", "test")
            }
            if light_manifest is not None
            else None
        ),
        "light_leaf_class_pixel_counts": (
            {
                split: [
                    int(
                        light_manifest.loc[
                            light_manifest.split == split, f"leaf_pixels_class_{i}"
                        ].sum()
                    )
                    for i in range(14)
                ]
                for split in ("train", "val", "test")
            }
            if light_manifest is not None
            else None
        ),
        "warnings": warnings,
    }
    (out_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
