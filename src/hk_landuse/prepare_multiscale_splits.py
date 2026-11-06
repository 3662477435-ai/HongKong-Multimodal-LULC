from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window

from .config import load_config
from .labels import IGNORE_INDEX, LEAF_CLASS_NAMES


def optimize_split(
    pixel_hist: dict[str, np.ndarray],
    dominant_hist: dict[str, np.ndarray],
    seed: int,
    trials: int,
) -> tuple[dict[str, str], float]:
    groups = sorted(pixel_hist)
    n = len(groups)
    n_train, n_val = round(0.70 * n), round(0.15 * n)
    global_pixels = sum(pixel_hist.values(), start=np.zeros(14, dtype=np.int64))
    global_dominant = sum(dominant_hist.values(), start=np.zeros(14, dtype=np.int64))
    group_support = np.array([sum(dominant_hist[g][c] > 0 for g in groups) for c in range(14)])
    rng = np.random.default_rng(seed)
    best_score = math.inf
    best_map: dict[str, str] | None = None
    for _ in range(trials):
        permutation = list(rng.permutation(groups))
        sets = {
            "train": permutation[:n_train],
            "val": permutation[n_train : n_train + n_val],
            "test": permutation[n_train + n_val :],
        }
        pixels = {
            split: sum((pixel_hist[g] for g in members), start=np.zeros(14, dtype=np.int64))
            for split, members in sets.items()
        }
        dominant = {
            split: sum((dominant_hist[g] for g in members), start=np.zeros(14, dtype=np.int64))
            for split, members in sets.items()
        }
        score = 0.0
        for split, fraction in (("train", 0.70), ("val", 0.15), ("test", 0.15)):
            score += 8 * abs(pixels[split].sum() / global_pixels.sum() - fraction)
            pixel_prop = pixels[split] / max(pixels[split].sum(), 1)
            global_pixel_prop = global_pixels / global_pixels.sum()
            score += float(np.abs(pixel_prop - global_pixel_prop).mean())
            dominant_prop = dominant[split] / max(dominant[split].sum(), 1)
            global_dominant_prop = global_dominant / global_dominant.sum()
            score += 2 * float(np.abs(dominant_prop - global_dominant_prop).mean())
        for class_id in range(14):
            if dominant["train"][class_id] == 0:
                score += 10_000
            if group_support[class_id] >= 3:
                if dominant["val"][class_id] == 0:
                    score += 2_000
                if dominant["test"][class_id] == 0:
                    score += 2_000
            # Stronger penalty for tiny validation/test support.
            if group_support[class_id] >= 5:
                score += 20 / max(dominant["val"][class_id], 1)
                score += 20 / max(dominant["test"][class_id], 1)
            if global_dominant[class_id] > 0:
                class_fractions = np.array(
                    [dominant[s][class_id] for s in ("train", "val", "test")], dtype=float
                ) / global_dominant[class_id]
                score += 30 * float(np.abs(class_fractions - np.array([0.70, 0.15, 0.15])).mean())
                desired_train = min(10, max(2, int(np.floor(0.50 * global_dominant[class_id]))))
                if dominant["train"][class_id] < desired_train:
                    score += 500 * (desired_train - dominant["train"][class_id])
        if score < best_score:
            best_score = score
            best_map = {group: split for split, members in sets.items() for group in members}
    if best_map is None:
        raise RuntimeError("No split candidate was generated")
    return best_map, best_score


def balanced_blocks(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    limits = {"train": 160, "val": 80, "test": 80}
    parts = []
    for split in ("train", "val", "test"):
        split_frame = frame[frame.split == split]
        for class_id in range(14):
            candidates = split_frame[split_frame.leaf_target == class_id]
            if candidates.empty:
                continue
            count = min(limits[split], len(candidates))
            chosen = rng.choice(candidates.index.to_numpy(), size=count, replace=False)
            parts.append(frame.loc[chosen])
    return (
        pd.concat(parts)
        .drop_duplicates("patch_id")
        .sort_values(["split", "leaf_target", "row_off", "col_off"])
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create spatial splits supporting rare dominant leaf classes.")
    parser.add_argument("--config", default="config_paper_replication.yaml")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--trials", type=int, default=30000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    prepared = Path(cfg["paths"]["prepared_dir"])
    label_path = prepared / cfg["data"]["label_name"]
    block = args.block_size
    group_size = 512
    records: list[dict] = []
    pixel_hist: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(14, dtype=np.int64))
    dominant_hist: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(14, dtype=np.int64))
    with rasterio.open(label_path) as labels:
        for row_off in range(0, labels.height, block):
            for col_off in range(0, labels.width, block):
                height = min(block, labels.height - row_off)
                width = min(block, labels.width - col_off)
                target = labels.read(1, window=Window(col_off, row_off, width, height))
                valid = target != IGNORE_INDEX
                if valid.sum() < 0.10 * block * block:
                    continue
                hist = np.bincount(target[valid], minlength=14).astype(np.int64)
                leaf_target = int(hist.argmax())
                purity = float(hist.max() / hist.sum())
                group = f"r{row_off // group_size}_c{col_off // group_size}"
                pixel_hist[group] += hist
                if purity >= 0.25:
                    dominant_hist[group][leaf_target] += 1
                record = {
                    "patch_id": f"b{block}_r{row_off}_c{col_off}",
                    "row_off": row_off,
                    "col_off": col_off,
                    "height": height,
                    "width": width,
                    "group": group,
                    "label_fraction": float(valid.sum() / (block * block)),
                    "leaf_target": leaf_target,
                    "leaf_purity": purity,
                }
                record.update({f"leaf_pixels_class_{i}": int(hist[i]) for i in range(14)})
                records.append(record)
    split_map, score = optimize_split(
        pixel_hist,
        dominant_hist,
        seed=int(cfg["data"]["seed"]),
        trials=args.trials,
    )
    frame = pd.DataFrame(records)
    frame["split"] = frame.group.map(split_map)
    frame.to_csv(prepared / "manifest_blocks32.csv", index=False, encoding="utf-8-sig")
    eligible = frame[frame.leaf_purity >= 0.25].copy()
    light = balanced_blocks(eligible, int(cfg["data"]["seed"]))
    light.to_csv(prepared / "manifest_blocks32_light.csv", index=False, encoding="utf-8-sig")
    table = {
        split: [int(((light.split == split) & (light.leaf_target == i)).sum()) for i in range(14)]
        for split in ("train", "val", "test")
    }
    report = {
        "optimization_score": score,
        "group_counts": pd.Series(split_map).value_counts().to_dict(),
        "all_block_counts": frame.split.value_counts().to_dict(),
        "light_block_counts": light.split.value_counts().to_dict(),
        "light_dominant_class_counts": table,
        "leaf_class_names": LEAF_CLASS_NAMES,
        "spatial_leakage_check": int(frame.groupby("group").split.nunique().max()),
    }
    (prepared / "multiscale_split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
