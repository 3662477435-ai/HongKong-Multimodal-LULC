from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hk_landuse.config import load_config  # noqa: E402
from hk_landuse.prepare_data import _compute_normalization, choose_group_splits  # noqa: E402
from hk_landuse.text import build_vocab  # noqa: E402


# Keep this lightweight preprocessing script independent of PyTorch. These are
# the contiguous eight-class experiment's retained official class ids (the
# commercial/office and barren-land groups are excluded by protocol).
ACTIVE_OFFICIAL_IDS = (0, 2, 3, 4, 5, 6, 7, 9)


def group_histogram(frame: pd.DataFrame, group_column: str = "group") -> dict[str, np.ndarray]:
    columns = [f"pixels_class_{class_id}" for class_id in ACTIVE_OFFICIAL_IDS]
    return {
        str(group): part[columns].sum().to_numpy(dtype=np.int64)
        for group, part in frame.groupby(group_column)
    }


def block_protocol(
    source: pd.DataFrame,
    block_pixels: int,
    train_fraction: float,
    val_fraction: float,
    seed: int,
) -> pd.DataFrame:
    frame = source.copy()
    frame["group"] = (
        "r"
        + (frame["row_off"] // block_pixels).astype(str)
        + "_c"
        + (frame["col_off"] // block_pixels).astype(str)
    )
    assignment = choose_group_splits(
        group_histogram(frame), train_fraction, val_fraction, seed, trials=5000
    )
    frame["split"] = frame["group"].map(assignment)
    return frame


def rectangle_gap(a: pd.DataFrame, b: pd.DataFrame, patch_size: int) -> np.ndarray:
    """Minimum Euclidean edge-to-edge gap from every row in a to rectangles in b."""
    ax = a["col_off"].to_numpy(dtype=float)[:, None]
    ay = a["row_off"].to_numpy(dtype=float)[:, None]
    bx = b["col_off"].to_numpy(dtype=float)[None, :]
    by = b["row_off"].to_numpy(dtype=float)[None, :]
    dx = np.maximum(np.abs(ax - bx) - patch_size, 0.0)
    dy = np.maximum(np.abs(ay - by) - patch_size, 0.0)
    return np.sqrt(dx * dx + dy * dy).min(axis=1)


def buffered_protocol(source: pd.DataFrame, patch_size: int, buffer_pixels: int) -> pd.DataFrame:
    train = source[source.split == "train"].copy()
    val = source[source.split == "val"].copy()
    test = source[source.split == "test"].copy()
    # Protect the final test set first, then protect validation from training.
    val = val[rectangle_gap(val, test, patch_size) >= buffer_pixels].copy()
    protected = pd.concat([val, test], ignore_index=True)
    train = train[rectangle_gap(train, protected, patch_size) >= buffer_pixels].copy()
    return pd.concat([train, val, test], ignore_index=True)


def directional_candidates(source: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    center_x = source.col_off.to_numpy() + source.width.to_numpy() / 2
    center_y = source.row_off.to_numpy() + source.height.to_numpy() / 2
    return [
        ("east", center_x >= np.quantile(center_x, 0.82)),
        ("west", center_x <= np.quantile(center_x, 0.18)),
        ("south", center_y >= np.quantile(center_y, 0.82)),
        ("north", center_y <= np.quantile(center_y, 0.18)),
    ]


def leave_region_out_protocol(source: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    class_columns = [f"pixels_class_{class_id}" for class_id in ACTIVE_OFFICIAL_IDS]
    global_prop = source[class_columns].sum().to_numpy(dtype=float)
    global_prop /= global_prop.sum()
    best = None
    for name, mask in directional_candidates(source):
        test = source[mask]
        hist = test[class_columns].sum().to_numpy(dtype=float)
        coverage = int((hist > 0).sum())
        prop = hist / max(hist.sum(), 1.0)
        score = 20.0 * (8 - coverage) + float(np.abs(prop - global_prop).mean())
        candidate = (score, name, mask)
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    _, direction, test_mask = best
    remaining = source[~test_mask].copy()
    # Use the opposite geographic tail for validation so validation does not
    # tune hyperparameters on the held-out region.
    center_x = remaining.col_off + remaining.width / 2
    center_y = remaining.row_off + remaining.height / 2
    if direction == "east":
        val_mask = center_x <= center_x.quantile(0.18)
    elif direction == "west":
        val_mask = center_x >= center_x.quantile(0.82)
    elif direction == "south":
        val_mask = center_y <= center_y.quantile(0.18)
    else:
        val_mask = center_y >= center_y.quantile(0.82)
    frame = source.copy()
    frame["split"] = "train"
    frame.loc[test_mask, "split"] = "test"
    frame.loc[remaining.index[val_mask], "split"] = "val"
    # Retain 5.12-km groups for clustered test uncertainty.
    frame["group"] = (
        "r" + (frame.row_off // 512).astype(str) + "_c" + (frame.col_off // 512).astype(str)
    )
    return frame, direction


def summarize(frame: pd.DataFrame) -> dict:
    columns = [f"pixels_class_{class_id}" for class_id in ACTIVE_OFFICIAL_IDS]
    result = {}
    for split in ("train", "val", "test"):
        part = frame[frame.split == split]
        result[split] = {
            "patches": int(len(part)),
            "groups": int(part.group.nunique()),
            "pixels_by_class": {
                str(class_id): int(part[column].sum())
                for class_id, column in zip(ACTIVE_OFFICIAL_IDS, columns)
            },
        }
    return result


def write_protocol(
    prepared: Path,
    image_path: str,
    name: str,
    frame: pd.DataFrame,
    metadata: dict,
) -> None:
    frame = frame.sort_values(["split", "row_off", "col_off"]).reset_index(drop=True)
    manifest_name = f"manifest_{name}.csv"
    vocab_name = f"vocab_{name}.json"
    normalization_name = f"normalization_{name}.json"
    frame.to_csv(prepared / manifest_name, index=False)
    vocab = build_vocab(frame.loc[frame.split == "train", "text"].fillna("").tolist())
    (prepared / vocab_name).write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    normalization = _compute_normalization(image_path, frame, sample_stride=4)
    (prepared / normalization_name).write_text(
        json.dumps(normalization, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    audit = {
        "protocol": name,
        "manifest": manifest_name,
        "vocab": vocab_name,
        "normalization": normalization_name,
        "summary": summarize(frame),
        **metadata,
    }
    (prepared / f"protocol_{name}.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


def main() -> None:
    cfg = load_config(REPO / "config_hierarchical.yaml")
    prepared = Path(cfg["paths"]["prepared_dir"])
    if not prepared.is_absolute():
        prepared = REPO / prepared
    source = pd.read_csv(prepared / "manifest_hier_v2.csv")
    train_fraction = float(cfg["data"]["train_fraction"])
    val_fraction = float(cfg["data"]["val_fraction"])
    seed = int(cfg["data"]["seed"])
    protocols = {
        "block256": (
            block_protocol(source, 256, train_fraction, val_fraction, seed),
            {"design": "spatial blocks", "block_pixels": 256, "block_km": 2.56},
        ),
        "block1024": (
            block_protocol(source, 1024, train_fraction, val_fraction, seed),
            {"design": "spatial blocks", "block_pixels": 1024, "block_km": 10.24},
        ),
        "block512_buffer128": (
            buffered_protocol(source, patch_size=128, buffer_pixels=128),
            {
                "design": "fixed 5.12-km groups with exclusion buffer",
                "block_pixels": 512,
                "block_km": 5.12,
                "buffer_pixels": 128,
                "buffer_km": 1.28,
            },
        ),
    }
    loro, direction = leave_region_out_protocol(source)
    protocols[f"loro_{direction}"] = (
        loro,
        {
            "design": "directional leave-region-out",
            "held_out_direction": direction,
            "test_quantile": 0.18,
        },
    )
    for name, (frame, metadata) in protocols.items():
        write_protocol(prepared, cfg["paths"]["image"], name, frame, metadata)


if __name__ == "__main__":
    main()
