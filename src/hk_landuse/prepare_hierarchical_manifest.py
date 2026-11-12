from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .config import load_config
from .prepare_data import _balanced_subset, _compute_normalization
from .text import build_vocab


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer the optimized 32-pixel spatial split to 128-pixel neural patches."
    )
    parser.add_argument("--config", default="config_release.yaml")
    parser.add_argument("--split-manifest", default="manifest_blocks32.csv")
    args = parser.parse_args()
    cfg = load_config(args.config)
    prepared = Path(cfg["paths"]["prepared_dir"])

    blocks = pd.read_csv(prepared / args.split_manifest)
    unique = blocks.groupby("group")["split"].nunique()
    if int(unique.max()) != 1:
        raise ValueError("The source block manifest contains spatial split leakage")
    split_map = blocks.groupby("group")["split"].first().to_dict()

    manifest = pd.read_csv(prepared / "manifest.csv")
    manifest["split"] = manifest["group"].map(split_map)
    missing = manifest["split"].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} neural patches have no optimized group assignment")
    manifest = manifest.sort_values(["split", "row_off", "col_off"]).reset_index(drop=True)
    manifest.to_csv(prepared / "manifest_hier_v2.csv", index=False, encoding="utf-8-sig")
    full_vocab = build_vocab(manifest.loc[manifest.split == "train", "text"].fillna("").tolist())
    (prepared / "vocab_hier_v2.json").write_text(
        json.dumps(full_vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    full_normalization = _compute_normalization(
        cfg["paths"]["image"],
        manifest,
        int(cfg["data"].get("normalization_sample_stride", 4)),
    )
    (prepared / "normalization_hier_v2.json").write_text(
        json.dumps(full_normalization, indent=2), encoding="utf-8"
    )

    light_cfg = cfg["light_experiment"]
    light = _balanced_subset(
        manifest,
        limits={
            "train": int(light_cfg["max_train_patches"]),
            "val": int(light_cfg["max_val_patches"]),
            "test": int(light_cfg["max_test_patches"]),
        },
        min_per_class=int(light_cfg.get("min_patches_per_class", 12)),
        seed=int(cfg["data"]["seed"]),
        class_prefix="leaf_pixels_class_",
        num_classes=14,
    )
    light.to_csv(prepared / "manifest_hier_v2_light.csv", index=False, encoding="utf-8-sig")

    vocab = build_vocab(light.loc[light.split == "train", "text"].fillna("").tolist())
    (prepared / "vocab_hier_v2_light.json").write_text(
        json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    normalization = _compute_normalization(
        cfg["paths"]["image"],
        light,
        int(cfg["data"].get("normalization_sample_stride", 4)),
    )
    (prepared / "normalization_hier_v2_light.json").write_text(
        json.dumps(normalization, indent=2), encoding="utf-8"
    )

    report = {
        "all_patch_counts": manifest.split.value_counts().to_dict(),
        "light_patch_counts": light.split.value_counts().to_dict(),
        "spatial_group_counts": manifest.groupby("split").group.nunique().to_dict(),
        "spatial_leakage_check": int(manifest.groupby("group").split.nunique().max()),
        "light_leaf_patch_support": {
            split: [
                int(((light.split == split) & (light[f"leaf_pixels_class_{i}"] > 0)).sum())
                for i in range(14)
            ]
            for split in ("train", "val", "test")
        },
        "vocabulary_size": len(vocab),
        "full_vocabulary_size": len(full_vocab),
    }
    (prepared / "hierarchical_v2_split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
