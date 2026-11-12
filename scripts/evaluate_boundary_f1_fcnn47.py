"""Evaluate the saved FCNN-47 checkpoint with the manuscript Boundary-F1 protocol.

This is inference-only: it does not alter the checkpoint or run optimisation.
For each 128 x 128 test patch, class boundaries are extracted from the
prediction and target independently.  A one-pixel binary dilation provides the
10-m tolerance.  Class-wise precision/recall are accumulated across patches
and then macro-averaged over classes with reference support.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, binary_erosion
from torch.utils.data import DataLoader

from hk_landuse.config import load_config
from hk_landuse.task8 import CLASS_NAMES_8, leaf_target_to_official8
from hk_landuse.train_comparison import make_dataset
from hk_landuse.train_comparison_8class import ALL_MODALITIES, build_model


def class_boundary(labels: np.ndarray, class_id: int) -> np.ndarray:
    """Inner morphological boundary for one class within an individual patch."""
    mask = labels == class_id
    if not mask.any():
        return mask
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_release.yaml")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-name", default="manifest_hier_v2.csv")
    parser.add_argument("--vocab-name", default="vocab_hier_v2.json")
    parser.add_argument("--normalization-name", default="normalization_hier_v2.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["manifest_name"] = args.manifest_name
    cfg["data"]["vocab_name"] = args.vocab_name
    cfg["data"]["normalization_name"] = args.normalization_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modalities = set(ALL_MODALITIES)
    dataset = make_dataset(cfg, "test", False, modalities, "full")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model, multimodal = build_model(cfg, dataset, "fcnn", modalities, pretrained=False)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    matched_pred = np.zeros(8, dtype=np.int64)
    matched_ref = np.zeros(8, dtype=np.int64)
    total_pred = np.zeros(8, dtype=np.int64)
    total_ref = np.zeros(8, dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            s2 = batch["s2"].to(device, non_blocking=True)
            n, _, h, w = s2.shape
            logits = model(
                s2,
                batch["s1"].to(device, non_blocking=True),
                batch["urban"].to(device, non_blocking=True),
                batch["roads"].to(device, non_blocking=True),
                batch["osm"].to(device, non_blocking=True),
                batch["input_ids"].to(device, non_blocking=True),
                batch["text_mask"].to(device, non_blocking=True),
            )
            pred = logits.argmax(dim=1).cpu().numpy()
            target = leaf_target_to_official8(batch["label"].to(device)).cpu().numpy()
            for p, y in zip(pred, target):
                valid = y != 255
                # Invalid reference pixels are excluded from both boundaries.
                p = p.copy()
                p[~valid] = -1
                y = y.copy()
                y[~valid] = -1
                for c in range(8):
                    pb = class_boundary(p, c)
                    rb = class_boundary(y, c)
                    pd = binary_dilation(pb, structure=np.ones((3, 3), dtype=bool))
                    rd = binary_dilation(rb, structure=np.ones((3, 3), dtype=bool))
                    total_pred[c] += int(pb.sum())
                    total_ref[c] += int(rb.sum())
                    matched_pred[c] += int((pb & rd).sum())
                    matched_ref[c] += int((rb & pd).sum())

    precision = np.divide(matched_pred, total_pred, out=np.zeros(8), where=total_pred > 0)
    recall = np.divide(matched_ref, total_ref, out=np.zeros(8), where=total_ref > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(8), where=(precision + recall) > 0)
    supported = total_ref > 0
    result = {
        "model": "FCNN equal-budget 47 epochs",
        "checkpoint": str(args.checkpoint.resolve()),
        "best_epoch": int(checkpoint.get("epoch", -1)),
        "protocol": "patch-local inner boundary; one-pixel (10 m) dilation tolerance; class macro average",
        "macro_boundary_f1": float(f1[supported].mean()),
        "classwise": [
            {
                "class_id": int(i + 1),
                "class_name": CLASS_NAMES_8[i],
                "boundary_precision": float(precision[i]),
                "boundary_recall": float(recall[i]),
                "boundary_f1": float(f1[i]),
                "reference_boundary_pixels": int(total_ref[i]),
            }
            for i in range(8)
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
