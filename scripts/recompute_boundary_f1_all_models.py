"""Recompute Boundary F1 for every reported model under one fixed protocol.

Inference only: no checkpoint is updated.  Boundaries are inner 8-connected
morphological boundaries, matched with a one-pixel (10 m) dilation tolerance
within each 128 x 128 test patch.  Per-class matches are aggregated across all
test patches before computing the unweighted eight-class mean.
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
from hk_landuse.predict_experiment_models_8class import make_ours
from hk_landuse.task8 import CLASS_NAMES_8, leaf_target_to_official8
from hk_landuse.train_comparison import make_dataset
from hk_landuse.train_comparison_8class import ALL_MODALITIES, build_model


def inner_boundary(labels: np.ndarray, class_id: int) -> np.ndarray:
    mask = labels == class_id
    if not mask.any():
        return mask
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3), dtype=bool), border_value=0)


def predict(model, hierarchical: bool, batch: dict[str, torch.Tensor], device: torch.device) -> np.ndarray:
    args = (
        batch["s2"].to(device, non_blocking=True),
        batch["s1"].to(device, non_blocking=True),
        batch["urban"].to(device, non_blocking=True),
        batch["roads"].to(device, non_blocking=True),
        batch["osm"].to(device, non_blocking=True),
        batch["input_ids"].to(device, non_blocking=True),
        batch["text_mask"].to(device, non_blocking=True),
    )
    out = model(*args)
    logits = out["combined_log_probs"] if hierarchical else out
    return logits.argmax(dim=1).cpu().numpy()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config_hierarchical.yaml")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=2)
    args = p.parse_args()
    cfg = load_config(args.config)
    cfg["data"].update({
        "manifest_name": "manifest_hier_v2.csv",
        "vocab_name": "vocab_hier_v2.json",
        "normalization_name": "normalization_hier_v2.json",
    })
    root = Path(cfg["paths"]["output_dir"]).resolve()
    if root.name != "full":
        root = root / "full"
    exp = root / "eight_class_20260725"
    fair = root / "eight_class_fairness_supplement_20260802"
    specs = [
        ("本文模型", exp / "ours_fcnn_hierarchy_semantic_seed42" / "best_final.pt", "hierarchical", True),
        ("CNN", exp / "cnn_full_seed42" / "best_model.pt", "cnn", False),
        ("FCNN（20 epoch）", exp / "fcnn_full_seed42" / "best_model.pt", "fcnn", False),
        ("FCNN（47 epoch）", fair / "fcnn_equal_budget_47ep_seed42" / "best_model.pt", "fcnn", False),
        ("Transformer", exp / "transformer_full_seed42" / "best_model.pt", "transformer", False),
        ("FCN-ResNet50", exp / "fcn_resnet50_full_seed42" / "best_model.pt", "fcn_resnet50", False),
        ("VGG-UNet-Transformer", exp / "vgg_unet_transformer_full_seed42" / "best_model.pt", "vgg_unet_transformer", False),
    ]
    missing = [str(path) for _, path, _, _ in specs if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing checkpoints:\n" + "\n".join(missing))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = make_dataset(cfg, "test", False, set(ALL_MODALITIES), "full")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda")
    results = []
    kernel = np.ones((3, 3), dtype=bool)
    for name, checkpoint_path, backbone, hierarchical in specs:
        if hierarchical:
            model = make_ours(cfg, dataset, checkpoint_path)
        else:
            model, _ = build_model(cfg, dataset, backbone, set(ALL_MODALITIES), pretrained=False)
            state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state["model_state"])
        model.to(device).eval()
        matched_pred = np.zeros(8, dtype=np.int64)
        matched_ref = np.zeros(8, dtype=np.int64)
        total_pred = np.zeros(8, dtype=np.int64)
        total_ref = np.zeros(8, dtype=np.int64)
        with torch.no_grad():
            for batch in loader:
                pred = predict(model, hierarchical, batch, device)
                target = leaf_target_to_official8(batch["label"].to(device)).cpu().numpy()
                for pr, ref in zip(pred, target):
                    valid = ref != 255
                    pr, ref = pr.copy(), ref.copy()
                    pr[~valid], ref[~valid] = -1, -1
                    for class_id in range(8):
                        pb, rb = inner_boundary(pr, class_id), inner_boundary(ref, class_id)
                        pd, rd = binary_dilation(pb, kernel), binary_dilation(rb, kernel)
                        total_pred[class_id] += pb.sum()
                        total_ref[class_id] += rb.sum()
                        matched_pred[class_id] += (pb & rd).sum()
                        matched_ref[class_id] += (rb & pd).sum()
        precision = np.divide(matched_pred, total_pred, out=np.zeros(8), where=total_pred > 0)
        recall = np.divide(matched_ref, total_ref, out=np.zeros(8), where=total_ref > 0)
        f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(8), where=(precision + recall) > 0)
        result = {
            "model": name,
            "checkpoint": str(checkpoint_path),
            "macro_boundary_f1": float(f1[total_ref > 0].mean()),
            "classwise_boundary_f1": {CLASS_NAMES_8[i]: float(f1[i]) for i in range(8)},
        }
        results.append(result)
        print(f"{name}: {result['macro_boundary_f1']:.6f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "protocol": "128x128 patch-local inner morphological boundaries; one-pixel (10 m) dilation tolerance; macro-average over eight classes",
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
