from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .calibrate_hybrid_ensemble_8class import (
    combine,
    metrics,
    optimize_classwise_gate,
)
from .config import load_config
from .hierarchical_train import make_dataset
from .predict_experiment_models_8class import make_ours
from .task8 import leaf_target_to_official8


def collect_heads(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    patch_to_group: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    model.eval()
    hierarchy_blocks: list[np.ndarray] = []
    direct_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    group_blocks: list[np.ndarray] = []
    group_names = sorted(set(patch_to_group.values())) if patch_to_group else []
    group_to_id = {name: index for index, name in enumerate(group_names)}
    with torch.inference_mode():
        for batch in loader:
            target = leaf_target_to_official8(
                batch["label"].to(device, non_blocking=True)
            )
            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
                dtype=torch.bfloat16,
            ):
                output = model(
                    batch["s1"].to(device, non_blocking=True),
                    batch["s2"].to(device, non_blocking=True),
                    batch["input_ids"].to(device, non_blocking=True),
                    batch["text_mask"].to(device, non_blocking=True),
                    osm=batch["osm"].to(device, non_blocking=True),
                    urban=batch["urban"].to(device, non_blocking=True),
                    roads=batch["roads"].to(device, non_blocking=True),
                )
            valid = target != 255
            hierarchy_blocks.append(
                output["hierarchical_official8_log_probs"]
                .exp()
                .permute(0, 2, 3, 1)[valid]
                .float()
                .cpu()
                .numpy()
            )
            direct_blocks.append(
                output["direct_official8_log_probs"]
                .exp()
                .permute(0, 2, 3, 1)[valid]
                .float()
                .cpu()
                .numpy()
            )
            target_blocks.append(target[valid].cpu().numpy())
            if patch_to_group is not None:
                for sample_index, patch_id in enumerate(batch["patch_id"]):
                    count = int(valid[sample_index].sum())
                    group_blocks.append(
                        np.full(
                            count,
                            group_to_id[patch_to_group[str(patch_id)]],
                            dtype=np.int16,
                        )
                    )
    return (
        np.concatenate(hierarchy_blocks),
        np.concatenate(direct_blocks),
        np.concatenate(target_blocks),
        np.concatenate(group_blocks) if group_blocks else np.empty(0, np.int16),
        group_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate two heads that share one Swin V2-Tiny encoder."
    )
    parser.add_argument("--config", default="config_hierarchical.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["data"]["manifest_name"] = "manifest_hier_v2.csv"
    cfg["data"]["vocab_name"] = "vocab_hier_v2.json"
    cfg["data"]["normalization_name"] = "normalization_hier_v2.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader_args = {
        "batch_size": int(cfg["train"]["batch_size"]),
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    validation_dataset = make_dataset(cfg, "val", False)
    checkpoint_metadata = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    image_backbone = checkpoint_metadata.get(
        "image_backbone", "swin_v2_tiny"
    )
    model = make_ours(
        cfg, validation_dataset, Path(args.checkpoint)
    ).to(device)
    hierarchy_val, direct_val, target_val, _, _ = collect_heads(
        model,
        DataLoader(validation_dataset, shuffle=False, **loader_args),
        device,
    )
    candidates: list[dict] = []
    for alpha in np.linspace(0.0, 1.0, 41):
        prediction = combine(
            hierarchy_val,
            direct_val,
            np.full(8, alpha, dtype=np.float32),
            np.zeros(8, dtype=np.float32),
        )
        candidates.append(
            {
                "kind": "global",
                "alpha": float(alpha),
                "gates": [float(alpha)] * 8,
                "biases": [0.0] * 8,
                "validation": metrics(target_val, prediction),
            }
        )
    best_global = max(
        candidates, key=lambda row: row["validation"]["macro_iou_present"]
    )
    gates, biases = optimize_classwise_gate(
        hierarchy_val,
        direct_val,
        target_val,
        float(best_global["alpha"]),
        args.seed,
    )
    candidates.append(
        {
            "kind": "classwise",
            "gates": gates.tolist(),
            "biases": biases.tolist(),
            "validation": metrics(
                target_val,
                combine(hierarchy_val, direct_val, gates, biases),
            ),
        }
    )
    selected = max(
        candidates, key=lambda row: row["validation"]["macro_iou_present"]
    )

    test_dataset = make_dataset(cfg, "test", False)
    patch_to_group = {
        str(row.patch_id): str(row.group)
        for row in test_dataset.frame.itertuples()
    }
    hierarchy_test, direct_test, target_test, group_ids, group_names = collect_heads(
        model,
        DataLoader(test_dataset, shuffle=False, **loader_args),
        device,
        patch_to_group=patch_to_group,
    )
    prediction = combine(
        hierarchy_test,
        direct_test,
        np.asarray(selected["gates"], dtype=np.float32),
        np.asarray(selected["biases"], dtype=np.float32),
    )
    test_metrics = metrics(target_test, prediction)
    test_metrics["spatial_group_confusions"] = {
        group_name: metrics(
            target_test[group_ids == group_index],
            prediction[group_ids == group_index],
        )["confusion_matrix"]
        for group_index, group_name in enumerate(group_names)
    }
    report = {
        "task": "official8_without_commercial_and_barren",
        "model": f"hierarchical_multimodal_{image_backbone}_dual_head",
        "backbone": image_backbone,
        "external_expert": None,
        "seed": args.seed,
        "selection_rule": "maximum validation macro_mIoU",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "selected": selected,
        "test": test_metrics,
        "global_candidates": candidates[:-1],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "selected_kind": selected["kind"],
        "validation_mIoU": selected["validation"]["macro_iou_present"],
        "test_OA": test_metrics["overall_accuracy"],
        "test_mIoU": test_metrics["macro_iou_present"],
        "test_macro_F1": test_metrics["macro_f1_present"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
