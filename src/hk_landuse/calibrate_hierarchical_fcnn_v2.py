from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .calibrate_hybrid_ensemble_8class import combine, metrics, optimize_classwise_gate
from .config import load_config
from .hierarchical_fcnn_model import HierarchicalMultiModalFCNN
from .task8 import leaf_target_to_official8
from .train_comparison import make_dataset


def collect(
    model: HierarchicalMultiModalFCNN,
    loader: DataLoader,
    device: torch.device,
    patch_to_group: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    direct_blocks, hierarchy_blocks, target_blocks, group_blocks = [], [], [], []
    names = sorted(set(patch_to_group.values())) if patch_to_group else []
    name_to_id = {name: index for index, name in enumerate(names)}
    model.eval()
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
                    batch["s2"].to(device, non_blocking=True),
                    batch["s1"].to(device, non_blocking=True),
                    batch["urban"].to(device, non_blocking=True),
                    batch["roads"].to(device, non_blocking=True),
                    batch["osm"].to(device, non_blocking=True),
                    batch["input_ids"].to(device, non_blocking=True),
                    batch["text_mask"].to(device, non_blocking=True),
                )
            valid = target != 255
            direct_blocks.append(
                output["direct_log_probs"].exp().permute(0, 2, 3, 1)[valid]
                .float().cpu().numpy()
            )
            hierarchy_blocks.append(
                output["hierarchy_log_probs"].exp().permute(0, 2, 3, 1)[valid]
                .float().cpu().numpy()
            )
            target_blocks.append(target[valid].cpu().numpy())
            if patch_to_group:
                for index, patch_id in enumerate(batch["patch_id"]):
                    group_blocks.append(np.full(
                        int(valid[index].sum()),
                        name_to_id[patch_to_group[str(patch_id)]],
                        dtype=np.int16,
                    ))
    return (
        np.concatenate(direct_blocks),
        np.concatenate(hierarchy_blocks),
        np.concatenate(target_blocks),
        np.concatenate(group_blocks) if group_blocks else np.empty(0, np.int16),
        names,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_hierarchical.yaml")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["data"]["manifest_name"] = "manifest_hier_v2.csv"
    cfg["data"]["vocab_name"] = "vocab_hier_v2.json"
    cfg["data"]["normalization_name"] = "normalization_hier_v2.json"
    modalities = {
        "s2", "s1", "highres", "nighttime", "building",
        "roads", "poi", "aoi", "text",
    }
    validation_dataset = make_dataset(cfg, "val", False, modalities)
    test_dataset = make_dataset(cfg, "test", False, modalities)
    prepared = Path(cfg["paths"]["prepared_dir"])
    vocab = json.loads((prepared / "vocab_hier_v2.json").read_text(encoding="utf-8"))
    model = HierarchicalMultiModalFCNN(
        vocab_size=len(vocab),
        s2_channels=validation_dataset.s2_output_channels,
        s1_channels=validation_dataset.s1_output_channels,
        urban_channels=validation_dataset.urban_channels,
        road_channels=validation_dataset.road_channels,
        text_dim=int(cfg["model"]["text_dim"]),
        text_heads=int(cfg["model"]["text_heads"]),
        text_layers=int(cfg["model"]["text_layers"]),
        max_tokens=int(cfg["data"]["max_text_tokens"]),
        dropout=float(cfg["model"].get("dropout", 0.1)),
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    loader_args = {
        "batch_size": 4,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    direct_val, hierarchy_val, target_val, _, _ = collect(
        model, DataLoader(validation_dataset, **loader_args), device
    )
    candidates = []
    for alpha in np.linspace(0, 1, 41):
        gates = np.full(8, alpha, np.float32)
        prediction = combine(
            direct_val, hierarchy_val, gates, np.zeros(8, np.float32)
        )
        candidates.append({
            "kind": "global", "alpha": float(alpha),
            "gates": gates.tolist(), "biases": [0.0] * 8,
            "validation": metrics(target_val, prediction),
        })
    best_global = max(candidates, key=lambda row: row["validation"]["macro_iou_present"])
    gates, biases = optimize_classwise_gate(
        direct_val, hierarchy_val, target_val, best_global["alpha"], args.seed
    )
    candidates.append({
        "kind": "classwise", "gates": gates.tolist(), "biases": biases.tolist(),
        "validation": metrics(
            target_val, combine(direct_val, hierarchy_val, gates, biases)
        ),
    })
    selected = max(candidates, key=lambda row: row["validation"]["macro_iou_present"])
    patch_to_group = {
        str(row.patch_id): str(row.group) for row in test_dataset.frame.itertuples()
    }
    direct_test, hierarchy_test, target_test, group_ids, group_names = collect(
        model, DataLoader(test_dataset, **loader_args), device, patch_to_group
    )
    prediction = combine(
        direct_test, hierarchy_test,
        np.asarray(selected["gates"], np.float32),
        np.asarray(selected["biases"], np.float32),
    )
    test = metrics(target_test, prediction)
    test["spatial_group_confusions"] = {
        name: metrics(
            target_test[group_ids == index], prediction[group_ids == index]
        )["confusion_matrix"]
        for index, name in enumerate(group_names)
    }
    report = {
        "model": "hierarchical_multimodal_fcnn_v2",
        "seed": args.seed,
        "selection_rule": "maximum validation macro_mIoU",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "selected": selected,
        "test": test,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "selected_kind": selected["kind"],
        "validation_mIoU": selected["validation"]["macro_iou_present"],
        "test_OA": test["overall_accuracy"],
        "test_mIoU": test["macro_iou_present"],
        "test_macro_F1": test["macro_f1_present"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
