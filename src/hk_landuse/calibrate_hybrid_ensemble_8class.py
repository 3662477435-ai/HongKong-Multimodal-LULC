from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import load_config
from .hierarchical_train import make_dataset
from .metrics import ConfusionMetrics
from .predict_experiment_models_8class import make_comparator, make_ours
from .task8 import CLASS_NAMES_8, leaf_target_to_official8


def collect_probabilities(
    ours: torch.nn.Module,
    expert: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    patch_to_group: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    ours.eval()
    expert.eval()
    ours_blocks: list[np.ndarray] = []
    expert_blocks: list[np.ndarray] = []
    target_blocks: list[np.ndarray] = []
    group_blocks: list[np.ndarray] = []
    group_names = sorted(set(patch_to_group.values())) if patch_to_group else []
    group_to_id = {name: index for index, name in enumerate(group_names)}
    with torch.inference_mode():
        for batch in loader:
            s2 = batch["s2"].to(device, non_blocking=True)
            s1 = batch["s1"].to(device, non_blocking=True)
            urban = batch["urban"].to(device, non_blocking=True)
            roads = batch["roads"].to(device, non_blocking=True)
            osm = batch["osm"].to(device, non_blocking=True)
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            text_mask = batch["text_mask"].to(device, non_blocking=True)
            target = leaf_target_to_official8(
                batch["label"].to(device, non_blocking=True)
            )
            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
                dtype=torch.bfloat16,
            ):
                ours_output = ours(
                    s1,
                    s2,
                    input_ids,
                    text_mask,
                    osm=osm,
                    urban=urban,
                    roads=roads,
                )
                ours_probability = ours_output["official8_log_probs"].exp()
                expert_probability = expert(
                    s2, s1, urban, roads, osm, input_ids, text_mask
                ).softmax(dim=1)
            valid = target != 255
            ours_blocks.append(
                ours_probability.permute(0, 2, 3, 1)[valid]
                .float()
                .cpu()
                .numpy()
            )
            expert_blocks.append(
                expert_probability.permute(0, 2, 3, 1)[valid]
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
        np.concatenate(ours_blocks),
        np.concatenate(expert_blocks),
        np.concatenate(target_blocks),
        (
            np.concatenate(group_blocks)
            if group_blocks
            else np.empty(0, dtype=np.int16)
        ),
        group_names,
    )


def combine(
    ours_probability: np.ndarray,
    expert_probability: np.ndarray,
    gates: np.ndarray,
    biases: np.ndarray,
) -> np.ndarray:
    probability = (
        (1.0 - gates[None]) * ours_probability
        + gates[None] * expert_probability
    )
    logits = np.log(np.clip(probability, 1e-8, 1.0)) + biases[None]
    return logits.argmax(axis=1)


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    metric = ConfusionMetrics(8, CLASS_NAMES_8)
    metric.update(
        torch.as_tensor(prediction, dtype=torch.long),
        torch.as_tensor(target, dtype=torch.long),
    )
    return metric.compute()


def optimize_classwise_gate(
    ours_probability: np.ndarray,
    expert_probability: np.ndarray,
    target: np.ndarray,
    initial_gate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples = []
    for class_id in range(8):
        indices = np.flatnonzero(target == class_id)
        take = min(len(indices), 30_000)
        samples.append(rng.choice(indices, take, replace=False))
    selected = np.concatenate(samples)
    ours = torch.from_numpy(ours_probability[selected]).float()
    expert = torch.from_numpy(expert_probability[selected]).float()
    labels = torch.from_numpy(target[selected]).long()
    gate_logits = torch.nn.Parameter(
        torch.full((8,), float(np.log(initial_gate / (1.0 - initial_gate))))
    )
    biases = torch.nn.Parameter(torch.zeros(8))
    optimizer = torch.optim.LBFGS(
        [gate_logits, biases],
        lr=0.5,
        max_iter=80,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        gates = gate_logits.sigmoid()
        probability = (1.0 - gates) * ours + gates * expert
        logits = probability.clamp_min(1e-8).log() + biases
        loss = torch.nn.functional.cross_entropy(logits, labels)
        # Keep calibration conservative and identifiable.
        loss = loss + 0.002 * biases.square().mean()
        loss.backward()
        return loss

    optimizer.step(closure)
    return (
        gate_logits.sigmoid().detach().numpy(),
        biases.detach().numpy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validation-calibrated hierarchical/FCNN mixture of experts."
    )
    parser.add_argument("--config", default="config_hierarchical.yaml")
    parser.add_argument("--ours-checkpoint", required=True)
    parser.add_argument("--expert-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["manifest_name"] = "manifest_hier_v2.csv"
    cfg["data"]["vocab_name"] = "vocab_hier_v2.json"
    cfg["data"]["normalization_name"] = "normalization_hier_v2.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_dataset = make_dataset(cfg, "val", False)
    loader_args = {
        "batch_size": int(cfg["train"]["batch_size"]),
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_args
    )
    ours = make_ours(
        cfg, validation_dataset, Path(args.ours_checkpoint)
    ).to(device)
    expert = make_comparator(
        cfg, validation_dataset, "fcnn", Path(args.expert_checkpoint)
    ).to(device)
    ours_val, expert_val, target_val, _, _ = collect_probabilities(
        ours, expert, validation_loader, device
    )

    candidates: list[dict] = []
    for alpha in np.linspace(0.0, 1.0, 41):
        prediction = combine(
            ours_val,
            expert_val,
            np.full(8, alpha, dtype=np.float32),
            np.zeros(8, dtype=np.float32),
        )
        result = metrics(target_val, prediction)
        candidates.append(
            {
                "kind": "global",
                "alpha": float(alpha),
                "gates": [float(alpha)] * 8,
                "biases": [0.0] * 8,
                "validation": result,
            }
        )
    best_global = max(
        candidates, key=lambda row: row["validation"]["macro_iou_present"]
    )
    classwise_gates, classwise_biases = optimize_classwise_gate(
        ours_val,
        expert_val,
        target_val,
        float(best_global["alpha"]),
        args.seed,
    )
    classwise_prediction = combine(
        ours_val,
        expert_val,
        classwise_gates,
        classwise_biases,
    )
    candidates.append(
        {
            "kind": "classwise",
            "gates": classwise_gates.tolist(),
            "biases": classwise_biases.tolist(),
            "validation": metrics(target_val, classwise_prediction),
        }
    )
    selected = max(
        candidates, key=lambda row: row["validation"]["macro_iou_present"]
    )

    # The spatial test set is opened only after the validation rule is frozen.
    test_dataset = make_dataset(cfg, "test", False)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)
    patch_to_group = {
        str(row.patch_id): str(row.group)
        for row in test_dataset.frame.itertuples()
    }
    ours_test, expert_test, target_test, group_ids, group_names = collect_probabilities(
        ours, expert, test_loader, device, patch_to_group=patch_to_group
    )
    test_prediction = combine(
        ours_test,
        expert_test,
        np.asarray(selected["gates"], dtype=np.float32),
        np.asarray(selected["biases"], dtype=np.float32),
    )
    test_metrics = metrics(target_test, test_prediction)
    test_metrics["spatial_group_confusions"] = {
        group_name: metrics(
            target_test[group_ids == group_index],
            test_prediction[group_ids == group_index],
        )["confusion_matrix"]
        for group_index, group_name in enumerate(group_names)
    }
    report = {
        "task": "official8_without_commercial_and_barren",
        "model": "hierarchical_fcnn_mixture_of_experts",
        "seed": args.seed,
        "selection_rule": "maximum validation macro_mIoU",
        "ours_checkpoint": str(Path(args.ours_checkpoint).resolve()),
        "expert_checkpoint": str(Path(args.expert_checkpoint).resolve()),
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
    print(
        json.dumps(
            {
                "selected_kind": selected["kind"],
                "validation_mIoU": selected["validation"]["macro_iou_present"],
                "test_OA": report["test"]["overall_accuracy"],
                "test_mIoU": report["test"]["macro_iou_present"],
                "test_macro_F1": report["test"]["macro_f1_present"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
