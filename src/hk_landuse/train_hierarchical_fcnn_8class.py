from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .config import load_config
from .hierarchical_fcnn_model import HierarchicalMultiModalFCNN
from .metrics import ConfusionMetrics
from .task8 import CLASS_NAMES_8, leaf_target_to_official8
from .train_comparison import make_dataset
from .train_comparison_8class import balanced_weights, dice_loss, patch_sampler

ALL_MODALITIES = {
    "s2", "s1", "highres", "nighttime", "building",
    "roads", "poi", "aoi", "text",
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def coarse_target(target: torch.Tensor) -> torch.Tensor:
    result = torch.full_like(target, 255)
    result[(target >= 0) & (target <= 4)] = 0
    result[(target == 5) | (target == 6)] = 1
    result[target == 7] = 2
    return result


def dice_from_log_probs(log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return dice_loss(log_probs, target)


def loss_function(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    class_weights: torch.Tensor,
    coarse_weights: torch.Tensor,
    stage: str,
    use_hierarchy: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = target != 255
    coarse = coarse_target(target)
    direct_ce = F.cross_entropy(
        output["direct_logits"], target, weight=class_weights, ignore_index=255
    )
    direct_dice = dice_loss(output["direct_logits"], target)
    combined_ce = F.nll_loss(
        output["combined_log_probs"], target, weight=class_weights, ignore_index=255
    )
    combined_dice = dice_from_log_probs(output["combined_log_probs"], target)
    hierarchy_ce = F.nll_loss(
        output["hierarchy_log_probs"], target, weight=class_weights, ignore_index=255
    )
    hierarchy_dice = dice_from_log_probs(output["hierarchy_log_probs"], target)
    coarse_ce = F.cross_entropy(
        output["coarse_logits"], coarse, weight=coarse_weights, ignore_index=255
    )
    built = valid & (target <= 4)
    natural = valid & ((target == 5) | (target == 6))
    zero = output["coarse_logits"].sum() * 0.0
    built_ce = (
        F.cross_entropy(
            output["built_logits"].permute(0, 2, 3, 1)[built],
            target[built],
        )
        if built.any()
        else zero
    )
    natural_ce = (
        F.cross_entropy(
            output["natural_logits"].permute(0, 2, 3, 1)[natural],
            target[natural] - 5,
        )
        if natural.any()
        else zero
    )
    hierarchy_objective = (
        0.7 * hierarchy_ce
        + 0.3 * hierarchy_dice
        + 0.35 * coarse_ce
        + 0.25 * (built_ce + natural_ce)
    )
    direct_objective = 0.7 * direct_ce + 0.3 * direct_dice
    combined_objective = 0.7 * combined_ce + 0.3 * combined_dice
    if stage == "hierarchy":
        loss = hierarchy_objective
    elif stage == "gate":
        loss = combined_objective
    elif stage == "joint":
        loss = (
            combined_objective
            + 0.20 * direct_objective
            + (0.30 * hierarchy_objective if use_hierarchy else zero)
        )
    else:
        raise ValueError(stage)
    return loss, {
        "direct_ce": float(direct_ce.detach()),
        "direct_dice": float(direct_dice.detach()),
        "combined_ce": float(combined_ce.detach()),
        "combined_dice": float(combined_dice.detach()),
        "hierarchy_ce": float(hierarchy_ce.detach()),
        "hierarchy_dice": float(hierarchy_dice.detach()),
        "coarse_ce": float(coarse_ce.detach()),
        "built_ce": float(built_ce.detach()),
        "natural_ce": float(natural_ce.detach()),
    }


def run_epoch(
    model: HierarchicalMultiModalFCNN,
    loader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    coarse_weights: torch.Tensor,
    stage: str,
    optimizer: torch.optim.Optimizer | None = None,
    patch_to_group: dict[str, str] | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    metric = ConfusionMetrics(8, CLASS_NAMES_8)
    group_metrics: dict[str, ConfusionMetrics] = {}
    total_loss = 0.0
    batches = 0
    components: dict[str, float] = {}
    for batch in loader:
        s2 = batch["s2"].to(device, non_blocking=True)
        batch_size, _, height, width = s2.shape
        target = leaf_target_to_official8(
            batch["label"].to(device, non_blocking=True)
        )
        urban = batch.get("urban")
        roads = batch.get("roads")
        osm = batch.get("osm")
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            output = model(
                s2,
                batch["s1"].to(device, non_blocking=True),
                urban.to(device, non_blocking=True)
                if urban is not None else s2.new_zeros((batch_size, 0, height, width)),
                roads.to(device, non_blocking=True)
                if roads is not None else s2.new_zeros((batch_size, 0, height, width)),
                osm.to(device, non_blocking=True)
                if osm is not None else s2.new_zeros((batch_size, 0, height, width)),
                batch["input_ids"].to(device, non_blocking=True),
                batch["text_mask"].to(device, non_blocking=True),
            )
            loss, parts = loss_function(
                output,
                target,
                class_weights,
                coarse_weights,
                stage,
                use_hierarchy=model.use_hierarchy,
            )
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                5.0,
            )
            optimizer.step()
        prediction_source = (
            output["hierarchy_log_probs"]
            if stage == "hierarchy"
            else output["combined_log_probs"]
        )
        metric.update(prediction_source, target)
        if patch_to_group is not None:
            predictions = prediction_source.argmax(dim=1)
            for index, patch_id in enumerate(batch["patch_id"]):
                group = patch_to_group[str(patch_id)]
                group_metrics.setdefault(
                    group, ConfusionMetrics(8, CLASS_NAMES_8)
                ).update(predictions[index], target[index])
        total_loss += float(loss.detach())
        for key, value in parts.items():
            components[key] = components.get(key, 0.0) + value
        batches += 1
    result = metric.compute()
    result["loss"] = total_loss / max(batches, 1)
    result["loss_components"] = {
        key: value / max(batches, 1) for key, value in components.items()
    }
    if patch_to_group is not None:
        result["spatial_group_confusions"] = {
            group: group_metric.compute()["confusion_matrix"]
            for group, group_metric in group_metrics.items()
        }
    return result


def set_trainability(model: HierarchicalMultiModalFCNN, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = stage == "joint"
    if stage == "hierarchy":
        for name, parameter in model.named_parameters():
            if (
                model.use_hierarchy
                and not name.startswith("base.")
                and name != "hierarchy_gate_logit"
                and (model.use_semantic_residual or not name.startswith("semantic_"))
            ):
                parameter.requires_grad = True
    elif stage == "gate":
        if model.use_hierarchy:
            model.hierarchy_gate_logit.requires_grad = True
        for name, parameter in model.named_parameters():
            if model.use_semantic_residual and name.startswith("semantic_"):
                parameter.requires_grad = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_hierarchical.yaml")
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage-epochs", default="8,8,12")
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--component-mode",
        choices=("full", "no_hierarchy", "no_residual", "shared_gate"),
        default="full",
        help=(
            "Structural ablation: remove the conditional hierarchy, remove the "
            "semantic residual, or replace class-wise gates by one shared gate."
        ),
    )
    parser.add_argument("--manifest-name", default="manifest_hier_v2.csv")
    parser.add_argument("--vocab-name", default="vocab_hier_v2.json")
    parser.add_argument("--normalization-name", default="normalization_hier_v2.json")
    parser.add_argument(
        "--building-feature-mode",
        choices=("full", "function_only", "morphology_only", "none"),
        default="full",
        help="Controlled building lineage ablation while retaining the 95-channel fusion geometry.",
    )
    parser.add_argument(
        "--modalities",
        default=",".join(sorted(ALL_MODALITIES)),
        help="Comma-separated modalities used by both the FCNN base and hierarchy.",
    )
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--evaluate-checkpoint", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["manifest_name"] = args.manifest_name
    cfg["data"]["vocab_name"] = args.vocab_name
    cfg["data"]["normalization_name"] = args.normalization_name
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modalities = {
        value.strip() for value in args.modalities.split(",") if value.strip()
    }
    unknown = modalities - ALL_MODALITIES
    if unknown:
        raise ValueError(f"Unknown modalities: {sorted(unknown)}")
    if "s2" not in modalities:
        raise ValueError("S2 is required as the shared spatial reference input")
    train_dataset = make_dataset(cfg, "train", True, modalities, args.building_feature_mode)
    validation_dataset = make_dataset(cfg, "val", False, modalities, args.building_feature_mode)
    test_dataset = make_dataset(cfg, "test", False, modalities, args.building_feature_mode)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=patch_sampler(train_dataset, args.seed),
        **loader_args,
    )
    validation_loader = DataLoader(
        validation_dataset, shuffle=False, **loader_args
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)
    prepared = Path(cfg["paths"]["prepared_dir"])
    vocab = json.loads((prepared / args.vocab_name).read_text(encoding="utf-8"))
    use_semantic_residual = args.component_mode != "no_residual"
    use_hierarchy = args.component_mode != "no_hierarchy"
    gate_mode = "shared" if args.component_mode == "shared_gate" else "class"
    model = HierarchicalMultiModalFCNN(
        vocab_size=len(vocab),
        s2_channels=train_dataset.s2_output_channels,
        s1_channels=train_dataset.s1_output_channels if "s1" in modalities else 0,
        urban_channels=train_dataset.urban_channels,
        road_channels=train_dataset.road_channels,
        osm_channels=20 if {"poi", "aoi"} & modalities else 0,
        text_dim=int(cfg["model"]["text_dim"]),
        text_heads=int(cfg["model"]["text_heads"]),
        text_layers=int(cfg["model"]["text_layers"]),
        max_tokens=int(cfg["data"]["max_text_tokens"]),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        use_text="text" in modalities,
        use_semantic_residual=use_semantic_residual,
        use_hierarchy=use_hierarchy,
        gate_mode=gate_mode,
    ).to(device)
    baseline = torch.load(
        args.baseline_checkpoint, map_location=device, weights_only=False
    )
    model.load_fcnn_baseline(baseline["model_state"])
    counts = np.asarray(
        [
            train_dataset.frame[f"pixels_class_{old_id}"].sum()
            for old_id in (0, 2, 3, 4, 5, 6, 7, 9)
        ],
        dtype=float,
    )
    class_weights = torch.as_tensor(
        balanced_weights(counts), device=device
    )
    coarse_counts = np.asarray(
        [counts[:5].sum(), counts[5:7].sum(), counts[7]], dtype=float
    )
    coarse_weights = torch.as_tensor(
        balanced_weights(coarse_counts), device=device
    )
    output_dir = Path(cfg["paths"]["output_dir"]) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    patch_to_group = {
        str(row.patch_id): str(row.group)
        for row in test_dataset.frame.itertuples()
    }
    if args.evaluate_checkpoint:
        checkpoint = torch.load(
            args.evaluate_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        validation = run_epoch(
            model, validation_loader, device, class_weights, coarse_weights, "joint"
        )
        test = run_epoch(
            model,
            test_loader,
            device,
            class_weights,
            coarse_weights,
            "joint",
            patch_to_group=patch_to_group,
        )
        report = {
            "model": "hierarchical_multimodal_fcnn",
            "seed": args.seed,
            "best_stage": checkpoint["stage"],
            "best_epoch": checkpoint["epoch"],
            "component_mode": args.component_mode,
            "modalities": sorted(modalities),
            "validation": {"official8": validation},
            "test": {"official8": test},
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({
            "test_OA": test["overall_accuracy"],
            "test_mIoU": test["macro_iou_present"],
            "test_macro_F1": test["macro_f1_present"],
        }, ensure_ascii=False, indent=2))
        return

    initial = run_epoch(
        model, validation_loader, device, class_weights, coarse_weights, "joint"
    )
    print(
        f"initial_FCNN val_OA={initial['overall_accuracy']:.4f} "
        f"val_mIoU={initial['macro_iou_present']:.4f}",
        flush=True,
    )
    stage_epochs = [int(value) for value in args.stage_epochs.split(",")]
    if len(stage_epochs) != 3:
        raise ValueError("--stage-epochs must be hierarchy,gate,joint")
    history: list[dict] = []
    candidates: list[dict] = []
    stage_names = (
        ("hierarchy", "gate", "joint")
        if use_hierarchy
        else ("gate", "joint")
    )
    stage_epoch_map = dict(zip(("hierarchy", "gate", "joint"), stage_epochs))
    for stage in stage_names:
        epochs = stage_epoch_map[stage]
        set_trainability(model, stage)
        new_named_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and not name.startswith("base.")
        ]
        base_parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith("base.")
        ]
        if stage == "gate":
            gate_parameters = [
                parameter for name, parameter in new_named_parameters
                if name == "hierarchy_gate_logit"
            ]
            semantic_parameters = [
                parameter for name, parameter in new_named_parameters
                if name != "hierarchy_gate_logit"
            ]
            groups = [
                {"params": semantic_parameters, "lr": 3e-4},
                {"params": gate_parameters, "lr": 2e-2},
            ]
        else:
            groups = [{
                "params": [parameter for _, parameter in new_named_parameters],
                "lr": 1e-4,
            }]
        if base_parameters:
            groups.append({"params": base_parameters, "lr": 1e-5})
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        best_score = -1.0
        stale = 0
        best_path = output_dir / f"best_{stage}.pt"
        for epoch in range(1, epochs + 1):
            train = run_epoch(
                model,
                train_loader,
                device,
                class_weights,
                coarse_weights,
                stage,
                optimizer=optimizer,
            )
            validation = run_epoch(
                model,
                validation_loader,
                device,
                class_weights,
                coarse_weights,
                stage,
            )
            score = validation["macro_iou_present"]
            print(
                f"stage={stage:9s} epoch={epoch:02d} "
                f"val_OA={validation['overall_accuracy']:.4f} "
                f"val_mIoU={score:.4f}",
                flush=True,
            )
            history.append({
                "stage": stage,
                "epoch": epoch,
                "train": train,
                "validation": validation,
            })
            if score > best_score:
                best_score = score
                stale = 0
                torch.save({
                    "model_state": model.state_dict(),
                    "stage": stage,
                    "epoch": epoch,
                    "seed": args.seed,
                "backbone": "full_modality_fcnn",
                "modalities": sorted(modalities),
                "component_mode": args.component_mode,
                "building_feature_mode": args.building_feature_mode,
                }, best_path)
            else:
                stale += 1
            if stale >= args.patience:
                break
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        validation = run_epoch(
            model,
            validation_loader,
            device,
            class_weights,
            coarse_weights,
            "joint" if stage != "hierarchy" else "hierarchy",
        )
        if stage != "hierarchy":
            candidates.append({
                "score": validation["macro_iou_present"],
                "checkpoint": str(best_path),
                "stage": stage,
                "epoch": checkpoint["epoch"],
            })
    selected = max(candidates, key=lambda row: row["score"])
    selected_checkpoint = torch.load(
        selected["checkpoint"], map_location="cpu", weights_only=False
    )
    torch.save(selected_checkpoint, output_dir / "best_final.pt")
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "selection.json").write_text(
        json.dumps(
            {
                "initial_validation": initial,
                "candidates": candidates,
                "selected": selected,
                "building_feature_mode": args.building_feature_mode,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(selected, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
