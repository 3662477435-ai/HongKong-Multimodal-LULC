from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .config import load_config
from .hierarchical_loss import HierarchicalLoss
from .hierarchical_model import HierarchicalSwinFusion
from .hierarchical_train import (
    _balanced_weights,
    make_dataset,
    seed_everything,
    set_stage_trainability,
)
from .labels import (
    COARSE_CLASS_NAMES,
    IGNORE_INDEX,
    LEAF_CLASS_NAMES,
    LEAF_TO_COARSE,
)
from .metrics import ConfusionMetrics
from .task8 import (
    CLASS_NAMES_8,
    EXCLUDED_LEAF_IDS,
    LEAF_TO_OFFICIAL8,
    filter_leaf_target,
    leaf_prediction_to_official8,
    leaf_target_to_official8,
)


STAGES = ("coarse", "natural", "built", "joint")


def patch_sampler(dataset, seed: int) -> WeightedRandomSampler:
    frame = dataset.frame
    active = [index for index in range(14) if index not in EXCLUDED_LEAF_IDS]
    support = np.asarray(
        [(frame[f"leaf_pixels_class_{index}"] > 0).sum() for index in active],
        dtype=float,
    )
    rarity = np.full(len(frame), 0.05, dtype=float)
    for leaf_id, count in zip(active, support):
        rarity += (
            frame[f"leaf_pixels_class_{leaf_id}"].to_numpy() > 0
        ) / np.sqrt(max(count, 1.0))
    return WeightedRandomSampler(
        torch.as_tensor(rarity, dtype=torch.double),
        num_samples=len(frame),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


class MetricBundle8:
    def __init__(self, patch_to_group: dict[str, str] | None = None) -> None:
        self.coarse = ConfusionMetrics(4, COARSE_CLASS_NAMES)
        self.built = ConfusionMetrics(6, LEAF_CLASS_NAMES[:6])
        self.natural = ConfusionMetrics(8, LEAF_CLASS_NAMES[6:])
        self.leaf = ConfusionMetrics(14, LEAF_CLASS_NAMES)
        self.official = ConfusionMetrics(8, CLASS_NAMES_8)
        self.patch_to_group = patch_to_group
        self.group_metrics: dict[str, ConfusionMetrics] = {}
        self.coarse_lut = torch.as_tensor(LEAF_TO_COARSE, dtype=torch.long)

    def update(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        patch_ids: list[str] | tuple[str, ...],
    ) -> None:
        valid = target != IGNORE_INDEX
        safe = target.masked_fill(~valid, 0)
        coarse_target = self.coarse_lut.to(target.device)[safe].masked_fill(
            ~valid, IGNORE_INDEX
        )
        self.coarse.update(outputs["coarse_logits"], coarse_target)
        self.built.update(outputs["built_logits"], safe, mask=valid & (safe < 6))
        self.natural.update(
            outputs["natural_logits"], safe - 6, mask=valid & (safe >= 6)
        )
        self.leaf.update(outputs["leaf_log_probs"], target)
        leaf_prediction = outputs["leaf_log_probs"].argmax(dim=1)
        if "official8_log_probs" in outputs:
            official_prediction = outputs["official8_log_probs"].argmax(dim=1)
        else:
            official_prediction = leaf_prediction_to_official8(leaf_prediction)
        official_target = leaf_target_to_official8(target)
        self.official.update(official_prediction, official_target)
        if self.patch_to_group is None:
            return
        for index, patch_id in enumerate(patch_ids):
            group = self.patch_to_group[str(patch_id)]
            metric = self.group_metrics.setdefault(
                group, ConfusionMetrics(8, CLASS_NAMES_8)
            )
            metric.update(
                official_prediction[index:index + 1],
                official_target[index:index + 1],
            )

    def compute(self) -> dict:
        result = {
            "coarse3_present": self.coarse.compute(),
            "built5_present": self.built.compute(),
            "natural7_present": self.natural.compute(),
            "leaf12_present": self.leaf.compute(),
            "official8_projection": self.official.compute(),
        }
        if self.patch_to_group is not None:
            result["official8_projection"]["spatial_group_confusions"] = {
                group: metric.matrix.tolist()
                for group, metric in sorted(self.group_metrics.items())
            }
        return result


def run_epoch(
    model: HierarchicalSwinFusion,
    loader: DataLoader,
    criterion: HierarchicalLoss,
    stage: str,
    device: torch.device,
    optimizer=None,
    scaler=None,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.bfloat16,
    patch_to_group: dict[str, str] | None = None,
) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)
    metrics = MetricBundle8(patch_to_group=patch_to_group)
    total_loss = 0.0
    component_sums: dict[str, float] = {}
    batches = 0
    skipped = 0
    for batch in loader:
        s1 = batch["s1"].to(device, non_blocking=True)
        s2 = batch["s2"].to(device, non_blocking=True)
        ids = batch["input_ids"].to(device, non_blocking=True)
        text_mask = batch["text_mask"].to(device, non_blocking=True)
        target = filter_leaf_target(batch["label"].to(device, non_blocking=True))
        osm = batch.get("osm")
        urban = batch.get("urban")
        roads = batch.get("roads")
        if osm is not None:
            osm = osm.to(device, non_blocking=True)
        if urban is not None:
            urban = urban.to(device, non_blocking=True)
        if roads is not None:
            roads = roads.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type, enabled=use_amp, dtype=amp_dtype
        ):
            outputs = model(
                s1, s2, ids, text_mask, osm=osm, urban=urban, roads=roads
            )
            loss, components = criterion(outputs, target, stage=stage, osm=osm)
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            finite = bool(torch.isfinite(loss)) and all(
                bool(torch.isfinite(gradient).all()) for gradient in gradients
            )
            if finite:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            else:
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.update()
        total_loss += float(loss.detach())
        for key, value in components.items():
            component_sums[key] = component_sums.get(key, 0.0) + value
        batches += 1
        metrics.update(outputs, target, batch["patch_id"])
    result = metrics.compute()
    result["loss"] = total_loss / max(batches, 1)
    result["loss_components"] = {
        key: value / max(batches, 1) for key, value in component_sums.items()
    }
    result["skipped_nonfinite_steps"] = skipped
    return result["loss"], result


def selection_score(metrics: dict, stage: str) -> float:
    key = {
        "coarse": "coarse3_present",
        "natural": "natural7_present",
        "built": "built5_present",
        "joint": "official8_projection",
    }[stage]
    return float(metrics[key]["macro_iou_present"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the eight-class soft-routed model.")
    parser.add_argument("--config", default="config_release.yaml")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--stage-epochs", default="1,1,2,10")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-name", default="manifest_hier_v2.csv")
    parser.add_argument("--vocab-name", default="vocab_hier_v2.json")
    parser.add_argument("--normalization-name", default="normalization_hier_v2.json")
    parser.add_argument("--modalities", default=None)
    parser.add_argument("--focal-gamma", type=float, default=0.0)
    parser.add_argument("--leaf-aux-weight", type=float, default=0.30)
    parser.add_argument("--consistency-weight", type=float, default=0.05)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--only-joint", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--evaluate-checkpoint", default=None)
    parser.add_argument("--head-warmup-epochs", type=int, default=0)
    parser.add_argument("--head-learning-rate", type=float, default=0.0003)
    parser.add_argument("--hierarchy-only", action="store_true")
    parser.add_argument("--head-only", action="store_true")
    parser.add_argument(
        "--image-backbone",
        choices=("swin_v2_tiny", "fcnn"),
        default="swin_v2_tiny",
    )
    parser.add_argument("--joint-hierarchy-weight", type=float, default=1.0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["manifest_name"] = args.manifest_name
    cfg["data"]["vocab_name"] = args.vocab_name
    cfg["data"]["normalization_name"] = args.normalization_name
    cfg["train"]["seed"] = args.seed
    if args.modalities:
        cfg["model"]["modalities"] = [
            item.strip() for item in args.modalities.split(",") if item.strip()
        ]
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if str(cfg["train"].get("amp_dtype", "bfloat16")).lower() == "bfloat16"
        else torch.float16
    )

    train_ds = make_dataset(cfg, "train", True)
    val_ds = make_dataset(cfg, "val", False)
    test_ds = make_dataset(cfg, "test", False)
    worker_count = (
        args.num_workers
        if args.num_workers is not None
        else int(cfg["train"].get("num_workers", 0))
    )
    loader_args = {
        "batch_size": int(cfg["train"]["batch_size"]),
        "num_workers": worker_count,
        "pin_memory": device.type == "cuda",
        "persistent_workers": worker_count > 0,
    }
    train_loader = DataLoader(
        train_ds, sampler=patch_sampler(train_ds, args.seed), **loader_args
    )
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    prepared = Path(cfg["paths"]["prepared_dir"])
    vocab = json.loads(
        (prepared / cfg["data"]["vocab_name"]).read_text(encoding="utf-8")
    )
    modalities = set(cfg["model"]["modalities"])
    model = HierarchicalSwinFusion(
        vocab_size=len(vocab),
        s2_channels=train_ds.s2_output_channels,
        s1_channels=train_ds.s1_output_channels,
        text_dim=int(cfg["model"]["text_dim"]),
        text_heads=int(cfg["model"]["text_heads"]),
        text_layers=int(cfg["model"]["text_layers"]),
        max_tokens=int(cfg["data"]["max_text_tokens"]),
        fpn_channels=int(cfg["model"].get("fpn_channels", 128)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        pretrained=bool(cfg["model"].get("pretrained", True)),
        use_s2="s2" in modalities,
        use_s1="s1" in modalities,
        use_poi="poi" in modalities,
        use_aoi="aoi" in modalities,
        use_text="text" in modalities,
        use_highres="highres" in modalities,
        use_nighttime="nighttime" in modalities,
        use_building="building" in modalities,
        use_roads="roads" in modalities,
        building_channels=train_ds.building_channels,
        separate_optical_indices=bool(
            cfg["model"].get("separate_optical_indices", False)
        ),
        excluded_leaf_ids=EXCLUDED_LEAF_IDS,
        official_projection=(
            None
            if args.hierarchy_only
            else tuple(int(x) for x in LEAF_TO_OFFICIAL8)
        ),
        image_backbone=args.image_backbone,
    ).to(device)
    if args.resume_checkpoint and not args.evaluate_checkpoint:
        resume = torch.load(
            args.resume_checkpoint, map_location=device, weights_only=False
        )
        incompatible = model.load_state_dict(resume["model_state"], strict=False)
        print(
            "resume_checkpoint="
            f"{args.resume_checkpoint} missing={list(incompatible.missing_keys)} "
            f"unexpected={list(incompatible.unexpected_keys)}",
            flush=True,
        )

    leaf_counts = np.asarray(
        [
            train_ds.frame[f"leaf_pixels_class_{leaf_id}"].sum()
            for leaf_id in range(14)
        ],
        dtype=float,
    )
    leaf_counts[list(EXCLUDED_LEAF_IDS)] = 0.0
    coarse_counts = np.zeros(4, dtype=float)
    for leaf_id, count in enumerate(leaf_counts):
        coarse_counts[LEAF_TO_COARSE[leaf_id]] += count
    official_counts = np.zeros(8, dtype=float)
    for leaf_id, count in enumerate(leaf_counts):
        official_id = int(LEAF_TO_OFFICIAL8[leaf_id])
        if official_id >= 0:
            official_counts[official_id] += count
    criterion = HierarchicalLoss(
        torch.as_tensor(
            _balanced_weights(
                leaf_counts,
                float(cfg["train"].get("class_weight_power", 0.5)),
                float(cfg["train"].get("max_class_weight", 6.0)),
            ),
            device=device,
        ),
        torch.as_tensor(
            _balanced_weights(
                coarse_counts,
                float(cfg["train"].get("class_weight_power", 0.5)),
                float(cfg["train"].get("max_class_weight", 6.0)),
            ),
            device=device,
        ),
        dice_weight=float(cfg["train"].get("dice_weight", 0.3)),
        coarse_weight=float(cfg["train"].get("coarse_weight", 0.5)),
        weak_weight=float(cfg["train"].get("weak_weight", 0.05)),
        conflict_weight=float(cfg["train"].get("aoi_conflict_weight", 0.3)),
        focal_gamma=args.focal_gamma,
        ignored_built_classes=(1,),
        official_weights=(
            None
            if args.hierarchy_only
            else torch.as_tensor(
                _balanced_weights(
                    official_counts,
                    float(cfg["train"].get("class_weight_power", 0.5)),
                    float(cfg["train"].get("max_class_weight", 6.0)),
                ),
                device=device,
            )
        ),
        official_lut=(
            None
            if args.hierarchy_only
            else torch.as_tensor(
                LEAF_TO_OFFICIAL8, dtype=torch.long, device=device
            )
        ),
        leaf_aux_weight=args.leaf_aux_weight,
        consistency_weight=args.consistency_weight,
        joint_hierarchy_weight=args.joint_hierarchy_weight,
    )
    stage_epochs = [int(value) for value in args.stage_epochs.split(",")]
    if len(stage_epochs) != 4:
        raise ValueError("stage-epochs must contain four integers")
    output = Path(cfg["paths"]["output_dir"]) / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    results: dict[str, dict] = {}
    patch_to_group = {
        str(row.patch_id): str(row.group) for row in test_ds.frame.itertuples()
    }
    if args.evaluate_checkpoint:
        checkpoint = torch.load(
            args.evaluate_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        _, validation = run_epoch(
            model, val_loader, criterion, "joint", device,
            use_amp=use_amp, amp_dtype=amp_dtype,
        )
        _, test = run_epoch(
            model, test_loader, criterion, "joint", device,
            use_amp=use_amp, amp_dtype=amp_dtype,
            patch_to_group=patch_to_group,
        )
        evaluated = {
            "best_epoch": int(checkpoint.get("epoch", 0)),
            "validation": validation,
            "test": test,
            "evaluation_only": True,
        }
        (output / "metrics_joint.json").write_text(
            json.dumps(evaluated, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output / "stage_metrics.json").write_text(
            json.dumps({"joint": evaluated}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        joint = test["official8_projection"]
        print(json.dumps({
            "run": args.run_name,
            "test_OA": joint["overall_accuracy"],
            "test_mIoU": joint["macro_iou_present"],
            "test_macro_F1": joint["macro_f1_present"],
        }, ensure_ascii=False, indent=2))
        return

    if args.head_only:
        if args.hierarchy_only:
            raise ValueError("--head-only and --hierarchy-only are incompatible")
        if args.head_warmup_epochs <= 0:
            raise ValueError("--head-only requires --head-warmup-epochs > 0")
        stage_plan = (("head", args.head_warmup_epochs),)
    elif args.only_joint:
        stage_plan = []
        if args.head_warmup_epochs > 0:
            stage_plan.append(("head", args.head_warmup_epochs))
        stage_plan.append(("joint", stage_epochs[-1]))
        stage_plan = tuple(stage_plan)
    else:
        stage_plan = tuple(zip(STAGES, stage_epochs))
    for stage, epochs in stage_plan:
        criterion_stage = "joint" if stage == "head" else stage
        if stage == "head":
            for parameter in model.parameters():
                parameter.requires_grad = False
            for name, parameter in model.named_parameters():
                if name.startswith("official_"):
                    parameter.requires_grad = True
        else:
            set_stage_trainability(model, stage)
        if stage not in ("joint", "head"):
            for name, parameter in model.named_parameters():
                if name.startswith("official_"):
                    parameter.requires_grad = False
        stage_learning_rate = (
            args.head_learning_rate
            if stage == "head"
            else float(args.learning_rate or cfg["train"]["learning_rate"])
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=stage_learning_rate,
            weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1)
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=use_amp and amp_dtype == torch.float16
        )
        best = -1.0
        stale = 0
        best_path = output / f"best_{stage}.pt"
        for epoch in range(1, epochs + 1):
            _, train = run_epoch(
                model, train_loader, criterion, criterion_stage, device,
                optimizer=optimizer, scaler=scaler, use_amp=use_amp,
                amp_dtype=amp_dtype,
            )
            _, validation = run_epoch(
                model, val_loader, criterion, criterion_stage, device,
                use_amp=use_amp, amp_dtype=amp_dtype,
            )
            scheduler.step()
            score = (
                float(validation["official8_projection"]["macro_iou_present"])
                if stage == "head"
                else selection_score(validation, stage)
            )
            history.append({
                "stage": stage, "epoch": epoch,
                "train": train, "validation": validation,
            })
            key = {
                "coarse": "coarse3_present",
                "natural": "natural7_present",
                "built": "built5_present",
                "joint": "official8_projection",
                "head": "official8_projection",
            }[stage]
            print(
                f"task=official8 stage={stage:7s} seed={args.seed} epoch={epoch:02d} "
                f"val_OA={validation[key]['overall_accuracy']:.4f} "
                f"val_mIoU={score:.4f}",
                flush=True,
            )
            if score > best:
                best = score
                stale = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "stage": stage,
                        "epoch": epoch,
                        "modalities": sorted(modalities),
                        "num_classes": 8,
                        "class_names": CLASS_NAMES_8,
                        "excluded_leaf_ids": list(EXCLUDED_LEAF_IDS),
                        "seed": args.seed,
                        "config": cfg,
                        "image_backbone": args.image_backbone,
                    },
                    best_path,
                )
            else:
                stale += 1
            if args.patience > 0 and stale >= args.patience:
                print(f"stage={stage} early_stop best_val_mIoU={best:.4f}", flush=True)
                break
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        _, validation = run_epoch(
            model, val_loader, criterion, criterion_stage, device,
            use_amp=use_amp, amp_dtype=amp_dtype,
        )
        test = None
        if not args.skip_test:
            _, test = run_epoch(
                model, test_loader, criterion, criterion_stage, device,
                # A head-only stage still optimizes and evaluates the joint
                # eight-class objective.
                use_amp=use_amp, amp_dtype=amp_dtype,
                patch_to_group=patch_to_group,
            )
        results[stage] = {
            "best_epoch": int(checkpoint["epoch"]),
            "validation": validation,
            "test": test,
        }
        (output / f"metrics_{stage}.json").write_text(
            json.dumps(results[stage], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "stage_metrics.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    final_stage = "joint" if "joint" in results else "head"
    joint_test = results[final_stage]["test"]
    if joint_test is None:
        joint = results[final_stage]["validation"]["official8_projection"]
        print(json.dumps({
            "run": args.run_name,
            "selection_split": "validation",
            "validation_OA": joint["overall_accuracy"],
            "validation_mIoU": joint["macro_iou_present"],
            "validation_macro_F1": joint["macro_f1_present"],
        }, ensure_ascii=False, indent=2))
    else:
        joint = joint_test["official8_projection"]
        print(json.dumps({
            "run": args.run_name,
            "test_OA": joint["overall_accuracy"],
            "test_mIoU": joint["macro_iou_present"],
            "test_macro_F1": joint["macro_f1_present"],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
