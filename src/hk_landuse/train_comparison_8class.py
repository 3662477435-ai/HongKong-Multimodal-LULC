from __future__ import annotations

import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from .comparison_models import (
    CNNBaseline,
    FCNNBaseline,
    FCNResNet50Baseline,
    FCNResNet50DirectBaseline,
    MultiModalEarlyFusionComparator,
    TransformerBaseline,
    VGGUNetTransformerBaseline,
    VGGUNetTransformerDirectBaseline,
    SegFormerB1Direct48,
    DeepLabV3PlusResNet50Direct48,
    SwinTUPerNetDirect48,
)
from .config import load_config
from .metrics import ConfusionMetrics
from .task8 import ACTIVE_OFFICIAL_IDS, CLASS_NAMES_8, leaf_target_to_official8
from .train_comparison import make_dataset


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


def balanced_weights(counts: np.ndarray, power: float = 0.5, maximum: float = 6.0) -> np.ndarray:
    weights = np.zeros(len(counts), dtype=np.float32)
    present = counts > 0
    weights[present] = (counts[present].sum() / (present.sum() * counts[present])) ** power
    weights[present] = np.clip(weights[present], 0.25, maximum)
    return weights


def patch_sampler(dataset, seed: int) -> WeightedRandomSampler:
    frame = dataset.frame
    support = np.asarray(
        [(frame[f"pixels_class_{old_id}"] > 0).sum() for old_id in ACTIVE_OFFICIAL_IDS],
        dtype=float,
    )
    rarity = np.full(len(frame), 0.05, dtype=float)
    for old_id, count in zip(ACTIVE_OFFICIAL_IDS, support):
        rarity += (frame[f"pixels_class_{old_id}"].to_numpy() > 0) / np.sqrt(max(count, 1.0))
    return WeightedRandomSampler(
        torch.as_tensor(rarity, dtype=torch.double),
        num_samples=len(frame),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != 255
    safe = target.masked_fill(~valid, 0)
    onehot = F.one_hot(safe, 8).permute(0, 3, 1, 2).to(logits.dtype)
    mask = valid[:, None]
    probabilities = logits.softmax(dim=1) * mask
    onehot = onehot * mask
    intersection = (probabilities * onehot).sum((0, 2, 3))
    denominator = probabilities.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    present = onehot.sum((0, 2, 3)) > 0
    return 1.0 - ((2 * intersection[present] + 1) / (denominator[present] + 1)).mean()


def run_epoch(
    model,
    loader,
    device,
    weights,
    dice_weight,
    optimizer=None,
    multimodal: bool = True,
    patch_to_group: dict[str, str] | None = None,
) -> dict:
    training = optimizer is not None
    model.train(training)
    metric = ConfusionMetrics(8, CLASS_NAMES_8)
    group_metrics: dict[str, ConfusionMetrics] = {}
    total_loss = 0.0
    batches = 0
    for batch in loader:
        s2 = batch["s2"].to(device, non_blocking=True)
        target = leaf_target_to_official8(batch["label"].to(device, non_blocking=True))
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            if multimodal:
                batch_size, _, height, width = s2.shape
                urban = batch.get("urban")
                roads = batch.get("roads")
                osm = batch.get("osm")
                logits = model(
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
            else:
                logits = model(s2)
            ce = F.cross_entropy(logits, target, weight=weights, ignore_index=255)
            loss = (1.0 - dice_weight) * ce + dice_weight * dice_loss(logits, target)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.detach())
        batches += 1
        metric.update(logits, target)
        if patch_to_group is not None:
            prediction = logits.argmax(dim=1)
            for index, patch_id in enumerate(batch["patch_id"]):
                group = patch_to_group[str(patch_id)]
                group_metric = group_metrics.setdefault(
                    group, ConfusionMetrics(8, CLASS_NAMES_8)
                )
                group_metric.update(prediction[index:index + 1], target[index:index + 1])
    result = metric.compute()
    result["loss"] = total_loss / max(batches, 1)
    if patch_to_group is not None:
        result["spatial_group_confusions"] = {
            group: value.matrix.tolist() for group, value in sorted(group_metrics.items())
        }
    return result


def build_model(cfg: dict, dataset, backbone: str, modalities: set[str], pretrained: bool):
    constructors = {
        "cnn": CNNBaseline,
        "fcnn": FCNNBaseline,
        "transformer": TransformerBaseline,
        "fcn_resnet50": FCNResNet50Baseline,
        "vgg_unet_transformer": VGGUNetTransformerBaseline,
        "fcn_resnet50_direct48": FCNResNet50DirectBaseline,
        "vgg_unet_transformer_direct48": VGGUNetTransformerDirectBaseline,
        "segformer_b1_direct48": SegFormerB1Direct48,
        "deeplabv3plus_resnet50_direct48": DeepLabV3PlusResNet50Direct48,
        "swin_t_upernet_direct48": SwinTUPerNetDirect48,
    }
    use_fusion = modalities != {"s2"}
    if not use_fusion:
        return constructors[backbone](
            dataset.s2_output_channels, num_classes=8, pretrained=pretrained
        ), False
    prepared = Path(cfg["paths"]["prepared_dir"])
    vocab = json.loads(
        (prepared / cfg["data"].get("vocab_name", "vocab_hier_v2.json")).read_text(
            encoding="utf-8"
        )
    )
    model = MultiModalEarlyFusionComparator(
        backbone=backbone,
        vocab_size=len(vocab),
        s2_channels=dataset.s2_output_channels,
        s1_channels=dataset.s1_output_channels if "s1" in modalities else 0,
        urban_channels=dataset.urban_channels,
        road_channels=dataset.road_channels,
        osm_channels=20 if {"poi", "aoi"} & modalities else 0,
        text_dim=int(cfg["model"]["text_dim"]),
        text_heads=int(cfg["model"]["text_heads"]),
        text_layers=int(cfg["model"]["text_layers"]),
        max_tokens=int(cfg["data"]["max_text_tokens"]),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        pretrained=pretrained,
        use_text="text" in modalities,
        num_classes=8,
    )
    return model, True


def optimizer_with_backbone_rate(model, backbone_rate: float):
    """Use a 10x rate for newly initialized adapters/decoders and text modules."""
    # `model` is used by SegFormer, `swin` by the Swin wrapper and `backbone`
    # by the direct-48 DeepLabV3+ wrapper.  The expanded first convolution is
    # deliberately excluded: it is initialized by channel replication and is
    # optimized together with the newly initialized adaptation/decoder layers.
    pretrained_tokens = ("segmenter.model", "segmenter.swin", "segmenter.backbone")
    expanded_first_layer_tokens = (
        "segmenter.model.segformer.encoder.patch_embeddings.0.proj.weight",
        "segmenter.swin.features.0.0.weight",
        "segmenter.backbone.conv1.weight",
    )
    backbone, new = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_backbone = any(token in name for token in pretrained_tokens)
        is_expanded_first_layer = any(token in name for token in expanded_first_layer_tokens)
        (backbone if is_backbone and not is_expanded_first_layer else new).append(parameter)
    groups = [{"params": new, "lr": backbone_rate * 10.0}]
    if backbone:
        groups.append({"params": backbone, "lr": backbone_rate})
    return torch.optim.AdamW(groups, weight_decay=1e-4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an eight-class full-modal comparator.")
    parser.add_argument("--config", default="config_release.yaml")
    parser.add_argument(
        "--model",
        choices=(
            "cnn", "fcnn", "transformer", "fcn_resnet50", "vgg_unet_transformer",
            "fcn_resnet50_direct48", "vgg_unet_transformer_direct48",
            "segformer_b1_direct48", "deeplabv3plus_resnet50_direct48", "swin_t_upernet_direct48",
        ),
        required=True,
    )
    parser.add_argument("--modalities", default=",".join(sorted(ALL_MODALITIES)))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--backbone-learning-rate", type=float, default=None,
        help="For direct-48 pretrained strong baselines: backbone LR, while newly initialized modules use 10x this rate.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--dice-weight", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--manifest-name", default="manifest_hier_v2.csv")
    parser.add_argument("--vocab-name", default="vocab_hier_v2.json")
    parser.add_argument("--normalization-name", default="normalization_hier_v2.json")
    parser.add_argument(
        "--building-feature-mode",
        choices=("full", "function_only", "morphology_only", "none"),
        default="full",
        help="Retain both building groups, only 9 functional bands, only 5 morphology/height bands, or zero all 14 while preserving fusion width.",
    )
    parser.add_argument(
        "--evaluate-checkpoint",
        type=Path,
        default=None,
        help="Skip training and evaluate an existing eight-class checkpoint.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg["data"]["manifest_name"] = args.manifest_name
    cfg["data"]["vocab_name"] = args.vocab_name
    cfg["data"]["normalization_name"] = args.normalization_name
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modalities = {value.strip() for value in args.modalities.split(",") if value.strip()}
    if "s2" not in modalities:
        raise ValueError("S2 is required as the reference modality")
    train_ds = make_dataset(cfg, "train", True, modalities, args.building_feature_mode)
    val_ds = make_dataset(cfg, "val", False, modalities, args.building_feature_mode)
    test_ds = make_dataset(cfg, "test", False, modalities, args.building_feature_mode)
    worker_count = (
        args.num_workers
        if args.num_workers is not None
        else int(cfg["train"].get("num_workers", 0))
    )
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": worker_count,
        "pin_memory": device.type == "cuda",
        "persistent_workers": worker_count > 0,
    }
    train_loader = DataLoader(train_ds, sampler=patch_sampler(train_ds, args.seed), **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    model, multimodal = build_model(
        cfg, train_ds, args.model, modalities, pretrained=not args.no_pretrained
    )
    model = model.to(device)
    counts = np.asarray(
        [train_ds.frame[f"pixels_class_{old_id}"].sum() for old_id in ACTIVE_OFFICIAL_IDS],
        dtype=float,
    )
    weights = torch.as_tensor(balanced_weights(counts), device=device)
    output = Path(cfg["paths"]["output_dir"]) / args.run_name
    output.mkdir(parents=True, exist_ok=True)
    if args.evaluate_checkpoint is not None:
        checkpoint = torch.load(
            args.evaluate_checkpoint, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state"])
        validation = run_epoch(
            model, val_loader, device, weights, args.dice_weight,
            multimodal=multimodal,
        )
        patch_to_group = {
            str(row.patch_id): str(row.group) for row in test_ds.frame.itertuples()
        }
        test = run_epoch(
            model, test_loader, device, weights, args.dice_weight,
            multimodal=multimodal, patch_to_group=patch_to_group,
        )
        result = {
            "task": "official8_without_commercial_and_barren",
            "model": args.model,
            "seed": args.seed,
            "modalities": sorted(modalities),
            "building_feature_mode": args.building_feature_mode,
            "best_epoch": int(checkpoint["epoch"]),
            "evaluation_only": True,
            "validation": {"official8": validation},
            "test": {"official8": test},
        }
        (output / "metrics.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({
            "run": args.run_name,
            "best_epoch": result["best_epoch"],
            "test_OA": test["overall_accuracy"],
            "test_mIoU": test["macro_iou_present"],
            "test_macro_F1": test["macro_f1_present"],
        }, ensure_ascii=False, indent=2))
        return
    optimizer = (
        optimizer_with_backbone_rate(model, args.backbone_learning_rate)
        if args.backbone_learning_rate is not None
        else torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1)
    )
    best_path = output / "best_model.pt"
    history: list[dict] = []
    best = -1.0
    stale = 0
    training_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        train = run_epoch(
            model, train_loader, device, weights, args.dice_weight,
            optimizer=optimizer, multimodal=multimodal,
        )
        validation = run_epoch(
            model, val_loader, device, weights, args.dice_weight,
            multimodal=multimodal,
        )
        scheduler.step()
        score = float(validation["macro_iou_present"])
        history.append({
            "epoch": epoch,
            "epoch_wall_seconds": time.perf_counter() - epoch_started,
            "train": train,
            "validation": validation,
        })
        print(
            f"task=official8 model={args.model} seed={args.seed} epoch={epoch:02d} "
            f"val_OA={validation['overall_accuracy']:.4f} val_mIoU={score:.4f}",
            flush=True,
        )
        if score > best:
            best = score
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "model": args.model,
                    "modalities": sorted(modalities),
                    "multimodal": multimodal,
                    "num_classes": 8,
                    "class_names": CLASS_NAMES_8,
                    "excluded_official_ids": [1, 8],
                    "seed": args.seed,
                },
                best_path,
            )
        else:
            stale += 1
        if args.patience > 0 and stale >= args.patience:
            print(f"early_stop best_val_mIoU={best:.4f}", flush=True)
            break

    training_wall_seconds = time.perf_counter() - training_started
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation = run_epoch(
        model, val_loader, device, weights, args.dice_weight, multimodal=multimodal
    )
    patch_to_group = {
        str(row.patch_id): str(row.group) for row in test_ds.frame.itertuples()
    }
    test = run_epoch(
        model, test_loader, device, weights, args.dice_weight,
        multimodal=multimodal, patch_to_group=patch_to_group,
    )
    result = {
        "task": "official8_without_commercial_and_barren",
        "model": args.model,
        "seed": args.seed,
        "modalities": sorted(modalities),
        "building_feature_mode": args.building_feature_mode,
        "best_epoch": int(checkpoint["epoch"]),
        "max_epochs": int(args.epochs),
        "actual_epochs_run": int(len(history)),
        "training_wall_seconds": float(training_wall_seconds),
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "validation": {"official8": validation},
        "test": {"official8": test},
    }
    (output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "run": args.run_name,
        "best_epoch": result["best_epoch"],
        "test_OA": test["overall_accuracy"],
        "test_mIoU": test["macro_iou_present"],
        "test_macro_F1": test["macro_f1_present"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
