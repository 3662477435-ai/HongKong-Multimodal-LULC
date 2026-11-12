from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .comparison_models import (
    CNNBaseline,
    FCNNBaseline,
    FCNResNet50Baseline,
    MultiModalEarlyFusionComparator,
    TransformerBaseline,
    VGGUNetTransformerBaseline,
)
from .config import load_config
from .dataset import LandUsePatchDataset
from .hierarchical_train import _balanced_weights, _patch_sampler
from .labels import CLASS_NAMES, IGNORE_INDEX, LEAF_CLASS_NAMES, LEAF_TO_OFFICIAL
from .metrics import ConfusionMetrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(
    cfg: dict,
    split: str,
    augment: bool,
    modalities: set[str] | None = None,
    building_feature_mode: str = "full",
) -> LandUsePatchDataset:
    prepared = Path(cfg["paths"]["prepared_dir"])
    modalities = modalities or {"s2"}
    use_osm = bool({"poi", "aoi"} & modalities)
    return LandUsePatchDataset(
        image_path=cfg["paths"]["image"],
        label_path=prepared / "labels_leaf14.tif",
        manifest_path=prepared / cfg["data"].get("manifest_name", "manifest_hier_v2.csv"),
        vocab_path=prepared / cfg["data"].get("vocab_name", "vocab_hier_v2.json"),
        normalization_path=prepared / cfg["data"].get(
            "normalization_name", "normalization_hier_v2.json"
        ),
        patch_size=int(cfg["data"]["patch_size"]),
        max_text_tokens=int(cfg["data"]["max_text_tokens"]),
        osm_poi_path=cfg["paths"].get("osm_poi_raster") if use_osm else None,
        osm_aoi_path=cfg["paths"].get("osm_aoi_raster") if use_osm else None,
        use_poi="poi" in modalities,
        use_aoi="aoi" in modalities,
        highres_path=cfg["paths"].get("highres_aligned") if "highres" in modalities else None,
        nighttime_path=cfg["paths"].get("nighttime_aligned") if "nighttime" in modalities else None,
        building_path=cfg["paths"].get("building_aligned") if "building" in modalities else None,
        building_feature_mode=building_feature_mode,
        road_features_path=cfg["paths"].get("roads_aligned") if "roads" in modalities else None,
        split=split,
        augment=augment,
    )


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    safe = target.masked_fill(~valid, 0)
    onehot = F.one_hot(safe, 14).permute(0, 3, 1, 2).to(logits.dtype)
    mask = valid[:, None]
    probs = logits.softmax(dim=1) * mask
    onehot = onehot * mask
    intersection = (probs * onehot).sum((0, 2, 3))
    denominator = probs.sum((0, 2, 3)) + onehot.sum((0, 2, 3))
    present = onehot.sum((0, 2, 3)) > 0
    return 1.0 - ((2 * intersection[present] + 1) / (denominator[present] + 1)).mean()


def run_epoch(
    model, loader, device, weights, dice_weight, optimizer=None, amp=True,
    multimodal: bool = False,
):
    training = optimizer is not None
    model.train(training)
    leaf_metric = ConfusionMetrics(14, LEAF_CLASS_NAMES)
    official_metric = ConfusionMetrics(10, CLASS_NAMES)
    official_lut = torch.as_tensor(LEAF_TO_OFFICIAL, device=device)
    total_loss = 0.0
    batches = 0
    for batch in loader:
        image = batch["s2"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type,
            enabled=amp and device.type == "cuda",
            dtype=torch.bfloat16,
        ):
            if multimodal:
                batch_size, _, height, width = image.shape
                urban = batch.get("urban")
                roads = batch.get("roads")
                osm = batch.get("osm")
                logits = model(
                    image,
                    batch["s1"].to(device, non_blocking=True),
                    (
                        urban.to(device, non_blocking=True) if urban is not None
                        else image.new_zeros((batch_size, 0, height, width))
                    ),
                    (
                        roads.to(device, non_blocking=True) if roads is not None
                        else image.new_zeros((batch_size, 0, height, width))
                    ),
                    (
                        osm.to(device, non_blocking=True) if osm is not None
                        else image.new_zeros((batch_size, 0, height, width))
                    ),
                    batch["input_ids"].to(device, non_blocking=True),
                    batch["text_mask"].to(device, non_blocking=True),
                )
            else:
                logits = model(image)
            ce = F.cross_entropy(logits, target, weight=weights, ignore_index=IGNORE_INDEX)
            loss = (1.0 - dice_weight) * ce + dice_weight * dice_loss(logits, target)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += float(loss.detach())
        batches += 1
        leaf_metric.update(logits, target)
        pred = logits.argmax(dim=1)
        valid = target != IGNORE_INDEX
        safe = target.masked_fill(~valid, 0)
        official_pred = official_lut[pred]
        official_target = official_lut[safe].masked_fill(~valid, IGNORE_INDEX)
        official_metric.update(official_pred, official_target)
    return {
        "loss": total_loss / max(batches, 1),
        "leaf14": leaf_metric.compute(),
        "official10": official_metric.compute(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train paper semantic-segmentation comparators.")
    parser.add_argument("--config", default="config_release.yaml")
    parser.add_argument(
        "--model",
        choices=("cnn", "fcnn", "transformer", "fcn_resnet50", "vgg_unet_transformer"),
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dice-weight", type=float, default=0.3)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--multimodal", action="store_true",
        help="Use S2, S1, high-res, NTL, buildings, roads, POI/AOI and OSM text.",
    )
    parser.add_argument(
        "--modalities",
        default=None,
        help="Explicit comma-separated subset. Supplying this enables the shared fusion head.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed = int(cfg["train"]["seed"])
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_modalities = {
        "s2", "s1", "highres", "nighttime", "building",
        "roads", "poi", "aoi", "text",
    }
    if args.modalities:
        modalities = {value.strip() for value in args.modalities.split(",") if value.strip()}
        use_fusion_head = True
    elif args.multimodal:
        modalities = all_modalities
        use_fusion_head = True
    else:
        modalities = {"s2"}
        use_fusion_head = False
    if "s2" not in modalities:
        raise ValueError("Every comparator requires s2 as the reference image modality")
    train_ds = make_dataset(cfg, "train", True, modalities)
    val_ds = make_dataset(cfg, "val", False, modalities)
    test_ds = make_dataset(cfg, "test", False, modalities)
    loader_args = dict(
        batch_size=args.batch_size,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        pin_memory=device.type == "cuda",
    )
    train_loader = DataLoader(train_ds, sampler=_patch_sampler(train_ds, seed), **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    constructors = {
        "cnn": CNNBaseline,
        "fcnn": FCNNBaseline,
        "transformer": TransformerBaseline,
        "fcn_resnet50": FCNResNet50Baseline,
        "vgg_unet_transformer": VGGUNetTransformerBaseline,
    }
    if use_fusion_head:
        vocab = json.loads(
            (Path(cfg["paths"]["prepared_dir"]) / "vocab_hier_v2.json").read_text(
                encoding="utf-8"
            )
        )
        model = MultiModalEarlyFusionComparator(
            backbone=args.model,
            vocab_size=len(vocab),
            s2_channels=train_ds.s2_output_channels,
            s1_channels=train_ds.s1_output_channels if "s1" in modalities else 0,
            urban_channels=train_ds.urban_channels,
            road_channels=train_ds.road_channels,
            osm_channels=20 if {"poi", "aoi"} & modalities else 0,
            text_dim=int(cfg["model"]["text_dim"]),
            text_heads=int(cfg["model"]["text_heads"]),
            text_layers=int(cfg["model"]["text_layers"]),
            max_tokens=int(cfg["data"]["max_text_tokens"]),
            dropout=float(cfg["model"].get("dropout", 0.1)),
            pretrained=True,
            use_text="text" in modalities,
        ).to(device)
    else:
        model = constructors[args.model](train_ds.s2_output_channels, pretrained=True).to(device)
    counts = np.array(
        [train_ds.frame[f"leaf_pixels_class_{i}"].sum() for i in range(14)], dtype=float
    )
    weights = torch.as_tensor(_balanced_weights(counts, 0.5, 6.0), device=device)
    dice_weight = float(args.dice_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    run_name = args.run_name or f"comparison_{args.model}"
    output = Path(cfg["paths"]["output_dir"]) / run_name
    output.mkdir(parents=True, exist_ok=True)
    best = -1.0
    stale = 0
    history = []
    best_path = output / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, train_loader, device, weights, dice_weight, optimizer,
            multimodal=use_fusion_head,
        )
        val_metrics = run_epoch(
            model, val_loader, device, weights, dice_weight,
            multimodal=use_fusion_head,
        )
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        score = float(val_metrics["official10"]["macro_iou_present"])
        print(
            f"model={args.model} epoch={epoch:02d} val_OA="
            f"{val_metrics['official10']['overall_accuracy']:.4f} val_mIoU={score:.4f}",
            flush=True,
        )
        if score > best:
            best = score
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "multimodal": use_fusion_head,
                    "modalities": sorted(modalities),
                    "model": args.model,
                    "training": {
                        "epochs": args.epochs,
                        "patience": args.patience,
                        "learning_rate": args.learning_rate,
                        "batch_size": args.batch_size,
                        "dice_weight": dice_weight,
                        "optimizer": "AdamW",
                        "weight_decay": 1e-4,
                        "seed": seed,
                    },
                },
                best_path,
            )
        else:
            stale += 1
        if stale >= args.patience:
            print(f"early_stop best_val_mIoU={best:.4f}", flush=True)
            break
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    validation = run_epoch(
        model, val_loader, device, weights, dice_weight, multimodal=use_fusion_head
    )
    test = run_epoch(
        model, test_loader, device, weights, dice_weight, multimodal=use_fusion_head
    )
    result = {
        "model": args.model,
        "multimodal": use_fusion_head,
        "modalities": sorted(modalities),
        "best_epoch": checkpoint["epoch"],
        "validation": validation,
        "test": test,
    }
    (output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "best_epoch": result["best_epoch"],
        "validation_OA": validation["official10"]["overall_accuracy"],
        "validation_mIoU": validation["official10"]["macro_iou_present"],
        "test_OA": test["official10"]["overall_accuracy"],
        "test_mIoU": test["official10"]["macro_iou_present"],
    }, indent=2))


if __name__ == "__main__":
    main()
