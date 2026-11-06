from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .config import load_config
from .advanced_model import SwinUPerNetFusion
from .dataset import LandUsePatchDataset
from .losses import CrossEntropyDiceLoss
from .metrics import ConfusionMetrics
from .model import MultiModalUNet


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(cfg: dict, split: str, augment: bool) -> LandUsePatchDataset:
    prepared = Path(cfg["paths"]["prepared_dir"])
    return LandUsePatchDataset(
        image_path=cfg["paths"]["image"],
        label_path=prepared / "labels_10class.tif",
        manifest_path=prepared / cfg["data"].get("manifest_name", "manifest.csv"),
        vocab_path=prepared / cfg["data"].get("vocab_name", "vocab.json"),
        normalization_path=prepared / cfg["data"].get("normalization_name", "normalization.json"),
        patch_size=int(cfg["data"]["patch_size"]),
        max_text_tokens=int(cfg["data"]["max_text_tokens"]),
        osm_poi_path=cfg["paths"].get("osm_poi_raster"),
        osm_aoi_path=cfg["paths"].get("osm_aoi_raster"),
        split=split,
        augment=augment,
    )


def build_model(cfg: dict, vocab_size: int) -> MultiModalUNet:
    model_cfg = cfg["model"]
    if model_cfg.get("architecture", "multimodal_unet") == "swin_upernet":
        return SwinUPerNetFusion(
            vocab_size=vocab_size,
            num_classes=int(model_cfg["num_classes"]),
            text_dim=int(model_cfg["text_dim"]),
            text_heads=int(model_cfg["text_heads"]),
            text_layers=int(model_cfg["text_layers"]),
            max_tokens=int(cfg["data"]["max_text_tokens"]),
            fpn_channels=int(model_cfg.get("fpn_channels", 128)),
            osm_channels=int(model_cfg.get("osm_channels", 20)),
            dropout=float(model_cfg["dropout"]),
            pretrained=bool(model_cfg.get("pretrained", True)),
        )
    return MultiModalUNet(
        vocab_size=vocab_size,
        num_classes=int(model_cfg["num_classes"]),
        base=int(model_cfg["base_channels"]),
        text_dim=int(model_cfg["text_dim"]),
        text_heads=int(model_cfg["text_heads"]),
        text_layers=int(model_cfg["text_layers"]),
        max_tokens=int(cfg["data"]["max_text_tokens"]),
        dropout=float(model_cfg["dropout"]),
    )


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None, use_amp=False) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)
    metrics = ConfusionMetrics(num_classes=10)
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        s1 = batch["s1"].to(device, non_blocking=True)
        s2 = batch["s2"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        text_mask = batch["text_mask"].to(device, non_blocking=True)
        osm = batch.get("osm")
        if osm is not None:
            osm = osm.to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(s1, s2, input_ids, text_mask, osm=osm)
            loss = criterion(logits, target)
        if training:
            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach())
        total_batches += 1
        metrics.update(logits, target)
    result = metrics.compute()
    result["loss"] = total_loss / max(total_batches, 1)
    return result["loss"], result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multimodal Sentinel + OSM segmentation model.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override config; useful for smoke tests.")
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--text-dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--dice-weight", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--class-weight-power", type=float, default=None)
    parser.add_argument("--max-class-weight", type=float, default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.base_channels is not None:
        cfg["model"]["base_channels"] = args.base_channels
    if args.text_dim is not None:
        cfg["model"]["text_dim"] = args.text_dim
    if args.dropout is not None:
        cfg["model"]["dropout"] = args.dropout
    if args.learning_rate is not None:
        cfg["train"]["learning_rate"] = args.learning_rate
    if args.dice_weight is not None:
        cfg["train"]["dice_weight"] = args.dice_weight
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size
    if args.class_weight_power is not None:
        cfg["train"]["class_weight_power"] = args.class_weight_power
    if args.max_class_weight is not None:
        cfg["train"]["max_class_weight"] = args.max_class_weight
    seed = int(cfg["train"]["seed"])
    seed_everything(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(cfg["train"]["amp"]) and device.type == "cuda"

    train_ds = make_dataset(cfg, "train", augment=True)
    val_ds = make_dataset(cfg, "val", augment=False)
    test_ds = make_dataset(cfg, "test", augment=False)
    loader_args = {
        "batch_size": int(cfg["train"]["batch_size"]),
        "num_workers": int(cfg["train"]["num_workers"]),
        "pin_memory": device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    prepared = Path(cfg["paths"]["prepared_dir"])
    vocab_path = prepared / cfg["data"].get("vocab_name", "vocab.json")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    model = build_model(cfg, len(vocab)).to(device)
    manifest = pd.read_csv(prepared / cfg["data"].get("manifest_name", "manifest.csv"))
    counts = np.array(
        [manifest.loc[manifest.split == "train", f"pixels_class_{i}"].sum() for i in range(10)],
        dtype=float,
    )
    weights = np.zeros(10, dtype=np.float32)
    present = counts > 0
    power = float(cfg["train"].get("class_weight_power", 0.5))
    max_weight = float(cfg["train"].get("max_class_weight", 4.0))
    weights[present] = (counts[present].sum() / (present.sum() * counts[present])) ** power
    weights[present] = np.clip(weights[present], 0.25, max_weight)
    criterion = CrossEntropyDiceLoss(
        torch.tensor(weights, device=device),
        dice_weight=float(cfg["train"]["dice_weight"]),
        hierarchy_weight=float(cfg["train"].get("hierarchy_weight", 0.15)),
    )
    base_lr = float(cfg["train"]["learning_rate"])
    if hasattr(model, "backbone"):
        backbone_params = list(model.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
        param_groups = [
            {"params": backbone_params, "lr": base_lr * float(cfg["train"].get("backbone_lr_multiplier", 0.1))},
            {"params": head_params, "lr": base_lr},
        ]
    else:
        param_groups = model.parameters()
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=float(cfg["train"]["weight_decay"]),
    )
    epochs = args.epochs or int(cfg["train"]["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = Path(cfg["paths"]["output_dir"])
    if args.run_name:
        output_dir = output_dir / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best_model.pt"
    history = []
    best_miou = -1.0
    stale = 0
    for epoch in range(1, epochs + 1):
        _, train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler, use_amp)
        _, val_metrics = run_epoch(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step()
        record = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": optimizer.param_groups[0]["lr"]}
        history.append(record)
        print(
            f"epoch={epoch:03d} train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_mIoU={val_metrics['macro_iou_present']:.4f}"
        )
        if val_metrics["macro_iou_present"] > best_miou:
            best_miou = val_metrics["macro_iou_present"]
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "vocab_size": len(vocab),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= int(cfg["train"]["patience"]):
                print(f"early_stopping epoch={epoch}")
                break

    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    _, test_metrics = run_epoch(model, test_loader, criterion, device, use_amp=use_amp)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"device": str(device), "best_epoch": checkpoint["epoch"], "test": test_metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
