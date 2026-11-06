from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.nn import functional as F

from .config import load_config
from .dataset import LandUsePatchDataset
from .hierarchical_loss import HierarchicalLoss
from .hierarchical_model import HierarchicalSwinFusion
from .labels import (
    CLASS_NAMES,
    COARSE_CLASS_NAMES,
    IGNORE_INDEX,
    LEAF_CLASS_NAMES,
    LEAF_TO_COARSE,
    LEAF_TO_OFFICIAL,
)
from .metrics import ConfusionMetrics


STAGES = ("coarse", "natural", "built", "joint")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(cfg: dict, split: str, augment: bool) -> LandUsePatchDataset:
    prepared = Path(cfg["paths"]["prepared_dir"])
    modalities = set(cfg["model"].get("modalities", ["s2", "s1", "poi", "aoi", "text"]))
    use_osm = bool({"poi", "aoi"} & modalities)
    return LandUsePatchDataset(
        image_path=cfg["paths"]["image"],
        label_path=prepared / "labels_leaf14.tif",
        manifest_path=prepared / cfg["data"].get("manifest_name", "manifest_light.csv"),
        vocab_path=prepared / cfg["data"].get("vocab_name", "vocab_light.json"),
        normalization_path=prepared / cfg["data"].get("normalization_name", "normalization_light.json"),
        patch_size=int(cfg["data"]["patch_size"]),
        max_text_tokens=int(cfg["data"]["max_text_tokens"]),
        osm_poi_path=cfg["paths"].get("osm_poi_raster") if use_osm else None,
        osm_aoi_path=cfg["paths"].get("osm_aoi_raster") if use_osm else None,
        highres_path=cfg["paths"].get("highres_aligned") if "highres" in modalities else None,
        nighttime_path=cfg["paths"].get("nighttime_aligned") if "nighttime" in modalities else None,
        building_path=cfg["paths"].get("building_aligned") if "building" in modalities else None,
        road_features_path=cfg["paths"].get("roads_aligned") if "roads" in modalities else None,
        split=split,
        augment=augment,
    )


def _balanced_weights(counts: np.ndarray, power: float, max_weight: float) -> np.ndarray:
    weights = np.zeros(len(counts), dtype=np.float32)
    present = counts > 0
    weights[present] = (counts[present].sum() / (present.sum() * counts[present])) ** power
    weights[present] = np.clip(weights[present], 0.25, max_weight)
    return weights


def _patch_sampler(dataset: LandUsePatchDataset, seed: int) -> WeightedRandomSampler:
    frame = dataset.frame
    columns = [f"leaf_pixels_class_{i}" for i in range(14)]
    support = np.array([(frame[c] > 0).sum() for c in columns], dtype=float)
    rarity = np.zeros(len(frame), dtype=float)
    for i, column in enumerate(columns):
        rarity += (frame[column].to_numpy() > 0) / np.sqrt(max(support[i], 1.0))
    rarity += 0.05
    return WeightedRandomSampler(
        torch.as_tensor(rarity, dtype=torch.double),
        num_samples=len(frame),
        replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


class MetricBundle:
    def __init__(self, subsets: bool = False) -> None:
        self.coarse = ConfusionMetrics(4, COARSE_CLASS_NAMES)
        self.built = ConfusionMetrics(6, LEAF_CLASS_NAMES[:6])
        self.natural = ConfusionMetrics(8, LEAF_CLASS_NAMES[6:])
        self.leaf = ConfusionMetrics(14, LEAF_CLASS_NAMES)
        self.official = ConfusionMetrics(10, CLASS_NAMES)
        self.subsets = {
            name: ConfusionMetrics(10, CLASS_NAMES)
            for name in (
                "AOI覆盖区",
                "无AOI区",
                "POI_250m邻域",
                "POI稀疏区",
                "真实建成区",
                "真实非建成区",
            )
        } if subsets else {}
        self.coarse_lut = torch.as_tensor(LEAF_TO_COARSE, dtype=torch.long)
        self.official_lut = torch.as_tensor(LEAF_TO_OFFICIAL, dtype=torch.long)

    def update(self, outputs: dict[str, torch.Tensor], target: torch.Tensor, osm: torch.Tensor | None) -> None:
        valid = target != IGNORE_INDEX
        safe = target.masked_fill(~valid, 0)
        coarse_target = self.coarse_lut.to(target.device)[safe].masked_fill(~valid, IGNORE_INDEX)
        self.coarse.update(outputs["coarse_logits"], coarse_target)
        self.built.update(outputs["built_logits"], safe, mask=valid & (safe < 6))
        self.natural.update(outputs["natural_logits"], safe - 6, mask=valid & (safe >= 6))
        self.leaf.update(outputs["leaf_log_probs"], target)

        leaf_pred = outputs["leaf_log_probs"].argmax(dim=1)
        lut = self.official_lut.to(target.device)
        official_pred = lut[leaf_pred]
        official_target = lut[safe].masked_fill(~valid, IGNORE_INDEX)
        self.official.update(official_pred, official_target)
        if not self.subsets or osm is None:
            return
        aoi_covered = osm[:, 14:20].sum(dim=1) > 0
        # Presence of at least one rasterised POI within a 250 m neighbourhood.
        poi_context = F.avg_pool2d(osm[:, :14].sum(dim=1, keepdim=True), 25, 1, 12)[:, 0] > 0
        masks = {
            "AOI覆盖区": aoi_covered,
            "无AOI区": ~aoi_covered,
            "POI_250m邻域": poi_context,
            "POI稀疏区": ~poi_context,
            "真实建成区": valid & (safe < 6),
            "真实非建成区": valid & (safe >= 6),
        }
        for name, mask in masks.items():
            self.subsets[name].update(official_pred, official_target, mask=mask)

    def compute(self) -> dict:
        result = {
            "coarse4": self.coarse.compute(),
            "built6_conditional": self.built.compute(),
            "natural8_conditional": self.natural.compute(),
            "leaf14_soft_route": self.leaf.compute(),
            "official10_projection": self.official.compute(),
        }
        if self.subsets:
            result["condition_subsets_official10"] = {
                name: metric.compute() for name, metric in self.subsets.items()
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
    subsets: bool = False,
) -> tuple[float, dict]:
    training = optimizer is not None
    model.train(training)
    metrics = MetricBundle(subsets=subsets)
    total_loss = 0.0
    parts_sum: dict[str, float] = {}
    batches = 0
    skipped_steps = 0
    for batch in loader:
        s1 = batch["s1"].to(device, non_blocking=True)
        s2 = batch["s2"].to(device, non_blocking=True)
        ids = batch["input_ids"].to(device, non_blocking=True)
        text_mask = batch["text_mask"].to(device, non_blocking=True)
        target = batch["label"].to(device, non_blocking=True)
        osm = batch.get("osm")
        if osm is not None:
            osm = osm.to(device, non_blocking=True)
        urban = batch.get("urban")
        if urban is not None:
            urban = urban.to(device, non_blocking=True)
        roads = batch.get("roads")
        if roads is not None:
            roads = roads.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.amp.autocast(
            device_type=device.type, enabled=use_amp, dtype=amp_dtype
        ):
            outputs = model(s1, s2, ids, text_mask, osm=osm, urban=urban, roads=roads)
            loss, parts = criterion(outputs, target, stage=stage, osm=osm)
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            gradients_finite = bool(torch.isfinite(loss)) and all(
                bool(torch.isfinite(grad).all()) for grad in gradients
            )
            if gradients_finite:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
            else:
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None and scaler.is_enabled():
                    scaler.update()
        total_loss += float(loss.detach())
        for key, value in parts.items():
            parts_sum[key] = parts_sum.get(key, 0.0) + value
        batches += 1
        metrics.update(outputs, target, osm)
    result = metrics.compute()
    result["loss"] = total_loss / max(batches, 1)
    result["loss_components"] = {key: value / max(batches, 1) for key, value in parts_sum.items()}
    result["skipped_nonfinite_steps"] = skipped_steps
    return result["loss"], result


def set_stage_trainability(model: HierarchicalSwinFusion, stage: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True
    if stage in ("coarse", "natural"):
        prefixes = ("osm_", "text_", "road_", "built_", "aoi_prior")
        for name, parameter in model.named_parameters():
            if name.startswith(prefixes):
                parameter.requires_grad = False
    if stage == "coarse":
        for parameter in model.natural_head.parameters():
            parameter.requires_grad = False
    if stage == "built":
        for parameter in model.natural_head.parameters():
            parameter.requires_grad = False


def selection_score(metrics: dict, stage: str) -> float:
    key = {
        "coarse": "coarse4",
        "natural": "natural8_conditional",
        "built": "built6_conditional",
        "joint": "official10_projection",
    }[stage]
    return float(metrics[key]["macro_iou_present"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential soft-routed hierarchical training.")
    parser.add_argument("--config", default="config_hierarchical.yaml")
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--run-name", default="hierarchical_light")
    parser.add_argument("--stage-epochs", default=None, help="Override, e.g. 2,2,3,5")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--only-joint", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--class-weight-power", type=float, default=None)
    parser.add_argument("--max-class-weight", type=float, default=None)
    parser.add_argument("--manifest-name", default=None)
    parser.add_argument("--vocab-name", default=None)
    parser.add_argument("--normalization-name", default=None)
    parser.add_argument(
        "--modalities",
        default=None,
        help="Comma-separated model inputs, e.g. s2 or s2,s1,highres,nighttime,building,roads,poi,aoi,text",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.manifest_name:
        cfg["data"]["manifest_name"] = args.manifest_name
    if args.vocab_name:
        cfg["data"]["vocab_name"] = args.vocab_name
    if args.normalization_name:
        cfg["data"]["normalization_name"] = args.normalization_name
    if args.modalities:
        cfg["model"]["modalities"] = [
            value.strip() for value in args.modalities.split(",") if value.strip()
        ]
    seed = int(cfg["train"]["seed"])
    seed_everything(seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if str(cfg["train"].get("amp_dtype", "bfloat16")).lower() == "bfloat16"
        else torch.float16
    )

    train_ds = make_dataset(cfg, "train", True)
    val_ds = make_dataset(cfg, "val", False)
    test_ds = make_dataset(cfg, "test", False)
    loader_args = {
        "batch_size": int(cfg["train"]["batch_size"]),
        "num_workers": int(cfg["train"].get("num_workers", 0)),
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, sampler=_patch_sampler(train_ds, seed), **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)

    prepared = Path(cfg["paths"]["prepared_dir"])
    vocab = json.loads((prepared / cfg["data"]["vocab_name"]).read_text(encoding="utf-8"))
    modalities = set(cfg["model"].get("modalities", ["s2", "s1", "poi", "aoi", "text"]))
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
        separate_optical_indices=bool(cfg["model"].get("separate_optical_indices", False)),
    ).to(device)
    if args.resume_checkpoint:
        resume = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        resume_state = resume["model_state"]
        current_state = model.state_dict()
        for key, value in list(resume_state.items()):
            if key not in current_state or value.shape == current_state[key].shape:
                continue
            # A full-data vocabulary can contain additional training-only OSM
            # tokens. Preserve all learned rows and leave new rows initialized.
            if value.ndim == current_state[key].ndim:
                expanded = current_state[key].clone()
                overlap = tuple(slice(0, min(a, b)) for a, b in zip(value.shape, expanded.shape))
                expanded[overlap] = value[overlap]
                resume_state[key] = expanded
            else:
                raise ValueError(
                    f"Cannot resume parameter {key}: {tuple(value.shape)} -> "
                    f"{tuple(current_state[key].shape)}"
                )
        incompatible = model.load_state_dict(resume_state, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(
                f"resume_missing={len(incompatible.missing_keys)} "
                f"resume_unexpected={len(incompatible.unexpected_keys)}",
                flush=True,
            )

    manifest = train_ds.frame
    leaf_counts = np.array([manifest[f"leaf_pixels_class_{i}"].sum() for i in range(14)], dtype=float)
    coarse_counts = np.zeros(4, dtype=float)
    for leaf_id, count in enumerate(leaf_counts):
        coarse_counts[LEAF_TO_COARSE[leaf_id]] += count
    power = float(
        args.class_weight_power
        if args.class_weight_power is not None
        else cfg["train"].get("class_weight_power", 0.5)
    )
    maximum = float(
        args.max_class_weight
        if args.max_class_weight is not None
        else cfg["train"].get("max_class_weight", 6.0)
    )
    criterion = HierarchicalLoss(
        torch.as_tensor(_balanced_weights(leaf_counts, power, maximum), device=device),
        torch.as_tensor(_balanced_weights(coarse_counts, power, maximum), device=device),
        dice_weight=float(cfg["train"].get("dice_weight", 0.3)),
        coarse_weight=float(cfg["train"].get("coarse_weight", 0.5)),
        weak_weight=float(cfg["train"].get("weak_weight", 0.05)),
        conflict_weight=float(cfg["train"].get("aoi_conflict_weight", 0.3)),
        focal_gamma=float(
            args.focal_gamma
            if args.focal_gamma is not None
            else cfg["train"].get("focal_gamma", 0.0)
        ),
    )
    stage_epochs = cfg["train"].get("stage_epochs", [2, 2, 3, 5])
    if args.stage_epochs:
        stage_epochs = [int(x) for x in args.stage_epochs.split(",")]
    if len(stage_epochs) != 4:
        raise ValueError("stage_epochs must contain four integers")
    active_stages = STAGES
    if args.only_joint:
        active_stages = ("joint",)
        stage_epochs = [stage_epochs[-1]]

    output_dir = Path(cfg["paths"]["output_dir"]) / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict] = {}
    history: list[dict] = []
    for stage, epochs in zip(active_stages, stage_epochs):
        set_stage_trainability(model, stage)
        trainable = [p for p in model.parameters() if p.requires_grad]
        effective_learning_rate = float(args.learning_rate or cfg["train"]["learning_rate"])
        optimizer = torch.optim.AdamW(
            trainable,
            lr=effective_learning_rate,
            weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
        best_score = -1.0
        stale_epochs = 0
        best_path = output_dir / f"best_{stage}.pt"
        for epoch in range(1, epochs + 1):
            _, train_metrics = run_epoch(
                model, train_loader, criterion, stage, device, optimizer, scaler, use_amp, amp_dtype
            )
            _, val_metrics = run_epoch(
                model, val_loader, criterion, stage, device, use_amp=use_amp, amp_dtype=amp_dtype
            )
            scheduler.step()
            score = selection_score(val_metrics, stage)
            history.append({"stage": stage, "epoch": epoch, "train": train_metrics, "val": val_metrics})
            key = {"coarse": "coarse4", "natural": "natural8_conditional", "built": "built6_conditional", "joint": "official10_projection"}[stage]
            print(
                f"stage={stage:7s} epoch={epoch:02d} loss={val_metrics['loss']:.4f} "
                f"val_OA={val_metrics[key]['overall_accuracy']:.4f} "
                f"val_mIoU={val_metrics[key]['macro_iou_present']:.4f}",
                flush=True,
            )
            if score > best_score:
                best_score = score
                stale_epochs = 0
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "stage": stage,
                        "epoch": epoch,
                        "config": cfg,
                        "training": {
                            "stage_epochs": stage_epochs,
                            "learning_rate": effective_learning_rate,
                            "patience": args.patience,
                            "focal_gamma": criterion.focal_gamma,
                            "class_weight_power": power,
                            "max_class_weight": maximum,
                        },
                    },
                    best_path,
                )
            else:
                stale_epochs += 1
            if args.patience > 0 and stale_epochs >= args.patience:
                print(
                    f"stage={stage:7s} early_stop epoch={epoch:02d} "
                    f"best_val_mIoU={best_score:.4f}",
                    flush=True,
                )
                break
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        _, val_metrics = run_epoch(
            model, val_loader, criterion, stage, device, use_amp=use_amp, amp_dtype=amp_dtype, subsets=True
        )
        _, test_metrics = run_epoch(
            model, test_loader, criterion, stage, device, use_amp=use_amp, amp_dtype=amp_dtype, subsets=True
        )
        all_results[stage] = {
            "best_epoch": checkpoint["epoch"],
            "validation": val_metrics,
            "test": test_metrics,
        }
        (output_dir / f"metrics_{stage}.json").write_text(
            json.dumps(all_results[stage], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (output_dir / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "stage_metrics.json").write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({stage: {"best_epoch": value["best_epoch"], "test": {
        "coarse4": value["test"]["coarse4"]["macro_iou_present"],
        "built6": value["test"]["built6_conditional"]["macro_iou_present"],
        "natural8": value["test"]["natural8_conditional"]["macro_iou_present"],
        "official10": value["test"]["official10_projection"]["macro_iou_present"],
    }} for stage, value in all_results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
