"""Evaluate direct, conditional-hierarchy and fused outputs of the final model.

This is evaluation only: all three outputs are obtained from the same frozen
checkpoint and the original 128 x 128 test patches.  It therefore separates
the probabilistic roles of the two paths without creating a new training run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hk_landuse.config import load_config
from hk_landuse.hierarchical_fcnn_model import HierarchicalMultiModalFCNN
from hk_landuse.metrics import ConfusionMetrics
from hk_landuse.task8 import CLASS_NAMES_8, leaf_target_to_official8
from hk_landuse.train_comparison import make_dataset


ALL_MODALITIES = {"s2", "s1", "highres", "nighttime", "building", "roads", "poi", "aoi", "text"}


def calibration(log_probs: torch.Tensor, target: torch.Tensor, bins: int = 15) -> tuple[float, float, int]:
    valid = target != 255
    probabilities = log_probs.exp().permute(0, 2, 3, 1)[valid].float()
    labels = target[valid].long()
    if labels.numel() == 0:
        return float("nan"), float("nan"), 0
    onehot = torch.nn.functional.one_hot(labels, probabilities.shape[1]).float()
    brier_sum = ((probabilities - onehot) ** 2).sum(dim=1).mean()
    confidence, predicted = probabilities.max(dim=1)
    correct = predicted.eq(labels).float()
    ece = confidence.new_zeros(())
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        selected = (confidence >= lower) & ((confidence < upper) if index < bins - 1 else (confidence <= upper))
        if selected.any():
            ece += selected.float().mean() * (confidence[selected].mean() - correct[selected].mean()).abs()
    return float(brier_sum.cpu()), float(ece.cpu()), int(labels.numel())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_release.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/full/eight_class_20260725/ours_fcnn_hierarchy_semantic_seed42/best_final.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/full/eight_class_review_revision_20260801/final_results/hierarchy_probability_audit"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["data"]["manifest_name"] = "manifest_hier_v2.csv"
    cfg["data"]["vocab_name"] = "vocab_hier_v2.json"
    cfg["data"]["normalization_name"] = "normalization_hier_v2.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modalities = ALL_MODALITIES
    dataset = make_dataset(cfg, "test", False, modalities)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    prepared = Path(cfg["paths"]["prepared_dir"])
    vocab = json.loads((prepared / cfg["data"]["vocab_name"]).read_text(encoding="utf-8"))
    model = HierarchicalMultiModalFCNN(
        vocab_size=len(vocab), s2_channels=dataset.s2_output_channels,
        s1_channels=dataset.s1_output_channels, urban_channels=dataset.urban_channels,
        road_channels=dataset.road_channels, osm_channels=20,
        text_dim=int(cfg["model"]["text_dim"]), text_heads=int(cfg["model"]["text_heads"]),
        text_layers=int(cfg["model"]["text_layers"]), max_tokens=int(cfg["data"]["max_text_tokens"]),
        dropout=float(cfg["model"].get("dropout", 0.1)), use_text=True,
        use_semantic_residual=True, use_hierarchy=True, gate_mode="class",
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    metrics = {key: ConfusionMetrics(8, CLASS_NAMES_8) for key in ("direct", "hierarchy", "fused")}
    brier, ece, pixels = {key: [] for key in metrics}, {key: [] for key in metrics}, {key: 0 for key in metrics}
    with torch.no_grad():
        for batch in loader:
            s2 = batch["s2"].to(device)
            n, _, h, w = s2.shape
            target = leaf_target_to_official8(batch["label"].to(device))
            output = model(
                s2, batch["s1"].to(device), batch["urban"].to(device), batch["roads"].to(device),
                batch["osm"].to(device), batch["input_ids"].to(device), batch["text_mask"].to(device),
            )
            source = {
                "direct": output["direct_log_probs"],
                "hierarchy": output["hierarchy_log_probs"],
                "fused": output["combined_log_probs"],
            }
            for key, log_probs in source.items():
                metrics[key].update(log_probs, target)
                current_brier, current_ece, count = calibration(log_probs, target)
                brier[key].append((current_brier, count))
                ece[key].append((current_ece, count))
                pixels[key] += count
    output = {}
    for key, metric in metrics.items():
        result = metric.compute()
        count = max(pixels[key], 1)
        result["multiclass_brier"] = sum(value * n for value, n in brier[key]) / count
        result["ece_15bin"] = sum(value * n for value, n in ece[key]) / count
        output[key] = result
    gates = torch.sigmoid(model.hierarchy_gate_logit).detach().cpu().tolist()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_stage": checkpoint.get("stage"),
        "checkpoint_epoch": checkpoint.get("epoch"), "input_protocol": "original fixed 128x128 spatially blocked test patches",
        "outputs": output,
        "classwise_hierarchy_gate_alpha": dict(zip(CLASS_NAMES_8, gates)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
    }
    (args.output_dir / "hierarchy_probability_paths.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = []
    for key, value in output.items():
        rows.append({"output": key, "OA": value["overall_accuracy"], "macro_mIoU": value["macro_iou_present"], "macro_F1": value["macro_f1_present"], "Brier": value["multiclass_brier"], "ECE_15bin": value["ece_15bin"]})
    import csv
    with (args.output_dir / "hierarchy_probability_paths.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
