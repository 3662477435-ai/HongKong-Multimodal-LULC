from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.enums import Resampling
from rasterio.windows import Window
from torch.utils.data import DataLoader

from .comparison_models import MultiModalEarlyFusionComparator
from .config import load_config
from .hierarchical_fcnn_model import HierarchicalMultiModalFCNN
from .predict_experiment_models import build_context_manifest, make_inference_dataset
from .task8 import (
    CLASS_NAMES_8,
    EXCLUDED_LEAF_IDS,
    LEAF_TO_OFFICIAL8,
    RAW_GROUPS_8,
)


MODEL_KEYS = (
    "ours", "cnn", "fcnn", "transformer", "fcn_resnet50", "vgg_unet_transformer"
)
COLORS = {
    0: (0, 0, 0, 0),
    1: (78, 121, 167, 255),
    2: (225, 87, 89, 255),
    3: (118, 183, 178, 255),
    4: (89, 161, 79, 255),
    5: (237, 201, 72, 255),
    6: (176, 122, 161, 255),
    7: (255, 157, 167, 255),
    8: (41, 128, 185, 255),
}


def make_ours(cfg: dict, dataset, checkpoint: Path) -> torch.nn.Module:
    prepared = Path(cfg["paths"]["prepared_dir"])
    vocab = json.loads((prepared / "vocab_hier_v2.json").read_text(encoding="utf-8"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    modalities = set(state.get("modalities", (
        "s2", "s1", "highres", "nighttime", "building",
        "roads", "poi", "aoi", "text",
    )))
    model = HierarchicalMultiModalFCNN(
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
        use_text="text" in modalities,
    )
    model.load_state_dict(state["model_state"])
    return model


def make_comparator(cfg: dict, dataset, backbone: str, checkpoint: Path) -> torch.nn.Module:
    prepared = Path(cfg["paths"]["prepared_dir"])
    vocab = json.loads((prepared / "vocab_hier_v2.json").read_text(encoding="utf-8"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    modalities = set(state["modalities"])
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
        pretrained=False,
        use_text="text" in modalities,
        num_classes=8,
    )
    model.load_state_dict(state["model_state"])
    return model


def leaf_probabilities_to_official8(probabilities: torch.Tensor) -> torch.Tensor:
    result = probabilities.new_zeros(
        (probabilities.shape[0], 8, *probabilities.shape[-2:])
    )
    for leaf_id, official_id in enumerate(LEAF_TO_OFFICIAL8):
        if official_id >= 0:
            result[:, int(official_id)] += probabilities[:, leaf_id]
    return result


def write_prediction(
    name: str,
    model: torch.nn.Module,
    hierarchical: bool,
    loader: DataLoader,
    dataset,
    reference_path: str,
    output_dir: Path,
    device: torch.device,
) -> dict[str, str]:
    class_path = output_dir / f"{name}_official8_classification.tif"
    confidence_path = output_dir / f"{name}_official8_confidence.tif"
    with rasterio.open(reference_path) as reference:
        profile = reference.profile.copy()
    profile.update(
        count=1, dtype="uint8", nodata=0, compress="deflate", predictor=2,
        tiled=True, blockxsize=256, blockysize=256,
    )
    confidence_profile = profile.copy()
    confidence_profile.update(dtype="float32", nodata=0.0, predictor=3)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = dataset.frame
    cursor = 0
    model = model.to(device).eval()
    with rasterio.open(class_path, "w", **profile) as class_dst, rasterio.open(
        confidence_path, "w", **confidence_profile
    ) as confidence_dst:
        # Set palette metadata before the first block write. GeoTIFF does not
        # allow PhotometricInterpretation to be changed after pixel data has
        # started, and deferred GDAL errors can otherwise surface on the next
        # model's input read.
        class_dst.write_colormap(1, COLORS)
        class_dst.update_tags(
            class_schema="1-based eight active classes; 0=excluded/BLU0/nodata",
            excluded_official_classes="commercial_office,barren",
            **{f"class_{index + 1}": name for index, name in enumerate(CLASS_NAMES_8)},
            raw_groups=json.dumps(RAW_GROUPS_8, ensure_ascii=False),
        )
        confidence_dst.update_tags(
            definition="maximum probability among the eight active classes",
            excluded_official_classes="commercial_office,barren",
        )
        for _, block in class_dst.block_windows(1):
            shape = (int(block.height), int(block.width))
            class_dst.write(np.zeros(shape, dtype=np.uint8), 1, window=block)
            confidence_dst.write(np.zeros(shape, dtype=np.float32), 1, window=block)
        with torch.inference_mode():
            for batch in loader:
                s2 = batch["s2"].to(device, non_blocking=True)
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda",
                    dtype=torch.bfloat16,
                ):
                    if hierarchical:
                        outputs = model(
                            s2,
                            batch["s1"].to(device, non_blocking=True),
                            batch["urban"].to(device, non_blocking=True),
                            batch["roads"].to(device, non_blocking=True),
                            batch["osm"].to(device, non_blocking=True),
                            batch["input_ids"].to(device, non_blocking=True),
                            batch["text_mask"].to(device, non_blocking=True),
                        )
                        probabilities = outputs["combined_log_probs"].exp()
                    else:
                        probabilities = model(
                            s2,
                            batch["s1"].to(device, non_blocking=True),
                            batch["urban"].to(device, non_blocking=True),
                            batch["roads"].to(device, non_blocking=True),
                            batch["osm"].to(device, non_blocking=True),
                            batch["input_ids"].to(device, non_blocking=True),
                            batch["text_mask"].to(device, non_blocking=True),
                        ).softmax(dim=1)
                confidence, prediction = probabilities.max(dim=1)
                prediction = (prediction + 1).cpu().numpy().astype(np.uint8)
                confidence = confidence.float().cpu().numpy()
                for index in range(prediction.shape[0]):
                    row = frame.iloc[cursor]
                    height, width = int(row.height), int(row.width)
                    target_row = int(getattr(row, "target_row_off", row.row_off))
                    target_col = int(getattr(row, "target_col_off", row.col_off))
                    crop_row = int(getattr(row, "crop_row_off", 0))
                    crop_col = int(getattr(row, "crop_col_off", 0))
                    window = Window(target_col, target_row, width, height)
                    label = batch["label"][
                        index,
                        crop_row:crop_row + height,
                        crop_col:crop_col + width,
                    ].numpy()
                    valid = (label != 255) & (label != 1) & (label != 12)
                    pred = prediction[
                        index,
                        crop_row:crop_row + height,
                        crop_col:crop_col + width,
                    ]
                    conf = confidence[
                        index,
                        crop_row:crop_row + height,
                        crop_col:crop_col + width,
                    ]
                    class_dst.write(np.where(valid, pred, 0), 1, window=window)
                    confidence_dst.write(
                        np.where(valid, conf, 0.0), 1, window=window
                    )
                    cursor += 1
        class_dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
        confidence_dst.build_overviews([2, 4, 8, 16], Resampling.average)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"classification": str(class_path), "confidence": str(confidence_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_hierarchical.yaml")
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--context-margin", type=int, default=16)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = build_context_manifest(cfg, args.context_margin)
    dataset = make_inference_dataset(cfg, manifest)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    outputs = {}
    for model_key in MODEL_KEYS:
        run = args.experiment_root / f"{model_key}_full_seed{args.seed}"
        if model_key == "ours":
            run = args.experiment_root / (
                "ours_fcnn_hierarchy_semantic_seed42"
                if args.seed == 42 else f"ours_hfcnn_full_train_seed{args.seed}"
            )
            checkpoint = run / "best_final.pt"
            model = make_ours(cfg, dataset, checkpoint)
        else:
            checkpoint = run / "best_model.pt"
            model = make_comparator(cfg, dataset, model_key, checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        print(f"predicting={model_key}", flush=True)
        outputs[model_key] = write_prediction(
            model_key, model, model_key == "ours", loader, dataset,
            cfg["paths"]["image"], args.output_dir, device,
        )
    (args.output_dir / "prediction_manifest.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
