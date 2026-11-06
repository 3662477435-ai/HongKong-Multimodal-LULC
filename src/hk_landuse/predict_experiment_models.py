from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from torch.utils.data import DataLoader

from .comparison_models import (
    FCNResNet50Baseline,
    MultiModalEarlyFusionComparator,
    VGGUNetTransformerBaseline,
)
from .config import load_config
from .dataset import LandUsePatchDataset
from .hierarchical_model import HierarchicalSwinFusion
from .labels import CLASS_NAMES, IGNORE_INDEX, LEAF_TO_OFFICIAL, RAW_GROUPS


COLORS = {
    0: (0, 0, 0, 0),
    1: (230, 76, 60, 255),
    2: (241, 196, 15, 255),
    3: (155, 89, 182, 255),
    4: (52, 152, 219, 255),
    5: (127, 140, 141, 255),
    6: (211, 84, 0, 255),
    7: (166, 206, 57, 255),
    8: (39, 174, 96, 255),
    9: (189, 195, 199, 255),
    10: (41, 128, 185, 255),
}


def build_context_manifest(cfg: dict, margin: int) -> Path:
    """Make valid-core tiles so predictions never use a patch's outer margin."""
    prepared = Path(cfg["paths"]["prepared_dir"])
    patch_size = int(cfg["data"]["patch_size"])
    core_size = patch_size - 2 * margin
    if core_size <= 0:
        raise ValueError("context margin must be smaller than half the patch size")
    source_manifest = pd.read_csv(prepared / "inference_manifest.csv")
    text_lookup = {
        (int(row.row_off), int(row.col_off)): str(row.text)
        for row in source_manifest.itertuples()
    }
    label_path = prepared / "labels_leaf14.tif"
    with rasterio.open(label_path) as label_src:
        height, width = label_src.height, label_src.width
        valid = label_src.read(1) != IGNORE_INDEX
    records = []
    for target_row in range(0, height, core_size):
        output_height = min(core_size, height - target_row)
        for target_col in range(0, width, core_size):
            output_width = min(core_size, width - target_col)
            if not valid[
                target_row : target_row + output_height,
                target_col : target_col + output_width,
            ].any():
                continue
            source_row = (target_row // patch_size) * patch_size
            source_col = (target_col // patch_size) * patch_size
            records.append(
                {
                    "patch_id": f"context_r{target_row}_c{target_col}",
                    "row_off": target_row - margin,
                    "col_off": target_col - margin,
                    "height": output_height,
                    "width": output_width,
                    "target_row_off": target_row,
                    "target_col_off": target_col,
                    "crop_row_off": margin,
                    "crop_col_off": margin,
                    "text": text_lookup.get((source_row, source_col), "no_osm_context"),
                }
            )
    path = prepared / f"inference_manifest_context{margin}_blu_masked.csv"
    pd.DataFrame.from_records(records).to_csv(path, index=False, encoding="utf-8")
    print(
        f"context_manifest={path} tiles={len(records)} patch={patch_size} "
        f"core={core_size} margin={margin}", flush=True
    )
    return path


def make_inference_dataset(cfg: dict, manifest_path: Path | None = None) -> LandUsePatchDataset:
    prepared = Path(cfg["paths"]["prepared_dir"])
    return LandUsePatchDataset(
        image_path=cfg["paths"]["image"],
        label_path=prepared / "labels_leaf14.tif",
        manifest_path=manifest_path or prepared / "inference_manifest.csv",
        vocab_path=prepared / "vocab_hier_v2.json",
        normalization_path=prepared / "normalization_hier_v2.json",
        patch_size=int(cfg["data"]["patch_size"]),
        max_text_tokens=int(cfg["data"]["max_text_tokens"]),
        osm_poi_path=cfg["paths"].get("osm_poi_raster"),
        osm_aoi_path=cfg["paths"].get("osm_aoi_raster"),
        highres_path=cfg["paths"].get("highres_aligned"),
        nighttime_path=cfg["paths"].get("nighttime_aligned"),
        building_path=cfg["paths"].get("building_aligned"),
        road_features_path=cfg["paths"].get("roads_aligned"),
        split=None,
        augment=False,
    )


def make_main_model(cfg: dict, dataset: LandUsePatchDataset, checkpoint: Path) -> torch.nn.Module:
    vocab = json.loads(
        (Path(cfg["paths"]["prepared_dir"]) / "vocab_hier_v2.json").read_text(encoding="utf-8")
    )
    modalities = set(cfg["model"]["modalities"])
    model = HierarchicalSwinFusion(
        vocab_size=len(vocab),
        s2_channels=dataset.s2_output_channels,
        s1_channels=dataset.s1_output_channels,
        text_dim=int(cfg["model"]["text_dim"]),
        text_heads=int(cfg["model"]["text_heads"]),
        text_layers=int(cfg["model"]["text_layers"]),
        max_tokens=int(cfg["data"]["max_text_tokens"]),
        fpn_channels=int(cfg["model"].get("fpn_channels", 128)),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        pretrained=False,
        use_s2="s2" in modalities,
        use_s1="s1" in modalities,
        use_poi="poi" in modalities,
        use_aoi="aoi" in modalities,
        use_text="text" in modalities,
        use_highres="highres" in modalities,
        use_nighttime="nighttime" in modalities,
        use_building="building" in modalities,
        use_roads="roads" in modalities,
        building_channels=dataset.building_channels,
        separate_optical_indices=bool(cfg["model"].get("separate_optical_indices", False)),
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
    model.load_state_dict(state)
    return model


def make_comparator(name: str, channels: int, checkpoint: Path) -> torch.nn.Module:
    cls = FCNResNet50Baseline if name == "fcn_resnet50" else VGGUNetTransformerBaseline
    model = cls(channels, pretrained=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
    model.load_state_dict(state)
    return model


def make_multimodal_comparator(
    cfg: dict, dataset: LandUsePatchDataset, name: str, checkpoint: Path
) -> torch.nn.Module:
    vocab = json.loads(
        (Path(cfg["paths"]["prepared_dir"]) / "vocab_hier_v2.json").read_text(
            encoding="utf-8"
        )
    )
    model = MultiModalEarlyFusionComparator(
        backbone=name,
        vocab_size=len(vocab),
        s2_channels=dataset.s2_output_channels,
        s1_channels=dataset.s1_output_channels,
        urban_channels=dataset.urban_channels,
        road_channels=dataset.road_channels,
        text_dim=int(cfg["model"]["text_dim"]),
        text_heads=int(cfg["model"]["text_heads"]),
        text_layers=int(cfg["model"]["text_layers"]),
        max_tokens=int(cfg["data"]["max_text_tokens"]),
        dropout=float(cfg["model"].get("dropout", 0.1)),
        pretrained=False,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state"]
    model.load_state_dict(state)
    return model


def official_probabilities(leaf_probabilities: torch.Tensor) -> torch.Tensor:
    result = leaf_probabilities.new_zeros(
        (leaf_probabilities.shape[0], 10, *leaf_probabilities.shape[-2:])
    )
    lut = torch.as_tensor(LEAF_TO_OFFICIAL, device=leaf_probabilities.device)
    index = lut[None, :, None, None].expand_as(leaf_probabilities)
    return result.scatter_add(1, index, leaf_probabilities)


def write_prediction(
    name: str,
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: LandUsePatchDataset,
    reference_path: str,
    output_dir: Path,
    device: torch.device,
    hierarchical: bool,
    multimodal: bool = False,
) -> dict[str, str]:
    class_path = output_dir / f"{name}_official10_classification.tif"
    confidence_path = output_dir / f"{name}_official10_confidence.tif"
    with rasterio.open(reference_path) as reference:
        class_profile = reference.profile.copy()
        class_profile.update(
            count=1, dtype="uint8", nodata=0, compress="deflate", predictor=2,
            tiled=True, blockxsize=256, blockysize=256,
        )
        confidence_profile = class_profile.copy()
        confidence_profile.update(dtype="float32", nodata=0.0, predictor=3)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device).eval()
    frame = dataset.frame
    cursor = 0
    with rasterio.open(class_path, "w", **class_profile) as class_dst, rasterio.open(
        confidence_path, "w", **confidence_profile
    ) as confidence_dst:
        # Some GeoTIFF readers do not guarantee the value of never-written
        # blocks.  Initialise them explicitly so skipped all-BLU-zero cores are
        # unambiguously NoData rather than driver-dependent garbage.
        for _, empty_window in class_dst.block_windows(1):
            shape = (int(empty_window.height), int(empty_window.width))
            class_dst.write(np.zeros(shape, dtype=np.uint8), 1, window=empty_window)
            confidence_dst.write(
                np.zeros(shape, dtype=np.float32), 1, window=empty_window
            )
        class_dst.write_colormap(1, COLORS)
        class_dst.update_tags(
            class_schema="1-based official 10 classes; 0=BLU raw 0/unmapped/nodata",
            spatial_mask="labels_leaf14 != 255 (equivalent to supported nonzero BLU classes)",
            inference_tiling="128x128 input with context-center output; see prediction manifest",
            **{f"class_{i + 1}": value for i, value in enumerate(CLASS_NAMES)},
            raw_groups=json.dumps({i + 1: RAW_GROUPS[i] for i in range(10)}, ensure_ascii=False),
        )
        confidence_dst.update_tags(
            definition="maximum 14-leaf probability; selected leaf is projected to official10",
            range="0..1",
            spatial_mask="0 outside supported nonzero BLU classes",
        )
        with torch.inference_mode():
            for batch in loader:
                s2 = batch["s2"].to(device, non_blocking=True)
                with torch.amp.autocast(
                    device_type=device.type, enabled=device.type == "cuda", dtype=torch.bfloat16
                ):
                    if hierarchical:
                        outputs = model(
                            batch["s1"].to(device, non_blocking=True),
                            s2,
                            batch["input_ids"].to(device, non_blocking=True),
                            batch["text_mask"].to(device, non_blocking=True),
                            osm=batch["osm"].to(device, non_blocking=True),
                            urban=batch["urban"].to(device, non_blocking=True),
                            roads=batch["roads"].to(device, non_blocking=True),
                        )
                        leaf_probs = outputs["leaf_log_probs"].exp()
                    elif multimodal:
                        leaf_probs = model(
                            s2,
                            batch["s1"].to(device, non_blocking=True),
                            batch["urban"].to(device, non_blocking=True),
                            batch["roads"].to(device, non_blocking=True),
                            batch["osm"].to(device, non_blocking=True),
                            batch["input_ids"].to(device, non_blocking=True),
                            batch["text_mask"].to(device, non_blocking=True),
                        ).softmax(dim=1)
                    else:
                        leaf_probs = model(s2).softmax(dim=1)
                # Exactly match the validation protocol used to select every
                # checkpoint: leaf argmax, followed by the fixed ten-class map.
                confidence, leaf_prediction = leaf_probs.max(dim=1)
                lut = torch.as_tensor(LEAF_TO_OFFICIAL, device=device)
                prediction = (lut[leaf_prediction] + 1).cpu().numpy().astype("uint8")
                confidence = confidence.float().cpu().numpy()
                for i in range(prediction.shape[0]):
                    row = frame.iloc[cursor]
                    height, width = int(row.height), int(row.width)
                    target_row = int(getattr(row, "target_row_off", row.row_off))
                    target_col = int(getattr(row, "target_col_off", row.col_off))
                    crop_row = int(getattr(row, "crop_row_off", 0))
                    crop_col = int(getattr(row, "crop_col_off", 0))
                    window = Window(target_col, target_row, width, height)
                    pred_core = prediction[
                        i, crop_row : crop_row + height, crop_col : crop_col + width
                    ]
                    confidence_core = confidence[
                        i, crop_row : crop_row + height, crop_col : crop_col + width
                    ]
                    valid_core = (
                        batch["label"][
                            i, crop_row : crop_row + height, crop_col : crop_col + width
                        ].numpy()
                        != IGNORE_INDEX
                    )
                    class_dst.write(np.where(valid_core, pred_core, 0), 1, window=window)
                    confidence_dst.write(
                        np.where(valid_core, confidence_core, 0.0), 1, window=window
                    )
                    cursor += 1
        class_dst.build_overviews([2, 4, 8, 16], Resampling.nearest)
        confidence_dst.build_overviews([2, 4, 8, 16], Resampling.average)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"classification": str(class_path), "confidence": str(confidence_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write full-area maps for the final three-model experiment.")
    parser.add_argument("--config", default="config_hierarchical.yaml")
    parser.add_argument("--main-checkpoint", required=True, type=Path)
    parser.add_argument("--fcn-checkpoint", required=True, type=Path)
    parser.add_argument("--vgg-checkpoint", required=True, type=Path)
    parser.add_argument("--fcn-all-checkpoint", type=Path)
    parser.add_argument("--vgg-all-checkpoint", type=Path)
    parser.add_argument("--cnn-all-checkpoint", type=Path)
    parser.add_argument("--fcnn-all-checkpoint", type=Path)
    parser.add_argument("--transformer-all-checkpoint", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--context-margin", type=int, default=16,
        help="Discard this many pixels on every patch edge; 16 gives a 96x96 valid core.",
    )
    parser.add_argument(
        "--only", default=None,
        help="Comma-separated product keys; e.g. fcn_resnet50_all_modalities,vgg_unet_transformer_all_modalities",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = build_context_manifest(cfg, args.context_margin)
    dataset = make_inference_dataset(cfg, manifest_path)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)), pin_memory=device.type == "cuda",
    )
    specs = [
        ("ours_height_hierarchical", "main", args.main_checkpoint, True, False),
        ("fcn_resnet50", "fcn_resnet50", args.fcn_checkpoint, False, False),
        ("vgg_unet_transformer", "vgg_unet_transformer", args.vgg_checkpoint, False, False),
    ]
    if args.fcn_all_checkpoint:
        specs.append((
            "fcn_resnet50_all_modalities", "fcn_resnet50",
            args.fcn_all_checkpoint, False, True,
        ))
    if args.vgg_all_checkpoint:
        specs.append((
            "vgg_unet_transformer_all_modalities", "vgg_unet_transformer",
            args.vgg_all_checkpoint, False, True,
        ))
    for product_name, backbone, checkpoint in (
        ("cnn_all_modalities", "cnn", args.cnn_all_checkpoint),
        ("fcnn_all_modalities", "fcnn", args.fcnn_all_checkpoint),
        ("transformer_all_modalities", "transformer", args.transformer_all_checkpoint),
    ):
        if checkpoint:
            specs.append((product_name, backbone, checkpoint, False, True))
    if args.only:
        selected = set(args.only.split(","))
        specs = [spec for spec in specs if spec[0] in selected]
    manifest_output = args.output_dir / "prediction_manifest.json"
    outputs = {}
    if manifest_output.exists():
        outputs = json.loads(manifest_output.read_text(encoding="utf-8"))
    outputs["inference"] = {
            "input_patch_size": int(cfg["data"]["patch_size"]),
            "context_margin": args.context_margin,
            "output_core_size": int(cfg["data"]["patch_size"]) - 2 * args.context_margin,
            "BLU_zero_policy": "not inferred where a core is entirely invalid; written as nodata=0 everywhere",
            "manifest": str(manifest_path),
        }
    for name, kind, checkpoint, hierarchical, multimodal in specs:
        model = (
            make_main_model(cfg, dataset, checkpoint)
            if kind == "main"
            else (
                make_multimodal_comparator(cfg, dataset, kind, checkpoint)
                if multimodal
                else make_comparator(kind, dataset.s2_output_channels, checkpoint)
            )
        )
        print(f"predicting={name}", flush=True)
        outputs[name] = write_prediction(
            name, model, loader, dataset, cfg["paths"]["image"], args.output_dir,
            device, hierarchical, multimodal,
        )
    manifest_output.write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
