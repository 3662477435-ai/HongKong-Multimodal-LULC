from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
import rasterio
from torch.utils.data import DataLoader

from .config import load_config
from .dataset import LandUsePatchDataset
from .labels import CLASS_NAMES
from .train import build_model


CLASS_COLORS = {
    0: (0, 0, 0, 0),
    1: (230, 85, 105, 255),
    2: (155, 89, 182, 255),
    3: (110, 110, 110, 255),
    4: (243, 156, 18, 255),
    5: (241, 196, 15, 255),
    6: (166, 103, 76, 255),
    7: (126, 190, 75, 255),
    8: (39, 130, 73, 255),
    9: (194, 165, 116, 255),
    10: (52, 152, 219, 255),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict the full S1/S2 reference extent.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    prepared = Path(cfg["paths"]["prepared_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else output_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_path = prepared / cfg["data"].get("vocab_name", "vocab.json")
    normalization_path = prepared / cfg["data"].get("normalization_name", "normalization.json")
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    model = build_model(cfg, len(vocab)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    dataset = LandUsePatchDataset(
        image_path=cfg["paths"]["image"],
        label_path=prepared / "labels_10class.tif",
        manifest_path=prepared / "inference_manifest.csv",
        vocab_path=vocab_path,
        normalization_path=normalization_path,
        patch_size=int(cfg["data"]["patch_size"]),
        max_text_tokens=int(cfg["data"]["max_text_tokens"]),
        osm_poi_path=cfg["paths"].get("osm_poi_raster"),
        osm_aoi_path=cfg["paths"].get("osm_aoi_raster"),
        split=None,
        augment=False,
    )
    loader = DataLoader(dataset, batch_size=int(cfg["train"]["batch_size"]), shuffle=False)
    manifest = pd.read_csv(prepared / "inference_manifest.csv").set_index("patch_id")
    with rasterio.open(cfg["paths"]["image"]) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        finite = np.isfinite(src.read()).all(axis=0)
    prediction = np.zeros((height, width), dtype=np.uint8)
    confidence = np.zeros((height, width), dtype=np.float32)

    with torch.inference_mode():
        for batch in loader:
            logits = model(
                batch["s1"].to(device),
                batch["s2"].to(device),
                batch["input_ids"].to(device),
                batch["text_mask"].to(device),
                osm=batch.get("osm").to(device) if batch.get("osm") is not None else None,
            )
            probs = logits.softmax(dim=1)
            conf, pred = probs.max(dim=1)
            pred = pred.cpu().numpy().astype(np.uint8) + 1  # output 1..10; 0 is nodata
            conf = conf.cpu().numpy().astype(np.float32)
            for i, patch_id in enumerate(batch["patch_id"]):
                row = manifest.loc[patch_id]
                r0, c0 = int(row.row_off), int(row.col_off)
                h, w = int(row.height), int(row.width)
                prediction[r0 : r0 + h, c0 : c0 + w] = pred[i, :h, :w]
                confidence[r0 : r0 + h, c0 : c0 + w] = conf[i, :h, :w]
    prediction[~finite] = 0
    confidence[~finite] = np.nan

    class_profile = profile.copy()
    class_profile.update(count=1, dtype="uint8", nodata=0, compress="deflate", tiled=True)
    with rasterio.open(output_dir / "prediction_10class.tif", "w", **class_profile) as dst:
        dst.write(prediction, 1)
        dst.set_band_description(1, "predicted_class_1_to_10")
        dst.write_colormap(1, CLASS_COLORS)
        dst.update_tags(**{f"CLASS_{i + 1}": name for i, name in enumerate(CLASS_NAMES)})
    confidence_profile = profile.copy()
    confidence_profile.update(count=1, dtype="float32", nodata=np.nan, compress="deflate", tiled=True)
    with rasterio.open(output_dir / "prediction_confidence.tif", "w", **confidence_profile) as dst:
        dst.write(confidence, 1)
        dst.set_band_description(1, "maximum_softmax_probability")
    print(output_dir / "prediction_10class.tif")


if __name__ == "__main__":
    main()
