from __future__ import annotations

import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd

from hk_landuse.labels import IGNORE_INDEX, remap_blu
from hk_landuse.model import MultiModalUNet


PROJECT = Path(__file__).resolve().parents[1]


def test_label_mapping() -> None:
    raw = np.array([0, 1, 2, 3, 11, 21, 31, 41, 51, 61, 71, 81, 91], dtype=np.uint8)
    mapped = remap_blu(raw)
    assert mapped.tolist() == [IGNORE_INDEX, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def test_model_forward_shape() -> None:
    model = MultiModalUNet(vocab_size=50, base=8, text_dim=32, text_heads=4, text_layers=1, max_tokens=16)
    logits = model(
        torch.randn(2, 3, 64, 64),
        torch.randn(2, 5, 64, 64),
        torch.randint(1, 50, (2, 16)),
        torch.ones(2, 16, dtype=torch.bool),
    )
    assert logits.shape == (2, 10, 64, 64)
    assert torch.isfinite(logits).all()


def test_spatial_groups_do_not_cross_splits() -> None:
    manifest_path = PROJECT / "artifacts" / "prepared" / "manifest.csv"
    if not manifest_path.exists():
        return
    manifest = pd.read_csv(manifest_path)
    assert manifest.groupby("group")["split"].nunique().max() == 1
    assert set(manifest["split"]) == {"train", "val", "test"}


def test_dataset_report_grid() -> None:
    report_path = PROJECT / "artifacts" / "prepared" / "dataset_report.json"
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["reference_grid"]["width"] == 372
    assert report["reference_grid"]["height"] == 562
    assert report["reference_grid"]["crs"] == "EPSG:32650"
