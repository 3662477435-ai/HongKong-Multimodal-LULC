from __future__ import annotations

import numpy as np
import torch

from .labels import IGNORE_INDEX, RAW_GROUPS


# The eight-class experiment removes official class 1 (commercial/office)
# and official class 8 (barren land) before loss computation and prediction.
EXCLUDED_OFFICIAL_IDS = (1, 8)
ACTIVE_OFFICIAL_IDS = (0, 2, 3, 4, 5, 6, 7, 9)
EXCLUDED_LEAF_IDS = (1, 12)

CLASS_NAMES_8 = (
    "住宅",
    "工业仓储",
    "公共机构与休闲",
    "交通设施",
    "其他建设用地",
    "农业",
    "林地/灌丛/草地/湿地",
    "水体",
)

RAW_GROUPS_8 = {
    new_id: RAW_GROUPS[old_id]
    for new_id, old_id in enumerate(ACTIVE_OFFICIAL_IDS)
}

# Old fourteen hierarchy leaves -> contiguous eight-class target.
# -1 denotes a deliberately excluded class.
LEAF_TO_OFFICIAL8 = np.asarray(
    [0, -1, 1, 2, 3, 4, 5, 5, 6, 6, 6, 6, -1, 7],
    dtype=np.int64,
)


def filter_leaf_target(target: torch.Tensor) -> torch.Tensor:
    """Return a copy with commercial and barren leaves set to IGNORE_INDEX."""
    filtered = target.clone()
    for leaf_id in EXCLUDED_LEAF_IDS:
        filtered[filtered == leaf_id] = IGNORE_INDEX
    return filtered


def leaf_target_to_official8(target: torch.Tensor) -> torch.Tensor:
    """Project fourteen-leaf labels to the contiguous eight-class task."""
    valid = target != IGNORE_INDEX
    safe = target.masked_fill(~valid, 0)
    lut = torch.as_tensor(LEAF_TO_OFFICIAL8, dtype=torch.long, device=target.device)
    projected = lut[safe]
    valid &= projected >= 0
    return projected.masked_fill(~valid, IGNORE_INDEX)


def leaf_prediction_to_official8(prediction: torch.Tensor) -> torch.Tensor:
    """Project active leaf predictions; raise if an excluded leaf is emitted."""
    lut = torch.as_tensor(LEAF_TO_OFFICIAL8, dtype=torch.long, device=prediction.device)
    projected = lut[prediction]
    if bool((projected < 0).any()):
        bad = torch.unique(prediction[projected < 0]).tolist()
        raise RuntimeError(f"Eight-class model emitted excluded leaf ids: {bad}")
    return projected

