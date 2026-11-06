from __future__ import annotations

import numpy as np


IGNORE_INDEX = 255

# Official ten groups used in every comparable OA/mIoU report.
CLASS_NAMES = [
    "住宅",
    "商业办公",
    "工业仓储",
    "公共机构与休闲",
    "交通设施",
    "其他建设用地",
    "农业",
    "林地／灌丛／草地／湿地",
    "荒地",
    "水体",
]

RAW_GROUPS = {
    0: (1, 2, 3),
    1: (11,),
    2: (21, 22, 23),
    3: (31, 32),
    4: (41, 42, 43, 44),
    5: (51, 52, 53, 54),
    6: (61, 62),
    7: (71, 72, 73, 74),
    8: (81, 83),
    9: (91, 92),
}

RAW_TO_TRAIN = {
    raw_value: train_id
    for train_id, raw_values in RAW_GROUPS.items()
    for raw_value in raw_values
}

COARSE_CLASS_NAMES = ["建设地表", "植被与农地", "水体与湿地", "裸地与岩岸"]

LEAF_CLASS_NAMES = [
    "住宅",
    "商业办公",
    "工业仓储",
    "公共机构与休闲",
    "交通设施",
    "其他建设用地",
    "农地",
    "鱼塘／基围",
    "林地",
    "灌丛",
    "草地",
    "红树林／沼泽",
    "荒地／岩岸",
    "水塘／河道／明渠",
]

LEAF_RAW_GROUPS = {
    0: (1, 2, 3),
    1: (11,),
    2: (21, 22, 23),
    3: (31, 32),
    4: (41, 42, 43, 44),
    5: (51, 52, 53, 54),
    6: (61,),
    7: (62,),
    8: (71,),
    9: (72,),
    10: (73,),
    11: (74,),
    12: (81, 83),
    13: (91, 92),
}

RAW_TO_LEAF = {
    raw_value: leaf_id
    for leaf_id, raw_values in LEAF_RAW_GROUPS.items()
    for raw_value in raw_values
}

# leaf -> coarse physical group: built, vegetation, water/wetland, barren/rock.
LEAF_TO_COARSE = np.array([0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 1, 2, 3, 2], dtype=np.int64)
BUILT_LEAF_IDS = tuple(range(6))
NATURAL_LEAF_IDS = tuple(range(6, 14))
VEGETATION_LEAF_IDS = (6, 8, 9, 10)
WATER_LEAF_IDS = (7, 11, 13)
BARREN_LEAF_IDS = (12,)

# leaf -> official ten-class projection.
LEAF_TO_OFFICIAL = np.array([0, 1, 2, 3, 4, 5, 6, 6, 7, 7, 7, 7, 8, 9], dtype=np.int64)


def _remap(raw: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    mapped = np.full(raw.shape, IGNORE_INDEX, dtype=np.uint8)
    for raw_value, train_id in mapping.items():
        mapped[raw == raw_value] = train_id
    return mapped


def remap_blu(raw: np.ndarray) -> np.ndarray:
    """Map BLU values to the official ten contiguous classes."""
    return _remap(raw, RAW_TO_TRAIN)


def remap_blu_leaf(raw: np.ndarray) -> np.ndarray:
    """Map BLU values to fourteen hierarchy leaves; background is ignored."""
    return _remap(raw, RAW_TO_LEAF)


def leaf_to_coarse(target: np.ndarray) -> np.ndarray:
    result = np.full(target.shape, IGNORE_INDEX, dtype=np.uint8)
    valid = target != IGNORE_INDEX
    result[valid] = LEAF_TO_COARSE[target[valid]]
    return result


def leaf_to_official(target: np.ndarray) -> np.ndarray:
    result = np.full(target.shape, IGNORE_INDEX, dtype=np.uint8)
    valid = target != IGNORE_INDEX
    result[valid] = LEAF_TO_OFFICIAL[target[valid]]
    return result


def train_to_raw_group(train_id: int) -> tuple[int, ...]:
    return RAW_GROUPS[train_id]
