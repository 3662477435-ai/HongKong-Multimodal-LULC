from __future__ import annotations

import numpy as np


OPTICAL_INDEX_NAMES = (
    "S2_SAVI",
    "S2_BSI",
    "S2_NDRGRI",
    "S2_NBR2",
    "S2_NDBI",
    "S2_MNDWI",
)


def _ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return numerator / np.where(np.abs(denominator) > 1e-6, denominator, np.nan)


def bare_soil_indices(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    swir2: np.ndarray,
) -> np.ndarray:
    """Return bounded S2 indices used as classifier inputs, not hard labels."""
    savi = 1.5 * _ratio(nir - red, nir + red + 0.5)
    bsi = _ratio((swir1 + red) - (nir + blue), (swir1 + red) + (nir + blue))
    ndrgri = _ratio(red - green, red + green)
    nbr2 = _ratio(swir1 - swir2, swir1 + swir2)
    ndbi = _ratio(swir1 - nir, swir1 + nir)
    mndwi = _ratio(green - swir1, green + swir1)
    result = np.stack([savi, bsi, ndrgri, nbr2, ndbi, mndwi]).astype(np.float32)
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    result[0] = np.clip(result[0], -1.5, 1.5) / 1.5
    result[1:] = np.clip(result[1:], -1.0, 1.0)
    return result
