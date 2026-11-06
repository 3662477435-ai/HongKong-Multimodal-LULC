from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def dual_pol_numpy(vv_db: np.ndarray, vh_db: np.ndarray) -> dict[str, np.ndarray]:
    """Physically valid dual-pol transforms from dB backscatter mosaics."""
    vv = np.power(10.0, np.clip(vv_db, -60.0, 40.0) / 10.0, dtype=np.float32)
    vh = np.power(10.0, np.clip(vh_db, -60.0, 40.0) / 10.0, dtype=np.float32)
    total = vv + vh
    eps = np.float32(1e-8)
    p_vv = vv / (total + eps)
    p_vh = vh / (total + eps)
    return {
        "vh_over_vv": np.clip(vh / (vv + eps), 0.0, 10.0),
        "vv_over_vh": np.clip(vv / (vh + eps), 0.0, 100.0),
        "total_power_db": 10.0 * np.log10(total + eps),
        "normalized_pol_difference": (vv - vh) / (total + eps),
        "dual_pol_rvi": 4.0 * vh / (total + eps),
        "intensity_entropy": -(p_vv * np.log2(p_vv + eps) + p_vh * np.log2(p_vh + eps)),
    }


def dual_pol_torch(vv_db: "torch.Tensor", vh_db: "torch.Tensor") -> "torch.Tensor":
    """Four bounded index channels for a neural S1 encoder.

    The raw VV, VH and VV-VH channels remain separate; this function adds total
    power, normalized polarization difference, dual-pol RVI and intensity entropy.
    """
    import torch

    vv = torch.pow(10.0, vv_db.clamp(-60.0, 40.0) / 10.0)
    vh = torch.pow(10.0, vh_db.clamp(-60.0, 40.0) / 10.0)
    total = vv + vh
    eps = 1e-8
    p_vv = vv / (total + eps)
    p_vh = vh / (total + eps)
    total_db_scaled = ((10.0 * torch.log10(total + eps)) + 20.0).clamp(-30.0, 30.0) / 10.0
    normalized_difference = (vv - vh) / (total + eps)
    rvi_scaled = (4.0 * vh / (total + eps)) / 2.0 - 1.0
    entropy = -(p_vv * torch.log2(p_vv + eps) + p_vh * torch.log2(p_vh + eps))
    return torch.stack([total_db_scaled, normalized_difference, rvi_scaled, entropy], dim=0)
