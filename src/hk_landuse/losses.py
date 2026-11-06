from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .labels import IGNORE_INDEX


class CrossEntropyDiceLoss(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor,
        dice_weight: float = 0.3,
        hierarchy_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.dice_weight = dice_weight
        self.hierarchy_weight = hierarchy_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.class_weights, ignore_index=IGNORE_INDEX)
        valid = target != IGNORE_INDEX
        safe_target = target.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe_target, logits.shape[1]).permute(0, 3, 1, 2).to(logits.dtype)
        mask = valid.unsqueeze(1)
        probs = logits.softmax(dim=1) * mask
        one_hot = one_hot * mask
        intersection = (probs * one_hot).sum(dim=(0, 2, 3))
        denominator = probs.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
        present = one_hot.sum(dim=(0, 2, 3)) > 0
        dice = (2 * intersection + 1.0) / (denominator + 1.0)
        dice_loss = 1.0 - dice[present].mean() if present.any() else logits.sum() * 0.0
        fine_loss = (1.0 - self.dice_weight) * ce + self.dice_weight * dice_loss
        built_logit = torch.logsumexp(logits[:, :6], dim=1)
        natural_logit = torch.logsumexp(logits[:, 6:], dim=1)
        hierarchy_logits = torch.stack([built_logit, natural_logit], dim=1)
        hierarchy_target = (safe_target >= 6).long().masked_fill(~valid, IGNORE_INDEX)
        hierarchy_loss = F.cross_entropy(
            hierarchy_logits,
            hierarchy_target,
            ignore_index=IGNORE_INDEX,
        )
        return fine_loss + self.hierarchy_weight * hierarchy_loss
