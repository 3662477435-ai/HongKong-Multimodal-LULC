from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .labels import IGNORE_INDEX, LEAF_TO_COARSE


class HierarchicalLoss(nn.Module):
    def __init__(
        self,
        leaf_weights: torch.Tensor,
        coarse_weights: torch.Tensor,
        dice_weight: float = 0.3,
        coarse_weight: float = 0.5,
        weak_weight: float = 0.05,
        conflict_weight: float = 0.3,
        focal_gamma: float = 0.0,
        ignored_built_classes: tuple[int, ...] = (),
        official_weights: torch.Tensor | None = None,
        official_lut: torch.Tensor | None = None,
        leaf_aux_weight: float = 0.30,
        consistency_weight: float = 0.05,
        joint_hierarchy_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("leaf_weights", leaf_weights)
        self.register_buffer("coarse_weights", coarse_weights)
        self.register_buffer("coarse_lut", torch.as_tensor(LEAF_TO_COARSE, dtype=torch.long))
        self.dice_weight = dice_weight
        self.coarse_weight = coarse_weight
        self.weak_weight = weak_weight
        self.conflict_weight = conflict_weight
        self.focal_gamma = focal_gamma
        self.ignored_built_classes = tuple(
            sorted(set(int(x) for x in ignored_built_classes))
        )
        self.register_buffer("official_weights", official_weights)
        self.register_buffer("official_lut", official_lut)
        self.leaf_aux_weight = leaf_aux_weight
        self.consistency_weight = consistency_weight
        self.joint_hierarchy_weight = joint_hierarchy_weight

    def _focal(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Class-balanced focal loss; gamma=0 recovers weighted cross entropy."""
        log_probs = F.log_softmax(logits, dim=-1)
        chosen = log_probs.gather(1, target[:, None]).squeeze(1)
        probability = chosen.exp()
        alpha = weights[target]
        loss = -alpha * (1.0 - probability).pow(self.focal_gamma) * chosen
        return loss.sum() / alpha.sum().clamp_min(1e-6)

    def _targets(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = target != IGNORE_INDEX
        safe = target.masked_fill(~valid, 0)
        coarse = self.coarse_lut.to(target.device)[safe].masked_fill(~valid, IGNORE_INDEX)
        return safe, coarse

    def _dice(self, log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != IGNORE_INDEX
        safe = target.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe, 14).permute(0, 3, 1, 2).to(log_probs.dtype)
        mask = valid[:, None]
        probs = log_probs.exp() * mask
        one_hot = one_hot * mask
        intersection = (probs * one_hot).sum((0, 2, 3))
        denominator = probs.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
        present = one_hot.sum((0, 2, 3)) > 0
        return 1.0 - ((2 * intersection[present] + 1) / (denominator[present] + 1)).mean()

    @staticmethod
    def _dice_n(
        log_probs: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
    ) -> torch.Tensor:
        valid = target != IGNORE_INDEX
        safe = target.masked_fill(~valid, 0)
        one_hot = F.one_hot(safe, num_classes).permute(0, 3, 1, 2).to(log_probs.dtype)
        mask = valid[:, None]
        probs = log_probs.exp() * mask
        one_hot = one_hot * mask
        intersection = (probs * one_hot).sum((0, 2, 3))
        denominator = probs.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
        present = one_hot.sum((0, 2, 3)) > 0
        return 1.0 - (
            (2 * intersection[present] + 1) / (denominator[present] + 1)
        ).mean()

    def _weak_aoi(
        self,
        built_logits: torch.Tensor,
        target: torch.Tensor,
        osm: torch.Tensor | None,
    ) -> torch.Tensor:
        if osm is None:
            return built_logits.sum() * 0.0
        aoi = osm[:, 14:20]
        covered = aoi.sum(dim=1) > 0
        weak_target = aoi.argmax(dim=1)
        for class_id in self.ignored_built_classes:
            covered &= weak_target != class_id
        if not covered.any():
            return built_logits.sum() * 0.0
        valid = target != IGNORE_INDEX
        agrees = valid & (target < 6) & (target == weak_target)
        conflicts = valid & ~agrees
        weights = torch.ones_like(target, dtype=built_logits.dtype)
        weights[agrees] = 1.0
        weights[conflicts] = self.conflict_weight
        pixel_loss = F.cross_entropy(built_logits, weak_target, reduction="none")
        return (pixel_loss[covered] * weights[covered]).sum() / weights[covered].sum().clamp_min(1.0)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        target: torch.Tensor,
        stage: str,
        osm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        safe, coarse_target = self._targets(target)
        coarse_loss = F.cross_entropy(
            outputs["coarse_logits"],
            coarse_target,
            weight=self.coarse_weights,
            ignore_index=IGNORE_INDEX,
        )
        built_mask = (target >= 0) & (target < 6)
        natural_mask = (target >= 6) & (target < 14)
        zero = outputs["coarse_logits"].sum() * 0.0
        built_loss = (
            self._focal(
                outputs["built_logits"].permute(0, 2, 3, 1)[built_mask],
                target[built_mask],
                self.leaf_weights[:6],
            )
            if built_mask.any()
            else zero
        )
        natural_loss = (
            self._focal(
                outputs["natural_logits"].permute(0, 2, 3, 1)[natural_mask],
                target[natural_mask] - 6,
                self.leaf_weights[6:],
            )
            if natural_mask.any()
            else zero
        )
        fine_valid = target != IGNORE_INDEX
        fine_nll = self._focal(
            outputs["leaf_log_probs"].permute(0, 2, 3, 1)[fine_valid],
            target[fine_valid],
            self.leaf_weights,
        )
        dice = self._dice(outputs["leaf_log_probs"], target)
        weak = self._weak_aoi(outputs["built_logits"], target, osm)
        official_nll = zero
        official_dice = zero
        consistency = zero
        if (
            "official8_log_probs" in outputs
            and self.official_weights is not None
            and self.official_lut is not None
        ):
            official_target = self.official_lut[safe]
            official_valid = fine_valid & (official_target >= 0)
            official_target = official_target.masked_fill(
                ~official_valid, IGNORE_INDEX
            )
            official_nll = self._focal(
                outputs["official8_log_probs"]
                .permute(0, 2, 3, 1)[official_valid],
                official_target[official_valid],
                self.official_weights,
            )
            official_dice = self._dice_n(
                outputs["official8_log_probs"],
                official_target,
                len(self.official_weights),
            )
            hierarchical = outputs["hierarchical_official8_log_probs"]
            direct = outputs["direct_official8_log_probs"]
            hierarchical_probability = hierarchical.exp().detach()
            direct_probability = direct.exp()
            consistency = (
                hierarchical_probability
                * (
                    hierarchical
                    .detach()
                    - direct
                )
            ).sum(dim=1)[official_valid].mean()

        if stage == "coarse":
            loss = coarse_loss
        elif stage == "natural":
            loss = self.coarse_weight * coarse_loss + natural_loss
        elif stage == "built":
            loss = self.coarse_weight * coarse_loss + built_loss + self.weak_weight * weak
        elif stage == "joint":
            if "official8_log_probs" in outputs and self.official_weights is not None:
                fine = (
                    (1.0 - self.dice_weight) * official_nll
                    + self.dice_weight * official_dice
                    + self.leaf_aux_weight
                    * ((1.0 - self.dice_weight) * fine_nll + self.dice_weight * dice)
                    + self.consistency_weight * consistency
                )
            else:
                fine = (1.0 - self.dice_weight) * fine_nll + self.dice_weight * dice
            hierarchy_auxiliary = (
                self.coarse_weight * coarse_loss
                + 0.5 * (built_loss + natural_loss)
                + self.weak_weight * weak
            )
            loss = fine + self.joint_hierarchy_weight * hierarchy_auxiliary
        else:
            raise ValueError(f"Unknown hierarchy stage: {stage}")
        parts = {
            "coarse": float(coarse_loss.detach()),
            "built": float(built_loss.detach()),
            "natural": float(natural_loss.detach()),
            "fine_nll": float(fine_nll.detach()),
            "dice": float(dice.detach()),
            "weak_aoi": float(weak.detach()),
            "official_nll": float(official_nll.detach()),
            "official_dice": float(official_dice.detach()),
            "hierarchy_direct_consistency": float(consistency.detach()),
        }
        return loss, parts
