from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .advanced_model import ConvGNAct
from .comparison_models import FCNNBaseline, MultiModalEarlyFusionComparator


class HierarchicalMultiModalFCNN(nn.Module):
    """Full-modality FCNN augmented with conditional land-use/cover heads."""

    def __init__(
        self,
        vocab_size: int,
        s2_channels: int,
        s1_channels: int,
        urban_channels: int,
        road_channels: int,
        osm_channels: int,
        text_dim: int,
        text_heads: int,
        text_layers: int,
        max_tokens: int,
        dropout: float = 0.1,
        use_text: bool = True,
        use_semantic_residual: bool = True,
        use_hierarchy: bool = True,
        gate_mode: str = "class",
    ) -> None:
        super().__init__()
        if gate_mode not in {"class", "shared"}:
            raise ValueError(f"Unsupported gate_mode={gate_mode!r}")
        self.use_semantic_residual = bool(use_semantic_residual)
        self.use_hierarchy = bool(use_hierarchy)
        self.gate_mode = gate_mode
        self.s2_only_direct = (
            s1_channels == 0
            and urban_channels == 0
            and road_channels == 0
            and osm_channels == 0
            and not use_text
        )
        if self.s2_only_direct:
            self.base = FCNNBaseline(s2_channels, num_classes=8, pretrained=False)
        else:
            self.base = MultiModalEarlyFusionComparator(
                backbone="fcnn",
                vocab_size=vocab_size,
                s2_channels=s2_channels,
                s1_channels=s1_channels,
                urban_channels=urban_channels,
                road_channels=road_channels,
                osm_channels=osm_channels,
                text_dim=text_dim,
                text_heads=text_heads,
                text_layers=text_layers,
                max_tokens=max_tokens,
                dropout=dropout,
                pretrained=False,
                use_text=use_text,
                num_classes=8,
            )
        if not isinstance(self._base_segmenter(), FCNNBaseline):
            raise TypeError("The hierarchical extension requires an FCNN segmenter")
        self.proj4 = ConvGNAct(384, 128, 1)
        self.proj3 = ConvGNAct(256, 128, 1)
        self.proj2 = ConvGNAct(128, 96, 1)
        self.proj1 = ConvGNAct(64, 64, 1)
        self.decode3 = ConvGNAct(128, 128)
        self.decode2 = ConvGNAct(128 + 96, 96)
        self.decode1 = nn.Sequential(
            ConvGNAct(96 + 64, 96),
            nn.Dropout2d(dropout),
            ConvGNAct(96, 64),
        )
        # Coarse: built, vegetation/agriculture, water.
        self.coarse_head = nn.Conv2d(64, 3, 1)
        self.built_head = nn.Conv2d(64, 5, 1)
        self.natural_head = nn.Conv2d(64, 2, 1)
        self.osm_channels = osm_channels
        self.urban_channels = urban_channels
        self.road_channels = road_channels
        self.semantic_osm = self._semantic_encoder(osm_channels)
        self.semantic_urban = self._semantic_encoder(urban_channels)
        self.semantic_roads = self._semantic_encoder(road_channels)
        self.semantic_residual = nn.Sequential(
            ConvGNAct(64 + 32 + 32 + 32, 96),
            nn.Dropout2d(dropout),
            ConvGNAct(96, 64),
            nn.Conv2d(64, 8, 1),
        )
        nn.init.zeros_(self.semantic_residual[-1].weight)
        nn.init.zeros_(self.semantic_residual[-1].bias)
        self.semantic_residual_scale = nn.Parameter(torch.tensor(-1.0))
        # The initialized model reproduces the FCNN baseline almost exactly.
        gate_shape = (8,) if gate_mode == "class" else (1,)
        self.hierarchy_gate_logit = nn.Parameter(torch.full(gate_shape, -3.0))

    @staticmethod
    def _semantic_encoder(channels: int) -> nn.Module | None:
        if channels <= 0:
            return None
        return nn.Sequential(
            ConvGNAct(channels, 32),
            ConvGNAct(32, 32),
        )

    @staticmethod
    def _semantic_feature(
        encoder: nn.Module | None,
        tensor: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if encoder is None:
            return reference.new_zeros(
                (reference.shape[0], 32, reference.shape[2], reference.shape[3])
            )
        return encoder(tensor)

    def load_fcnn_baseline(self, state: dict[str, torch.Tensor]) -> None:
        incompatible = self.base.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"FCNN baseline mismatch: {incompatible}")

    def _base_segmenter(self) -> FCNNBaseline:
        if self.s2_only_direct:
            if not isinstance(self.base, FCNNBaseline):
                raise TypeError("Expected a direct FCNN base")
            return self.base
        if not isinstance(self.base, MultiModalEarlyFusionComparator):
            raise TypeError("Expected a multimodal FCNN comparator")
        if not isinstance(self.base.segmenter, FCNNBaseline):
            raise TypeError("Expected an FCNN segmenter")
        return self.base.segmenter

    def _hierarchy_feature(
        self,
        features: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        e1, e2, e3, bottleneck = features
        d3 = self.decode3(
            F.interpolate(
                self.proj4(bottleneck),
                size=e3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            + self.proj3(e3)
        )
        d2 = self.decode2(
            torch.cat(
                [
                    F.interpolate(
                        d3, size=e2.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    self.proj2(e2),
                ],
                dim=1,
            )
        )
        return self.decode1(
            torch.cat(
                [
                    F.interpolate(
                        d2, size=e1.shape[-2:], mode="bilinear", align_corners=False
                    ),
                    self.proj1(e1),
                ],
                dim=1,
            )
        )

    @staticmethod
    def _route(
        coarse_logits: torch.Tensor,
        built_logits: torch.Tensor,
        natural_logits: torch.Tensor,
    ) -> torch.Tensor:
        coarse = F.log_softmax(coarse_logits, dim=1)
        built = F.log_softmax(built_logits, dim=1)
        natural = F.log_softmax(natural_logits, dim=1)
        return torch.stack(
            [
                *[coarse[:, 0] + built[:, index] for index in range(5)],
                coarse[:, 1] + natural[:, 0],
                coarse[:, 1] + natural[:, 1],
                coarse[:, 2],
            ],
            dim=1,
        )

    def forward(
        self,
        s2: torch.Tensor,
        s1: torch.Tensor,
        urban: torch.Tensor,
        roads: torch.Tensor,
        osm: torch.Tensor,
        input_ids: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.s2_only_direct:
            fused = s2
        else:
            fused = self.base.fuse_modalities(
                s2, s1, urban, roads, osm, input_ids, text_mask
            )
        segmenter = self._base_segmenter()
        features = segmenter.encode(fused)
        base_direct_logits = segmenter.logits_from_features(
            features, s2.shape[-2:]
        )
        if self.use_semantic_residual:
            semantic_feature = torch.cat(
                [
                    features[0],
                    self._semantic_feature(self.semantic_osm, osm, features[0]),
                    self._semantic_feature(self.semantic_urban, urban, features[0]),
                    self._semantic_feature(self.semantic_roads, roads, features[0]),
                ],
                dim=1,
            )
            semantic_residual = self.semantic_residual(semantic_feature)
            direct_logits = base_direct_logits + F.softplus(
                self.semantic_residual_scale
            ) * semantic_residual
        else:
            semantic_residual = torch.zeros_like(base_direct_logits)
            direct_logits = base_direct_logits
        hierarchy_feature = self._hierarchy_feature(features)
        coarse_logits = self.coarse_head(hierarchy_feature)
        built_logits = self.built_head(hierarchy_feature)
        natural_logits = self.natural_head(hierarchy_feature)
        hierarchy_log_probs = self._route(
            coarse_logits, built_logits, natural_logits
        )
        direct_log_probs = F.log_softmax(direct_logits, dim=1)
        gate = torch.sigmoid(self.hierarchy_gate_logit)
        if gate.numel() == 1:
            gate = gate.expand(8)
        if self.use_hierarchy:
            combined = torch.logaddexp(
                direct_log_probs
                + torch.log((1.0 - gate).clamp_min(1e-6))[None, :, None, None],
                hierarchy_log_probs
                + torch.log(gate.clamp_min(1e-6))[None, :, None, None],
            )
            combined = combined - torch.logsumexp(combined, dim=1, keepdim=True)
        else:
            combined = direct_log_probs
        return {
            "combined_log_probs": combined,
            "direct_logits": direct_logits,
            "base_direct_logits": base_direct_logits,
            "semantic_residual_logits": semantic_residual,
            "direct_log_probs": direct_log_probs,
            "hierarchy_log_probs": hierarchy_log_probs,
            "coarse_logits": coarse_logits,
            "built_logits": built_logits,
            "natural_logits": natural_logits,
            "hierarchy_gates": gate,
        }
