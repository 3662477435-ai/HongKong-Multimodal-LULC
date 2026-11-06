from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import Swin_V2_T_Weights, swin_v2_t

from .advanced_model import ConvGNAct, PyramidPooling
from .model import TextEncoder


class MultiScaleOSM(nn.Module):
    """Build POI TF-like context and AOI area-density proxies at three scales."""

    def forward(self, osm: torch.Tensor) -> torch.Tensor:
        poi, aoi = osm[:, :14], osm[:, 14:]
        poi_50 = F.avg_pool2d(poi, kernel_size=5, stride=1, padding=2)
        poi_250 = F.avg_pool2d(poi, kernel_size=25, stride=1, padding=12)
        aoi_50 = F.avg_pool2d(aoi, kernel_size=5, stride=1, padding=2)
        aoi_250 = F.avg_pool2d(aoi, kernel_size=25, stride=1, padding=12)
        return torch.cat([poi, poi_50, poi_250, aoi, aoi_50, aoi_250], dim=1)


class RoadContextEncoder(nn.Module):
    """Encode road hierarchy, width, proximity and 50/250 m network density."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvGNAct(23, 64),
            ConvGNAct(64, 64),
            ConvGNAct(64, 64),
        )

    def forward(self, roads: torch.Tensor) -> torch.Tensor:
        hierarchy = torch.stack(
            [
                roads[:, 0:4].sum(dim=1).clamp_max(1.0),
                roads[:, 4],
                roads[:, 5],
                roads[:, 6],
                roads[:, 7:9].sum(dim=1).clamp_max(1.0),
            ],
            dim=1,
        )
        density_50 = F.avg_pool2d(hierarchy, kernel_size=5, stride=1, padding=2)
        density_250 = F.avg_pool2d(hierarchy, kernel_size=25, stride=1, padding=12)
        return self.encoder(torch.cat([roads, density_50, density_250], dim=1))


class SARDualPolEncoder(nn.Module):
    """Encode raw/derived dual-pol channels together with local SAR texture."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            ConvGNAct(channels * 3, 64),
            ConvGNAct(64, 64),
            ConvGNAct(64, 64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local_mean = F.avg_pool2d(x, kernel_size=5, stride=1, padding=2)
        local_second = F.avg_pool2d(x.square(), kernel_size=5, stride=1, padding=2)
        local_std = (local_second - local_mean.square()).clamp_min(1e-6).sqrt()
        return self.encoder(torch.cat([x, local_mean, local_std], dim=1))


class FCNNOpticalEncoder(nn.Module):
    """FCNN-style multi-scale optical backbone returning dense 10 m features."""

    def __init__(self, input_channels: int, dropout: float) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(
            ConvGNAct(input_channels, 64),
            ConvGNAct(64, 64),
        )
        self.enc2 = nn.Sequential(
            ConvGNAct(64, 128),
            ConvGNAct(128, 128),
        )
        self.enc3 = nn.Sequential(
            ConvGNAct(128, 256),
            ConvGNAct(256, 256),
        )
        self.bottleneck = nn.Sequential(
            ConvGNAct(256, 384),
            nn.Dropout2d(dropout),
            ConvGNAct(384, 384),
        )
        self.pool = nn.MaxPool2d(2)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        bottleneck = self.bottleneck(self.pool(e3))
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


class BranchPhysicalFusion(nn.Module):
    """Learnable branch-specific gates prevent SAR from dominating semantics."""

    def __init__(
        self,
        sar_gate: float,
        highres_gate: float,
        index_gate: float,
        dropout: float,
        index_channels: int = 32,
    ) -> None:
        super().__init__()
        self.sar_gate_logit = nn.Parameter(torch.tensor(float(sar_gate)))
        self.highres_gate_logit = nn.Parameter(torch.tensor(float(highres_gate)))
        self.index_channels = index_channels
        if index_channels:
            self.index_gate_logit = nn.Parameter(torch.tensor(float(index_gate)))
        else:
            self.register_parameter("index_gate_logit", None)
        hidden_channels = 112 if index_channels else 96
        self.refine = nn.Sequential(
            ConvGNAct(64 + 64 + 32 + index_channels, hidden_channels),
            nn.Dropout2d(dropout),
            ConvGNAct(hidden_channels, 64),
        )

    def forward(
        self,
        optical: torch.Tensor,
        sar: torch.Tensor,
        highres: torch.Tensor,
        indices: torch.Tensor | None,
    ) -> torch.Tensor:
        sar = torch.sigmoid(self.sar_gate_logit) * sar
        highres = torch.sigmoid(self.highres_gate_logit) * highres
        parts = [optical, sar, highres]
        if self.index_channels:
            if indices is None or self.index_gate_logit is None:
                raise ValueError("Index-gated fusion requires index features")
            parts.append(torch.sigmoid(self.index_gate_logit) * indices)
        return self.refine(torch.cat(parts, dim=1))


class HierarchicalSwinFusion(nn.Module):
    """Image-only physical heads and semantic-conditioned built-use head.

    The returned leaf probabilities are exactly normalized soft routes.  No
    threshold is used, so a coarse prediction cannot irreversibly discard a
    fine-class hypothesis.
    """

    def __init__(
        self,
        vocab_size: int,
        s2_channels: int = 13,
        s1_channels: int = 3,
        text_dim: int = 128,
        text_heads: int = 4,
        text_layers: int = 2,
        max_tokens: int = 96,
        fpn_channels: int = 128,
        dropout: float = 0.1,
        pretrained: bool = True,
        use_s2: bool = True,
        use_s1: bool = True,
        use_poi: bool = True,
        use_aoi: bool = True,
        use_text: bool = True,
        use_highres: bool = True,
        use_nighttime: bool = True,
        use_building: bool = True,
        use_roads: bool = True,
        building_channels: int = 9,
        optical_index_channels: int = 6,
        separate_optical_indices: bool = False,
        excluded_leaf_ids: tuple[int, ...] = (),
        official_projection: tuple[int, ...] | None = None,
        image_backbone: str = "swin_v2_tiny",
    ) -> None:
        super().__init__()
        self.s2_channels = s2_channels
        self.s1_channels = s1_channels
        self.optical_index_channels = optical_index_channels
        self.separate_optical_indices = separate_optical_indices
        self.base_s2_channels = (
            s2_channels - optical_index_channels if separate_optical_indices else s2_channels
        )
        if self.base_s2_channels <= 0:
            raise ValueError("s2_channels must exceed optical_index_channels")
        self.use_s2 = use_s2
        self.use_s1 = use_s1
        self.use_poi = use_poi
        self.use_aoi = use_aoi
        self.use_text = use_text
        self.use_highres = use_highres
        self.use_nighttime = use_nighttime
        self.use_building = use_building
        self.use_roads = use_roads
        self.building_channels = building_channels
        self.excluded_leaf_ids = tuple(sorted(set(int(x) for x in excluded_leaf_ids)))
        self.image_backbone_name = image_backbone
        self.official_projection = (
            tuple(int(x) for x in official_projection)
            if official_projection is not None
            else None
        )
        if self.official_projection is not None and len(self.official_projection) != 14:
            raise ValueError("official_projection must contain one entry for each of 14 leaves")

        if image_backbone == "swin_v2_tiny":
            weights = Swin_V2_T_Weights.DEFAULT if pretrained else None
            self.backbone = swin_v2_t(weights=weights).features
            self.spectral_adapter = nn.Conv2d(self.base_s2_channels, 3, 1)
            with torch.no_grad():
                self.spectral_adapter.weight.zero_()
                self.spectral_adapter.bias.zero_()
                # S2 order: B1, B2, B3, B4, ...; initialise as RGB.
                self.spectral_adapter.weight[0, 3, 0, 0] = 1.0
                self.spectral_adapter.weight[1, 2, 0, 0] = 1.0
                self.spectral_adapter.weight[2, 1, 0, 0] = 1.0
            stage_channels = (96, 192, 384, 768)
            self.laterals = nn.ModuleList(
                [ConvGNAct(ch, fpn_channels, 1) for ch in stage_channels[:3]]
            )
            self.ppm = PyramidPooling(stage_channels[3], fpn_channels)
            self.fpn_smooth = nn.ModuleList(
                [ConvGNAct(fpn_channels, fpn_channels) for _ in range(3)]
            )
            self.image_fusion = nn.Sequential(
                ConvGNAct(fpn_channels * 4, fpn_channels * 2),
                nn.Dropout2d(dropout),
                ConvGNAct(fpn_channels * 2, fpn_channels),
            )
            self.detail_encoder = nn.Sequential(
                ConvGNAct(self.base_s2_channels, 64),
                ConvGNAct(64, 64),
            )
            self.image_refine = nn.Sequential(
                ConvGNAct(fpn_channels + 64, 96),
                nn.Dropout2d(dropout),
                ConvGNAct(96, 64),
            )
        elif image_backbone == "fcnn":
            self.fcnn_image_encoder = FCNNOpticalEncoder(
                self.base_s2_channels, dropout
            )
        else:
            raise ValueError(f"Unsupported image_backbone: {image_backbone}")
        self.sar_encoder = SARDualPolEncoder(s1_channels)
        self.index_encoder = (
            nn.Sequential(
                ConvGNAct(optical_index_channels, 32),
                ConvGNAct(32, 32),
            )
            if separate_optical_indices
            else None
        )
        self.highres_encoder = nn.Sequential(
            ConvGNAct(3, 32),
            ConvGNAct(32, 32),
        )
        self.socioeconomic_encoder = nn.Sequential(
            # NTL (3), building channels, and three paper-inspired explicit
            # NTL x morphology interactions (coverage, mean height, FAR proxy).
            ConvGNAct(3 + building_channels + 3, 64),
            ConvGNAct(64, 64),
        )
        # Initial gates follow the measured block ablation: S1 is strongest for
        # physical/coarse classes, useful for natural classes and weak for built use.
        gated_channels = 32 if separate_optical_indices else 0
        self.coarse_fusion = BranchPhysicalFusion(
            2.0, -1.0, -1.0, dropout, gated_channels
        )
        self.natural_fusion = BranchPhysicalFusion(
            1.0, -0.5, 1.2, dropout, gated_channels
        )
        self.built_physical_fusion = BranchPhysicalFusion(
            -1.5, 1.0, 0.0, dropout, gated_channels
        )

        # 14 POI x 3 scales + 6 AOI x 3 scales = 60 channels.
        self.osm_pyramid = MultiScaleOSM()
        self.osm_encoder = nn.Sequential(
            ConvGNAct(60, 64, stride=2),
            ConvGNAct(64, fpn_channels, stride=2),
            ConvGNAct(fpn_channels, fpn_channels),
        )
        self.road_encoder = RoadContextEncoder()
        self.text_encoder = TextEncoder(vocab_size, text_dim, text_heads, text_layers, max_tokens, dropout)
        self.text_film = nn.Linear(text_dim, fpn_channels * 2)
        self.built_refine = nn.Sequential(
            ConvGNAct(64 + fpn_channels + 64 + 64, 192),
            nn.Dropout2d(dropout),
            ConvGNAct(192, 64),
        )

        self.coarse_head = nn.Conv2d(64, 4, 1)
        self.natural_head = nn.Conv2d(64, 8, 1)
        self.built_head = nn.Conv2d(64, 6, 1)
        self.aoi_prior_strength = nn.Parameter(torch.tensor(-1.5))
        self.road_transport_prior_strength = nn.Parameter(torch.tensor(-2.0))
        if self.official_projection is not None:
            active_ids = sorted({x for x in self.official_projection if x >= 0})
            if active_ids != list(range(len(active_ids))):
                raise ValueError("official_projection active ids must be contiguous from zero")
            self.num_official_classes = len(active_ids)
            self.official_fusion = nn.Sequential(
                ConvGNAct(64 * 3, 128),
                nn.Dropout2d(dropout),
                ConvGNAct(128, 64),
            )
            self.official_head = nn.Conv2d(64, self.num_official_classes, 1)
            # Start conservatively: retain most of the hierarchical probability,
            # while allowing each official class to learn how much direct evidence
            # is needed to correct accumulated routing errors.
            self.official_direct_gate_logit = nn.Parameter(
                torch.full((self.num_official_classes,), -0.5)
            )

    @staticmethod
    def _nchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    def _image_features(self, s2: torch.Tensor) -> torch.Tensor:
        if self.image_backbone_name == "fcnn":
            return self.fcnn_image_encoder(s2)
        input_size = s2.shape[-2:]
        x = self.spectral_adapter(s2)
        x = self.backbone[0](x)
        x = self.backbone[1](x)
        c1 = self._nchw(x)
        x = self.backbone[2](x)
        x = self.backbone[3](x)
        c2 = self._nchw(x)
        x = self.backbone[4](x)
        x = self.backbone[5](x)
        c3 = self._nchw(x)
        x = self.backbone[6](x)
        x = self.backbone[7](x)
        c4 = self._nchw(x)

        p4 = self.ppm(c4)
        p3 = self.fpn_smooth[2](
            self.laterals[2](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        )
        p2 = self.fpn_smooth[1](
            self.laterals[1](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        )
        p1 = self.fpn_smooth[0](
            self.laterals[0](c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        )
        pyramid = [p1] + [
            F.interpolate(p, size=p1.shape[-2:], mode="bilinear", align_corners=False)
            for p in (p2, p3, p4)
        ]
        fused = self.image_fusion(torch.cat(pyramid, dim=1))
        fused = F.interpolate(fused, size=input_size, mode="bilinear", align_corners=False)
        detail = self.detail_encoder(s2)
        return self.image_refine(torch.cat([fused, detail], dim=1))

    def _mask_excluded_logits(
        self,
        coarse_logits: torch.Tensor,
        built_logits: torch.Tensor,
        natural_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Remove excluded leaves from every conditional softmax denominator."""
        if not self.excluded_leaf_ids:
            return coarse_logits, built_logits, natural_logits
        coarse_logits = coarse_logits.clone()
        built_logits = built_logits.clone()
        natural_logits = natural_logits.clone()
        for leaf_id in self.excluded_leaf_ids:
            if 0 <= leaf_id < 6:
                built_logits[:, leaf_id] = -1.0e4
            elif 6 <= leaf_id < 14:
                natural_logits[:, leaf_id - 6] = -1.0e4
        # Coarse barren/rock contains only leaf 12, so it must also be removed
        # when the barren class is excluded from the task.
        if 12 in self.excluded_leaf_ids:
            coarse_logits[:, 3] = -1.0e4
        return coarse_logits, built_logits, natural_logits

    @staticmethod
    def _route(
        coarse_logits: torch.Tensor,
        built_logits: torch.Tensor,
        natural_logits: torch.Tensor,
    ) -> torch.Tensor:
        coarse = F.log_softmax(coarse_logits, dim=1)
        built = F.log_softmax(built_logits, dim=1)
        # natural local indices: farm, fishpond, forest, shrub, grass, wetland, barren, water
        vegetation_local = (0, 2, 3, 4)
        water_local = (1, 5, 7)
        vegetation = F.log_softmax(natural_logits[:, vegetation_local], dim=1)
        water = F.log_softmax(natural_logits[:, water_local], dim=1)
        leaves: list[torch.Tensor | None] = [None] * 14
        for i in range(6):
            leaves[i] = coarse[:, 0] + built[:, i]
        for j, leaf_id in enumerate((6, 8, 9, 10)):
            leaves[leaf_id] = coarse[:, 1] + vegetation[:, j]
        for j, leaf_id in enumerate((7, 11, 13)):
            leaves[leaf_id] = coarse[:, 2] + water[:, j]
        leaves[12] = coarse[:, 3]
        return torch.stack([x for x in leaves if x is not None], dim=1)

    def _aggregate_official(self, leaf_log_probs: torch.Tensor) -> torch.Tensor:
        if self.official_projection is None:
            raise RuntimeError("Official projection was not configured")
        groups = []
        for class_id in range(self.num_official_classes):
            leaf_ids = [
                leaf_id
                for leaf_id, projected in enumerate(self.official_projection)
                if projected == class_id
            ]
            groups.append(torch.logsumexp(leaf_log_probs[:, leaf_ids], dim=1))
        return torch.stack(groups, dim=1)

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        osm: torch.Tensor | None = None,
        urban: torch.Tensor | None = None,
        roads: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if not self.use_s2:
            s2 = torch.zeros_like(s2)
        if not self.use_s1:
            s1 = torch.zeros_like(s1)
        s2_base = s2[:, : self.base_s2_channels]
        optical_feature = self._image_features(s2_base)
        sar_feature = self.sar_encoder(s1)
        index_feature = None
        if self.separate_optical_indices:
            if self.index_encoder is None:
                raise RuntimeError("Missing optical index encoder")
            optical_indices = s2[:, self.base_s2_channels :]
            index_feature = self.index_encoder(optical_indices)

        batch, _, height, width = s2.shape
        if urban is None:
            urban = s2.new_zeros((batch, 0, height, width))
        cursor = 0
        if self.use_highres:
            highres = urban[:, cursor : cursor + 3]
            cursor += 3
        else:
            highres = s2.new_zeros((batch, 3, height, width))
        if self.use_nighttime:
            nighttime = urban[:, cursor : cursor + 3]
            cursor += 3
        else:
            nighttime = s2.new_zeros((batch, 3, height, width))
        if self.use_building:
            buildings = urban[:, cursor : cursor + self.building_channels]
            cursor += self.building_channels
        else:
            buildings = s2.new_zeros((batch, self.building_channels, height, width))
        if cursor != urban.shape[1]:
            raise ValueError(
                f"Unexpected urban channel count: consumed {cursor}, received {urban.shape[1]}"
            )
        highres_feature = self.highres_encoder(highres)
        if self.building_channels >= 13:
            # Band 2 is the source log1p VIIRS layer after dataset scaling.
            ntl = nighttime[:, 1:2]
            interactions = torch.cat(
                [ntl * buildings[:, i : i + 1] for i in (9, 10, 12)], dim=1
            )
        else:
            interactions = urban.new_zeros((batch, 3, height, width))
        socioeconomic_feature = self.socioeconomic_encoder(
            torch.cat([nighttime, buildings, interactions], dim=1)
        )
        coarse_feature = self.coarse_fusion(
            optical_feature, sar_feature, highres_feature, index_feature
        )
        natural_feature = self.natural_fusion(
            optical_feature, sar_feature, highres_feature, index_feature
        )
        built_physical = self.built_physical_fusion(
            optical_feature, sar_feature, highres_feature, index_feature
        )
        coarse_logits = self.coarse_head(coarse_feature)
        natural_logits = self.natural_head(natural_feature)

        if osm is None:
            osm = s2.new_zeros((batch, 20, height, width))
        poi = osm[:, :14] if self.use_poi else torch.zeros_like(osm[:, :14])
        aoi = osm[:, 14:] if self.use_aoi else torch.zeros_like(osm[:, 14:])
        semantic = self.osm_pyramid(torch.cat([poi, aoi], dim=1))
        semantic = self.osm_encoder(semantic)
        semantic = F.interpolate(semantic, size=(height, width), mode="bilinear", align_corners=False)
        if self.use_text:
            text = self.text_encoder(input_ids, text_mask)
            gamma, beta = self.text_film(text).chunk(2, dim=1)
            semantic = semantic * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        if roads is None:
            roads = s2.new_zeros((batch, 13, height, width))
        if not self.use_roads:
            roads = torch.zeros_like(roads)
        road_feature = self.road_encoder(roads)
        built_feature = self.built_refine(
            torch.cat([built_physical, semantic, socioeconomic_feature, road_feature], dim=1)
        )
        built_logits = self.built_head(built_feature)
        if self.use_aoi:
            built_logits = built_logits + F.softplus(self.aoi_prior_strength) * aoi
        if self.use_roads:
            road_prior = torch.zeros_like(built_logits)
            road_prior[:, 4] = roads[:, 9]
            built_logits = built_logits + F.softplus(self.road_transport_prior_strength) * road_prior
        coarse_logits, built_logits, natural_logits = self._mask_excluded_logits(
            coarse_logits, built_logits, natural_logits
        )
        leaf_log_probs = self._route(coarse_logits, built_logits, natural_logits)
        result = {
            "coarse_logits": coarse_logits,
            "built_logits": built_logits,
            "natural_logits": natural_logits,
            "leaf_log_probs": leaf_log_probs,
            "sar_gates": torch.stack([
                torch.sigmoid(self.coarse_fusion.sar_gate_logit),
                torch.sigmoid(self.natural_fusion.sar_gate_logit),
                torch.sigmoid(self.built_physical_fusion.sar_gate_logit),
            ]),
        }
        if self.official_projection is not None:
            hierarchical_official = self._aggregate_official(leaf_log_probs)
            direct_feature = self.official_fusion(
                torch.cat([coarse_feature, natural_feature, built_feature], dim=1)
            )
            direct_official = F.log_softmax(self.official_head(direct_feature), dim=1)
            direct_gate = torch.sigmoid(self.official_direct_gate_logit)
            log_direct_gate = torch.log(direct_gate.clamp_min(1e-6))[None, :, None, None]
            log_hierarchical_gate = torch.log(
                (1.0 - direct_gate).clamp_min(1e-6)
            )[None, :, None, None]
            combined = torch.logaddexp(
                hierarchical_official + log_hierarchical_gate,
                direct_official + log_direct_gate,
            )
            # Class-wise mixing weights do not necessarily sum the mixture to
            # exactly one across classes, so renormalise before loss/inference.
            combined = combined - torch.logsumexp(combined, dim=1, keepdim=True)
            result.update(
                {
                    "official8_log_probs": combined,
                    "hierarchical_official8_log_probs": hierarchical_official,
                    "direct_official8_log_probs": direct_official,
                    "official8_direct_gates": direct_gate,
                }
            )
        if self.separate_optical_indices:
            result["index_gates"] = torch.stack([
                torch.sigmoid(self.coarse_fusion.index_gate_logit),
                torch.sigmoid(self.natural_fusion.index_gate_logit),
                torch.sigmoid(self.built_physical_fusion.index_gate_logit),
            ])
        return result
