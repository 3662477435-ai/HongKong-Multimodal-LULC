from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import Swin_V2_T_Weights, swin_v2_t

from .model import TextEncoder


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        padding = kernel_size // 2
        # Keep at least two channels per group.  A one-channel group fails for
        # 1x1 pyramid-pooled features when the final validation batch has N=1.
        groups = min(16, max(1, out_channels // 2))
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )


class PyramidPooling(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        branch_channels = out_channels // 4
        self.scales = (1, 2, 3, 6)
        self.branches = nn.ModuleList(
            [nn.Sequential(nn.AdaptiveAvgPool2d(scale), ConvGNAct(in_channels, branch_channels, 1)) for scale in self.scales]
        )
        self.fuse = ConvGNAct(in_channels + branch_channels * len(self.scales), out_channels, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        pooled = [F.interpolate(branch(x), size=size, mode="bilinear", align_corners=False) for branch in self.branches]
        return self.fuse(torch.cat([x, *pooled], dim=1))


class SwinUPerNetFusion(nn.Module):
    """Pretrained Swin-V2 + UPerNet with dense OSM and OSM-text conditioning."""

    def __init__(
        self,
        vocab_size: int,
        num_classes: int = 10,
        text_dim: int = 128,
        text_heads: int = 4,
        text_layers: int = 2,
        max_tokens: int = 96,
        fpn_channels: int = 128,
        osm_channels: int = 20,
        dropout: float = 0.1,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        weights = Swin_V2_T_Weights.DEFAULT if pretrained else None
        swin = swin_v2_t(weights=weights)
        self.backbone = swin.features
        self.spectral_adapter = nn.Conv2d(8, 3, 1, bias=True)
        with torch.no_grad():
            self.spectral_adapter.weight.zero_()
            self.spectral_adapter.bias.zero_()
            # Input order is Blue, Green, Red, NIR, NDVI, VV, VH, VV-VH.
            self.spectral_adapter.weight[0, 2, 0, 0] = 1.0
            self.spectral_adapter.weight[1, 1, 0, 0] = 1.0
            self.spectral_adapter.weight[2, 0, 0, 0] = 1.0

        stage_channels = (96, 192, 384, 768)
        self.laterals = nn.ModuleList([ConvGNAct(ch, fpn_channels, 1) for ch in stage_channels[:3]])
        self.ppm = PyramidPooling(stage_channels[3], fpn_channels)
        self.fpn_smooth = nn.ModuleList([ConvGNAct(fpn_channels, fpn_channels, 3) for _ in range(3)])
        self.osm_encoder = nn.Sequential(
            ConvGNAct(osm_channels, 48, 3, stride=2),
            ConvGNAct(48, fpn_channels, 3, stride=2),
            ConvGNAct(fpn_channels, fpn_channels, 3),
        )
        self.text_encoder = TextEncoder(vocab_size, text_dim, text_heads, text_layers, max_tokens, dropout)
        self.text_film = nn.Linear(text_dim, fpn_channels * 2)
        self.fusion = nn.Sequential(
            ConvGNAct(fpn_channels * 5, fpn_channels * 2, 3),
            nn.Dropout2d(dropout),
            ConvGNAct(fpn_channels * 2, fpn_channels, 3),
        )
        self.detail_encoder = nn.Sequential(
            ConvGNAct(8 + osm_channels, 48, 3),
            ConvGNAct(48, 48, 3),
        )
        self.refine = nn.Sequential(
            ConvGNAct(fpn_channels + 48, 96, 3),
            nn.Dropout2d(dropout),
            ConvGNAct(96, 64, 3),
        )
        self.classifier = nn.Conv2d(64, num_classes, 1)
        self.aoi_prior_strength = nn.Parameter(torch.tensor(2.0))

    @staticmethod
    def _nchw(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        s1: torch.Tensor,
        s2: torch.Tensor,
        input_ids: torch.Tensor,
        text_mask: torch.Tensor,
        osm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_size = s1.shape[-2:]
        image_all = torch.cat([s2, s1], dim=1)
        x = self.spectral_adapter(image_all)
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
        p3 = self.laterals[2](c3) + F.interpolate(p4, size=c3.shape[-2:], mode="bilinear", align_corners=False)
        p3 = self.fpn_smooth[2](p3)
        p2 = self.laterals[1](c2) + F.interpolate(p3, size=c2.shape[-2:], mode="bilinear", align_corners=False)
        p2 = self.fpn_smooth[1](p2)
        p1 = self.laterals[0](c1) + F.interpolate(p2, size=c1.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.fpn_smooth[0](p1)
        target_size = p1.shape[-2:]
        pyramid = [p1] + [F.interpolate(p, size=target_size, mode="bilinear", align_corners=False) for p in (p2, p3, p4)]
        if osm is None:
            osm_feature = torch.zeros_like(p1)
            osm_full = torch.zeros(
                image_all.shape[0], 20, *input_size, device=image_all.device, dtype=image_all.dtype
            )
        else:
            osm_feature = self.osm_encoder(osm)
            osm_feature = F.interpolate(osm_feature, size=target_size, mode="bilinear", align_corners=False)
            osm_full = osm
        fused = self.fusion(torch.cat([*pyramid, osm_feature], dim=1))
        text = self.text_encoder(input_ids, text_mask)
        gamma, beta = self.text_film(text).chunk(2, dim=1)
        fused = fused * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        fused = F.interpolate(fused, size=input_size, mode="bilinear", align_corners=False)
        detail = self.detail_encoder(torch.cat([image_all, osm_full], dim=1))
        logits = self.classifier(self.refine(torch.cat([fused, detail], dim=1)))
        # AOI bands are the final six OSM channels and map directly to train classes 0..5.
        prior_logits = torch.zeros_like(logits)
        prior_logits[:, :6] = osm_full[:, -6:]
        return logits + F.softplus(self.aoi_prior_strength) * prior_logits
